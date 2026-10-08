"""HTTP range integrity and budgets, using offline urllib-compatible responses."""

import hashlib
import io
import json
import re
from email.message import Message

import pytest

import flowstate.http_ranges as ranges_module
from flowstate.http_ranges import HTTPRangeReader

SOURCE_URL = "https://example.org/public/burgers.hdf5"
REDIRECT_URL = "https://download.example.org/object?temporary-token=do-not-persist"
HDF5_BYTES = b"\x89HDF\r\n\x1a\n" + bytes(range(248))
ETAG = '"stable-object-version"'


class FakeResponse:
    def __init__(self, body, headers, *, status=206, url=REDIRECT_URL):
        self.status = status
        self.headers = headers
        self.body = io.BytesIO(body)
        self.url = url
        self.read_sizes = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        self.body.close()

    def geturl(self):
        return self.url

    def read(self, size=-1):
        self.read_sizes.append(size)
        return self.body.read(size)


class FakeServer:
    def __init__(self, monkeypatch, *, payload=HDF5_BYTES, size=None, mutate=None):
        self.payload = payload
        self.size = len(payload) if size is None else size
        self.mutate = mutate
        self.requests = []
        self.responses = []

        def offline_tls_context(*, cafile):
            assert cafile == ranges_module.certifi.where()
            return object()

        monkeypatch.setattr(ranges_module.urllib.request, "urlopen", self.urlopen)
        monkeypatch.setattr(ranges_module.ssl, "create_default_context", offline_tls_context)

    def urlopen(self, request, *, timeout, context):
        assert timeout == 30
        assert context is not None
        request_headers = {name.lower(): value for name, value in request.header_items()}
        assert request_headers["accept-encoding"] == "identity"
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request_headers["range"])
        assert match, "Only explicit bounded byte ranges are permitted"
        start, end = map(int, match.groups())
        assert 0 <= start <= end < self.size
        self.requests.append(
            {"url": request.full_url, "headers": request_headers, "start": start, "end": end}
        )
        headers = Message()
        headers["Content-Range"] = f"bytes {start}-{end}/{self.size}"
        headers["Content-Length"] = str(end - start + 1)
        headers["ETag"] = ETAG
        response = FakeResponse(self.payload[start : end + 1], headers)
        if self.mutate is not None:
            self.mutate(response, len(self.requests))
        self.responses.append(response)
        return response


def test_random_seek_readinto_and_cached_block_boundaries(monkeypatch):
    server = FakeServer(monkeypatch)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32) as reader:
        assert reader.readable() and reader.seekable()
        assert reader.tell() == 0
        assert reader.read(0) == b""
        assert len(server.requests) == 1  # Constructor probes exactly the signature.
        assert reader.seek(27) == 27
        destination = bytearray(13)
        assert reader.readinto(destination) == 13
        assert destination == HDF5_BYTES[27:40]
        assert reader.tell() == 40
        assert [(item["start"], item["end"]) for item in server.requests] == [
            (0, 7),
            (0, 31),
            (32, 63),
        ]
        assert reader.seek(-10, io.SEEK_CUR) == 30
        assert reader.read(20) == HDF5_BYTES[30:50]
        assert len(server.requests) == 3  # Both boundary blocks remain cached.
        assert reader.seek(-6, io.SEEK_END) == 250
        destination = bytearray(b"?" * 12)
        assert reader.readinto(destination) == 6
        assert destination == HDF5_BYTES[250:] + b"?" * 6
        assert reader.tell() == 256
        assert reader.read(2) == b""
        assert reader.readinto(bytearray(3)) == 0
        assert len(server.requests) == 4
    assert all(response.closed for response in server.responses)
    with pytest.raises(ValueError, match="closed"):
        reader.read(1)


def test_four_block_lru_reuses_recent_blocks_and_refetches_evicted_block(monkeypatch):
    server = FakeServer(monkeypatch)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32) as reader:
        for start in (0, 32, 64, 96):
            reader.seek(start)
            assert reader.read(1) == HDF5_BYTES[start : start + 1]
        reader.seek(0)
        assert reader.read(1) == HDF5_BYTES[:1]  # Refresh block zero's recency.
        assert len(server.requests) == 5
        reader.seek(128)
        assert reader.read(1) == HDF5_BYTES[128:129]  # Evicts block one, not zero.
        reader.seek(0)
        assert reader.read(1) == HDF5_BYTES[:1]
        assert len(server.requests) == 6
        reader.seek(32)
        assert reader.read(1) == HDF5_BYTES[32:33]
        assert len(server.requests) == 7


def test_final_short_block_and_full_small_read_are_exact(monkeypatch):
    payload = HDF5_BYTES[:250]
    server = FakeServer(monkeypatch, payload=payload)
    with HTTPRangeReader(SOURCE_URL, len(payload), block_size=32) as reader:
        assert reader.read() == payload
        assert server.requests[-1]["start"] == 224
        assert server.requests[-1]["end"] == 249
        assert reader.receipt()["bytes_received"] == 8 + len(payload)


def test_strong_if_match_and_range_hashes_are_recorded_without_redirect_url(monkeypatch):
    server = FakeServer(monkeypatch)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32) as reader:
        reader.seek(34)
        assert reader.read(4) == HDF5_BYTES[34:38]
        receipt = reader.receipt()
    assert "if-match" not in server.requests[0]["headers"]
    assert server.requests[1]["headers"]["if-match"] == ETAG
    assert server.requests[1]["url"] == REDIRECT_URL
    assert receipt["source_url"] == SOURCE_URL
    assert receipt["strong_etag"] == ETAG
    assert receipt["requests"] == 2
    assert receipt["bytes_received"] == 40
    assert receipt["full_remote_checksum_verified"] is False
    assert REDIRECT_URL not in json.dumps(receipt)
    assert "temporary-token" not in json.dumps(receipt)
    assert receipt["range_sha256"] == [
        {"start": 0, "end": 7, "sha256": hashlib.sha256(HDF5_BYTES[:8]).hexdigest()},
        {"start": 32, "end": 63, "sha256": hashlib.sha256(HDF5_BYTES[32:64]).hexdigest()},
    ]


@pytest.mark.parametrize("status", [200, 204, 301, 404, 416])
def test_server_must_return_206_without_reading_unbounded_fallback_body(monkeypatch, status):
    def mutate(response, request_number):
        response.status = status

    server = FakeServer(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError, match="206"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)
    assert len(server.requests) == 1
    assert server.responses[0].read_sizes == []
    assert server.responses[0].closed


@pytest.mark.parametrize(
    "content_range",
    ["bytes 1-8/256", "bytes 0-6/256", "bytes 0-7/257", "bytes 0-7/*", "garbage", None],
)
def test_exact_content_range_and_declared_source_size_are_required(monkeypatch, content_range):
    def mutate(response, request_number):
        del response.headers["Content-Range"]
        if content_range is not None:
            response.headers["Content-Range"] = content_range

    server = FakeServer(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError, match="Content-Range"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)
    assert server.responses[0].read_sizes == []


@pytest.mark.parametrize("etag", ['W/"version"', "unquoted", "", None])
def test_weak_or_missing_etags_are_rejected(monkeypatch, etag):
    def mutate(response, request_number):
        del response.headers["ETag"]
        if etag is not None:
            response.headers["ETag"] = etag

    server = FakeServer(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError, match="strong ETag"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)
    assert server.responses[0].read_sizes == []


def test_changed_etag_aborts_before_accepting_body(monkeypatch):
    def mutate(response, request_number):
        if request_number == 2:
            response.headers.replace_header("ETag", '"changed-version"')

    server = FakeServer(monkeypatch, mutate=mutate)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32) as reader:
        with pytest.raises(ValueError, match="changed during"):
            reader.read(1)
        assert reader.tell() == 0
        assert reader.receipt()["strong_etag"] == ETAG
        assert reader.receipt()["bytes_received"] == 8
        assert len(reader.receipt()["range_sha256"]) == 1
    assert server.responses[1].read_sizes == []
    assert server.requests[1]["headers"]["if-match"] == ETAG


@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate"])
def test_encoded_response_body_is_rejected_without_read(monkeypatch, encoding):
    def mutate(response, request_number):
        response.headers["Content-Encoding"] = encoding

    server = FakeServer(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError, match="Encoded"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)
    assert server.responses[0].read_sizes == []


@pytest.mark.parametrize("length", ["7", "9", None])
def test_exact_content_length_is_required(monkeypatch, length):
    def mutate(response, request_number):
        del response.headers["Content-Length"]
        if length is not None:
            response.headers["Content-Length"] = length

    server = FakeServer(monkeypatch, mutate=mutate)
    with pytest.raises(ValueError, match="Content-Length"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)
    assert server.responses[0].read_sizes == []


def test_truncated_range_counts_received_bytes_but_never_hashes_partial_data(monkeypatch):
    def mutate(response, request_number):
        if request_number == 2:
            response.body = io.BytesIO(HDF5_BYTES[:9])

    server = FakeServer(monkeypatch, mutate=mutate)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32) as reader:
        with pytest.raises(ValueError, match="Truncated"):
            reader.read(1)
        assert reader.tell() == 0
        assert reader.receipt()["bytes_received"] == 17
        assert len(reader.receipt()["range_sha256"]) == 1
    assert server.responses[1].read_sizes == [32]


@pytest.mark.parametrize("budget", [{"max_requests": 1}, {"max_bytes": 8}])
def test_transfer_caps_stop_before_another_network_request(monkeypatch, budget):
    server = FakeServer(monkeypatch)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32, **budget) as reader:
        with pytest.raises(ValueError, match="budget exhausted"):
            reader.read(1)
        assert reader.tell() == 0
        assert reader.receipt()["requests"] == 1
        assert reader.receipt()["bytes_received"] == 8
    assert len(server.requests) == 1


def test_budget_smaller_than_signature_prevents_any_network_request(monkeypatch):
    server = FakeServer(monkeypatch)
    with pytest.raises(ValueError, match="budget exhausted"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), max_bytes=7, block_size=32)
    assert server.requests == []


def test_single_huge_read_fails_before_fetching_blocks(monkeypatch):
    size = 32 * 1024**2
    server = FakeServer(monkeypatch, size=size)
    with HTTPRangeReader(SOURCE_URL, size, block_size=32) as reader:
        with pytest.raises(ValueError, match="16 MiB"):
            reader.read(16 * 1024**2 + 1)
        with pytest.raises(ValueError, match="16 MiB"):
            reader.read()
        assert reader.tell() == 0
    assert len(server.requests) == 1


def test_non_hdf5_signature_and_non_https_redirect_are_rejected(monkeypatch):
    FakeServer(monkeypatch, payload=b"NOT-HDF5" + HDF5_BYTES[8:])
    with pytest.raises(ValueError, match="signature"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)

    def redirect_http(response, request_number):
        response.url = "http://example.org/insecure-object"

    server = FakeServer(monkeypatch, mutate=redirect_http)
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32)
    assert server.responses[0].read_sizes == []


@pytest.mark.parametrize("offset, whence", [(-1, 0), (257, 0), (1, 2), (0, 7)])
def test_seek_rejects_invalid_offsets_without_network_io(monkeypatch, offset, whence):
    server = FakeServer(monkeypatch)
    with HTTPRangeReader(SOURCE_URL, len(HDF5_BYTES), block_size=32) as reader:
        with pytest.raises(ValueError, match="Seek|seek"):
            reader.seek(offset, whence)
        assert reader.tell() == 0
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"expected_size": True},
        {"expected_size": 0},
        {"max_requests": 0},
        {"max_requests": 513},
        {"max_bytes": 256 * 1024**2 + 1},
        {"block_size": 4 * 1024**2 + 1},
        {"block_size": 0},
        {"block_size": 32.0},
    ],
)
def test_invalid_constructor_budgets_fail_without_network_io(monkeypatch, options):
    server = FakeServer(monkeypatch)
    arguments = {"expected_size": len(HDF5_BYTES), "block_size": 32, **options}
    with pytest.raises(ValueError):
        HTTPRangeReader(SOURCE_URL, **arguments)
    assert server.requests == []


@pytest.mark.parametrize("url", ["http://example.org/file", "file:///tmp/file", "https:///file"])
def test_source_url_must_be_public_https_shape(monkeypatch, url):
    server = FakeServer(monkeypatch)
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPRangeReader(url, len(HDF5_BYTES))
    assert server.requests == []
