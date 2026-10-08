"""Bounded, read-only HTTP ranges for HDF5; never falls back to a whole-file GET."""

from __future__ import annotations

import hashlib
import io
import re
import ssl
import urllib.request
from collections import OrderedDict
from urllib.parse import urlparse

import certifi


class HTTPRangeReader(io.RawIOBase):
    """Seekable remote file with a small LRU cache and a stable strong ETag.

    Range hashes describe bytes actually transferred, NOT a full-file checksum.
    A public redirect URL is held only in memory, never written to provenance.
    """

    def __init__(
        self,
        url: str,
        expected_size: int,
        *,
        max_bytes: int = 64 * 1024**2,
        max_requests: int = 128,
        block_size: int = 1024**2,
    ):
        super().__init__()
        if urlparse(url).scheme != "https" or not urlparse(url).netloc:
            raise ValueError("Public data URL must use HTTPS")
        for name, value in (
            ("expected_size", expected_size),
            ("max_bytes", max_bytes),
            ("max_requests", max_requests),
            ("block_size", block_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_bytes > 256 * 1024**2 or max_requests > 512 or block_size > 4 * 1024**2:
            raise ValueError("HTTP range budget exceeds the local acquisition limits")
        self.url = url
        self._download_url = url
        self.size = expected_size
        self.max_bytes, self.max_requests, self.block_size = max_bytes, max_requests, block_size
        self.bytes_received = self.requests = self.position = 0
        self.etag: str | None = None
        self.ranges: list[dict] = []
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._context = ssl.create_default_context(cafile=certifi.where())
        signature = self._fetch(0, 7)
        if signature != b"\x89HDF\r\n\x1a\n":
            raise ValueError("Remote source does not have the expected HDF5 signature")

    def _fetch(self, start: int, end: int) -> bytes:
        count = end - start + 1
        if self.requests >= self.max_requests or self.bytes_received + count > self.max_bytes:
            raise ValueError("HTTP range transfer budget exhausted")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag is not None:
            headers["If-Match"] = self.etag
        request = urllib.request.Request(self._download_url, headers=headers)
        self.requests += 1
        with urllib.request.urlopen(request, timeout=30, context=self._context) as response:
            if response.status != 206:
                raise ValueError("Server must honor ranges with HTTP 206; full download refused")
            content_range = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not content_range or tuple(map(int, content_range.groups())) != (
                start,
                end,
                self.size,
            ):
                raise ValueError("Unexpected Content-Range or remote source size")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Encoded HTTP ranges are not supported")
            if response.headers.get("Content-Length") != str(count):
                raise ValueError("Unexpected range Content-Length")
            etag = response.headers.get("ETag", "")
            if not (etag.startswith('"') and etag.endswith('"')):
                raise ValueError("A strong ETag is required for consistent remote reads")
            if self.etag is not None and etag != self.etag:
                raise ValueError("Remote source changed during acquisition")
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Remote data redirected away from HTTPS")
            self.etag = etag
            self._download_url = response.geturl()
            data = response.read(count)
            self.bytes_received += len(data)
            if len(data) != count:
                raise ValueError("Truncated HTTP range response")
        self.ranges.append({"start": start, "end": end, "sha256": hashlib.sha256(data).hexdigest()})
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= position <= self.size:
            raise ValueError("Seek outside remote file")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size > 16 * 1024**2:
            raise ValueError("Single remote read exceeds 16 MiB")
        parts = []
        remaining = size
        while remaining:
            block = self.position // self.block_size
            if block not in self._cache:
                start = block * self.block_size
                self._cache[block] = self._fetch(start, min(start + self.block_size, self.size) - 1)
                if len(self._cache) > 4:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(block)
            offset = self.position % self.block_size
            piece = self._cache[block][offset : offset + remaining]
            parts.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def receipt(self) -> dict:
        return {
            "source_url": self.url,
            "source_size_bytes": self.size,
            "strong_etag": self.etag,
            "requests": self.requests,
            "bytes_received": self.bytes_received,
            "max_bytes": self.max_bytes,
            "max_requests": self.max_requests,
            "range_sha256": [item.copy() for item in self.ranges],
            "full_remote_checksum_verified": False,
        }

    def close(self) -> None:
        if hasattr(self, "_cache"):
            self._cache.clear()
        super().close()
