import importlib.util
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from flowstate.dashboard import export_dashboard
from flowstate.lake import Lake

_SERVER_PATH = Path(__file__).resolve().parents[1] / "containers" / "viewer" / "server.py"
_SPEC = importlib.util.spec_from_file_location("flowstate_viewer_server", _SERVER_PATH)
viewer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(viewer)

HTML = b"<!doctype html><html><body>Immutable research snapshot</body></html>"


@pytest.fixture
def running_viewer(tmp_path):
    snapshot = tmp_path / "index.html"
    snapshot.write_bytes(HTML)
    (tmp_path / "secret.txt").write_text("DO NOT SERVE", encoding="utf-8")
    server = viewer.create_server(viewer.load_snapshot(snapshot), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address, snapshot
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(address, method, path):
    connection = HTTPConnection(*address, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_routes_headers_and_preloaded_immutable_snapshot(running_viewer):
    address, snapshot = running_viewer
    snapshot.write_bytes(b"<!doctype html>Changed file must not be served")
    for path in ("/", "/index.html", "/?filter=burgers", "/index.html?filter=burgers"):
        status, headers, body = request(address, "GET", path)
        assert status == 200
        assert body == HTML
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert int(headers["Content-Length"]) == len(HTML)
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert "connect-src 'none'" in headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert "script-src 'unsafe-inline'" in headers["Content-Security-Policy"]
    status, headers, body = request(address, "HEAD", "/")
    assert status == 200 and body == b""
    assert int(headers["Content-Length"]) == len(HTML)
    assert request(address, "GET", "/healthz")[::2] == (200, b"ok\n")
    assert request(address, "HEAD", "/healthz")[::2] == (200, b"")


@pytest.mark.parametrize(
    "path",
    [
        "/secret.txt", "/../secret.txt", "/%2e%2e/secret.txt", "/index.html/../secret.txt",
        "/app/server.py", "/snapshot/", "/healthz/", "/index.html%00", "/%2findex.html",
        "/..\\secret.txt", "http://example.com/index.html",
    ],
)
def test_unknown_and_traversal_paths_do_not_expose_files(running_viewer, path):
    status, _, body = request(running_viewer[0], "GET", path)
    assert status == 404
    assert body == b"Not found\n"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"])
def test_write_methods_are_rejected(running_viewer, method):
    status, headers, body = request(running_viewer[0], method, "/")
    assert status == 405
    assert headers["Allow"] == "GET, HEAD"
    assert body == b"Method not allowed\n"


@pytest.mark.parametrize("content", [b"", b"not html", b"<!doctype html>\xff"])
def test_invalid_snapshots_fail_closed(tmp_path, content):
    path = tmp_path / "snapshot.html"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        viewer.load_snapshot(path)


def test_oversized_snapshot_missing_file_and_directory_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "MAX_HTML_BYTES", 20)
    path = tmp_path / "snapshot.html"
    path.write_bytes(HTML)
    for item in (path, tmp_path, tmp_path / "missing.html"):
        with pytest.raises(ValueError):
            viewer.load_snapshot(item)


def test_real_dashboard_export_is_served_unchanged(tmp_path):
    lake = tmp_path / "lake"
    Lake(lake)
    snapshot = tmp_path / "research.html"
    export_dashboard(lake, snapshot)
    content = viewer.load_snapshot(snapshot)
    assert b'id="snapshot" type="application/json"' in content
    assert b"__PAYLOAD__" not in content
    server = viewer.create_server(content, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = request(server.server_address, "GET", "/")
        assert status == 200
        assert body == snapshot.read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("value", ["0", "80", "65536", "-1", "oops", "8080.0", "８０８０"])
def test_invalid_ports_fail_before_listening(value):
    with pytest.raises(ValueError):
        viewer.configured_port(value)


def test_startup_uses_environment_and_hides_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("FLOWSTATE_VIEWER_HTML", str(tmp_path / "missing-private-path.html"))
    assert viewer.main() == 1
    assert "missing-private-path" not in capsys.readouterr().err
    monkeypatch.setenv("PORT", "invalid")
    assert viewer.main() == 1


def test_environment_port_is_used_for_startup(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.html"
    path.write_bytes(HTML)
    monkeypatch.setenv("FLOWSTATE_VIEWER_HTML", str(path))
    monkeypatch.setenv("PORT", "9090")
    seen = {}

    class FakeServer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def serve_forever(self):
            raise KeyboardInterrupt

    def create(snapshot, *, port):
        seen.update(snapshot=snapshot, port=port)
        return FakeServer()

    monkeypatch.setattr(viewer, "create_server", create)
    assert viewer.main() == 0
    assert seen == {"snapshot": HTML, "port": 9090}
