"""Serve one immutable, preloaded Flowstate HTML export using the standard library."""

from __future__ import annotations

import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_HTML_BYTES = 40 * 1024**2
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def load_snapshot(path: Path) -> bytes:
    """Read a regular UTF-8 HTML file at startup; never open request-derived paths."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Snapshot must be a regular file, not a symbolic link")
    with path.open("rb") as stream:
        content = stream.read(MAX_HTML_BYTES + 1)
    if not content or len(content) > MAX_HTML_BYTES:
        raise ValueError("Snapshot must be nonempty and within the HTML size limit")
    text = content.decode("utf-8")
    if not text.lstrip().lower().startswith("<!doctype html>"):
        raise ValueError("Snapshot must be an exported HTML document")
    return content


def configured_port(value: str) -> int:
    """Use an unprivileged port for local containers and a platform-provided PORT."""
    if not value.isascii() or not value.isdecimal() or not 1024 <= int(value) <= 65535:
        raise ValueError("PORT must be an integer from 1024 through 65535")
    return int(value)


def create_server(snapshot: bytes, *, host: str = "0.0.0.0", port: int = 8080):
    """Create a server; callers own its lifecycle, including test-only ephemeral ports."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "FlowstateSnapshot"
        sys_version = ""

        def _respond(self, status: int, content: bytes, media_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Connection", "close")
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)
            self.close_connection = True

        def do_GET(self) -> None:
            # Exact route matching deliberately does not decode or normalize paths.
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                self._respond(200, snapshot, "text/html; charset=utf-8")
            elif route == "/healthz":
                self._respond(200, b"ok\n", "text/plain; charset=utf-8")
            else:
                self._respond(404, b"Not found\n", "text/plain; charset=utf-8")

        do_HEAD = do_GET

        def _unsupported(self) -> None:
            self._respond(405, b"Method not allowed\n", "text/plain; charset=utf-8")

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = _unsupported
        do_CONNECT = _unsupported

        def setup(self) -> None:
            self.request.settimeout(10)
            super().setup()

        def log_message(self, format: str, *args) -> None:
            # Request URLs can contain sensitive query strings; omit access logs.
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> int:
    try:
        port = configured_port(os.environ.get("PORT", "8080"))
        snapshot = load_snapshot(
            Path(os.environ.get("FLOWSTATE_VIEWER_HTML", "/snapshot/index.html"))
        )
        server = create_server(snapshot, port=port)
    except (OSError, ValueError):
        print("Viewer startup failed: check PORT and the readable HTML snapshot.", file=sys.stderr)
        return 1
    print(f"Flowstate snapshot ready on port {port}", flush=True)
    try:
        with server:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
