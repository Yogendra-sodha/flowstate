"""Check snapshot isolation and cleanup without Docker or cloud access."""

import importlib.util
import json
import os
import stat
import subprocess
import urllib.error
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "containers" / "viewer" / "smoke.py"
_SPEC = importlib.util.spec_from_file_location("flowstate_viewer_smoke", _PATH)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)

HTML = b"<!doctype html><html><body>Reviewed snapshot</body></html>"


@pytest.mark.parametrize("fail_start", [False, True])
def test_readable_copy_preserves_private_source_and_cleans_up(tmp_path, monkeypatch, fail_start):
    original = tmp_path / "reviewed.html"
    original.write_bytes(HTML)
    original.chmod(0o600)
    original_mode = stat.S_IMODE(original.stat().st_mode)
    seen = {}
    calls = []

    def docker(*args):
        calls.append(args)
        if args[0] == "create":
            mount = args[args.index("--mount") + 1]
            source = mount.split("source=", 1)[1].split(",target=", 1)[0]
            staged = Path(source)
            seen["staged"] = staged
            assert staged != original and staged.parent != original.parent
            assert staged.read_bytes() == HTML
            assert "target=/snapshot/index.html,readonly" in mount
            if os.name == "posix":
                assert stat.S_IMODE(staged.stat().st_mode) == 0o644
                assert stat.S_IMODE(staged.parent.stat().st_mode) == 0o700
            return "owned-container"
        if args[0] == "start":
            if fail_start:
                raise subprocess.CalledProcessError(1, "docker start")
            return "owned-container"
        if args[0] == "inspect":
            return json.dumps([{
                "Image": "sha256:test-image",
                "NetworkSettings": {"Ports": {"8080/tcp": [
                    {"HostPort": "12345", "HostIp": "127.0.0.1"},
                ]}},
                "HostConfig": {"ReadonlyRootfs": True, "CapDrop": ["ALL"],
                               "SecurityOpt": ["no-new-privileges"]},
            }])
        if args[0] == "exec":
            return "65532"
        if args[0] == "rm":
            assert args[1] == "--force"
            assert args[2] == calls[0][calls[0].index("--name") + 1]
            # The staged mount must outlive its container, then be removed.
            assert seen["staged"].read_bytes() == HTML
            return "owned-container"
        raise AssertionError(f"Unexpected Docker call: {args}")

    class Response:
        status = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return self.content

    class Opener:
        def open(self, url, *, timeout):
            assert url.startswith("http://127.0.0.1:12345/") and timeout > 0
            if url.endswith("/healthz"):
                return Response(b"ok\n")
            if url.endswith("/server.py"):
                raise urllib.error.HTTPError(url, 404, "Not found", {}, None)
            return Response(seen["staged"].read_bytes())

    monkeypatch.setattr(smoke, "docker", docker)
    monkeypatch.setattr(smoke.urllib.request, "build_opener", lambda *args: Opener())
    if fail_start:
        with pytest.raises(subprocess.CalledProcessError):
            smoke.check("test-image", original, "test-base")
    else:
        receipt = smoke.check("test-image", original, "test-base")
        assert receipt["snapshot_byte_identical"] and receipt["uid"] == "65532"
    assert original.read_bytes() == HTML
    assert stat.S_IMODE(original.stat().st_mode) == original_mode
    assert calls[-1][0] == "rm"
    assert not seen["staged"].exists()
    assert not seen["staged"].parent.exists()


def test_staged_snapshot_is_an_independent_copy(tmp_path):
    original = tmp_path / "reviewed.html"
    original.write_bytes(HTML)
    with smoke.readable_snapshot(original.read_bytes()) as staged:
        assert not staged.samefile(original)
        original.write_bytes(b"<!doctype html>Later edit")
        assert staged.read_bytes() == HTML
    assert original.read_bytes() == b"<!doctype html>Later edit"
    assert not staged.exists() and not staged.parent.exists()
