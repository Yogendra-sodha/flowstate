"""Check an already built container against an existing reviewed HTML snapshot.

No cloud calls. The temporary container serves only the passed file and is
stopped/removed afterward; the image and original snapshot remain available.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True,
                          text=True, timeout=45).stdout.strip()


@contextmanager
def readable_snapshot(content: bytes):
    """Expose only a reviewed copy to the container; preserve the private source."""
    directory = Path(tempfile.mkdtemp(prefix="flowstate-viewer-snapshot-")).resolve()
    snapshot = directory / "index.html"
    try:
        with snapshot.open("xb") as stream:
            stream.write(content)
        # The private temporary parent remains mode 0700 on POSIX. Docker mounts
        # only this file, which UID 65532 must be able to read inside the container.
        snapshot.chmod(0o644)
        yield snapshot
    finally:
        # Only our generated file and now-empty directory are removed.
        snapshot.unlink(missing_ok=True)
        directory.rmdir()


def check(image: str, snapshot: Path, base_image: str) -> dict:
    snapshot = snapshot.resolve(strict=True)
    expected = snapshot.read_bytes()
    with readable_snapshot(expected) as staged:
        return check_container(image, staged, expected, base_image)


def check_container(image: str, snapshot: Path, expected: bytes, base_image: str) -> dict:
    name = "flowstate-viewer-check-" + uuid.uuid4().hex
    created = False
    try:
        container = docker(
            "create", "--name", name, "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--publish", "127.0.0.1::8080",
            "--mount", f"type=bind,source={snapshot},target=/snapshot/index.html,readonly", image,
        )
        created = True
        docker("start", container)
        details = json.loads(docker("inspect", container))[0]
        port = details["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"]
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 20
        # Ignore configured corporate proxies for this strictly loopback check.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            try:
                with opener.open(url + "/healthz", timeout=2) as response:
                    if response.status == 200 and response.read() == b"ok\n":
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Container did not become ready")
            time.sleep(0.1)
        with opener.open(url + "/", timeout=5) as response:
            actual = response.read()
            assert actual == expected, "Container snapshot bytes differ"
            assert response.headers["Content-Type"] == "text/html; charset=utf-8"
        user = docker("exec", container, "id", "-u")
        assert user == "65532", "Viewer must run as the configured unprivileged user"
        try:
            opener.open(url + "/server.py", timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("Container unexpectedly exposed a file route")
        return {
            "status": "completed", "image_id": details["Image"], "image": image,
            "base_image": base_image,
            "snapshot_sha256": hashlib.sha256(actual).hexdigest(), "snapshot_bytes": len(actual),
            "snapshot_byte_identical": True, "healthcheck_passed": True,
            "uid": user, "read_only_root": details["HostConfig"]["ReadonlyRootfs"],
            "capabilities_dropped": details["HostConfig"]["CapDrop"],
            "security_options": details["HostConfig"]["SecurityOpt"],
            "host_binding": details["NetworkSettings"]["Ports"]["8080/tcp"],
        }
    finally:
        if created:
            docker("rm", "--force", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--image", default="flowstate-viewer:local")
    parser.add_argument("--base-image", required=True, help="Resolved build base digest")
    args = parser.parse_args()
    print(json.dumps(check(args.image, args.snapshot, args.base_image), indent=2))
