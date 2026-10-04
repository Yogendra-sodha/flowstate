"""Shared immutable publication and verified restore for object-storage providers."""

from __future__ import annotations

import errno
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol

from flowstate.lake import Lake, _rename_with_retry, _sha256

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class ObjectStore(Protocol):
    def read_optional(self, key: str) -> bytes | None: ...

    def put_immutable(self, key: str, source: Path | bytes, digest: str) -> bool: ...

    def download(self, key: str, destination: Path) -> str:
        """Stream an object to disk and return its SHA-256."""
        ...


def _prefix(prefix: str) -> str:
    prefix = prefix.strip("/")
    if "\\" in prefix or any(piece in (".", "..", "") for piece in prefix.split("/") if prefix):
        raise ValueError("Object prefix must contain ordinary slash-separated path components")
    return prefix


def _key(prefix: str, *parts: str) -> str:
    return "/".join(part for part in (prefix, *parts) if part)


def _manifest(data: bytes) -> dict[str, str]:
    try:
        document = json.loads(data)
        if not isinstance(document, dict):
            raise ValueError("Invalid experiment manifest")
        artifacts = document["artifacts"]
        if (
            document.get("version") != 1
            or document.get("algorithm") != "sha256"
            or not isinstance(artifacts, dict)
            or not {"record.json", "metadata.parquet"}.issubset(artifacts)
        ):
            raise ValueError("Invalid experiment manifest")
        for relative, digest in artifacts.items():
            path = PurePosixPath(relative)
            if (
                not relative
                or path.is_absolute()
                or any(piece in (".", "..", "") for piece in relative.split("/"))
                or "\\" in relative
                or ":" in relative
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise ValueError("Unsafe path or invalid hash in experiment manifest")
        return artifacts
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Invalid experiment manifest") from error


def upload(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str,
    store_factory: Callable[[], ObjectStore],
) -> dict:
    """Verify the source, upload content, then publish the completion manifest."""
    lake = Lake(lake_root)
    if problems := lake.verify(experiment_id):
        raise ValueError(f"Source experiment failed verification: {problems}")
    directory = lake.experiments / experiment_id
    manifest_data = (directory / "manifest.json").read_bytes()
    artifacts = _manifest(manifest_data)
    manifest_hash = hashlib.sha256(manifest_data).hexdigest()
    prefix = _prefix(prefix)
    marker = _key(prefix, "experiments", experiment_id, "manifest.json")
    content = _key(prefix, "artifacts", experiment_id, manifest_hash)
    store = store_factory()
    existing = store.read_optional(marker)
    if existing is not None and existing != manifest_data:
        raise FileExistsError(
            f"Remote experiment has a different committed manifest: {experiment_id}"
        )
    uploaded = reused = 0
    # Snapshot one artifact at a time: a local edit must not poison a create-only
    # remote key between preflight verification and the SDK reading the source.
    with tempfile.TemporaryDirectory(prefix=".staging-upload-", dir=lake.root) as temporary:
        snapshot = Path(temporary) / "artifact"
        for relative, digest in sorted(artifacts.items()):
            shutil.copyfile(directory / relative, snapshot)
            if _sha256(snapshot) != digest:
                raise ValueError(f"Source artifact changed during upload: {relative}")
            if store.put_immutable(_key(content, relative), snapshot, digest):
                uploaded += 1
            else:
                reused += 1
    # Readers treat this marker as the only indication that all artifacts were uploaded.
    committed_now = store.put_immutable(marker, manifest_data, manifest_hash)
    return {
        "id": experiment_id,
        "bucket": bucket,
        "prefix": prefix,
        "manifest_key": marker,
        "manifest_sha256": manifest_hash,
        "uploaded_objects": uploaded,
        "reused_objects": reused,
        "committed": True,
        "resumed": not committed_now,
    }


def download(
    lake_root: str | Path,
    experiment_id: str,
    prefix: str,
    store_factory: Callable[[], ObjectStore],
) -> dict:
    """Restore a committed mirror into staging, verify, then publish locally."""
    lake = Lake(lake_root)
    lake.exists(experiment_id)  # Validate the ID before composing paths or requesting credentials.
    prefix = _prefix(prefix)
    marker = _key(prefix, "experiments", experiment_id, "manifest.json")
    store = store_factory()
    manifest_data = store.read_optional(marker)
    if manifest_data is None:
        raise FileNotFoundError(f"No committed remote experiment: {experiment_id}")
    artifacts = _manifest(manifest_data)
    manifest_hash = hashlib.sha256(manifest_data).hexdigest()
    destination = lake.experiments / experiment_id
    if lake.exists(experiment_id):
        if lake.verify(experiment_id):
            raise ValueError("Existing local experiment failed verification")
        if _sha256(destination / "manifest.json") != manifest_hash:
            raise FileExistsError("Existing local experiment has a different manifest")
        return {"id": experiment_id, "path": str(destination), "resumed": True}
    staging = Path(
        tempfile.mkdtemp(prefix=f".staging-download-{experiment_id}-", dir=lake.experiments)
    )
    try:
        staged_lake = Lake(staging)
        candidate = staged_lake.experiments / experiment_id
        candidate.mkdir()
        content = _key(prefix, "artifacts", experiment_id, manifest_hash)
        for relative, expected in sorted(artifacts.items()):
            local = candidate / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            if store.download(_key(content, relative), local) != expected:
                raise ValueError(f"Downloaded artifact hash mismatch: {relative}")
        (candidate / "manifest.json").write_bytes(manifest_data)
        if problems := staged_lake.verify(experiment_id):
            raise ValueError(f"Downloaded experiment failed verification: {problems}")
        record = staged_lake.load_record(experiment_id)
        if not isinstance(record, dict) or record.get("id") != experiment_id:
            raise ValueError("Downloaded record ID does not match requested experiment")
        try:
            _rename_with_retry(candidate, destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if (
                lake.verify(experiment_id)
                or _sha256(destination / "manifest.json") != manifest_hash
            ):
                raise FileExistsError(
                    "Concurrent local publication has different content"
                ) from error
            return {"id": experiment_id, "path": str(destination), "resumed": True}
        return {"id": experiment_id, "path": str(destination), "resumed": False}
    finally:
        if staging.resolve().is_relative_to(lake.experiments.resolve()):
            shutil.rmtree(staging, ignore_errors=True)
