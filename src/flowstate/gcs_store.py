"""Native Google Cloud Storage mirrors using Application Default Credentials."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO

from flowstate import artifact_mirror

_CHUNK_SIZE = 1024 * 1024


def _client(project: str | None):
    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError("Install GCS support with: uv sync --extra gcs") from error
    # ADC supports local impersonation and attached service accounts on Google Cloud.
    # Credentials remain outside the lake and never become experiment provenance.
    return storage.Client(project=project)


def _read_chunks(blob, target: BinaryIO | None = None) -> str:
    # Pin a generation so ranged reads cannot mix different versions of an object.
    blob.reload(timeout=60)
    digest = hashlib.sha256()
    with blob.open(
        "rb",
        chunk_size=_CHUNK_SIZE,
        raw_download=True,
        if_generation_match=blob.generation,
        timeout=60,
    ) as source:
        for block in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(block)
            if target is not None:
                target.write(block)
    return digest.hexdigest()


class _GCSStore:
    def __init__(self, client, bucket: str):
        self.bucket = client.bucket(bucket)

    def read_optional(self, key: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(key).download_as_bytes(raw_download=True, timeout=60)
        except NotFound:
            return None

    def put_immutable(self, key: str, source: Path | bytes, digest: str) -> bool:
        from google.api_core.exceptions import PreconditionFailed
        from google.cloud.storage.retry import DEFAULT_RETRY_IF_GENERATION_SPECIFIED

        blob = self.bucket.blob(key)
        blob.metadata = {"sha256": digest}
        length = source.stat().st_size if isinstance(source, Path) else len(source)
        try:
            with source.open("rb") if isinstance(source, Path) else io.BytesIO(source) as stream:
                blob.upload_from_file(
                    stream,
                    size=length,
                    if_generation_match=0,
                    checksum="crc32c",
                    retry=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
                    timeout=60,
                )
            return True
        except PreconditionFailed as error:
            # Hash the stored bytes; user-controlled metadata is not an integrity check.
            if _read_chunks(blob) != digest:
                raise ValueError(f"Existing object has conflicting bytes: {key}") from error
            return False

    def download(self, key: str, destination: Path) -> str:
        with destination.open("wb") as target:
            return _read_chunks(self.bucket.blob(key), target)


def upload_experiment(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
) -> dict:
    """Publish immutable GCS content using create-only generation preconditions."""
    return artifact_mirror.upload(
        lake_root, experiment_id, bucket, prefix, lambda: _GCSStore(_client(project), bucket)
    )


def download_experiment(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
) -> dict:
    """Download and verify committed GCS artifacts before exposing them locally."""
    return artifact_mirror.download(
        lake_root, experiment_id, prefix, lambda: _GCSStore(_client(project), bucket)
    )
