"""Checksum-verified immutable experiment mirrors over the S3 object protocol."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.parse import urlparse

from flowstate import artifact_mirror

_SINGLE_PUT_LIMIT = 5 * 1024**3


def _client(endpoint_url: str | None):
    import boto3
    from botocore.config import Config

    if endpoint_url is not None:
        endpoint = urlparse(endpoint_url)
        if (
            endpoint.scheme not in ("http", "https")
            or not endpoint.netloc
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "endpoint_url must be an HTTP(S) endpoint without credentials or query"
            )
    # Standard SDK credential chain; neither credentials nor environment are persisted.
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )


def _read_optional(client, bucket: str, key: str) -> bytes | None:
    from botocore.exceptions import ClientError

    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    try:
        return body.read()
    finally:
        body.close()


def _remote_hash(client, bucket: str, key: str) -> str:
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    digest = hashlib.sha256()
    try:
        for block in body.iter_chunks(chunk_size=1024 * 1024):
            digest.update(block)
    finally:
        body.close()
    return digest.hexdigest()


def _put_immutable(client, bucket: str, key: str, source: Path | bytes, digest: str) -> bool:
    """Return True for a new object, False for an already identical object."""
    from botocore.exceptions import ClientError

    length = source.stat().st_size if isinstance(source, Path) else len(source)
    if length > _SINGLE_PUT_LIMIT:
        raise ValueError("One artifact exceeds the 5 GiB single-PUT limit; rechunk it first")
    for attempt in range(3):
        try:
            with source.open("rb") if isinstance(source, Path) else io.BytesIO(source) as stream:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=stream,
                    ContentLength=length,
                    Metadata={"sha256": digest},
                    IfNoneMatch="*",
                )
            return True
        except ClientError as error:
            code = error.response["Error"]["Code"]
            if code in ("ConditionalRequestConflict", "409") and attempt < 2:
                continue
            if code not in ("PreconditionFailed", "412"):
                raise
            # Metadata alone is not an integrity proof; hash the actual remote bytes.
            if _remote_hash(client, bucket, key) != digest:
                raise ValueError(f"Existing object has conflicting bytes: {key}") from error
            return False
    raise RuntimeError("Unreachable conditional write retry state")


class _S3Store:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def read_optional(self, key: str) -> bytes | None:
        return _read_optional(self.client, self.bucket, key)

    def put_immutable(self, key: str, source: Path | bytes, digest: str) -> bool:
        return _put_immutable(self.client, self.bucket, key, source, digest)

    def download(self, key: str, destination: Path) -> str:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        digest = hashlib.sha256()
        try:
            with destination.open("wb") as target:
                for block in body.iter_chunks(chunk_size=1024 * 1024):
                    digest.update(block)
                    target.write(block)
        finally:
            body.close()
        return digest.hexdigest()


def upload_experiment(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str = "",
    endpoint_url: str | None = None,
) -> dict:
    """Upload verified content with conditional writes and a manifest published last."""
    return artifact_mirror.upload(
        lake_root, experiment_id, bucket, prefix, lambda: _S3Store(_client(endpoint_url), bucket)
    )


def download_experiment(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str = "",
    endpoint_url: str | None = None,
) -> dict:
    """Restore a committed mirror and verify all hashes before local publication."""
    return artifact_mirror.download(
        lake_root, experiment_id, prefix, lambda: _S3Store(_client(endpoint_url), bucket)
    )
