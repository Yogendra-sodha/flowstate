"""Checksum-verified immutable experiment mirrors over the S3 object protocol."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from flowstate.lake import Lake, _rename_with_retry, _sha256

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
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


def _prefix(prefix: str) -> str:
    prefix = prefix.strip("/")
    if "\\" in prefix or any(piece in (".", "..", "") for piece in prefix.split("/") if prefix):
        raise ValueError("Object prefix must contain ordinary slash-separated path components")
    return prefix


def _key(prefix: str, *parts: str) -> str:
    return "/".join(part for part in (prefix, *parts) if part)


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


def _manifest(data: bytes) -> dict[str, str]:
    try:
        document = json.loads(data)
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


def upload_experiment(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str = "",
    endpoint_url: str | None = None,
) -> dict:
    """Upload immutable content objects, then conditionally publish the manifest.

    Repeating an interrupted upload verifies existing objects and sends missing
    ones. An existing commit with different content is never overwritten.
    """
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
    client = _client(endpoint_url)
    existing = _read_optional(client, bucket, marker)
    if existing is not None and existing != manifest_data:
        raise FileExistsError(
            f"Remote experiment has a different committed manifest: {experiment_id}"
        )
    uploaded = reused = 0
    for relative, digest in sorted(artifacts.items()):
        if _put_immutable(client, bucket, _key(content, relative), directory / relative, digest):
            uploaded += 1
        else:
            reused += 1
    # This small marker is the only signal that an upload is complete.
    committed_now = _put_immutable(client, bucket, marker, manifest_data, manifest_hash)
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


def download_experiment(
    lake_root: str | Path,
    experiment_id: str,
    bucket: str,
    prefix: str = "",
    endpoint_url: str | None = None,
) -> dict:
    """Download a committed mirror, verify all hashes, then atomically publish locally."""
    lake = Lake(lake_root)
    # Use public ID validation before composing object keys or filesystem paths.
    lake.exists(experiment_id)
    prefix = _prefix(prefix)
    marker = _key(prefix, "experiments", experiment_id, "manifest.json")
    client = _client(endpoint_url)
    manifest_data = _read_optional(client, bucket, marker)
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
            body = client.get_object(Bucket=bucket, Key=_key(content, relative))["Body"]
            digest = hashlib.sha256()
            try:
                with local.open("wb") as target:
                    for block in body.iter_chunks(chunk_size=1024 * 1024):
                        digest.update(block)
                        target.write(block)
            finally:
                body.close()
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded artifact hash mismatch: {relative}")
        (candidate / "manifest.json").write_bytes(manifest_data)
        if problems := staged_lake.verify(experiment_id):
            raise ValueError(f"Downloaded experiment failed verification: {problems}")
        if staged_lake.load_record(experiment_id).get("id") != experiment_id:
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
