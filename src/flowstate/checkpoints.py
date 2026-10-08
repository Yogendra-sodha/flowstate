"""Local immutable frame blobs and atomic solver-checkpoint commit markers.

Only committed markers are resumable. Pending files and unreferenced blobs are
retained after interruption; no shared mutable Zarr store is used by writers.
The guarantee is process-crash recovery, not power-loss or remote-FS durability.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
import zipfile
from pathlib import Path

import numpy as np

from flowstate.numerics import (
    SolverCheckpoint,
    _output_size,
    normalize_config,
    validate_checkpoint,
)

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MARKER = re.compile(r"step-([1-9][0-9]*)\.json\Z")
MAX_CHECKPOINTS = 256
MAX_FRAMES = 10_000
MAX_CHECKPOINT_BYTES = 4 * 1024**3


class CheckpointError(OSError):
    """Infrastructure/integrity failure, never a numerical anomaly."""


def validate_checkpoint_options(config: dict, checkpoint_every: int | None) -> None:
    if checkpoint_every is None:
        return
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if config["equation"] == "darcy2d":
        raise ValueError("Steady Darcy has no trajectory to checkpoint")
    if config["steps"] > np.iinfo(np.int64).max:
        raise ValueError("Checkpoint steps exceed int64 storage range")
    frames = 1 + config["steps"] // config["save_every"] + bool(
        config["steps"] % config["save_every"]
    )
    generations = (config["steps"] + checkpoint_every - 1) // checkpoint_every
    if generations > MAX_CHECKPOINTS or frames > MAX_FRAMES:
        raise ValueError("Checkpoint mode allows at most 256 generations and 10000 saved frames")
    state_bytes = config["grid_size"] * 8 if config["equation"] == "burgers1d" else (
        config["grid_size"] ** 2 * 16
    )
    # Upper bound for repeated diagnostic prefixes plus hash lists/NPY headers.
    estimate = _output_size(config) + generations * (state_bytes + frames * 128 + 65536)
    if estimate > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint history exceeds the 4 GiB uncompressed estimate")


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(provenance):
    return {k: v for k, v in provenance.items() if k != "git_dirty"}


def _plain(path: Path) -> None:
    if path.is_symlink():
        raise CheckpointError(f"Checkpoint paths must not be symbolic links: {path}")


def _publish(path: Path, content: bytes) -> None:
    """fsync a private file, then expose it with an exclusive atomic hard link."""
    _plain(path)
    pending = path.parent / f".pending-{uuid.uuid4().hex}"
    with pending.open("xb") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())
    try:
        os.link(pending, path)
    except FileExistsError:
        _plain(path)
        if path.read_bytes() != content:
            raise CheckpointError(
                f"Conflicting immutable checkpoint artifact: {path.name}"
            ) from None
    # Remove only this invocation's duplicate temporary link after publication.
    # On failure, leave the pending file as interruption evidence.
    pending.unlink()


class CheckpointStore:
    """Checkpoint state is bound to the same identity as its final experiment."""

    def __init__(self, lake_root: str | Path, run_id: str, config: dict, provenance: dict):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Invalid checkpoint experiment identity")
        self.config = normalize_config(config)
        self.provenance = _identity(provenance)
        self.run_id = run_id
        base = Path(lake_root) / "checkpoints"
        self.root = base / run_id
        self.blobs = self.root / "blobs"
        for path in (base, self.root, self.blobs):
            _plain(path)
            path.mkdir(parents=True, exist_ok=True)
        self.frame_hashes: list[str] = []
        self.loaded_step: int | None = None
        self.loaded_manifest_sha256: str | None = None
        self._frame_names = ("velocity",) if config["equation"] == "burgers1d" else (
            "u", "v", "vorticity"
        )
        self._shape = (config["grid_size"],) if config["equation"] == "burgers1d" else (
            config["grid_size"], config["grid_size"]
        )
        self._max_payload = max(
            16 * config["grid_size"] ** len(self._shape) + MAX_FRAMES * 48,
            8 * len(self._frame_names) * config["grid_size"] ** len(self._shape),
        ) + 65536

    def _write_blob(self, arrays: dict) -> str:
        stream = io.BytesIO()
        try:
            np.savez(stream, **arrays)
        except (ValueError, RuntimeError) as exc:
            raise CheckpointError(f"Cannot encode checkpoint arrays: {exc}") from exc
        content = stream.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        _publish(self.blobs / f"{digest}.npz", content)
        return digest

    def _read_blob(self, digest: str) -> dict:
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise CheckpointError("Invalid checkpoint blob hash")
        path = self.blobs / f"{digest}.npz"
        _plain(path)
        try:
            if path.stat().st_size > self._max_payload:
                raise CheckpointError("Checkpoint blob exceeds payload bound")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise CheckpointError(f"Checkpoint blob hash mismatch: {digest}")
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or len(names) > 4:
                    raise CheckpointError("Checkpoint archive has duplicate or unexpected entries")
                if sum(item.file_size for item in archive.infolist()) > self._max_payload:
                    raise CheckpointError("Checkpoint arrays exceed payload bound")
            with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                return {name: arrays[name] for name in arrays.files}
        except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
            raise CheckpointError(f"Cannot read checkpoint blob {digest}: {exc}") from exc

    def _frame(self, digest, expected_time):
        arrays = self._read_blob(digest)
        if set(arrays) != {"time", *self._frame_names}:
            raise CheckpointError("Checkpoint frame fields do not match the equation")
        time = arrays.pop("time")
        if time.dtype != np.dtype("float64") or time.shape != (1,) or time[0] != expected_time:
            raise CheckpointError("Checkpoint frame time does not match its saved schedule")
        for field in arrays.values():
            if field.dtype != np.dtype("float64") or field.shape != self._shape:
                raise CheckpointError("Checkpoint frame shape or dtype mismatch")
            if not np.isfinite(field).all():
                raise CheckpointError("Checkpoint frame is nonfinite")
        return arrays

    def append_frame(self, time: float, fields: dict) -> None:
        if len(self.frame_hashes) >= MAX_FRAMES:
            raise CheckpointError("Checkpoint frame budget exceeded")
        self.frame_hashes.append(self._write_blob({"time": np.array([time]), **fields}))

    def save(self, checkpoint: SolverCheckpoint) -> None:
        try:
            validate_checkpoint(self.config, checkpoint)
        except ValueError as exc:
            raise CheckpointError(f"Invalid solver checkpoint: {exc}") from exc
        if len(self.frame_hashes) != len(checkpoint.times):
            raise CheckpointError("Checkpoint frame and diagnostic prefixes disagree")
        state_hash = self._write_blob({
            "step": np.array([checkpoint.step], dtype=np.int64),
            "state": checkpoint.state,
            "times": checkpoint.times,
            "diagnostics": checkpoint.diagnostics,
        })
        payload = {
            "version": 1, "run_id": self.run_id, "config": self.config,
            "provenance": self.provenance, "step": checkpoint.step,
            "state": state_hash, "frames": list(self.frame_hashes),
        }
        document = {"payload": payload, "sha256": hashlib.sha256(_encoded(payload)).hexdigest()}
        _publish(self.root / f"step-{checkpoint.step}.json", _encoded(document))

    def load_latest(self) -> SolverCheckpoint | None:
        candidates = []
        for path in self.root.glob("step-*.json"):
            _plain(path)
            match = _MARKER.fullmatch(path.name)
            if not match:
                raise CheckpointError("Malformed published checkpoint name")
            candidates.append((int(match[1]), path))
        if not candidates:
            return None
        step, path = max(candidates)
        try:
            if path.stat().st_size > 2 * 1024**2:
                raise CheckpointError("Checkpoint manifest exceeds its bound")
            encoded = path.read_bytes()
            document = json.loads(encoded)
            payload = document["payload"]
            if hashlib.sha256(_encoded(payload)).hexdigest() != document["sha256"]:
                raise CheckpointError("Checkpoint manifest hash mismatch")
            if (
                payload["version"] != 1 or payload["run_id"] != self.run_id
                or payload["config"] != self.config or payload["provenance"] != self.provenance
                or type(payload["step"]) is not int or payload["step"] != step
            ):
                raise CheckpointError("Checkpoint identity, configuration, or runtime mismatch")
            arrays = self._read_blob(payload["state"])
            if set(arrays) != {"state", "step", "times", "diagnostics"}:
                raise CheckpointError("Checkpoint state fields mismatch")
            if arrays["step"].dtype != np.dtype("int64") or arrays["step"].shape != (1,):
                raise CheckpointError("Checkpoint step encoding mismatch")
            if arrays["step"][0] != step:
                raise CheckpointError("Checkpoint state step mismatch")
            checkpoint = SolverCheckpoint(step, arrays["state"], arrays["times"],
                                          arrays["diagnostics"])
            validate_checkpoint(self.config, checkpoint)
            frames = payload["frames"]
            if not isinstance(frames, list) or len(frames) != len(checkpoint.times):
                raise CheckpointError("Checkpoint frame prefix length mismatch")
            for digest, time in zip(frames, checkpoint.times, strict=True):
                self._frame(digest, time)
        except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
            raise CheckpointError(f"Cannot resume published checkpoint {path.name}: {exc}") from exc
        self.frame_hashes = list(frames)
        self.loaded_step = step
        self.loaded_manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        return checkpoint

    def replay(self, checkpoint: SolverCheckpoint, sink) -> None:
        for digest, time in zip(self.frame_hashes, checkpoint.times, strict=True):
            sink(float(time), self._frame(digest, time))
