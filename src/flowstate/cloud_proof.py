"""Prepare locally, then explicitly approve a bounded GCS removal/restore proof.

Preparation never creates a cloud client. Execution accepts the exact plan hash
and explicit cloud/deletion flags; the assistant still needs human authorization.
Only the newly generated experiment directory can be removed. No remote deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from flowstate import gcs_store
from flowstate.artifact_mirror import _prefix
from flowstate.dashboard import export_dashboard
from flowstate.engine import capture_provenance, run_experiment
from flowstate.lake import Lake

MAX_BYTES = 1024 * 1024
MAX_FILES = 128
CONFIG = {
    "equation": "burgers1d", "grid_size": 64, "steps": 20, "save_every": 5,
    "dt": 0.002, "viscosity": 0.05, "initial_condition": "random", "seed": 51,
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _plain(path: Path) -> None:
    """Inspect the lexical path first; resolving first would hide junctions."""
    for component in (path, *path.parents):
        if component.is_symlink() or component.is_junction():
            raise ValueError(f"Proof paths must not use links or junctions: {component}")
        if component.exists() and getattr(component.lstat(), "st_file_attributes", 0) & (
            stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"Proof paths must not use reparse points: {component}")


def _inventory(directory: Path) -> dict:
    _plain(directory)
    files = {}
    directories = []
    total_bytes = 0
    # follow_symlinks=False is explicit; reject a link before descending into it.
    for base, children, names in os.walk(directory, followlinks=False):
        for name in [*children, *names]:
            path = Path(base) / name
            _plain(path)
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                directories.append(path.relative_to(directory).as_posix())
            elif stat.S_ISREG(mode):
                size = path.stat().st_size
                total_bytes += size
                if len(files) >= MAX_FILES or total_bytes > MAX_BYTES:
                    raise ValueError("Proof exceeds its local transfer bound")
                files[path.relative_to(directory).as_posix()] = {
                    "bytes": size, "sha256": _hash(path),
                }
            else:
                raise ValueError("Proof artifacts must be ordinary files/directories")
    if len(files) > MAX_FILES or sum(item["bytes"] for item in files.values()) > MAX_BYTES:
        raise ValueError("Proof exceeds its local transfer bound")
    return {"files": files, "directories": sorted(directories)}


def _sdk_versions() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ("google-cloud-storage", "google-auth", "google-crc32c")}


def _runtime(provenance: dict) -> dict:
    # Documentation-only commits can advance while approval is pending. Source
    # bytes and runtime must match; original experiment provenance is preserved.
    return {k: v for k, v in provenance.items() if k not in {"git_commit", "git_dirty"}}


def _require_clean(provenance: dict) -> None:
    if not provenance.get("git_commit") or provenance.get("git_dirty") is not False:
        raise ValueError(
            "Commit source and use a clean checkout before proof preparation/execution"
        )


def prepare_proof(
    output: str | Path, *, project: str = "flowstate-510320",
    bucket: str = "flowstate-codex", prefix: str = "flowstate/",
    require_clean: bool = True,
) -> dict:
    """Create one disposable local run and a reviewable immutable plan; no cloud I/O."""
    if not isinstance(project, str) or not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ValueError("Invalid Google Cloud project ID")
    if not isinstance(bucket, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket
    ):
        raise ValueError("Invalid GCS bucket name")
    prefix = _prefix(prefix)
    if not prefix:
        raise ValueError("An explicit nonempty proof prefix is required")
    provenance = capture_provenance()
    if require_clean:
        _require_clean(provenance)
    libraries = _sdk_versions()
    root = Path(output).absolute()
    _plain(root)
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve(strict=True)
    proof_id = uuid.uuid4().hex
    _write_json(root / "owner.json", {"proof_id": proof_id, "root": str(root)})
    outcome = run_experiment(CONFIG, root / "source", stream=True)
    if outcome.record["status"] != "completed" or outcome.resumed:
        raise RuntimeError("Proof preparation requires a new completed numerical experiment")
    if _runtime(outcome.record["provenance"]) != _runtime(provenance):
        raise RuntimeError("Source or runtime changed during proof preparation")
    run_id = outcome.record["id"]
    directory = root / "source" / "experiments" / run_id
    if problems := Lake(root / "source").verify(run_id):
        raise ValueError(f"Generated proof failed verification: {problems}")
    inventory = _inventory(directory)
    viewer = export_dashboard(root / "source", root / "viewer.html")
    after = capture_provenance()
    if after != provenance:
        raise RuntimeError("Source/commit changed during proof preparation")
    plan = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "proof_id": proof_id, "root": str(root), "experiment_id": run_id,
        "config": outcome.record["config"], "provenance": provenance,
        "storage_libraries": libraries, "inventory": inventory,
        "file_count": len(inventory["files"]),
        "local_bytes": sum(item["bytes"] for item in inventory["files"].values()),
        "gcs": {"project": project, "bucket": bucket,
                "prefix": f"{prefix}/proofs/{proof_id}"},
        "local_removal_target": str(directory),
        "restore_lake": str(root / "restored"), "viewer": viewer,
        "bounds": {"max_files_including_manifest": MAX_FILES, "max_bytes": MAX_BYTES},
        "actions": ["upload", "repeat upload to hash remote bytes", "remove generated local run",
                    "download into fresh lake", "verify exact files and query restored metadata"],
        "cloud_accessed": False, "approved": False,
    }
    _write_json(root / "plan.json", plan)
    return {"plan": str(root / "plan.json"), "plan_sha256": _hash(root / "plan.json"),
            "experiment_id": run_id, "local_removal_target": str(directory),
            "file_count": plan["file_count"], "local_bytes": plan["local_bytes"],
            "gcs": plan["gcs"], "cloud_accessed": False}


def _load_plan(root: Path, expected_hash: str) -> tuple[dict, Path]:
    _plain(root)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("An exact prepared plan SHA-256 is required")
    for name in ("plan.json", "owner.json"):
        _plain(root / name)
    if _hash(root / "plan.json") != expected_hash:
        raise ValueError("Prepared plan hash changed; review it again")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    owner = json.loads((root / "owner.json").read_text(encoding="utf-8"))
    if plan["schema_version"] != 1 or plan["root"] != str(root.resolve(strict=True)):
        raise ValueError("Prepared proof root/schema differs from the approved plan")
    if owner != {"proof_id": plan["proof_id"], "root": plan["root"]}:
        raise ValueError("Prepared proof ownership marker changed")
    if not re.fullmatch(r"[0-9a-f]{32}", plan["experiment_id"]):
        raise ValueError("Invalid prepared experiment ID")
    directory = root / "source" / "experiments" / plan["experiment_id"]
    _plain(directory)
    if str(directory) != plan["local_removal_target"]:
        raise ValueError("Removal target differs from the approved plan")
    if not directory.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("Removal target escapes its proof directory")
    if _inventory(directory) != plan["inventory"]:
        raise ValueError("Prepared local files changed; refusing cloud access or removal")
    _plain(root / "restored")
    if (root / "restored").exists() or (root / "restored").is_symlink():
        raise FileExistsError("Restoration requires an absent destination lake")
    if str(root / "restored") != plan["restore_lake"]:
        raise ValueError("Restore destination differs from the approved plan")
    return plan, directory


def _remove_generated(root: Path, plan: dict, directory: Path, progress: dict) -> None:
    """Remove only the verified, inventoried generated subtree; never recursive rmtree."""
    _plain(directory)
    expected = root / "source" / "experiments" / plan["experiment_id"]
    if directory != expected or not directory.resolve(strict=True).is_relative_to(root.resolve()):
        raise ValueError("Refusing to remove outside the generated experiment")
    if _inventory(directory) != plan["inventory"]:
        raise ValueError("Generated files changed before removal")
    for relative, record in plan["inventory"]["files"].items():
        path = directory / relative
        _plain(path)
        if not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
            raise ValueError("Removal path escapes its experiment directory")
        if _hash(path) != record["sha256"]:
            raise ValueError("Generated artifact changed during removal")
        path.unlink()
        progress["removed_files"].append(relative)
    for relative in sorted(plan["inventory"]["directories"],
                           key=lambda item: len(Path(item).parts), reverse=True):
        path = directory / relative
        _plain(path)
        path.rmdir()
        progress["removed_directories"].append(relative)
    _plain(directory)
    directory.rmdir()
    progress["exists_after"] = directory.exists()


def execute_proof(
    output: str | Path, *, plan_sha256: str, allow_cloud: bool = False,
    allow_local_delete: bool = False, require_clean: bool = True,
) -> dict:
    """Run the already reviewed plan once; persist success or failure stage receipts."""
    if allow_cloud is not True or allow_local_delete is not True:
        raise PermissionError(
            "Explicit cloud access and generated-local-data deletion approval required"
        )
    root = Path(output).absolute()
    plan, directory = _load_plan(root, plan_sha256)
    before = capture_provenance()
    if require_clean:
        _require_clean(before)
    if _runtime(before) != _runtime(plan["provenance"]):
        raise ValueError("Source or runtime differs from the prepared proof")
    if _sdk_versions() != plan["storage_libraries"]:
        raise ValueError("Storage libraries differ from the prepared proof")
    for name in ("receipt.json", "before-removal.json", "after-removal.json",
                 "restored-viewer.html"):
        path = root / name
        if path.exists() or path.is_symlink() or path.is_junction():
            raise FileExistsError(f"Proof execution output already exists: {name}")
    # Exclusive marker prevents two executions or retries from deleting/restoring
    # the same proof. A failed or interrupted proof remains evidence, not input.
    _write_json(root / "execution-started.json", {
        "plan_sha256": plan_sha256, "started_at": datetime.now(UTC).isoformat(),
        "provenance": before, "cloud_approved": True, "local_delete_approved": True,
    })
    result = {
        "schema_version": 1, "status": "failed", "plan_sha256": plan_sha256,
        "plan": plan, "execution_provenance": before, "stages": {},
        "local_removal_started": False, "local_removal_completed": False,
        "remote_objects_deleted": False,
    }
    stage = "upload"
    started = time.perf_counter()
    try:
        options = plan["gcs"]
        source, restored = root / "source", root / "restored"
        run_id = plan["experiment_id"]
        upload = gcs_store.upload_experiment(source, run_id, **options)
        result["stages"][stage] = upload
        stage = "remote_verification"
        repeated = gcs_store.upload_experiment(source, run_id, **options)
        result["stages"][stage] = repeated
        expected_count = plan["file_count"] - 1  # Completion manifest is counted separately.
        if (
            not upload["committed"] or not repeated["committed"]
            or not repeated["resumed"] or repeated["uploaded_objects"] != 0
            or repeated["reused_objects"] != expected_count
            or repeated["manifest_sha256"] != upload["manifest_sha256"]
            or upload["manifest_sha256"] != plan["inventory"]["files"]["manifest.json"]["sha256"]
        ):
            raise ValueError("Remote read-back proof did not verify the complete expected mirror")
        stage = "local_removal"
        # Repeat all plan, ownership, path and inventory checks immediately before removal.
        _load_plan(root, plan_sha256)
        if _runtime(capture_provenance()) != _runtime(before):
            raise ValueError("Source/runtime changed before local removal")
        result["local_removal_started"] = True
        _write_json(root / "before-removal.json", {
            "plan_sha256": plan_sha256, "target": str(directory),
            "upload": upload, "remote_verification": repeated,
        })
        progress = {"target": str(directory), "existed_before": True,
                    "exists_after": None, "removed_files": [], "removed_directories": []}
        result["stages"][stage] = progress
        _remove_generated(root, plan, directory, progress)
        result["local_removal_completed"] = True
        _write_json(root / "after-removal.json", result["stages"][stage])
        stage = "download"
        if directory.exists():
            raise RuntimeError("Source experiment still exists after removal")
        downloaded = gcs_store.download_experiment(restored, run_id, **options)
        result["stages"][stage] = downloaded
        if downloaded["resumed"]:
            raise RuntimeError("Proof unexpectedly reused a local result instead of downloading")
        stage = "verification"
        lake = Lake(restored)
        inventory = _inventory(restored / "experiments" / run_id)
        problems = lake.verify(run_id)
        if problems or inventory != plan["inventory"]:
            raise ValueError("Restored artifact bytes differ from the prepared inventory")
        sql = "SELECT id, equation, status FROM experiments"
        rows = lake.query(sql)
        if rows != [{"id": run_id, "equation": "burgers1d", "status": "completed"}]:
            raise ValueError("Restored metadata query differs from the expected completed record")
        result["stages"][stage] = {
            "all_files_byte_identical": True, "file_count": len(inventory["files"]),
            "manifest_problems": problems, "inventory": inventory, "sql": sql, "rows": rows,
        }
        result["restored_viewer"] = export_dashboard(restored, root / "restored-viewer.html")
        if _runtime(capture_provenance()) != _runtime(before):
            raise ValueError("Source or runtime changed during proof execution")
        result["status"] = "completed"
    except Exception as exc:
        result["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        if stage == "local_removal" and stage in result["stages"]:
            result["stages"][stage]["exists_after"] = directory.exists()
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        _write_json(root / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create local artifacts only; no cloud access")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--project", default="flowstate-510320")
    prepare.add_argument("--bucket", default="flowstate-codex")
    prepare.add_argument("--prefix", default="flowstate/")
    execute = commands.add_parser("execute", help="Execute an explicitly approved prepared plan")
    execute.add_argument("output", type=Path)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--allow-cloud", action="store_true")
    execute.add_argument("--allow-local-delete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_proof(args.output, project=args.project, bucket=args.bucket,
                               prefix=args.prefix)
    else:
        result = execute_proof(args.output, plan_sha256=args.plan_sha256,
                               allow_cloud=args.allow_cloud,
                               allow_local_delete=args.allow_local_delete)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
