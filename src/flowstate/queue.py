"""Durable local-disk experiment queue with leased, fenced worker claims.

Multiple processes on one machine may share this SQLite file. Network filesystems
and cross-machine scheduling are outside this contract. Optional solver checkpoints
recover progress after a worker exits. A lost lease may cause duplicate computation,
but the immutable lake verifies/reuses the same experiment identity and a stale
worker cannot change the queue's job state.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from flowstate.checkpoints import validate_checkpoint_options
from flowstate.engine import (
    MAX_SWEEP_RUNS,
    capture_provenance,
    expand_sweep,
    experiment_id,
    run_experiment,
)

_APPLICATION_ID = 0x464C4F57
_SCHEMA_VERSION = 1
_STATES = ("queued", "running", "completed", "failed")


def _encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _runtime_identity(provenance: dict) -> dict:
    # Git dirtiness is an annotation; source bytes and commit remain part of identity.
    return {key: value for key, value in provenance.items() if key != "git_dirty"}


# A long-lived Python process keeps imported code even after its files change.
# Pin the runtime when this worker module loads; never relabel cached code using
# a later disk fingerprint. Restart worker processes after any checkout/edit.
_IMPORTED_RUNTIME_IDENTITY = _runtime_identity(capture_provenance())


def _identity(request: dict) -> str:
    value = {**request, "provenance": _runtime_identity(request["provenance"])}
    return hashlib.sha256(_encode(value).encode()).hexdigest()


@contextmanager
def _connection(path: str | Path, *, create: bool = False, readonly: bool = False):
    path = Path(path).resolve()
    if not create and not path.is_file():
        raise FileNotFoundError(f"Queue does not exist: {path}")
    uri = path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc" if create else "?mode=rw")
    with closing(sqlite3.connect(uri, uri=True, isolation_level=None, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        if not create:
            if (
                db.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
                or db.execute("PRAGMA user_version").fetchone()[0] != _SCHEMA_VERSION
            ):
                raise ValueError("Not a supported Flowstate queue database")
        yield db


@contextmanager
def _transaction(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.rollback()
        raise
    else:
        db.commit()


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connection(path, create=True) as db:
        with _transaction(db):
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if tables and (app_id != _APPLICATION_ID or version != _SCHEMA_VERSION):
                raise ValueError("Not a supported Flowstate queue database")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_json TEXT NOT NULL,
                expected_result_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                claim_count INTEGER NOT NULL DEFAULT 0,
                fencing_token TEXT, lease_until REAL,
                result_id TEXT, resumed INTEGER, error_json TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id), kind TEXT NOT NULL,
                at REAL NOT NULL, detail_json TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_state ON jobs(status, created_at)")
            db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        db.execute("PRAGMA journal_mode=WAL")


def _event(db, job_id: str, kind: str, now: float, detail: dict) -> None:
    db.execute(
        "INSERT INTO events(job_id,kind,at,detail_json) VALUES(?,?,?,?)",
        (job_id, kind, now, _encode(detail)),
    )


def submit_sweep(
    spec: dict,
    queue_path: str | Path,
    lake_root: str | Path,
    *,
    parent_id: str | None = None,
    attempt: int = 0,
    stream: bool = True,
    checkpoint_every: int | None = None,
) -> dict:
    """Validate the entire sweep, then insert new immutable jobs in one transaction.

    Identical requests share a job, including after completion or failure. Changing
    the attempt creates a new job; submitting again never resets terminal jobs.
    """
    configs = expand_sweep(spec)
    if type(attempt) is not int or attempt < 0:
        raise ValueError("attempt must be a nonnegative integer")
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        raise ValueError("parent_id must be a nonempty experiment ID")
    for config in configs:
        validate_checkpoint_options(config, checkpoint_every)
    provenance = capture_provenance()
    root = str(Path(lake_root).resolve())
    requests = []
    for config in configs:
        request = {
            "config": config, "lake_root": root, "parent_id": parent_id,
            "attempt": attempt, "stream": stream, "provenance": provenance,
        }
        if checkpoint_every is not None:
            request["checkpoint_every"] = checkpoint_every
        requests.append((_identity(request), request))
    path = Path(queue_path).resolve()
    _initialize(path)
    inserted, now = 0, time.time()
    with _connection(path) as db, _transaction(db):
        for job_id, request in requests:
            expected = experiment_id(request["config"], provenance, parent_id, attempt)
            cursor = db.execute(
                """INSERT OR IGNORE INTO jobs
                   (id,request_json,expected_result_id,status,created_at,updated_at)
                   VALUES(?,?,?,'queued',?,?)""",
                (job_id, _encode(request), expected, now, now),
            )
            if cursor.rowcount:
                inserted += 1
                _event(db, job_id, "submitted", now, {"expected_result_id": expected})
    return {
        "queue_path": str(path), "submitted": inserted, "existing": len(requests) - inserted,
        "job_ids": [job_id for job_id, _ in requests],
    }


def queue_status(queue_path: str | Path) -> dict:
    """Read a consistent snapshot without claiming jobs or recovering expired leases."""
    now = time.time()
    with _connection(queue_path, readonly=True) as db:
        db.execute("BEGIN")
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at,id").fetchall()
        events = db.execute("SELECT * FROM events ORDER BY sequence").fetchall()
    jobs, counts = [], dict.fromkeys(_STATES, 0)
    checkpoint_steps = {
        row["job_id"]: json.loads(row["detail_json"]).get("checkpoint_resumed_from")
        for row in events if row["kind"] in {"completed", "failed"}
    }
    for row in rows:
        request = json.loads(row["request_json"])
        counts[row["status"]] += 1
        jobs.append({
            **{key: row[key] for key in (
                "id", "status", "expected_result_id", "created_at", "updated_at", "claim_count",
                "lease_until", "result_id",
            )},
            "config": request["config"], "lake_root": request["lake_root"],
            "parent_id": request["parent_id"], "attempt": request["attempt"],
            "stream": request["stream"], "provenance": request["provenance"],
            "checkpoint_every": request.get("checkpoint_every"),
            "checkpoint_resumed_from": checkpoint_steps.get(row["id"]),
            "resumed": bool(row["resumed"]) if row["resumed"] is not None else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        })
    return {
        "queue_path": str(Path(queue_path).resolve()), "counts": counts, "jobs": jobs,
        "expired_leases": sum(
            j["status"] == "running" and j["lease_until"] <= now for j in jobs
        ),
        "events": [{
            "sequence": row["sequence"], "job_id": row["job_id"], "kind": row["kind"],
            "at": row["at"], "detail": json.loads(row["detail_json"]),
        } for row in events],
    }


def _claim(queue_path: str | Path, lease_seconds: float, *, now: float | None = None):
    with _connection(queue_path) as db, _transaction(db):
        # Obtain wall time after waiting for the write lock, never before: a
        # blocked connection must not claim or acknowledge using stale time.
        now = time.time() if now is None else now
        expired = db.execute(
            "SELECT id FROM jobs WHERE status='running' AND lease_until<=?", (now,)
        ).fetchall()
        for row in expired:
            db.execute(
                """UPDATE jobs SET status='queued',fencing_token=NULL,lease_until=NULL,
                   updated_at=? WHERE id=?""", (now, row["id"]),
            )
            _event(db, row["id"], "lease_expired", now, {})
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        token = uuid.uuid4().hex
        db.execute(
            """UPDATE jobs SET status='running',fencing_token=?,lease_until=?,updated_at=?,
               claim_count=claim_count+1 WHERE id=?""",
            (token, now + lease_seconds, now, row["id"]),
        )
        _event(db, row["id"], "claimed", now, {"claim_number": row["claim_count"] + 1})
        return {
            "id": row["id"], "token": token, "request": json.loads(row["request_json"]),
            "expected_result_id": row["expected_result_id"],
        }


def _renew(queue_path, claim: dict, lease_seconds: float, *, now: float | None = None) -> bool:
    with _connection(queue_path) as db, _transaction(db):
        now = time.time() if now is None else now
        return bool(db.execute(
            """UPDATE jobs SET lease_until=?,updated_at=? WHERE id=? AND status='running'
               AND fencing_token=? AND lease_until>?""",
            (now + lease_seconds, now, claim["id"], claim["token"], now),
        ).rowcount)


def _finish(queue_path, claim: dict, outcome: dict, *, now: float | None = None) -> bool:
    with _connection(queue_path) as db, _transaction(db):
        now = time.time() if now is None else now
        cursor = db.execute(
            """UPDATE jobs SET status=?,updated_at=?,fencing_token=NULL,lease_until=NULL,
               result_id=?,resumed=?,error_json=? WHERE id=? AND status='running'
               AND fencing_token=? AND lease_until>?""",
            (outcome["status"], now, outcome.get("result_id"), outcome.get("resumed"),
             _encode(outcome["error"]) if outcome.get("error") else None,
             claim["id"], claim["token"], now),
        )
        if cursor.rowcount:
            _event(db, claim["id"], outcome["status"], now, outcome)
        return bool(cursor.rowcount)


class _Heartbeat:
    def __init__(self, queue_path, claim: dict, lease_seconds: float):
        self.path, self.claim, self.lease = queue_path, claim, lease_seconds
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.error = None
        self.thread = threading.Thread(
            target=self._run, name="flowstate-queue-heartbeat", daemon=True,
        )

    def _run(self):
        while not self.stop.wait(self.lease / 3):
            try:
                renewed = _renew(self.path, self.claim, self.lease)
            except Exception as exc:
                self.error = {"type": type(exc).__name__, "message": str(exc)}
                self.lost.set()
                return
            if not renewed:
                self.lost.set()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def work_queue(
    queue_path: str | Path, *, max_jobs: int = 1, lease_seconds: float = 60,
) -> dict:
    """Drain up to max_jobs immediately available jobs, retaining all outcomes.

    Workers exit when no job is claimable; they do not wait for other leases to
    expire. Run again to recover an expired worker. Heartbeats renew ownership,
    but cannot cancel a solver that lost its lease. No failure is silently retried.
    """
    if type(max_jobs) is not int or not 1 <= max_jobs <= MAX_SWEEP_RUNS:
        raise ValueError(f"max_jobs must be an integer from 1 to {MAX_SWEEP_RUNS}")
    if (
        isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float))
        or not math.isfinite(lease_seconds) or not 1 <= lease_seconds <= 86400
    ):
        raise ValueError("lease_seconds must be finite and between 1 and 86400 seconds")
    results = []
    for _ in range(max_jobs):
        claim = _claim(queue_path, lease_seconds)
        if claim is None:
            break
        request = claim["request"]
        with _Heartbeat(queue_path, claim, lease_seconds) as heartbeat:
            try:
                current_identity = _runtime_identity(capture_provenance())
                if current_identity != _runtime_identity(request["provenance"]):
                    raise ValueError(
                        "Code or environment changed since submission; use the original runtime "
                        "or submit a new job with the current code and environment"
                    )
                if current_identity != _IMPORTED_RUNTIME_IDENTITY:
                    raise ValueError(
                        "Worker runtime changed after import; restart the worker process "
                        "before executing jobs from the updated code"
                    )
                result = run_experiment(
                    request["config"], request["lake_root"], parent_id=request["parent_id"],
                    attempt=request["attempt"], stream=request["stream"],
                    checkpoint_every=request.get("checkpoint_every"),
                )
                if result.record["id"] != claim["expected_result_id"]:
                    raise ValueError(
                        "Experiment identity changed after claim; result not acknowledged"
                    )
                if _runtime_identity(capture_provenance()) != current_identity:
                    raise ValueError(
                        "Worker runtime changed during execution; result not acknowledged. "
                        "Restart the worker process after restoring or updating the code"
                    )
                outcome = {
                    "status": result.record["status"], "result_id": result.record["id"],
                    "resumed": result.resumed, "error": result.record.get("error"),
                    "checkpoint_resumed_from": result.checkpoint_step,
                }
            except Exception as exc:
                outcome = {
                    "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            # Acknowledge before stopping the heartbeat, preserving the lease while
            # publication finishes. Every write still checks both token and expiry.
            acknowledged = not heartbeat.lost.is_set() and _finish(queue_path, claim, outcome)
        results.append({
            "job_id": claim["id"], **outcome,
            **({"status": "lost_claim", "heartbeat_error": heartbeat.error}
               if not acknowledged else {}),
        })
    return {
        "queue_path": str(Path(queue_path).resolve()), "processed": len(results),
        **{state: sum(r["status"] == state for r in results)
           for state in ("completed", "failed", "lost_claim")},
        "jobs": results,
    }
