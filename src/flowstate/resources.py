"""Sample current-process and descendant RSS without claiming a true memory peak."""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Real
from threading import Event, Lock

import psutil


class ProcessTreeSampler:
    """Periodically sample the calling process and its recursive children.

    Instances are single-use context managers. An initial and final sample are
    taken even when the body is shorter than the sampling interval. ``report``
    returns a thread-safe snapshot during or after the context. Known process
    disappearance/access errors produce counted partial samples; other failures
    propagate on exit after the worker has joined.
    """

    def __init__(self, interval: float = 0.02):
        if isinstance(interval, bool) or not isinstance(interval, Real):
            raise ValueError("interval must be a finite number between 0.001 and 60 seconds")
        try:
            interval = float(interval)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "interval must be a finite number between 0.001 and 60 seconds"
            ) from exc
        if not math.isfinite(interval) or not 0.001 <= interval <= 60:
            raise ValueError("interval must be a finite number between 0.001 and 60 seconds")
        self.interval = interval
        self._lock = Lock()
        self._stop = Event()
        self._started = False
        self._finished = False
        self._process = None
        self._executor: ThreadPoolExecutor | None = None
        self._worker: Future | None = None
        self._peak_aggregate: int | None = None
        self._peak_parent: int | None = None
        self._peak_children: int | None = None
        self._max_children = 0
        self._sample_count = 0
        self._partial_count = 0
        self._errors = {"process_gone": 0, "access_denied": 0}
        self._unexpected_errors: list[str] = []

    def __enter__(self) -> ProcessTreeSampler:
        with self._lock:
            if self._started:
                raise RuntimeError("ProcessTreeSampler instances cannot be reused or nested")
            self._started = True
        self._process = psutil.Process()
        self._sample()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowstate-rss")
        try:
            self._worker = self._executor.submit(self._run)
        except BaseException:
            # Starting a thread can itself fail. Never leave an owned executor behind.
            self._executor.shutdown(wait=True)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._stop.set()
        # The single worker completes any in-flight periodic sample, then takes
        # the queued final sample. shutdown(wait=True) always joins that thread.
        final = None
        try:
            final = self._executor.submit(self._sample)
        finally:
            self._executor.shutdown(wait=True)
            with self._lock:
                self._finished = True
        failures = [future.exception() for future in (self._worker, final) if future is not None]
        failures = [failure for failure in failures if failure is not None]
        if failures:
            with self._lock:
                self._unexpected_errors.extend(
                    f"{type(failure).__name__}: {failure}" for failure in failures
                )
            if exc_value is None:
                raise failures[0]
            # Preserve exceptions from the measured work; still make a sampler
            # failure visible rather than replace the user's original error.
            exc_value.add_note(f"Process-tree sampling also failed: {failures[0]}")
        return False

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        errors = {"process_gone": 0, "access_denied": 0}

        def rss(process):
            try:
                return int(process.memory_info().rss)
            except psutil.NoSuchProcess:
                errors["process_gone"] += 1
            except psutil.AccessDenied:
                errors["access_denied"] += 1
            return None

        parent_rss = rss(self._process)
        children = None
        try:
            children = self._process.children(recursive=True)
        except psutil.NoSuchProcess:
            errors["process_gone"] += 1
        except psutil.AccessDenied:
            errors["access_denied"] += 1
        observed = len(children) if children is not None else 0
        child_values = [rss(child) for child in children] if children is not None else []
        readable = [value for value in child_values if value is not None]
        # An enumerated empty child list is a measured zero. If enumeration or
        # every child read failed, child RSS is unavailable rather than zero.
        children_rss = sum(readable) if readable or children == [] else None
        aggregate = (
            (parent_rss if parent_rss is not None else 0) + sum(readable)
            if parent_rss is not None or readable
            else None
        )
        with self._lock:
            for attribute, value in (
                ("_peak_aggregate", aggregate),
                ("_peak_parent", parent_rss),
                ("_peak_children", children_rss),
            ):
                previous = getattr(self, attribute)
                if value is not None and (previous is None or value > previous):
                    setattr(self, attribute, value)
            self._max_children = max(self._max_children, observed)
            self._sample_count += 1
            self._partial_count += bool(sum(errors.values()))
            for name, count in errors.items():
                self._errors[name] += count

    def report(self) -> dict:
        """Return JSON-native measurements; unavailable values remain null."""
        with self._lock:
            if not self._started:
                raise RuntimeError("Enter ProcessTreeSampler before requesting a report")
            return {
                "sampled_peak_aggregate_rss_bytes": self._peak_aggregate,
                "sampled_peak_parent_rss_bytes": self._peak_parent,
                "sampled_peak_child_rss_bytes": self._peak_children,
                "max_observed_children": self._max_children,
                "sample_count": self._sample_count,
                "interval_seconds": self.interval,
                "sampling_errors": sum(self._errors.values()) + len(self._unexpected_errors),
                "sampling_error_counts": dict(self._errors),
                "partial_sample_count": self._partial_count,
                "unexpected_sampling_errors": list(self._unexpected_errors),
                "finished": self._finished,
                "scope_notes": [
                    "Sampled aggregate RSS sums the current process and observed descendants; "
                    "it is not PSS, and shared resident pages can be counted multiple times.",
                    "This is not a true peak: short-lived children, orphaned descendants, and "
                    "between-sample peaks may be missed. RSS reads are sequential, not atomic.",
                    "The interval is a requested wait between samples; sampling and OS scheduling "
                    "add latency. Initial and final samples are also taken.",
                    "Partial samples sum only readable processes; errors count failed operations, "
                    "not unique processes. Entirely unavailable components remain null.",
                    "Parent and child peaks can occur at different times; their sum need not equal "
                    "the sampled aggregate peak. Sampler and existing process memory is included.",
                ],
            }
