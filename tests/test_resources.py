"""Deterministic process-tree sampling checks with no timing-based peak assumptions."""

import json
import math
from threading import Event
from types import SimpleNamespace

import psutil
import pytest

import flowstate.resources as resources
from flowstate.resources import ProcessTreeSampler


class FakeProcess:
    def __init__(self, rss, children=None):
        self.rss = rss
        self.descendants = [] if children is None else children

    def memory_info(self):
        value = self.rss() if callable(self.rss) else self.rss
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(rss=value)

    def children(self, recursive=False):
        assert recursive is True
        value = self.descendants() if callable(self.descendants) else self.descendants
        if isinstance(value, Exception):
            raise value
        return value


def test_aggregation_uses_same_sample_not_sum_of_independent_peaks(monkeypatch):
    parent_values = iter([100, 200])
    child_groups = iter([[FakeProcess(30), FakeProcess(40)], [FakeProcess(10)]])
    parent = FakeProcess(lambda: next(parent_values), lambda: next(child_groups))
    monkeypatch.setattr(resources.psutil, "Process", lambda: parent)
    with ProcessTreeSampler(interval=60) as sampler:
        current = sampler.report()
        assert current["sample_count"] == 1
        assert current["finished"] is False
    report = sampler.report()
    assert report["sampled_peak_aggregate_rss_bytes"] == 210
    assert report["sampled_peak_parent_rss_bytes"] == 200
    assert report["sampled_peak_child_rss_bytes"] == 70
    assert report["max_observed_children"] == 2
    assert report["sample_count"] == 2
    assert report["sampling_errors"] == 0
    assert report["finished"] is True
    assert json.loads(json.dumps(report)) == report
    assert sampler._worker.done()
    assert all(not thread.is_alive() for thread in sampler._executor._threads)


def test_disappearing_or_inaccessible_children_produce_counted_partial_samples(monkeypatch):
    groups = iter(
        [
            [FakeProcess(psutil.NoSuchProcess(100)), FakeProcess(20)],
            [FakeProcess(psutil.AccessDenied(200)), FakeProcess(5)],
        ]
    )
    parent = FakeProcess(100, lambda: next(groups))
    monkeypatch.setattr(resources.psutil, "Process", lambda: parent)
    with ProcessTreeSampler(interval=60) as sampler:
        pass
    report = sampler.report()
    assert report["sampled_peak_aggregate_rss_bytes"] == 120
    assert report["sampled_peak_child_rss_bytes"] == 20
    assert report["max_observed_children"] == 2
    assert report["partial_sample_count"] == 2
    assert report["sampling_errors"] == 2
    assert report["sampling_error_counts"] == {"process_gone": 1, "access_denied": 1}


def test_unavailable_rss_remains_null_instead_of_becoming_measured_zero(monkeypatch):
    parent = FakeProcess(psutil.AccessDenied(1), [FakeProcess(psutil.NoSuchProcess(2))])
    monkeypatch.setattr(resources.psutil, "Process", lambda: parent)
    with ProcessTreeSampler(interval=60) as sampler:
        pass
    report = sampler.report()
    assert report["sampled_peak_parent_rss_bytes"] is None
    assert report["sampled_peak_child_rss_bytes"] is None
    assert report["sampled_peak_aggregate_rss_bytes"] is None
    assert report["sample_count"] == 2
    assert report["sampling_errors"] == 4


def test_failed_child_enumeration_is_recorded_and_later_empty_children_measure_zero(monkeypatch):
    groups = iter([psutil.AccessDenied(1), []])
    parent = FakeProcess(50, lambda: next(groups))
    monkeypatch.setattr(resources.psutil, "Process", lambda: parent)
    with ProcessTreeSampler(interval=60) as sampler:
        pass
    report = sampler.report()
    assert report["sampled_peak_aggregate_rss_bytes"] == 50
    assert report["sampled_peak_child_rss_bytes"] == 0
    assert report["partial_sample_count"] == 1
    assert report["sampling_errors"] == 1


def test_missing_parent_and_no_children_has_no_readable_aggregate(monkeypatch):
    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(psutil.AccessDenied(1)))
    with ProcessTreeSampler(interval=60) as sampler:
        pass
    report = sampler.report()
    assert report["sampled_peak_aggregate_rss_bytes"] is None
    assert report["sampled_peak_child_rss_bytes"] == 0


def test_context_exception_propagates_after_final_sample_and_worker_join(monkeypatch):
    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(42))
    failure = ValueError("The measured operation failed")
    sampler = ProcessTreeSampler(interval=60)
    with pytest.raises(ValueError) as caught, sampler:
        raise failure
    assert caught.value is failure
    assert sampler.report()["sample_count"] == 2
    assert sampler.report()["finished"]
    assert all(not thread.is_alive() for thread in sampler._executor._threads)


def test_unexpected_sampling_exception_is_not_silently_swallowed(monkeypatch):
    values = iter([42, RuntimeError("Unexpected sampler failure")])
    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(lambda: next(values)))
    sampler = ProcessTreeSampler(interval=60)
    with pytest.raises(RuntimeError, match="Unexpected sampler failure"), sampler:
        pass
    assert sampler.report()["sampling_errors"] == 1
    assert sampler.report()["unexpected_sampling_errors"] == [
        "RuntimeError: Unexpected sampler failure"
    ]
    assert all(not thread.is_alive() for thread in sampler._executor._threads)


def test_periodic_worker_samples_while_context_is_active(monkeypatch):
    second_sample = Event()
    reads = 0

    def memory():
        nonlocal reads
        reads += 1
        if reads >= 2:
            second_sample.set()
        return 100

    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(memory))
    with ProcessTreeSampler(interval=0.001) as sampler:
        assert second_sample.wait(timeout=5), "Worker did not sample"
    assert sampler.report()["sample_count"] >= 3


def test_background_worker_error_propagates_after_final_sample(monkeypatch):
    second_sample = Event()
    reads = 0

    def memory():
        nonlocal reads
        reads += 1
        if reads == 2:
            second_sample.set()
            raise RuntimeError("Unexpected background error")
        return 100

    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(memory))
    sampler = ProcessTreeSampler(interval=0.001)
    with pytest.raises(RuntimeError, match="background"), sampler:
        assert second_sample.wait(timeout=5), "Worker did not sample"
    assert sampler.report()["sample_count"] == 2  # Successful initial and final samples.
    assert sampler.report()["unexpected_sampling_errors"] == [
        "RuntimeError: Unexpected background error"
    ]
    assert all(not thread.is_alive() for thread in sampler._executor._threads)


def test_body_error_remains_primary_when_final_sample_also_fails(monkeypatch):
    values = iter([42, RuntimeError("Sampler failure")])
    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(lambda: next(values)))
    failure = ValueError("Body failure")
    sampler = ProcessTreeSampler(interval=60)
    with pytest.raises(ValueError) as caught, sampler:
        raise failure
    assert caught.value is failure
    assert any("Sampler failure" in note for note in failure.__notes__)
    assert sampler.report()["sampling_errors"] == 1


@pytest.mark.parametrize("interval", [0, -1, math.inf, math.nan, True, "0.02", 0.0001, 61])
def test_invalid_sampling_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="interval"):
        ProcessTreeSampler(interval=interval)


def test_sampler_is_single_use_and_report_does_not_expose_mutable_state(monkeypatch):
    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(42))
    sampler = ProcessTreeSampler(interval=60)
    with pytest.raises(RuntimeError, match="Enter"):
        sampler.report()
    with sampler:
        report = sampler.report()
        report["sampling_error_counts"]["process_gone"] = 999
    assert sampler.report()["sampling_errors"] == 0
    with pytest.raises(RuntimeError, match="reused"), sampler:
        pass


def test_real_current_process_sampler_smoke():
    with ProcessTreeSampler(interval=60) as sampler:
        allocation = bytearray(64 * 1024)
        allocation[0] = 1
    report = sampler.report()
    assert report["sample_count"] == 2
    assert report["sampled_peak_parent_rss_bytes"] > 0
    assert report["sampled_peak_aggregate_rss_bytes"] >= report["sampled_peak_parent_rss_bytes"]
    assert report["finished"]
