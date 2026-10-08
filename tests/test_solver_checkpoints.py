"""Checkpoint replay must preserve the exact trajectory, not just its tolerance."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from flowstate import numerics
from flowstate.numerics import SolverCheckpoint, solve, validate_checkpoint


class InterruptedRun(Exception):
    pass


def _config(equation="burgers1d", initial_condition="random"):
    return {
        "equation": equation,
        "grid_size": 16,
        "initial_condition": initial_condition,
        "seed": 51,
        "dt": 0.002,
        "steps": 11,
        "save_every": 4,
    }


def _hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.mark.parametrize(
    "equation,initial_condition",
    [
        ("burgers1d", "sine"),
        ("burgers1d", "random"),
        ("navier_stokes2d", "taylor_green"),
        ("navier_stokes2d", "random"),
    ],
)
def test_resume_preserves_every_array_and_skips_completed_steps(
    monkeypatch, equation, initial_condition
):
    config = _config(equation, initial_condition)
    reference = solve(config)
    frames = {name: [] for name in reference.fields}
    emitted_times = []
    snapshots = []

    def save_frame(time, values):
        emitted_times.append(time)
        for name, value in values.items():
            frames[name].append(value.copy())

    def interrupt(checkpoint):
        snapshots.append(checkpoint)
        if checkpoint.step == 6:
            raise InterruptedRun

    with pytest.raises(InterruptedRun):
        solve(
            config,
            frame_callback=save_frame,
            retain_fields=False,
            checkpoint_every=3,
            checkpoint_callback=interrupt,
        )
    checkpoint = snapshots[-1]
    assert checkpoint.step == 6  # Intentionally between saved frames 4 and 8.
    np.testing.assert_array_equal(checkpoint.times, reference.times[:2])
    original_state_hash = _hash(checkpoint.state)
    rk4 = numerics._rk4
    remaining_steps = []

    def count_step(*args):
        remaining_steps.append(1)
        return rk4(*args)

    monkeypatch.setattr(numerics, "_rk4", count_step)
    resumed = solve(
        config,
        checkpoint=checkpoint,
        retain_fields=False,
        frame_callback=save_frame,
    )
    assert len(remaining_steps) == config["steps"] - checkpoint.step
    assert resumed.fields == {}
    assert _hash(checkpoint.state) == original_state_hash
    np.testing.assert_array_equal(emitted_times, reference.times)
    assert _hash(resumed.times) == _hash(reference.times)
    for name, value in reference.fields.items():
        assert _hash(np.stack(frames[name])) == _hash(value)
    for name, value in reference.diagnostics.items():
        assert _hash(resumed.diagnostics[name]) == _hash(value)
    for name, value in reference.coordinates.items():
        assert _hash(resumed.coordinates[name]) == _hash(value)
    assert resumed.metadata == reference.metadata


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
@pytest.mark.parametrize("retain_fields", [True, False])
def test_snapshots_are_independent_and_follow_frame_publication(equation, retain_fields):
    config = _config(equation)
    baseline = solve(config)
    events = []

    def mutate_snapshot(checkpoint):
        validate_checkpoint(config, checkpoint)
        events.append(("checkpoint", checkpoint.step))
        # Mutating a consumer-owned snapshot must never modify the solver or its history.
        checkpoint.state[:] = 100
        checkpoint.times[:] = 100
        checkpoint.diagnostics[:] = 100

    result = solve(
        config,
        retain_fields=retain_fields,
        frame_callback=lambda time, _: events.append(("frame", round(time / config["dt"]))),
        checkpoint_every=3,
        checkpoint_callback=mutate_snapshot,
    )
    assert events == [
        ("frame", 0),
        ("checkpoint", 3),
        ("frame", 4),
        ("checkpoint", 6),
        ("frame", 8),
        ("checkpoint", 9),
        ("frame", 11),
        ("checkpoint", 11),
    ]
    assert _hash(result.times) == _hash(baseline.times)
    for name in baseline.diagnostics:
        assert _hash(result.diagnostics[name]) == _hash(baseline.diagnostics[name])
    if retain_fields:
        for name in baseline.fields:
            assert _hash(result.fields[name]) == _hash(baseline.fields[name])


@pytest.mark.parametrize("equation", ["burgers1d", "navier_stokes2d"])
def test_final_checkpoint_resume_does_not_repeat_a_frame_or_step(monkeypatch, equation):
    snapshots = []
    config = _config(equation)
    reference = solve(config, checkpoint_every=100, checkpoint_callback=snapshots.append)
    assert len(snapshots) == 1
    assert snapshots[0].step == config["steps"]

    def unexpected(*_args):
        pytest.fail("A completed checkpoint must not integrate or emit frames again")

    monkeypatch.setattr(numerics, "_rk4", unexpected)
    monkeypatch.setattr(np.fft, "fft2", unexpected)
    monkeypatch.setattr(np.fft, "ifft2", unexpected)
    resumed = solve(
        config,
        checkpoint=snapshots[0],
        retain_fields=False,
        frame_callback=unexpected,
        checkpoint_every=3,
        checkpoint_callback=unexpected,
    )
    assert _hash(resumed.times) == _hash(reference.times)
    for name in reference.diagnostics:
        assert _hash(resumed.diagnostics[name]) == _hash(reference.diagnostics[name])


@pytest.fixture
def checkpoint():
    snapshots = []
    solve(_config(), checkpoint_every=6, checkpoint_callback=snapshots.append)
    return snapshots[0]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"step": 0}, "checkpoint step"),
        ({"step": True}, "checkpoint step"),
        ({"step": 2.0}, "checkpoint step"),
        ({"step": 12}, "exceeds"),
        ({"state": np.zeros(15)}, "state"),
        ({"state": np.zeros(16, dtype=np.float32)}, "state"),
        ({"state": np.zeros(16, dtype=np.complex128)}, "state"),
        ({"state": np.full(16, np.nan)}, "state"),
        ({"state": [0.0] * 16}, "state"),
        ({"times": np.asarray([0, 0.006])}, "times"),
        ({"times": np.asarray([0, 0.008, 0.012])}, "times"),
        ({"times": np.asarray([0, np.nan])}, "times"),
        ({"times": np.asarray([0, 0.008], dtype=np.float32)}, "times"),
        ({"diagnostics": np.zeros((2, 2))}, "diagnostics"),
        ({"diagnostics": np.zeros((2, 3), dtype=np.float32)}, "diagnostics"),
        ({"diagnostics": np.full((2, 3), np.inf)}, "diagnostics"),
    ],
)
def test_invalid_checkpoint_rejected_before_integration(monkeypatch, checkpoint, change, message):
    def unexpected(*_args):
        pytest.fail("Invalid checkpoint reached numerical integration")

    monkeypatch.setattr(numerics, "_rk4", unexpected)
    with pytest.raises(ValueError, match=message):
        solve(
            _config(),
            checkpoint=replace(checkpoint, **change),
            retain_fields=False,
            frame_callback=unexpected,
        )


def test_navier_stokes_requires_full_precision_complex_spectral_state():
    snapshots = []
    config = _config("navier_stokes2d")
    solve(config, checkpoint_every=6, checkpoint_callback=snapshots.append)
    checkpoint = snapshots[0]
    assert checkpoint.state.dtype == np.complex128
    for state in (checkpoint.state.real, checkpoint.state.astype(np.complex64)):
        with pytest.raises(ValueError, match="state"):
            validate_checkpoint(config, replace(checkpoint, state=state))


@pytest.mark.parametrize("interval", [True, False, 0, -1, 1.5])
def test_invalid_checkpoint_interval(interval):
    with pytest.raises(ValueError, match="checkpoint_every"):
        solve(_config(), checkpoint_every=interval, checkpoint_callback=lambda _: None)


@pytest.mark.parametrize(
    "options",
    [
        {"checkpoint_every": 2},
        {"checkpoint_callback": lambda _: None},
        {"checkpoint_every": 2, "checkpoint_callback": 3},
    ],
)
def test_checkpoint_interval_and_callback_must_be_supplied_together(options):
    with pytest.raises(ValueError, match="checkpoint"):
        solve(_config(), **options)


def test_resume_requires_streaming_and_typed_checkpoint(checkpoint):
    with pytest.raises(ValueError, match="resume requires"):
        solve(_config(), checkpoint=checkpoint)
    with pytest.raises(ValueError, match="frame_callback"):
        solve(_config(), checkpoint=checkpoint, retain_fields=False)
    with pytest.raises(ValueError, match="SolverCheckpoint"):
        validate_checkpoint(_config(), {})


def test_darcy_rejects_time_dependent_checkpoint_options(checkpoint):
    config = {"equation": "darcy2d"}
    with pytest.raises(ValueError, match="Darcy"):
        validate_checkpoint(config, checkpoint)
    with pytest.raises(ValueError, match="Darcy"):
        solve(config, checkpoint=checkpoint)
    with pytest.raises(ValueError, match="Darcy"):
        solve(config, checkpoint_every=2, checkpoint_callback=lambda _: None)


def test_checkpoint_is_a_public_value_object():
    checkpoint = SolverCheckpoint(
        step=1,
        state=np.zeros(16, dtype=np.float64),
        times=np.asarray([0.0]),
        diagnostics=np.zeros((1, 3)),
    )
    validate_checkpoint(_config(), checkpoint)
