"""Tests for the parallel fallback helper."""

from unittest.mock import patch

import joblib
import pytest
from joblib.externals.loky.process_executor import TerminatedWorkerError

from studentprognose.utils.parallel import (
    FALLBACK_N_JOBS,
    run_parallel_with_fallback,
)


def _square(x):
    return x * x


def _delayed_squares(values):
    return [joblib.delayed(_square)(v) for v in values]


def test_happy_path_returns_results_without_fallback():
    result = run_parallel_with_fallback(_delayed_squares([1, 2, 3]), n_jobs=2)
    assert result == [1, 4, 9]


def test_terminated_worker_triggers_fallback_to_two_jobs():
    call_count = {"n": 0}
    captured_n_jobs = []

    class FakeParallel:
        def __init__(self, n_jobs):
            captured_n_jobs.append(n_jobs)

        def __call__(self, jobs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TerminatedWorkerError("simulated worker death")
            return ["fallback-result"]

    with patch("studentprognose.utils.parallel.joblib.Parallel", FakeParallel):
        result = run_parallel_with_fallback(_delayed_squares([1, 2, 3]), n_jobs=8)

    assert result == ["fallback-result"]
    assert call_count["n"] == 2
    assert captured_n_jobs == [8, FALLBACK_N_JOBS]


def test_fallback_does_not_swallow_when_n_jobs_already_at_threshold():
    class AlwaysCrash:
        def __init__(self, n_jobs):
            pass

        def __call__(self, jobs):
            raise TerminatedWorkerError("still dying")

    with patch("studentprognose.utils.parallel.joblib.Parallel", AlwaysCrash):
        with pytest.raises(TerminatedWorkerError):
            run_parallel_with_fallback(_delayed_squares([1]), n_jobs=FALLBACK_N_JOBS)


def test_other_exceptions_are_not_caught():
    class BoomParallel:
        def __init__(self, n_jobs):
            pass

        def __call__(self, jobs):
            raise ValueError("not a worker crash")

    with patch("studentprognose.utils.parallel.joblib.Parallel", BoomParallel):
        with pytest.raises(ValueError, match="not a worker crash"):
            run_parallel_with_fallback(_delayed_squares([1]), n_jobs=8)


def test_generator_jobs_can_be_retried():
    """Helper must materialise the iterable so a retry sees the same jobs."""
    call_count = {"n": 0}
    seen_job_counts = []

    class FakeParallel:
        def __init__(self, n_jobs):
            pass

        def __call__(self, jobs):
            seen_job_counts.append(len(list(jobs)))
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TerminatedWorkerError("die once")
            return ["ok"]

    gen = (joblib.delayed(_square)(v) for v in [1, 2, 3, 4])

    with patch("studentprognose.utils.parallel.joblib.Parallel", FakeParallel):
        run_parallel_with_fallback(gen, n_jobs=8)

    assert seen_job_counts == [4, 4]
