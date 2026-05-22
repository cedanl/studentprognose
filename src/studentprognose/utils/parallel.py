"""Parallel execution helper with crash fallback."""

import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError


FALLBACK_N_JOBS = 2


def run_parallel_with_fallback(delayed_jobs, n_jobs: int) -> list:
    """Run joblib jobs; on worker crash, retry with fewer workers.

    A ``TerminatedWorkerError`` indicates a worker process died on signal level
    (OOM-kill or native segfault). Both have the same symptom from joblib's
    perspective. Retrying with ``FALLBACK_N_JOBS`` workers strongly reduces
    memory pressure and is enough to recover from the typical Windows case
    documented in issue #197.

    Args:
        delayed_jobs: iterable of ``joblib.delayed(...)`` calls. Materialised to
            a list so a retry can re-iterate.
        n_jobs: number of workers for the initial attempt.

    Returns:
        list of results from ``joblib.Parallel``.

    Raises:
        TerminatedWorkerError: when the fallback also crashes, or when the
            initial ``n_jobs`` was already at or below ``FALLBACK_N_JOBS``.
    """
    jobs = list(delayed_jobs)
    try:
        return joblib.Parallel(n_jobs=n_jobs)(jobs)
    except TerminatedWorkerError:
        if n_jobs <= FALLBACK_N_JOBS:
            raise
        print(
            f"Worker proces crashte met n_jobs={n_jobs} (mogelijk OOM-kill of "
            f"segfault). Opnieuw proberen met n_jobs={FALLBACK_N_JOBS}. Als dit "
            "ook faalt: zet 'runtime.cpu_count': 1 in configuration.json."
        )
        return joblib.Parallel(n_jobs=FALLBACK_N_JOBS)(jobs)
