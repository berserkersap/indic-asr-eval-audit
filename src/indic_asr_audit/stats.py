"""Corpus error rates with cluster-bootstrap confidence intervals.

Utterances that share a sentence (the same text read by several speakers) are
not independent, so resampling is done over clusters, not utterances. A
speaker who reads several sentences also links them; FLEURS has no speaker
ids, so that dependence is not modelled and the intervals may be a little
narrow.

Resamples with nothing to score (a zero denominator) are dropped before the
percentiles are taken.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from indic_asr_audit.metrics import Counts

DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile confidence interval."""

    estimate: float
    low: float
    high: float


def corpus_rate(counts: Sequence[Counts]) -> float:
    """Return sum(errors) / sum(total), or nan when there is nothing to score."""
    total = sum(c.total for c in counts)
    return sum(c.errors for c in counts) / total if total else float("nan")


def _cluster_sums(
    counts: Sequence[Counts], clusters: Sequence[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(counts) != len(clusters):
        raise ValueError("counts and clusters must have the same length")
    if not counts:
        raise ValueError("nothing to resample")
    _, index = np.unique(np.asarray(clusters), return_inverse=True)
    errors = np.bincount(index, weights=[c.errors for c in counts])
    totals = np.bincount(index, weights=[c.total for c in counts])
    return errors, totals


def _resample_index(n_clusters: int, n_resamples: int, seed: int) -> NDArray[np.intp]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_resamples, n_clusters))


def _interval(estimate: float, samples: NDArray[np.float64], confidence: float) -> Interval:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if np.isnan(samples).all():
        return Interval(estimate, float("nan"), float("nan"))
    tail = (1 - confidence) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return Interval(estimate, float(low), float(high))


def bootstrap_rate(
    counts: Sequence[Counts],
    clusters: Sequence[str],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Corpus error rate with a cluster-bootstrap confidence interval."""
    errors, totals = _cluster_sums(counts, clusters)
    idx = _resample_index(len(errors), n_resamples, seed)
    with np.errstate(invalid="ignore"):
        samples = errors[idx].sum(axis=1) / totals[idx].sum(axis=1)
    return _interval(corpus_rate(counts), samples, confidence)


def paired_difference(
    counts_a: Sequence[Counts],
    counts_b: Sequence[Counts],
    clusters: Sequence[str],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Rate of A minus rate of B, both resampled with the same clusters.

    Pairing removes the shared difficulty of each sentence, so the interval is
    much tighter than comparing two independent intervals.
    """
    errors_a, totals_a = _cluster_sums(counts_a, clusters)
    errors_b, totals_b = _cluster_sums(counts_b, clusters)
    idx = _resample_index(len(errors_a), n_resamples, seed)
    with np.errstate(invalid="ignore"):
        samples = errors_a[idx].sum(axis=1) / totals_a[idx].sum(axis=1) - errors_b[idx].sum(
            axis=1
        ) / totals_b[idx].sum(axis=1)
    return _interval(corpus_rate(counts_a) - corpus_rate(counts_b), samples, confidence)


def bootstrap_rate_samples(
    counts_by_system: Mapping[str, Sequence[Counts]],
    clusters: Sequence[str],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Corpus rates of several systems on the same resamples.

    Returns an array of shape (n_resamples, n_systems), columns in the order
    of ``counts_by_system``, so statistics of the ranking (such as Kendall
    tau) can be bootstrapped.
    """
    sums = [_cluster_sums(counts, clusters) for counts in counts_by_system.values()]
    idx = _resample_index(len(sums[0][0]), n_resamples, seed)
    with np.errstate(invalid="ignore"):
        rates = [errors[idx].sum(axis=1) / totals[idx].sum(axis=1) for errors, totals in sums]
    return np.stack(rates, axis=1)


def percentile_interval(
    estimate: float, samples: NDArray[np.float64], confidence: float = 0.95
) -> Interval:
    """Percentile interval of bootstrap samples around an estimate."""
    return _interval(estimate, samples, confidence)
