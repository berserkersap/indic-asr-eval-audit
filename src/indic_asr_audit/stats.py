"""Corpus error rates with cluster-bootstrap confidence intervals.

Utterances that share a sentence (the same text read by several speakers) are
not independent, so resampling is done over clusters, not utterances.
"""

from collections.abc import Sequence
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
    _, index = np.unique(np.asarray(clusters), return_inverse=True)
    errors = np.bincount(index, weights=[c.errors for c in counts])
    totals = np.bincount(index, weights=[c.total for c in counts])
    return errors, totals


def _resample_index(n_clusters: int, n_resamples: int, seed: int) -> NDArray[np.intp]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_resamples, n_clusters))


def _interval(estimate: float, samples: NDArray[np.float64], confidence: float) -> Interval:
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
