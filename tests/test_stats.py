import math

import numpy as np
import pytest

from indic_asr_audit.metrics import Counts
from indic_asr_audit.stats import bootstrap_rate, corpus_rate, paired_difference


def test_corpus_rate_weights_by_length() -> None:
    assert corpus_rate([Counts(1, 1), Counts(0, 9)]) == pytest.approx(0.1)


def test_corpus_rate_empty_is_nan() -> None:
    assert math.isnan(corpus_rate([Counts(0, 0)]))


def test_bootstrap_covers_planted_rate() -> None:
    rng = np.random.default_rng(1)
    counts = [Counts(int(rng.binomial(20, 0.2)), 20) for _ in range(400)]
    clusters = [str(i // 2) for i in range(400)]
    interval = bootstrap_rate(counts, clusters, seed=0)
    assert interval.low < 0.2 < interval.high
    assert interval.low <= interval.estimate <= interval.high


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    counts = [Counts(i % 3, 5) for i in range(30)]
    clusters = [str(i) for i in range(30)]
    assert bootstrap_rate(counts, clusters, seed=7) == bootstrap_rate(counts, clusters, seed=7)


def test_paired_difference_of_identical_systems_is_zero() -> None:
    counts = [Counts(i % 4, 10) for i in range(50)]
    clusters = [str(i) for i in range(50)]
    interval = paired_difference(counts, counts, clusters)
    assert (interval.estimate, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_paired_difference_detects_planted_gap() -> None:
    a = [Counts(3, 10) for _ in range(100)]
    b = [Counts(2, 10) for _ in range(100)]
    interval = paired_difference(a, b, [str(i) for i in range(100)])
    assert interval.estimate == pytest.approx(0.1)
    assert interval.low > 0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_rate([Counts(1, 1)], ["a", "b"])
