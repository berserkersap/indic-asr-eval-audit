"""RQ2: does a TTS-generated test set rank ASR systems the way real speech does?

Every synthetic utterance has the id of the real recording of the same
sentence, so each comparison is paired: same sentences, same systems, real
audio versus synthetic audio. All scores use the numbers normalisation level,
so differences come from the audio, not from writing conventions.
"""

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from indic_asr_audit.analysis import SUBSETS
from indic_asr_audit.data import Utterance
from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.metrics import Counts, numeric_counts, word_counts
from indic_asr_audit.normalize import Level, normalize
from indic_asr_audit.stats import (
    bootstrap_rate_samples,
    corpus_rate,
    paired_difference,
    percentile_interval,
)


def _word_counts(
    utterances: Sequence[Utterance], hypotheses: dict[str, str], pack: LanguagePack
) -> list[Counts]:
    return [
        word_counts(
            normalize(u.text, pack, Level.NUMBERS),
            normalize(hypotheses[u.id], pack, Level.NUMBERS),
        )
        for u in utterances
    ]


def _rho(real: list[Counts], synth: list[Counts]) -> float:
    """Spearman correlation of per-utterance WER on real and synthetic audio."""
    pairs = [
        (r.errors / r.total, y.errors / y.total)
        for r, y in zip(real, synth, strict=True)
        if r.total and y.total
    ]
    if len(pairs) < 3:
        return float("nan")
    x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def rq2_tables(
    synthetic: Sequence[Utterance],
    real_hyps: dict[str, dict[str, str]],
    synth_hyps: dict[str, dict[str, str]],
    pack: LanguagePack,
    excluded_splits: dict[str, set[str]],
) -> dict[str, pd.DataFrame]:
    """Compare one synthetic set with the real recordings of the same sentences.

    Args:
        synthetic: Manifest of the synthetic set; ids match real utterances.
        real_hyps: {system: {id: hypothesis}} on the real audio.
        synth_hyps: {system: {id: hypothesis}} on the synthetic audio.
        pack: Language pack for normalisation.
        excluded_splits: Splits each system must not be scored on.

    Returns:
        Tables "gap" (per system), "agreement" (ranking) and "utterances"
        (per-utterance error correlation).
    """
    gap, agreement, per_utt = [], [], []
    for subset, keep in SUBSETS.items():
        utts = [u for u in synthetic if keep(u)]
        splits = {u.split for u in utts}
        systems = [
            s for s in synth_hyps if s in real_hyps and not splits & excluded_splits.get(s, set())
        ]
        if not utts or not systems:
            continue
        clusters = [u.sentence_id for u in utts]
        real = {s: _word_counts(utts, real_hyps[s], pack) for s in systems}
        synth = {s: _word_counts(utts, synth_hyps[s], pack) for s in systems}
        for s in systems:
            diff = paired_difference(synth[s], real[s], clusters)
            real_num = [numeric_counts(u.text, real_hyps[s][u.id], pack) for u in utts]
            synth_num = [numeric_counts(u.text, synth_hyps[s][u.id], pack) for u in utts]
            gap.append(
                {
                    "subset": subset,
                    "system": s,
                    "n": len(utts),
                    "wer_real": corpus_rate(real[s]),
                    "wer_synthetic": corpus_rate(synth[s]),
                    "wer_gap": diff.estimate,
                    "low": diff.low,
                    "high": diff.high,
                    "numeric_error_real": corpus_rate(real_num),
                    "numeric_error_synthetic": corpus_rate(synth_num),
                }
            )
            rho = _rho(real[s], synth[s])
            per_utt.append({"subset": subset, "system": s, "spearman_rho": rho})
        if len(systems) >= 2:
            agreement.append({"subset": subset, **_ranking_agreement(real, synth, clusters)})
    return {
        "gap": pd.DataFrame(gap),
        "agreement": pd.DataFrame(agreement),
        "utterances": pd.DataFrame(per_utt),
    }


def _ranking_agreement(
    real: dict[str, list[Counts]], synth: dict[str, list[Counts]], clusters: list[str]
) -> dict[str, object]:
    """Kendall tau between real and synthetic rankings, with a bootstrap interval.

    Both rankings are recomputed on the same sentence resamples, so the
    interval reflects how stable the agreement is, not just one draw.
    """
    systems = list(real)
    real_rates = [corpus_rate(real[s]) for s in systems]
    synth_rates = [corpus_rate(synth[s]) for s in systems]
    tau = float(kendalltau(real_rates, synth_rates).statistic)
    real_samples = bootstrap_rate_samples(real, clusters)
    synth_samples = bootstrap_rate_samples(synth, clusters)
    taus = np.array(
        [kendalltau(r, y).statistic for r, y in zip(real_samples, synth_samples, strict=True)]
    )
    interval = percentile_interval(tau, taus)
    flipped = [
        f"{a}/{b}"
        for (i, a), (j, b) in itertools.combinations(enumerate(systems), 2)
        if (real_rates[i] - real_rates[j]) * (synth_rates[i] - synth_rates[j]) < 0
    ]
    return {
        "n_systems": len(systems),
        "kendall_tau": tau,
        "low": interval.low,
        "high": interval.high,
        "pairs_flipped": len(flipped),
        "flipped": " ".join(flipped),
    }
