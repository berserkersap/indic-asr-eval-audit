"""RQ2: does a TTS-generated test set rank ASR systems the way real speech does?

Every synthetic utterance has the id of a real recording of the same
sentence, so each comparison is paired: same sentences, same systems, real
audio versus synthetic audio. All scores use the numbers normalisation level,
so differences come from the audio, not from writing conventions.

Agreement needs a yardstick: two real speakers reading the same sentence do
not produce identical rankings either. ``real_ceiling`` runs the same
comparison between two real recordings of each sentence, which bounds how
well any synthetic set could agree with one real speaker.

The intervals resample sentences. They do not include variation from the
choice of TTS voice or of the real speaker.
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

Hypotheses = dict[str, dict[str, str]]


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
    real_hyps: Hypotheses,
    synth_hyps: Hypotheses,
    pack: LanguagePack,
    excluded_splits: dict[str, set[str]],
    related: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compare one synthetic set with the real recordings of the same sentences.

    Args:
        synthetic: Manifest of the synthetic set; ids match real utterances.
        real_hyps: {system: {id: hypothesis}} on the real audio.
        synth_hyps: {system: {id: hypothesis}} on the synthetic audio.
        pack: Language pack for normalisation.
        excluded_splits: Splits each system must not be scored on.
        related: ASR systems from the same family as the TTS engine; ranking
            agreement is also reported without them.

    Returns:
        Tables "gap" (per system), "agreement" (ranking) and "utterances"
        (per-utterance error correlation).
    """
    related = related or set()
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
            per_utt.append({"subset": subset, "system": s, "spearman_rho": _rho(real[s], synth[s])})
        for label, chosen in (
            ("all", systems),
            ("unrelated", [s for s in systems if s not in related]),
        ):
            if len(chosen) >= 2 and (label == "all" or len(chosen) < len(systems)):
                rows = ranking_agreement(
                    {s: real[s] for s in chosen}, {s: synth[s] for s in chosen}, clusters
                )
                agreement.append({"subset": subset, "systems": label, **rows})
    return {
        "gap": pd.DataFrame(gap),
        "agreement": pd.DataFrame(agreement),
        "utterances": pd.DataFrame(per_utt),
    }


def ranking_agreement(
    real: dict[str, list[Counts]], synth: dict[str, list[Counts]], clusters: list[str]
) -> dict[str, object]:
    """Kendall tau between real and synthetic rankings, with a bootstrap interval.

    Real and synthetic rates come from one bootstrap call, so both rankings
    are recomputed on the same sentence resamples. Resamples where a ranking
    is all ties give no tau; their number is reported. A flipped pair counts
    as significant only if the two systems differ on real speech (paired
    interval excludes zero); flips among near-ties are noise.
    """
    systems = list(real)
    samples = bootstrap_rate_samples(
        {**{f"real:{s}": real[s] for s in systems}, **{f"synth:{s}": synth[s] for s in systems}},
        clusters,
    )
    real_samples, synth_samples = samples[:, : len(systems)], samples[:, len(systems) :]
    real_rates = [corpus_rate(real[s]) for s in systems]
    synth_rates = [corpus_rate(synth[s]) for s in systems]
    tau = float(kendalltau(real_rates, synth_rates).statistic)
    taus = np.array(
        [kendalltau(r, y).statistic for r, y in zip(real_samples, synth_samples, strict=True)]
    )
    interval = percentile_interval(tau, taus)
    flipped, significant = [], 0
    for (i, a), (j, b) in itertools.combinations(enumerate(systems), 2):
        if (real_rates[i] - real_rates[j]) * (synth_rates[i] - synth_rates[j]) < 0:
            flipped.append(f"{a}/{b}")
            real_diff = paired_difference(real[a], real[b], clusters)
            significant += real_diff.low > 0 or real_diff.high < 0
    return {
        "n_systems": len(systems),
        "kendall_tau": tau,
        "low": interval.low,
        "high": interval.high,
        "nan_resamples": int(np.isnan(taus).sum()),
        "pairs_flipped": len(flipped),
        "significant_flips": significant,
        "flipped": " ".join(flipped),
    }


def repeated_sentences(utterances: Sequence[Utterance]) -> dict[str, list[Utterance]]:
    """Recordings of every sentence read at least twice, in manifest order."""
    takes: dict[str, list[Utterance]] = {}
    for utt in utterances:
        takes.setdefault(utt.sentence_id, []).append(utt)
    return {sentence: group for sentence, group in takes.items() if len(group) >= 2}


def real_ceiling(
    utterances: Sequence[Utterance],
    real_hyps: Hypotheses,
    pack: LanguagePack,
    excluded_splits: dict[str, set[str]],
) -> dict[str, pd.DataFrame]:
    """The RQ2 comparison between two real recordings of the same sentence.

    For every sentence with at least two recordings, the first (in manifest
    order) plays "real" and the second plays "synthetic". The result is the
    agreement to expect from a second human speaker.
    """
    pairs = [(group[0], group[1]) for group in repeated_sentences(utterances).values()]
    first = [a for a, _ in pairs]
    second = {s: {a.id: hyps[b.id] for a, b in pairs} for s, hyps in real_hyps.items()}
    return rq2_tables(first, real_hyps, second, pack, excluded_splits)


def selection_table(
    real: Sequence[Utterance],
    kept_ids: set[str],
    real_hyps: Hypotheses,
    pack: LanguagePack,
    excluded_splits: dict[str, set[str]],
) -> pd.DataFrame:
    """Real-speech WER on sentences the synthetic set kept versus rejected.

    If rejected sentences are much harder, the synthetic set is easier than
    the real one for a reason that has nothing to do with TTS. Every
    sentence not in ``kept_ids`` counts as rejected, so pass the synthetic
    manifest of a full run, not a numeric-only one.
    """
    first_takes: dict[str, Utterance] = {}
    for utt in real:
        first_takes.setdefault(utt.sentence_id, utt)
    rows = []
    for system, hyps in real_hyps.items():
        takes = [
            u for u in first_takes.values() if u.split not in excluded_splits.get(system, set())
        ]
        for label, group in (
            ("kept", [u for u in takes if u.id in kept_ids]),
            ("rejected", [u for u in takes if u.id not in kept_ids]),
        ):
            if group:
                counts = _word_counts(group, hyps, pack)
                rows.append(
                    {
                        "system": system,
                        "sentences": label,
                        "n": len(group),
                        "wer": corpus_rate(counts),
                    }
                )
    return pd.DataFrame(rows)
