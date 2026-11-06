"""RQ1: does the way transcripts are normalised change what we conclude?

Every system is scored at every normalisation level on the same utterances,
with cluster-bootstrap intervals. The outputs answer three questions:

1. How much of the reported error is scoring convention, not recognition?
   (paired change in WER from one level to the next)
2. Does the ranking of systems change with the level? (Kendall tau)
3. Does a system's output convention matter? (Saaras v4 in "transcribe"
   mode writes digits, in "verbatim" mode writes number words)
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from indic_asr_audit.data import Utterance
from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.metrics import (
    Counts,
    char_counts,
    number_word_errors,
    numeric_counts,
    word_counts,
)
from indic_asr_audit.normalize import Level, normalize
from indic_asr_audit.stats import bootstrap_rate, corpus_rate, paired_difference

# "numeric" and "control" mean the reference does or does not contain a digit;
# a control may still contain a number written as a word. "held_out_numeric"
# leaves out FLEURS train, so systems trained on it (MMS) can be compared too;
# it is the headline subset, "numeric" is larger but has fewer systems.
SUBSETS: dict[str, Callable[[Utterance], bool]] = {
    "held_out_numeric": lambda u: u.has_number and u.split != "train",
    "numeric": lambda u: u.has_number,
    "control": lambda u: not u.has_number,
}


@dataclass(frozen=True)
class Scored:
    """Per-utterance counts for one system at one level."""

    words: list[Counts]
    chars: list[Counts]


def score_system(
    utterances: Sequence[Utterance], hypotheses: dict[str, str], pack: LanguagePack, level: Level
) -> Scored:
    """Word and character counts for every utterance at one level."""
    words, chars = [], []
    for utt in utterances:
        ref = normalize(utt.text, pack, level)
        hyp = normalize(hypotheses[utt.id], pack, level)
        words.append(word_counts(ref, hyp))
        chars.append(char_counts(ref, hyp))
    return Scored(words, chars)


def _row(
    system: str, subset: str, level: str, metric: str, counts: list[Counts], clusters: list[str]
) -> dict[str, object]:
    interval = bootstrap_rate(counts, clusters)
    return {
        "system": system,
        "subset": subset,
        "level": level,
        "metric": metric,
        "estimate": interval.estimate,
        "low": interval.low,
        "high": interval.high,
        "n": len(counts),
    }


def rq1_tables(
    utterances: Sequence[Utterance],
    transcripts: dict[str, dict[str, str]],
    pack: LanguagePack,
    excluded_splits: dict[str, set[str]],
) -> dict[str, pd.DataFrame]:
    """Compute every RQ1 table.

    Args:
        utterances: The manifest.
        transcripts: {system: {utterance id: hypothesis}}, complete per system.
        pack: Language pack used for normalisation and numbers.
        excluded_splits: Splits a system must not be scored on, e.g. the
            training split of a model trained on it.

    Returns:
        Tables named "scores", "level_steps", "ranks", "convention",
        "wer_vs_numeric" and "diagnostics".
    """
    levels = list(Level)
    scores, steps, ranks, disagreement, diagnostics = [], [], [], [], []
    scored: dict[tuple[str, str], dict[Level, Scored]] = {}

    for subset, keep in SUBSETS.items():
        subset_utts = [u for u in utterances if keep(u)]
        systems = [
            s
            for s in transcripts
            if not {u.split for u in subset_utts} & excluded_splits.get(s, set())
        ]
        clusters = [u.sentence_id for u in subset_utts]
        subset_scores = []
        numeric_by_system: dict[str, list[Counts]] = {}
        for system in systems:
            by_level = {
                level: score_system(subset_utts, transcripts[system], pack, level)
                for level in levels
            }
            scored[(system, subset)] = by_level
            for level, s in by_level.items():
                subset_scores.append(_row(system, subset, level, "wer", s.words, clusters))
                subset_scores.append(_row(system, subset, level, "cer", s.chars, clusters))
            hyps = transcripts[system]
            numbers = [numeric_counts(u.text, hyps[u.id], pack) for u in subset_utts]
            if any(c.total for c in numbers):
                lenient = [
                    numeric_counts(u.text, hyps[u.id], pack, lenient=True) for u in subset_utts
                ]
                subset_scores.append(_row(system, subset, "-", "numeric_error", numbers, clusters))
                subset_scores.append(
                    _row(system, subset, "-", "numeric_error_lenient", lenient, clusters)
                )
                numeric_by_system[system] = numbers
            words = by_level[Level.NUMBERS].words
            pairs = [
                (normalize(u.text, pack, Level.NUMBERS), normalize(hyps[u.id], pack, Level.NUMBERS))
                for u in subset_utts
            ]
            diagnostics.append(
                _diagnostics(
                    system,
                    subset,
                    words,
                    numbers,
                    [len(hyp.split()) for _, hyp in pairs],
                    [number_word_errors(ref, hyp) for ref, hyp in pairs],
                    clusters,
                )
            )
            for before, after in itertools.pairwise(levels):
                diff = paired_difference(by_level[after].words, by_level[before].words, clusters)
                steps.append(
                    {
                        "system": system,
                        "subset": subset,
                        "step": f"{before}->{after}",
                        "wer_change": diff.estimate,
                        "low": diff.low,
                        "high": diff.high,
                    }
                )

        scores += subset_scores
        if len(systems) >= 2:
            wer = pd.DataFrame([r for r in subset_scores if r["metric"] == "wer"])
            wide = wer.pivot(index="system", columns="level", values="estimate")
            ranks += [{"subset": subset, **row} for row in _rank_changes(wide)]
        if len(numeric_by_system) >= 2:
            wer_counts = {s: scored[(s, subset)][Level.NUMBERS].words for s in numeric_by_system}
            disagreement.append(
                {"subset": subset, **metric_disagreement(wer_counts, numeric_by_system, clusters)}
            )

    convention = _convention_gap(scored, levels, utterances)
    return {
        "scores": pd.DataFrame(scores),
        "level_steps": pd.DataFrame(steps),
        "ranks": pd.DataFrame(ranks),
        "convention": convention,
        "wer_vs_numeric": pd.DataFrame(disagreement),
        "diagnostics": pd.DataFrame(diagnostics),
    }


def _diagnostics(
    system: str,
    subset: str,
    words: list[Counts],
    numbers: list[Counts],
    hyp_lengths: list[int],
    number_errors: list[int],
    clusters: list[str],
) -> dict[str, object]:
    """Two checks on what drives a system's WER, at the numbers level.

    Long outputs: hypotheses more than twice as long as the reference, which
    for Whisper are repetition loops; WER is also given without them. Number
    share: numbers as a share of reference words, and the WER points from
    word errors that involve a word with a digit (``number_word_errors``).
    """
    long = [h > 2 * c.total for h, c in zip(hyp_lengths, words, strict=True)]
    kept = [c for c, is_long in zip(words, long, strict=True) if not is_long]
    ref_words = sum(c.total for c in words)
    return {
        "system": system,
        "subset": subset,
        "utterances": len(words),
        "sentences": len(set(clusters)),
        "long_outputs": sum(long),
        "wer": corpus_rate(words),
        "wer_without_long": corpus_rate(kept),
        "reference_words": ref_words,
        "reference_numbers": sum(c.total for c in numbers),
        "missed_numbers": sum(c.errors for c in numbers),
        "word_errors": sum(c.errors for c in words),
        "wer_from_number_words": sum(number_errors) / ref_words if ref_words else 0.0,
    }


def metric_disagreement(
    wer: dict[str, list[Counts]], numeric: dict[str, list[Counts]], clusters: list[str]
) -> dict[str, object]:
    """Do WER and the numeric value error rank the systems the same way?

    A pair of systems is reversed when one has the lower WER but the higher
    numeric error. The reversal is significant when both paired differences
    have intervals that exclude zero.
    """
    systems = list(wer)
    wer_rates = {s: corpus_rate(wer[s]) for s in systems}
    num_rates = {s: corpus_rate(numeric[s]) for s in systems}
    reversed_pairs, significant = [], []
    for a, b in itertools.combinations(systems, 2):
        if (wer_rates[a] - wer_rates[b]) * (num_rates[a] - num_rates[b]) >= 0:
            continue
        reversed_pairs.append(f"{a}/{b}")
        d_wer = paired_difference(wer[a], wer[b], clusters)
        d_num = paired_difference(numeric[a], numeric[b], clusters)
        if all(d.low > 0 or d.high < 0 for d in (d_wer, d_num)):
            significant.append(f"{a}/{b}")
    tau = kendalltau([wer_rates[s] for s in systems], [num_rates[s] for s in systems]).statistic
    return {
        "n_systems": len(systems),
        "kendall_tau": float(tau),
        "pairs_reversed": len(reversed_pairs),
        "significant_reversals": len(significant),
        "reversed": " ".join(reversed_pairs),
        "significant": " ".join(significant),
    }


def _rank_changes(wer: pd.DataFrame) -> list[dict[str, object]]:
    """Compare the ranking at the whisper level with every other level.

    With only a handful of systems Kendall tau has few possible values, so
    the flipped pairs are listed too; they are easier to interpret.
    """
    by_level = {level: {str(k): float(v) for k, v in wer[level].items()} for level in Level}
    base = by_level[Level.WHISPER]
    rows: list[dict[str, object]] = []
    for level in list(Level)[1:]:
        other = by_level[level]
        flipped = [
            f"{a}/{b}"
            for a, b in itertools.combinations(base, 2)
            if (base[a] - base[b]) * (other[a] - other[b]) < 0
        ]
        tau = kendalltau(list(base.values()), [other[k] for k in base]).statistic
        rows.append(
            {
                "versus": f"whisper vs {level}",
                "n_systems": len(base),
                "kendall_tau": float(tau),
                "pairs_flipped": len(flipped),
                "flipped": " ".join(flipped),
            }
        )
    return rows


def _convention_gap(
    scored: dict[tuple[str, str], dict[Level, Scored]],
    levels: list[Level],
    utterances: Sequence[Utterance],
    pair: tuple[str, str] = ("saaras-v4-verbatim", "saaras-v4-transcribe"),
) -> pd.DataFrame:
    """WER of verbatim minus transcribe output for the same model, per level."""
    rows = []
    for subset, keep in SUBSETS.items():
        a, b = scored.get((pair[0], subset)), scored.get((pair[1], subset))
        if a is None or b is None:
            continue
        clusters = [u.sentence_id for u in utterances if keep(u)]
        for level in levels:
            diff = paired_difference(a[level].words, b[level].words, clusters)
            rows.append(
                {
                    "subset": subset,
                    "level": level,
                    "wer_gap": diff.estimate,
                    "low": diff.low,
                    "high": diff.high,
                }
            )
    return pd.DataFrame(rows)


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write each table as CSV, rounded for readable diffs; an empty table removes its file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        if table.empty:
            path.unlink(missing_ok=True)
        else:
            table.round(4).to_csv(path, index=False, lineterminator="\n")
