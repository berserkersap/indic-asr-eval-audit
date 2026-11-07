"""Where the numeric value error comes from: how numbers are missed and written.

These tables explain the RQ1 and RQ2 results rather than test them, so only
the split comparison has intervals.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

import jiwer
import pandas as pd

from indic_asr_audit.data import Utterance
from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.metrics import missed_values, word_counts
from indic_asr_audit.normalize import Level, normalize, numeric_tokens
from indic_asr_audit.numbers import find_numbers
from indic_asr_audit.stats import bootstrap_rate

Transcripts = Mapping[str, Mapping[str, str]]  # {system: {utterance id: text}}
MISS_KINDS = ("another number", "a word", "dropped", "other")


def _numbers_level(text: str, pack: LanguagePack) -> str:
    return normalize(text, pack, Level.NUMBERS)


def _aligned(ref_words: list[str], hyp_words: list[str]) -> list[str | None]:
    """The hypothesis word aligned with each reference word; None where it was deleted."""
    slots: list[str | None] = [None] * len(ref_words)
    if not ref_words or not hyp_words:
        return slots
    for chunk in jiwer.process_words(" ".join(ref_words), " ".join(hyp_words)).alignments[0]:
        if chunk.type in ("equal", "substitute"):  # same length on both sides
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                slots[chunk.ref_start_idx + k] = hyp_words[chunk.hyp_start_idx + k]
    return slots


def miss_kinds(reference: str, hypothesis: str, pack: LanguagePack) -> Counter[str]:
    """How each missed reference number went wrong, from the word alignment.

    The texts are aligned at the ``numbers`` level, where numbers are digits.
    A missed number's slot holds "another number" (a wrong value, in digits
    or words), "a word" that is not a number (often a misspelled number
    word), or nothing ("dropped"). "other" is a number whose slot cannot be
    found, such as one with a leading zero.
    """
    missed = missed_values(reference, hypothesis, pack)
    if not missed:
        return Counter()
    ref_words = normalize(reference, pack, Level.NUMBERS).split()
    slots = _aligned(ref_words, normalize(hypothesis, pack, Level.NUMBERS).split())
    kinds: Counter[str] = Counter()
    used: set[int] = set()
    for value in sorted(missed.elements()):
        index = next((i for i, w in enumerate(ref_words) if w == value and i not in used), None)
        if index is None:
            kinds["other"] += 1
            continue
        used.add(index)
        word = slots[index]
        if word is None:
            kinds["dropped"] += 1
        elif find_numbers([word], pack):
            kinds["another number" if word != value else "other"] += 1
        else:
            kinds["a word"] += 1
    return kinds


def miss_kinds_table(
    utterances: Sequence[Utterance], transcripts: Transcripts, pack: LanguagePack
) -> pd.DataFrame:
    """Per system: numbers scored, missed, and how the missed ones went wrong."""
    rows = []
    for system, hyps in transcripts.items():
        kinds: Counter[str] = Counter()
        for u in utterances:
            kinds += miss_kinds(u.text, hyps[u.id], pack)
        total = sum(sum(missed_values(u.text, "", pack).values()) for u in utterances)
        rows.append(
            {"system": system, "numbers": total, "missed": sum(kinds.values())}
            | {kind: kinds[kind] for kind in MISS_KINDS}
        )
    return pd.DataFrame(rows)


def missed_by_all(
    utterances: Sequence[Utterance], transcripts: Transcripts, pack: LanguagePack
) -> pd.DataFrame:
    """Reference numbers that every system misses: candidates for a reference error."""
    rows = []
    for u in utterances:
        common: Counter[str] | None = None
        for hyps in transcripts.values():
            missed = missed_values(u.text, hyps[u.id], pack)
            common = missed if common is None else common & missed
        for value in sorted((common or Counter()).elements()):
            rows.append({"id": u.id, "sentence_id": u.sentence_id, "value": value})
    return pd.DataFrame(rows, columns=["id", "sentence_id", "value"])


def digit_share(texts: Sequence[str], pack: LanguagePack) -> tuple[int, int]:
    """Numbers written with digits and numbers written as words, over ``texts``.

    The article "ek" parses as the number 1, so the word count is an upper bound.
    """
    digits = words = 0
    for text in texts:
        for span in find_numbers(numeric_tokens(text, pack), pack):
            if span.written_as_digits:
                digits += 1
            else:
                words += 1
    return digits, words


def number_format_table(
    utterances: Sequence[Utterance], transcripts: Transcripts, pack: LanguagePack
) -> pd.DataFrame:
    """Per system, and for the reference: how many numbers are written as digits."""
    sources = {"reference": {u.id: u.text for u in utterances}, **transcripts}
    rows = []
    for name, texts in sources.items():
        digits, words = digit_share([texts[u.id] for u in utterances], pack)
        rows.append(
            {
                "system": name,
                "as_digits": digits,
                "as_words": words,
                "digit_share": digits / (digits + words) if digits + words else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def split_wer_table(
    utterances: Sequence[Utterance], transcripts: Transcripts, pack: LanguagePack
) -> pd.DataFrame:
    """WER at the numbers level on FLEURS train against dev and test, per system.

    A system trained on FLEURS train scores better there than on the other
    splits; one that did not see it scores about the same. This is a probe,
    so systems are scored on train here even where RQ1 leaves it out.
    """
    groups = {"train": {"train"}, "dev and test": {"dev", "test"}}
    rows = []
    for system, hyps in transcripts.items():
        for group, splits in groups.items():
            chosen = [u for u in utterances if u.split in splits]
            if not chosen:
                continue
            counts = [
                word_counts(_numbers_level(u.text, pack), _numbers_level(hyps[u.id], pack))
                for u in chosen
            ]
            interval = bootstrap_rate(counts, [u.sentence_id for u in chosen])
            rows.append(
                {
                    "system": system,
                    "splits": group,
                    "wer": interval.estimate,
                    "low": interval.low,
                    "high": interval.high,
                    "n": len(chosen),
                }
            )
    return pd.DataFrame(rows)


def number_format_change(
    utterances: Sequence[Utterance],
    real: Transcripts,
    synthetic: Transcripts,
    pack: LanguagePack,
) -> pd.DataFrame:
    """Per system: share of numbers written as digits on real and on synthetic audio."""
    rows = []
    for system in synthetic:
        if system not in real:
            continue
        shares = []
        for hyps in (real[system], synthetic[system]):
            digits, words = digit_share([hyps[u.id] for u in utterances], pack)
            shares.append(digits / (digits + words) if digits + words else float("nan"))
        rows.append(
            {"system": system, "digit_share_real": shares[0], "digit_share_synthetic": shares[1]}
        )
    return pd.DataFrame(rows)


def audit_tables(
    utterances: Sequence[Utterance], transcripts: Transcripts, pack: LanguagePack
) -> dict[str, pd.DataFrame]:
    """The RQ1 audit tables: held-out utterances with numbers, and the split probe."""
    held_out = [u for u in utterances if u.has_number and u.split != "train"]
    return {
        "missed_kinds": miss_kinds_table(held_out, transcripts, pack),
        "missed_by_all": missed_by_all(held_out, transcripts, pack),
        "number_format": number_format_table(held_out, transcripts, pack),
        "split_wer": split_wer_table([u for u in utterances if u.has_number], transcripts, pack),
    }
