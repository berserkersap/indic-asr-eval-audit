"""Per-utterance error counts. Corpus rates and confidence intervals live in stats."""

import unicodedata
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import jiwer

from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.normalize import numeric_tokens
from indic_asr_audit.numbers import find_numbers, spell_number


@dataclass(frozen=True)
class Counts:
    """Errors and reference size for one utterance.

    Corpus rates are sum(errors) / sum(total), never a mean of per-utterance
    rates, so long utterances weigh more, as in standard WER.
    """

    errors: int
    total: int


def word_counts(reference: str, hypothesis: str) -> Counts:
    """Word edit distance between two normalised strings."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return Counts(errors=len(hyp), total=0)
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return Counts(errors=out.substitutions + out.deletions + out.insertions, total=len(ref))


def _has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


def number_word_errors(reference: str, hypothesis: str) -> int:
    """Word errors in the alignment that involve a word containing a digit.

    Counts substitutions, deletions and insertions where the reference or the
    hypothesis word has a digit. A garbled number can cost several: "2 ी 0"
    for "20" is one substitution and two insertions, two of them with digits.
    Use on strings normalised at the numbers level, where numbers are digits.
    """
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return sum(map(_has_digit, hyp))
    errors = 0
    for chunk in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        ref_words = ref[chunk.ref_start_idx : chunk.ref_end_idx]
        hyp_words = hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]
        if chunk.type == "substitute":
            pairs = zip(ref_words, hyp_words, strict=True)
            errors += sum(_has_digit(r) or _has_digit(h) for r, h in pairs)
        elif chunk.type == "delete":
            errors += sum(map(_has_digit, ref_words))
        elif chunk.type == "insert":
            errors += sum(map(_has_digit, hyp_words))
    return errors


def char_counts(reference: str, hypothesis: str) -> Counts:
    """Character edit distance between two normalised strings.

    Characters are Unicode code points, not grapheme clusters, and spaces are
    excluded so that word splitting ("raaton-raat" vs "raaton raat") costs
    nothing.
    """
    ref, hyp = reference.replace(" ", ""), hypothesis.replace(" ", "")
    if not ref:
        return Counts(errors=len(hyp), total=0)
    out = jiwer.process_characters(ref, hyp)
    return Counts(errors=out.substitutions + out.deletions + out.insertions, total=len(ref))


def numeric_counts(
    reference: str, hypothesis: str, pack: LanguagePack, *, lenient: bool = False
) -> Counts:
    """Count reference numbers whose value is missing from the hypothesis.

    Only numbers written as digits in the raw reference are scored, because
    those are the values a downstream system must get right. Matching is by
    value and order-free (a multiset), so "53" is matched by "tirepan" or
    "53" anywhere in the hypothesis. Extra numbers in the hypothesis are not
    counted; WER already penalises them.

    Matching is lenient in two known ways: the article "ek" and the verb
    "do" (give) parse as 1 and 2, and leading zeros are dropped (007 is 7).

    With ``lenient``, a missed number still counts as found when the
    hypothesis has a misspelled number word for it, such as "pandra" for
    "pandrah" (see ``_spelled_run``); each such run forgives one number.
    Wrong digits are never forgiven.
    """
    wanted, found = _wanted_and_found(reference, hypothesis, pack)
    missed = wanted - found
    if lenient:
        words = _words(hypothesis, pack)
        taken: set[int] = set()
        for value in list(missed.elements()):
            run = _spelled_run(value, words, pack, taken)
            if run is not None:
                taken.update(run)
                missed[value] -= 1
    return Counts(errors=sum(missed.values()), total=sum(wanted.values()))


def _wanted_and_found(
    reference: str, hypothesis: str, pack: LanguagePack
) -> tuple[Counter[str], Counter[str]]:
    """Values written as digits in the reference, and every value in the hypothesis."""
    ref_spans = find_numbers(numeric_tokens(reference, pack), pack)
    wanted = Counter(s.value for s in ref_spans if s.written_as_digits)
    found = Counter(s.value for s in find_numbers(numeric_tokens(hypothesis, pack), pack))
    return wanted, found


def missed_values(reference: str, hypothesis: str, pack: LanguagePack) -> Counter[str]:
    """The reference numbers that ``numeric_counts`` counts as missed (strict)."""
    wanted, found = _wanted_and_found(reference, hypothesis, pack)
    return wanted - found


def _words(text: str, pack: LanguagePack) -> list[str]:
    """Words with punctuation removed but digits left attached ("7ttar" stays one word)."""
    cleaned = "".join(
        " " if unicodedata.category(c)[0] in "PS" else c for c in unicodedata.normalize("NFC", text)
    )
    return [pack.canonical_word(w) for w in cleaned.split()]


def _spelled_run(
    value: str, tokens: list[str], pack: LanguagePack, taken: set[int], cutoff: float = 0.25
) -> range | None:
    """The first free run of word tokens that is a near spelling of the value in words.

    The run must have as many words as the standard spelling, contain no
    digits, use no token in ``taken``, not overlap any word that already reads
    as a number (exact readings are matched before this, and a different
    value is a different number), and be within ``cutoff`` normalised
    character edit distance of the spelling. This forgives "pandra" for
    "pandrah", but not "1093" for 1963, "7ttar" for 70, or "taintees" (33)
    for "paintees" (35). A slip inside a multi-word number ("unnees so assi"
    for 1980) is not forgiven, because its other words read as numbers.
    """
    from rapidfuzz.distance import Levenshtein

    spelled = pack.canonical_word(spell_number(Fraction(value), pack))
    width = len(spelled.split())
    numbers = find_numbers(tokens, pack)
    for start in range(len(tokens) - width + 1):
        run = range(start, start + width)
        words = [tokens[i] for i in run]
        if taken.intersection(run) or any(ch.isdigit() for word in words for ch in word):
            continue
        if any(s.start < run.stop and start < s.end for s in numbers):
            continue
        if Levenshtein.normalized_distance(" ".join(words), spelled) <= cutoff:
            return run
    return None
