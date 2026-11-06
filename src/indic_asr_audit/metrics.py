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
    "pandrah" (see ``spelled_nearby``). Wrong digits are never forgiven.
    """
    ref_spans = find_numbers(numeric_tokens(reference, pack), pack)
    wanted = Counter(s.value for s in ref_spans if s.written_as_digits)
    hyp_tokens = numeric_tokens(hypothesis, pack)
    found = Counter(s.value for s in find_numbers(hyp_tokens, pack))
    missed = wanted - found
    if lenient:
        words = _words(hypothesis, pack)
        missed = Counter({v: n for v, n in missed.items() if not spelled_nearby(v, words, pack)})
    return Counts(errors=sum(missed.values()), total=sum(wanted.values()))


def _words(text: str, pack: LanguagePack) -> list[str]:
    """Words with punctuation removed but digits left attached ("7ttar" stays one word)."""
    cleaned = "".join(
        " " if unicodedata.category(c)[0] in "PS" else c for c in unicodedata.normalize("NFC", text)
    )
    return [pack.canonical_word(w) for w in cleaned.split()]


def spelled_nearby(value: str, tokens: list[str], pack: LanguagePack, cutoff: float = 0.25) -> bool:
    """Whether some run of word tokens is a near spelling of the value in words.

    The run must have as many words as the standard spelling, contain no
    digits, and be within ``cutoff`` normalised character edit distance of
    it. This forgives "pandra" for "pandrah" and "unnees so chhiyasath" for
    "unnees sau chhiyasath", but not "1093" for 1963 or "7ttar" for 70.
    """
    from rapidfuzz.distance import Levenshtein

    spelled = pack.canonical_word(spell_number(Fraction(value), pack))
    width = len(spelled.split())
    for start in range(len(tokens) - width + 1):
        run = tokens[start : start + width]
        if any(ch.isdigit() for word in run for ch in word):
            continue
        if Levenshtein.normalized_distance(" ".join(run), spelled) <= cutoff:
            return True
    return False
