"""Per-utterance error counts. Corpus rates and confidence intervals live in stats."""

from collections import Counter
from dataclasses import dataclass

import jiwer

from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.normalize import numeric_tokens
from indic_asr_audit.numbers import find_numbers


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
    """Character edit distance between two normalised strings, spaces excluded."""
    ref, hyp = reference.replace(" ", ""), hypothesis.replace(" ", "")
    if not ref:
        return Counts(errors=len(hyp), total=0)
    out = jiwer.process_characters(ref, hyp)
    return Counts(errors=out.substitutions + out.deletions + out.insertions, total=len(ref))


def numeric_counts(reference: str, hypothesis: str, pack: LanguagePack) -> Counts:
    """Count reference numbers whose value is missing from the hypothesis.

    Only numbers written as digits in the raw reference are scored, because
    those are the values a downstream system must get right. Matching is by
    value and order-free (a multiset), so "53" is matched by "tirepan" or
    "53" anywhere in the hypothesis. Extra numbers in the hypothesis are not
    counted; WER already penalises them.
    """
    ref_spans = find_numbers(numeric_tokens(reference, pack), pack)
    wanted = Counter(s.value for s in ref_spans if s.written_as_digits)
    found = Counter(s.value for s in find_numbers(numeric_tokens(hypothesis, pack), pack))
    missed = wanted - found
    return Counts(errors=sum(missed.values()), total=sum(wanted.values()))
