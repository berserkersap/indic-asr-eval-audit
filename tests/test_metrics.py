import pytest

from indic_asr_audit.languages import get_language
from indic_asr_audit.metrics import (
    Counts,
    char_counts,
    number_word_errors,
    numeric_counts,
    word_counts,
)

HINDI = get_language("hi")


def test_word_counts() -> None:
    assert word_counts("a b c d", "a x c") == Counts(errors=2, total=4)


def test_word_counts_empty_hypothesis() -> None:
    assert word_counts("a b", "") == Counts(errors=2, total=2)


def test_word_counts_empty_reference() -> None:
    assert word_counts("", "a b") == Counts(errors=2, total=0)


def test_char_counts_ignore_spaces() -> None:
    assert char_counts("रातों रात", "रातोंरात") == Counts(errors=0, total=8)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("53 साल के कुओमो", "तिरेपन साल के कुओमो", Counts(0, 1)),
        ("53 साल के कुओमो", "तिरपन साल के कुओमो", Counts(0, 1)),
        ("53 साल के कुओमो", "पचपन साल के कुओमो", Counts(1, 1)),
        ("53 साल के कुओमो", "साल के कुओमो", Counts(1, 1)),
        ("800,000 से ज़्यादा", "आठ लाख से ज्यादा", Counts(0, 1)),
        ("35mm और 35mm", "35 एमएम", Counts(1, 2)),
        ("15वीं सदी", "पंद्रहवीं सदी", Counts(0, 1)),
        ("एक आदमी", "दो आदमी", Counts(0, 0)),  # no digits in the reference
        ("53 और 53", "53", Counts(1, 2)),
        ("₹500 दिए", "पांच सौ रुपये दिए", Counts(0, 1)),
        ("सुबह 06:30 बजे", "सुबह 6.30 बजे", Counts(0, 2)),
        ("11:35 बजे", "11.35 बजे", Counts(0, 2)),
        # Known limitation: the article "ek" matches a reference "1".
        ("1 किताब", "मेरे पास एक कलम", Counts(0, 1)),
    ],
)
def test_numeric_counts(reference: str, hypothesis: str, expected: Counts) -> None:
    assert numeric_counts(reference, hypothesis, HINDI) == expected


@pytest.mark.parametrize(
    ("reference", "hypothesis", "strict", "lenient"),
    [
        ("लगभग 15 मील", "लगबग पंद्रा मिल", 1, 0),
        ("1963 में", "1093 में", 1, 1),
        ("करीब 70 किलोमीटर", "करीब 7त्तर किलोमीटर", 1, 1),
        ("कुल 35 लोग", "कुल तैंतीस लोग", 1, 1),
        ("सन 1900 में", "सन उन्नीस सौ एक में", 1, 1),
        ("5 और 5", "पांच और", 1, 1),  # one word forgives one number, not two
        ("15 और 15", "पंद्रा और", 2, 1),
    ],
)
def test_lenient_numeric_error_forgives_misspelled_words_only(
    reference: str, hypothesis: str, strict: int, lenient: int
) -> None:
    assert numeric_counts(reference, hypothesis, HINDI).errors == strict
    assert numeric_counts(reference, hypothesis, HINDI, lenient=True).errors == lenient


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("कुल 800 सैनिक", "कुल 800 सैनिक", 0),
        ("कुल 800 सैनिक", "कुल सैनिक", 1),  # deletion
        ("कुल 800 सैनिक", "कुल 8 0 0 सैनिक", 3),  # one substitution, two insertions
        ("कुल 800 सैनिक", "कुल 800 सेनिक", 0),  # an error on another word
        ("", "20", 1),
    ],
)
def test_number_word_errors_count_every_error_on_a_digit_word(
    reference: str, hypothesis: str, expected: int
) -> None:
    assert number_word_errors(reference, hypothesis) == expected
