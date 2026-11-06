import pytest

from indic_asr_audit.languages import get_language
from indic_asr_audit.normalize import Level, normalize, numeric_tokens
from indic_asr_audit.numbers import find_numbers, format_value

HINDI = get_language("hi")


def values(text: str) -> list[str]:
    return [span.value for span in find_numbers(numeric_tokens(text, HINDI), HINDI)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("तिरेपन साल", ["53"]),
        ("तिरपन साल", ["53"]),
        ("उन्नीस सौ नब्बे में", ["1990"]),
        ("दो हज़ार दस", ["2010"]),
        ("दो हजार दस", ["2010"]),
        ("दो लाख पचास हज़ार", ["250000"]),
        ("आठ लाख सैनिक", ["800000"]),
        ("दस हज़ार करोड़", ["100000000000"]),
        ("सौ", ["100"]),
        ("ढाई सौ रुपये", ["250"]),
        ("डेढ़ लाख", ["150000"]),
        ("साढ़े तीन", ["3.5"]),
        ("साढ़े तीन सौ", ["350"]),
        ("पौने दो सौ", ["175"]),
        ("सवा लाख", ["125000"]),
        ("दो दशमलव पांच", ["2.5"]),
        ("पाँच", ["5"]),
        ("दो तीन", ["2", "3"]),
        ("twenty five", ["25"]),
        ("पंद्रहवीं सदी", ["15"]),
        ("बीसवां", ["20"]),
        ("two thousand twenty", ["2020"]),
    ],
)
def test_spoken_numbers(text: str, expected: list[str]) -> None:
    assert values(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("53 साल", ["53"]),
        ("800,000 से", ["800000"]),
        ("8,00,000", ["800000"]),
        ("8 लाख", ["800000"]),
        ("2.5 करोड़", ["25000000"]),
        ("३५ प्रतिशत", ["35"]),
        ("35mm", ["35"]),
        ("1900 20", ["1900", "20"]),
        ("1.50", ["1.5"]),
        ("15वीं सदी", ["15"]),
        ("10:00-11:30", ["10", "11", "30"]),
    ],
)
def test_written_numbers(text: str, expected: list[str]) -> None:
    assert values(text) == expected


def test_written_as_digits_flag() -> None:
    tokens = normalize("53 साल और तीन दिन", HINDI, Level.SPELLING).split()
    spans = find_numbers(tokens, HINDI)
    assert [(s.value, s.written_as_digits) for s in spans] == [("53", True), ("3", False)]


def test_dangling_decimal_point_stays_a_word() -> None:
    assert normalize("दो दशमलव कुछ", HINDI, Level.NUMBERS) == "2 दशमलव कुछ"


def test_text_without_numbers_is_unchanged() -> None:
    text = "यह एनालॉग फिल्म दौर"
    assert normalize(text, HINDI, Level.NUMBERS) == normalize(text, HINDI, Level.SPELLING)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(5, 1, "5"), (5, 2, "2.5"), (7, 4, "1.75"), (1, 8, "0.125"), (-1, 4, "-0.25")],
)
def test_format_value(numerator: int, denominator: int, expected: str) -> None:
    from fractions import Fraction

    assert format_value(Fraction(numerator, denominator)) == expected
