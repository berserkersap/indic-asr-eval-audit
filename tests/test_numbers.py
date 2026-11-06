import itertools
import random
from fractions import Fraction

import pytest

from indic_asr_audit.languages import get_language
from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.normalize import Level, normalize, numeric_tokens
from indic_asr_audit.numbers import find_numbers, format_value, spell_number

HINDI = get_language("hi")


def values_of(text: str) -> list[str]:
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
        ("उन्नीस सौ सैंतालीस", ["1947"]),
        ("एक लाख दो हज़ार तीन सौ चार", ["102304"]),
        ("तीन दशमलव पांच लाख", ["350000"]),
        ("शून्य शून्य सात", ["0", "0", "7"]),
        ("शून्य पांच", ["0", "5"]),
        ("तीन सौ चार सौ", ["300", "400"]),
        ("पैतीस", ["35"]),
        ("बाइस", ["22"]),
        ("पाच", ["5"]),
        ("पंद्रहवी सदी", ["15"]),
        ("साढ़े", []),
        ("साढ़े साढ़े तीन", ["3.5"]),
        ("दशमलव", []),
        ("सात हज़ार दशमलव दो पांच", ["7000.25"]),
    ],
)
def test_spoken_numbers(text: str, expected: list[str]) -> None:
    assert values_of(text) == expected


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
        ("10:00:30", ["10", "0", "30"]),
        ("15.08.1947", ["15", "8", "1947"]),
        ("1,2,3", ["1", "2", "3"]),
        ("2,3 लोग", ["2", "3"]),
        ("₹500", ["500"]),
        ("में2 लोग", ["2"]),
        ("शून्य 5", ["0", "5"]),
        ("19 सौ 47", ["1947"]),
        ("दो सौ 5", ["205"]),
        ("दो हज़ार 21", ["2021"]),
        ("3 दशमलव 5", ["3.5"]),
    ],
)
def test_written_numbers(text: str, expected: list[str]) -> None:
    assert values_of(text) == expected


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
    assert format_value(Fraction(numerator, denominator)) == expected


def test_parser_never_fails_on_random_tokens() -> None:
    vocabulary = [*HINDI.number_words, "5", "2.5", "1900", "x", "और"]
    rng = random.Random(0)
    for _ in range(3000):
        tokens = rng.choices(vocabulary, k=rng.randint(0, 12))
        spans = find_numbers(tokens, HINDI)
        bounds = [(s.start, s.end) for s in spans]
        assert all(0 <= a < b <= len(tokens) for a, b in bounds)
        assert all(b <= c for (_, b), (c, _) in itertools.pairwise(bounds))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "शून्य"),
        (53, "तिरेपन"),
        (250, "दो सौ पचास"),
        (1990, "उन्नीस सौ नब्बे"),
        (2010, "दो हज़ार दस"),
        (800000, "आठ लाख"),
        (12500000, "एक करोड़ पच्चीस लाख"),
    ],
)
def test_spell_number(value: int, expected: str) -> None:
    assert spell_number(Fraction(value), HINDI) == expected


def test_spell_decimal_reads_digits_after_the_point() -> None:
    assert spell_number(Fraction("1.75"), HINDI) == "एक दशमलव सात पांच"


def test_spelled_numbers_parse_back_to_the_same_value() -> None:
    rng = random.Random(0)
    values: list[float] = [*range(2200), *(rng.randrange(10**10) for _ in range(3000))]
    values += [rng.randrange(10**4) + rng.choice([0.5, 0.25, 0.05]) for _ in range(300)]
    for value in values:
        spoken = spell_number(Fraction(str(value)), HINDI)
        assert values_of(spoken) == [format_value(Fraction(str(value)))], spoken


def test_spelling_needs_spoken_forms() -> None:
    pack = LanguagePack(code="xx", name="Test", native_digits="", units={}, scales={})
    with pytest.raises(ValueError, match="no spoken forms"):
        spell_number(Fraction(5), pack)
