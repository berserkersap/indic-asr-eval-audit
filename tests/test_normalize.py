import unicodedata

import pytest

from indic_asr_audit.languages import get_language
from indic_asr_audit.normalize import Level, normalize, whisper_basic

HINDI = get_language("hi")


def test_whisper_level_strips_vowel_signs() -> None:
    out = whisper_basic("किताब पढ़ो")
    assert not any(unicodedata.category(c).startswith("M") for c in out)
    assert len(out.split()) > 2  # removing the marks splits the words


def test_whisper_level_matches_transformers() -> None:
    english_normalizer = pytest.importorskip("transformers.models.whisper.english_normalizer")
    reference = english_normalizer.BasicTextNormalizer()
    for text in ["किताब पढ़ो।", "Hello, [noise] World (aside)!", "35mm है, 800,000 सैनिक"]:
        assert whisper_basic(text) == " ".join(reference(text).split())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("किताब पढ़ो।", "किताब पढ़ो"),
        ("Hello, World!", "hello world"),
        ("800,000 सैनिक", "800000 सैनिक"),
        ("35mm है.", "35 mm है"),
        ("2.5 करोड़", "2.5 करोड़"),
        ("३५ प्रतिशत", "35 प्रतिशत"),
        ("35% लोग", "35 प्रतिशत लोग"),
        ("रातों-रात", "रातों रात"),
        ("10:00 बजे", "10 बजे"),
        ("10:05 बजे", "10 05 बजे"),
    ],
)
def test_basic_level(text: str, expected: str) -> None:
    assert normalize(text, HINDI, Level.BASIC) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ज़्यादा", "ज्यादा"),
        ("फ़ोटोग्राफ़ी", "फोटोग्राफी"),
        ("पाँच", "पांच"),
        ("पढ़ाई", "पढ़ाई"),  # ड़ and ढ़ keep their nukta
    ],
)
def test_spelling_level(text: str, expected: str) -> None:
    assert normalize(text, HINDI, Level.SPELLING) == unicodedata.normalize("NFC", expected)


def test_precomposed_and_decomposed_nukta_match() -> None:
    precomposed, decomposed = "ज़", "ज़"  # ज़ two ways
    assert normalize(precomposed, HINDI, Level.SPELLING) == normalize(
        decomposed, HINDI, Level.SPELLING
    )


def test_numbers_level_makes_spoken_and_written_equal() -> None:
    written = normalize("इस साल 53 साल के कुओमो ने", HINDI, Level.NUMBERS)
    spoken = normalize("इस साल तिरेपन साल के कुओमो ने", HINDI, Level.NUMBERS)
    assert written == spoken
