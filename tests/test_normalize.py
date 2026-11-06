import random
import unicodedata

import pytest

from indic_asr_audit.languages import get_language
from indic_asr_audit.normalize import Level, has_latin, normalize, verbalize, whisper_basic

HINDI = get_language("hi")


def test_whisper_level_strips_vowel_signs() -> None:
    out = whisper_basic("किताब पढ़ो")
    assert not any(unicodedata.category(c).startswith("M") for c in out)
    assert len(out.split()) > 2  # removing the marks splits the words


def test_whisper_level_matches_transformers() -> None:
    english_normalizer = pytest.importorskip("transformers.models.whisper.english_normalizer")
    reference = english_normalizer.BasicTextNormalizer()
    alphabet = "कखपढ़ोि्ं।,.()[]<> abc123"
    rng = random.Random(0)
    texts = ["Hello, [noise] World (aside)!", "ब(भ।<नई)>[x(y]z)"]
    texts += ["".join(rng.choices(alphabet, k=30)) for _ in range(2000)]
    for text in texts:
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
        ("₹500 दिए", "500 रुपये दिए"),
        ("₹500और", "500 रुपये और"),
        ("1,2,3", "1 2 3"),
        ("में2 लोग", "में 2 लोग"),
        ("३.५", "3.5"),
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
        ("पढ़ाई", "पढाई"),
    ],
)
def test_spelling_level(text: str, expected: str) -> None:
    assert normalize(text, HINDI, Level.SPELLING) == unicodedata.normalize("NFC", expected)


def test_precomposed_and_decomposed_nukta_match() -> None:
    precomposed, decomposed = chr(0x095B), chr(0x091C) + chr(0x093C)  # ज़ two ways
    assert normalize(precomposed, HINDI, Level.SPELLING) == normalize(
        decomposed, HINDI, Level.SPELLING
    )


def test_numbers_level_makes_spoken_and_written_equal() -> None:
    written = normalize("इस साल 53 साल के कुओमो ने", HINDI, Level.NUMBERS)
    spoken = normalize("इस साल तिरेपन साल के कुओमो ने", HINDI, Level.NUMBERS)
    assert written == spoken


def test_zero_width_characters_are_removed() -> None:
    zwnj = chr(0x200C)
    assert normalize(f"तीन{zwnj} सौ", HINDI, Level.NUMBERS) == "300"


def test_ordinal_suffix_spellings_score_the_same() -> None:
    reference = normalize("15वीं सदी", HINDI, Level.NUMBERS)
    assert normalize("पंद्रहवी सदी", HINDI, Level.NUMBERS) == reference
    assert normalize("15 वी सदी", HINDI, Level.NUMBERS) == reference


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("इस साल 53 साल के.", "इस साल तिरेपन साल के."),
        ("सेना के 800,000 सैनिक", "सेना के आठ लाख सैनिक"),
        ("19वीं शताब्दी", "उन्नीसवीं शताब्दी"),
        ("₹500 दिए, 35% लोग", "पांच सौ रुपये दिए, पैंतीस प्रतिशत लोग"),
        ("सुबह 10:00 बजे", "सुबह दस बजे"),
        ("फॉर्मेट 35mm है", "फॉर्मेट पैंतीस mm है"),
        ("15.08.1947 को", "पंद्रह आठ उन्नीस सौ सैंतालीस को"),
    ],
)
def test_verbalize(text: str, expected: str) -> None:
    assert verbalize(text, HINDI) == expected


def test_verbalized_text_scores_like_the_reference() -> None:
    text = "1990 में 2.5 लाख लोग, 15वीं बार"
    spoken = verbalize(text, HINDI)
    assert normalize(spoken, HINDI, Level.NUMBERS) == normalize(text, HINDI, Level.NUMBERS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$45 दिए", "पैंतालीस डॉलर दिए"),
        ("¥130,000 का", "एक लाख तीस हज़ार येन का"),
        ("कोड 0230 है", "कोड शून्य दो तीन शून्य है"),
        ("133मी/सेकंड", "एक सौ तैंतीस मी/सेकंड"),
    ],
)
def test_verbalize_symbols_leading_zeros_and_units(text: str, expected: str) -> None:
    assert verbalize(text, HINDI) == expected


@pytest.mark.parametrize(
    ("text", "latin"), [("A1GP रेस", True), ("35mm है", True), ("53 साल", False)]
)
def test_has_latin_after_verbalize(text: str, latin: bool) -> None:
    assert has_latin(verbalize(text, HINDI)) is latin
