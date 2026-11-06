"""Text normalisation before scoring, at increasing levels of leniency.

Levels:
    whisper:  Whisper's BasicTextNormalizer, the default in many open ASR
              evaluations. It deletes every Unicode mark, which removes Indic
              vowel signs and the virama and so changes the words themselves.
    basic:    Unicode NFC, lower case, punctuation removed, digits unified.
              Vowel signs are kept.
    spelling: basic, plus the language pack's accepted spelling variants.
    numbers:  spelling, plus every number rewritten as digits, so "teen sau"
              and "300" score as the same word. Ordinal number words are
              split into number and suffix first.

Clock times lose a ":00" minute part at every level except whisper, since
"10:00" is read as "ten o'clock" with no spoken zero.
"""

import re
import unicodedata
from enum import StrEnum

from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.numbers import replace_numbers, split_ordinals


class Level(StrEnum):
    """Normalisation level, from the stock baseline to the most lenient."""

    WHISPER = "whisper"
    BASIC = "basic"
    SPELLING = "spelling"
    NUMBERS = "numbers"


_SPACES = re.compile(r"\s+")
_DIGIT_GROUPING = re.compile(r"(?<=\d),(?=\d)")
_ZERO_MINUTES = re.compile(r"(?<=\d):00(?!\d)")
_DIGIT_LETTER = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_DECIMAL_PLACEHOLDER = ""  # private-use character, never in real text
_WHISPER_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]|\(([^)]+?)\)")


def whisper_basic(text: str) -> str:
    """Reproduce Whisper's BasicTextNormalizer with its default arguments.

    Source: openai/whisper, whisper/normalizers/basic.py (MIT licence).
    """
    text = _WHISPER_BRACKETS.sub("", text.lower())
    text = "".join(
        " " if unicodedata.category(char)[0] in "MSP" else char
        for char in unicodedata.normalize("NFKC", text)
    )
    return _SPACES.sub(" ", text.lower()).strip()


def _basic(text: str, pack: LanguagePack) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    if pack.native_digits:
        text = text.translate(str.maketrans(pack.native_digits, "0123456789"))
    for symbol, word in pack.symbols.items():
        text = text.replace(symbol, f" {word} ")
    text = _DIGIT_GROUPING.sub("", text)
    text = _ZERO_MINUTES.sub("", text)
    text = _DIGIT_LETTER.sub(" ", text)
    text = _DECIMAL_POINT.sub(_DECIMAL_PLACEHOLDER, text)
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    text = text.replace(_DECIMAL_PLACEHOLDER, ".")
    return _SPACES.sub(" ", text).strip()


def normalize(text: str, pack: LanguagePack, level: Level) -> str:
    """Normalise text for scoring at the given level."""
    if level is Level.WHISPER:
        return whisper_basic(text)
    text = _basic(text, pack)
    if level is Level.BASIC:
        return text
    text = pack.fold(text)
    if level is Level.SPELLING:
        return text
    return " ".join(replace_numbers(numeric_tokens(text, pack), pack))


def numeric_tokens(text: str, pack: LanguagePack) -> list[str]:
    """Tokens ready for number parsing: spelling-normalised, ordinals split."""
    return split_ordinals(normalize(text, pack, Level.SPELLING).split(), pack)
