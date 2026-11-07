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

The basic level also rewrites written forms that are read differently from
how they are written: "₹500" is read "500 rupaye", "10:00" is read without
the zero minutes, "15.08.1947" is three numbers, not a decimal, and "6.30" next
to a time word ("6.30 baje") is the clock time 6:30, not the decimal 6.3.
"""

import re
import unicodedata
from enum import StrEnum
from fractions import Fraction
from functools import cache

from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.numbers import replace_numbers, spell_number, split_ordinals


class Level(StrEnum):
    """Normalisation level, from the stock baseline to the most lenient."""

    WHISPER = "whisper"
    BASIC = "basic"
    SPELLING = "spelling"
    NUMBERS = "numbers"


# A letter, or any character of the Indic script blocks (so vowel signs count).
_LETTER = r"[^\W\d_]|[\u0900-\u0dff]"
# A character inside a word: as _LETTER, but not the danda, double danda and
# abbreviation sign (U+0964, U+0965, U+0970), which end a word as punctuation.
_WORD_CHAR = r"[^\W\d_]|[\u0900-\u0963\u0966-\u096f\u0971-\u0dff]"

_SPACES = re.compile(r"\s+")
_ZERO_WIDTH = re.compile("[\u200b-\u200d\ufeff]")
# Thousands grouping only, Western (800,000) or Indian (8,00,000), never lists like "1,2,3".
_GROUPED_NUMBER = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{2,3})*,\d{3}(?!\d|,\d)")
_ZERO_MINUTES = re.compile(r"(?<=\d): ?00(?![\d:])")
_DOTTED_DATE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{2,4})(?![\d.])")
_DIGIT_LETTER = re.compile(rf"(?<=\d)(?={_LETTER})|(?<={_LETTER})(?=\d)")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CLOCK = re.compile(r"(?<![\d:])(\d{1,2}): ?(\d{2})(?![\d:])")
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_DECIMAL_PLACEHOLDER = "\ue000"  # private-use character, never in real text
_WHISPER_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]")
_WHISPER_PARENS = re.compile(r"\(([^)]+?)\)")


def whisper_basic(text: str) -> str:
    """Reproduce Whisper's BasicTextNormalizer with its default arguments.

    Source: openai/whisper, whisper/normalizers/basic.py (MIT licence). The
    only difference is that surrounding whitespace is stripped.
    """
    text = _WHISPER_BRACKETS.sub("", text.lower())
    text = _WHISPER_PARENS.sub("", text)
    text = "".join(
        " " if unicodedata.category(char)[0] in "MSP" else char
        for char in unicodedata.normalize("NFKC", text)
    )
    return _SPACES.sub(" ", text.lower()).strip()


@cache
def _dotted_time_patterns(
    before: tuple[str, ...], after: tuple[str, ...]
) -> tuple[re.Pattern[str], ...]:
    """Patterns for "h.mm" beside a time word.

    Every pattern has three groups, so one replacement serves all: (1) text
    kept before the time (empty when the time word follows), (2) the hour and
    (3) the minutes. Time words match whole words only, ignoring case:
    "barat 6.30" (a wedding procession) is not "raat 6.30", and "6.30
    istanbul" is not "6.30 IST"; a danda after the word ("6.30 baje।") ends it.
    """
    clock = r"(?<![\d.])([01]?\d|2[0-3])\.([0-5]\d)(?![\d.])"
    patterns = []
    if after:
        words = "|".join(map(re.escape, after))
        after_word = rf"(?=\s*(?:{words})(?!{_WORD_CHAR}))"
        patterns.append(re.compile(rf"()(?:{clock}){after_word}", re.IGNORECASE))
    if before:
        words = "|".join(map(re.escape, before))
        before_word = rf"((?<!{_WORD_CHAR})(?:{words})\s*)"
        patterns.append(re.compile(rf"{before_word}(?:{clock})", re.IGNORECASE))
    return tuple(patterns)


def _dotted_times(text: str, pack: LanguagePack) -> str:
    """Write a dotted clock time with a colon: "6.30 baje" is 6:30, not the decimal 6.3.

    Only a time word beside it tells a time from a decimal ("6.34 inch").
    """
    for pattern in _dotted_time_patterns(pack.time_words_before, pack.time_words_after):
        text = pattern.sub(r"\1\2:\3", text)
    return text


def _written_forms(text: str, pack: LanguagePack) -> str:
    """Rewrite written forms the way they are read; shared by scoring and TTS."""
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFC", text))
    if pack.native_digits:
        text = text.translate(str.maketrans(pack.native_digits, "0123456789"))
    text = _GROUPED_NUMBER.sub(lambda m: m.group().replace(",", ""), text)
    for symbol, word in pack.prefix_symbols.items():
        text = re.sub(rf"{re.escape(symbol)}\s*(\d+(?:\.\d+)?)", rf"\1 {word} ", text)
    for symbol, word in {**pack.prefix_symbols, **pack.symbols}.items():
        text = text.replace(symbol, f" {word} ")
    text = _ZERO_MINUTES.sub("", _dotted_times(text, pack))
    return _DOTTED_DATE.sub(lambda m: " ".join(str(int(part)) for part in m.groups()), text)


def _basic(text: str, pack: LanguagePack) -> str:
    text = _written_forms(text.lower(), pack)
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
    return " ".join(replace_numbers(split_ordinals(text.split(), pack), pack))


def numeric_tokens(text: str, pack: LanguagePack) -> list[str]:
    """Tokens ready for number parsing: spelling-normalised, ordinals split."""
    return split_ordinals(normalize(text, pack, Level.SPELLING).split(), pack)


def verbalize(text: str, pack: LanguagePack) -> str:
    """Rewrite text the way it is read aloud, for TTS engines that need words.

    Numbers are spelled out ("53" becomes "tirepan", "15vin" becomes
    "pandrahvin"), a number with leading zeros is read digit by digit
    ("007"), currency, percent and degree signs become words, and
    punctuation is kept so the engine can phrase the sentence. Latin letters
    are left as they are; see ``has_latin``.
    """
    text = _written_forms(text, pack)
    # Clock times are read without leading zeros: "07:19" is "saat unnees".
    text = _CLOCK.sub(lambda m: f"{int(m.group(1))} {int(m.group(2))}", text)
    return _SPACES.sub(" ", _NUMBER.sub(lambda m: _spoken(m, pack), text)).strip()


def _spoken(match: re.Match[str], pack: LanguagePack) -> str:
    digits = match.group()
    if digits.startswith("0") and digits.strip("0") and "." not in digits:
        words = " ".join(spell_number(Fraction(d), pack) for d in digits)
    else:
        words = spell_number(Fraction(digits), pack)
    before = match.string[match.start() - 1 : match.start()]
    after = match.string[match.end() :]
    is_ordinal = pack.fold(after[:4]).startswith(pack.ordinal_suffixes)
    # Keep a number apart from glued letters ("35mm", "H5N1"), except an ordinal suffix.
    lead = " " if before.isalpha() or unicodedata.category(before[:1] or " ")[0] == "M" else ""
    trail = " " if after[:1].isalpha() and not is_ordinal else ""
    return f"{lead}{words}{trail}"


def has_latin(text: str) -> bool:
    """True if text contains Latin letters, which a TTS voice may skip or misread."""
    return bool(_LATIN.search(text))
