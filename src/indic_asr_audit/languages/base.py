"""Language packs: the per-language data the normaliser and number parser need."""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from functools import cached_property


class WordKind(Enum):
    """Role of a word inside a spoken number."""

    UNIT = auto()
    SCALE = auto()
    FRACTION = auto()
    OFFSET = auto()
    POINT = auto()


@dataclass(frozen=True)
class LanguagePack:
    """Everything language-specific, kept as plain data.

    Adding a language means filling one of these in; see
    docs/extending-to-new-languages.md.

    Attributes:
        code: ISO 639-1 code, e.g. "hi".
        name: English name of the language.
        native_digits: The ten native digit characters, 0 to 9 ("" if none).
        units: Number words with values below 100, including spelling variants.
        scales: Multiplier words (hundred, thousand, lakh, ...).
        fractions: Words that are a value on their own and can take a scale,
            e.g. Hindi "dhai" (2.5) as in "dhai sau" (250).
        offsets: Prefixes that shift the number that follows, e.g. Hindi
            "saade" (+0.5) as in "saade teen" (3.5) and "paune" (-0.25).
        decimal_points: Words read for the decimal point.
        ordinal_suffixes: Suffixes that turn a number word into an ordinal,
            e.g. Hindi "vin" in "pandrahvin" (15th).
        symbols: Symbols replaced by a word before scoring, e.g. "%".
        prefix_symbols: Symbols written before a number but read after it,
            e.g. "₹" in "₹500", read "500 rupaye".
        spelling_folds: (regex, replacement) pairs that merge accepted
            spelling variants, applied after Unicode NFC.
        spoken_units: The standard word for each value 0 to 99, used to
            spell numbers out for TTS. Empty if spelling is not supported.
        spoken_scales: (value, word) pairs from the largest scale down.
        spoken_point: Word read for the decimal point.
        years_as_hundreds: Read 1100-1999 as hundreds ("nineteen hundred
            ninety"), as Hindi does.
    """

    code: str
    name: str
    native_digits: str
    units: Mapping[str, int]
    scales: Mapping[str, int]
    fractions: Mapping[str, Fraction] = field(default_factory=dict)
    offsets: Mapping[str, Fraction] = field(default_factory=dict)
    decimal_points: frozenset[str] = frozenset()
    ordinal_suffixes: tuple[str, ...] = ()
    symbols: Mapping[str, str] = field(default_factory=dict)
    prefix_symbols: Mapping[str, str] = field(default_factory=dict)
    spelling_folds: tuple[tuple[str, str], ...] = ()
    spoken_units: tuple[str, ...] = ()
    spoken_scales: tuple[tuple[int, str], ...] = ()
    spoken_point: str = ""
    years_as_hundreds: bool = False

    def __post_init__(self) -> None:
        if self.native_digits and len(self.native_digits) != 10:
            raise ValueError(f"{self.code}: native_digits must have 10 characters")
        if self.spoken_units and len(self.spoken_units) != 100:
            raise ValueError(f"{self.code}: spoken_units must have 100 entries")
        bad_units = {w: v for w, v in self.units.items() if not 0 <= v < 100}
        if bad_units:
            raise ValueError(f"{self.code}: units must be in [0, 100): {bad_units}")

    def fold(self, text: str) -> str:
        """Apply the spelling folds to text."""
        for pattern, replacement in self.spelling_folds:
            text = re.sub(pattern, replacement, text)
        return text

    def canonical_word(self, word: str) -> str:
        """Return word in the form the normaliser produces: NFC, lower case, folded."""
        return self.fold(unicodedata.normalize("NFC", word).lower())

    @cached_property
    def number_words(self) -> dict[str, tuple[WordKind, Fraction]]:
        """Lookup from canonical word to its role and value."""
        groups: list[tuple[WordKind, Mapping[str, int] | Mapping[str, Fraction]]] = [
            (WordKind.UNIT, self.units),
            (WordKind.SCALE, self.scales),
            (WordKind.FRACTION, self.fractions),
            (WordKind.OFFSET, self.offsets),
            (WordKind.POINT, dict.fromkeys(self.decimal_points, 0)),
        ]
        lookup: dict[str, tuple[WordKind, Fraction]] = {}
        for kind, words in groups:
            for word, value in words.items():
                key = self.canonical_word(word)
                entry = (kind, Fraction(value))
                if key in lookup and lookup[key] != entry:
                    raise ValueError(f"{self.code}: '{word}' is both {lookup[key]} and {entry}")
                lookup[key] = entry
        return lookup
