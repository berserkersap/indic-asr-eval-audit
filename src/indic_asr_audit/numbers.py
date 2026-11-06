"""Find spoken and written numbers in normalised text and rewrite them as digits.

The parser follows the Indian place-value system (hundred, thousand, lakh,
crore) and handles the fraction words that are common in everyday Hindi:
"dhai sau" is 250, "saade teen" is 3.5 and "paune do sau" is 175. Digits that
are already in the text take part too, so "8 lakh" and "aath lakh" both
become 800000.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from indic_asr_audit.languages.base import LanguagePack, WordKind

_DIGITS = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class NumberSpan:
    """A number found in a token list.

    Attributes:
        value: Canonical decimal string, e.g. "250" or "1.75".
        start: Index of the first token of the span.
        end: Index one past the last token.
        written_as_digits: True if any token in the span was digits.
    """

    value: str
    start: int
    end: int
    written_as_digits: bool


def format_value(value: Fraction) -> str:
    """Return a canonical decimal string: no trailing zeros, no exponent."""
    if value.denominator == 1:
        return str(value.numerator)
    text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".")


class _Builder:
    """Accumulates one number token by token.

    Each ``add_*`` method returns False when the token cannot extend the
    current number, which tells the caller to close it and start a new one.
    """

    def __init__(self) -> None:
        self.total = Fraction(0)  # value already closed by a scale above hundred
        self.current = Fraction(0)  # value still open to a following scale
        self.last_scale = 0  # smallest scale applied to total so far, 0 if none
        self.offset: Fraction | None = None  # pending "saade" / "sava" / "paune"
        self.has_value = False
        self.open_unit = False  # current ends in a unit that a scale may follow
        self.decimal: list[int] | None = None  # digits read after a decimal point
        self.written_as_digits = False
        self.last_was_digit = False

    def add_value(self, value: Fraction, *, is_digit: bool) -> bool:
        if self.decimal is not None:
            if is_digit or value.denominator != 1 or not 0 <= value <= 9:
                return False
            self.decimal.append(int(value))
            return True
        if self.last_was_digit or (is_digit and self.current):
            return False  # digits never merge with a neighbouring value
        if self.open_unit and not self._extends_open_unit(value):
            return False
        if self.last_scale and value >= self.last_scale:
            return False
        if self.offset is not None:
            value += self.offset
            self.offset = None
        self.current += value
        self.has_value = True
        self.open_unit = value < 100 or value.denominator != 1
        self.written_as_digits |= is_digit
        self.last_was_digit = is_digit
        return True

    def _extends_open_unit(self, value: Fraction) -> bool:
        """Whether a unit may follow the open unit, as in "unnees sau nabbe".

        A unit may follow a whole hundred (1900 + 90), or a ten when it is a
        single digit ("twenty five"). Otherwise, as in "do teen", it starts a
        new number.
        """
        if value.denominator != 1 or self.current.denominator != 1:
            return False
        rest = self.current % 100
        return (rest == 0 and value < 100) or (rest % 10 == 0 and rest > 0 and value < 10)

    def add_scale(self, scale: int) -> bool:
        if self.decimal is not None:
            return False
        # "sava lakh": an offset directly before a scale applies to one.
        if self.offset is not None and not self.add_value(Fraction(1), is_digit=False):
            return False
        if scale == 100:
            if self.current >= 100 or (self.has_value and not self.open_unit):
                return False
            self.current = (self.current or Fraction(1)) * 100
        else:
            if self.last_scale and scale >= self.last_scale and not self.current:
                # "das hazaar crore": a larger scale multiplies everything so far.
                self.total *= scale
            elif self.last_scale and scale >= self.last_scale:
                return False
            else:
                self.total += (self.current or Fraction(1)) * scale
            self.last_scale = scale
            self.current = Fraction(0)
        self.has_value = True
        self.open_unit = False
        self.last_was_digit = False
        return True

    def add_offset(self, offset: Fraction) -> bool:
        if self.has_value or self.offset is not None:
            return False
        self.offset = offset
        return True

    def add_point(self) -> bool:
        if not self.open_unit or self.decimal is not None or self.current.denominator != 1:
            return False
        self.decimal = []
        return True

    def value(self) -> Fraction | None:
        if not self.has_value:
            return None
        value = self.total + self.current
        if self.decimal:
            value += Fraction("0." + "".join(map(str, self.decimal)))
        return value


def split_ordinals(tokens: list[str], pack: LanguagePack) -> list[str]:
    """Split ordinal number words into number and suffix.

    "pandrahvin" becomes "pandrah vin", matching how "15vin" is split by
    the normaliser, so both are scored as the number 15.
    """
    out: list[str] = []
    for token in tokens:
        for suffix in pack.ordinal_suffixes:
            stem = token.removesuffix(suffix)
            if stem != token and stem in pack.number_words:
                out.extend((stem, suffix))
                break
        else:
            out.append(token)
    return out


def find_numbers(tokens: list[str], pack: LanguagePack) -> list[NumberSpan]:
    """Return every number in ``tokens``, which must already be normalised.

    Tokens are expected in the form ``normalize`` produces: lower case, NFC,
    folded spellings, ASCII digits and no punctuation except a decimal point.
    """
    spans: list[NumberSpan] = []
    builder = _Builder()
    start = 0
    i = 0

    def close(end: int) -> None:
        value = builder.value()
        if value is None:
            return
        if builder.decimal == []:
            end -= 1  # a trailing decimal-point word with no digits stays a word
        spans.append(NumberSpan(format_value(value), start, end, builder.written_as_digits))

    while i < len(tokens):
        token = tokens[i]
        if _DIGITS.fullmatch(token):
            accepted = builder.add_value(Fraction(token), is_digit=True)
        elif token in pack.number_words:
            kind, amount = pack.number_words[token]
            if kind in (WordKind.UNIT, WordKind.FRACTION):
                accepted = builder.add_value(amount, is_digit=False)
            elif kind is WordKind.SCALE:
                accepted = builder.add_scale(int(amount))
            elif kind is WordKind.OFFSET:
                accepted = builder.add_offset(amount)
            else:
                accepted = builder.add_point()
        else:
            close(i)
            builder = _Builder()
            start = i = i + 1
            continue

        if accepted:
            i += 1
            continue
        if not builder.has_value and builder.offset is None:
            # The token cannot start a number on its own (e.g. a stray scale).
            builder = _Builder()
            start = i = i + 1
            continue
        close(i)
        builder = _Builder()
        start = i
    close(len(tokens))
    return spans


def replace_numbers(tokens: list[str], pack: LanguagePack) -> list[str]:
    """Return tokens with each number span replaced by its digit form."""
    out: list[str] = []
    cursor = 0
    for span in find_numbers(tokens, pack):
        out.extend(tokens[cursor : span.start])
        out.append(span.value)
        cursor = span.end
    out.extend(tokens[cursor:])
    return out
