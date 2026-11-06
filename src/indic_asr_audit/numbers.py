"""Find spoken and written numbers in normalised text and rewrite them as digits.

The parser follows the Indian place-value system (hundred, thousand, lakh,
crore) and handles the fraction words that are common in everyday Hindi:
"dhai sau" is 250, "saade teen" is 3.5 and "paune do sau" is 175. Digits that
are already in the text take part too, so "8 lakh" and "aath lakh" both
become 800000.

Known gaps: English "and" inside numbers ("one hundred and five"), English
year style ("nineteen forty seven"), and irregular ordinals such as "pehla".
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from indic_asr_audit.languages.base import LanguagePack, WordKind

_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_HUNDRED = 100


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
    current number; the caller then closes it and starts a new one.
    """

    def __init__(self) -> None:
        self.total = Fraction(0)  # part already multiplied by a scale above hundred
        self.current = Fraction(0)  # part still open to a following scale
        self.last_scale = 0  # smallest scale above hundred applied so far
        self.offset: Fraction | None = None  # pending "saade", "sava" or "paune"
        self.has_value = False
        self.open_unit = False  # current ends in a unit, so a scale may follow
        self.decimal: str | None = None  # digits read after a decimal point
        self.written_as_digits = False
        self.last_was_digit = False
        self.last_was_english = False

    def add_value(
        self, value: Fraction, *, is_digit: bool, before_hundred: bool, english: bool = False
    ) -> bool:
        if self.decimal is not None:
            return self._add_decimal_digits(value)
        if self.last_was_digit or (is_digit and self.open_unit):
            return False  # "1900 20" and "shunya 5" are separate numbers
        if self.open_unit and not self._extends_open_unit(value, english):
            return False
        if before_hundred and self.current >= _HUNDRED:
            return False  # "teen sau, chaar sau" is 300 then 400
        if self.last_scale and value >= self.last_scale:
            return False
        if self.offset is not None:
            value += self.offset
            self.offset = None
        self.current += value
        self.has_value = True
        self.open_unit = value < _HUNDRED or value.denominator != 1
        self.written_as_digits |= is_digit
        self.last_was_digit = is_digit
        self.last_was_english = english
        return True

    def _add_decimal_digits(self, value: Fraction) -> bool:
        if value.denominator != 1 or value < 0 or self.decimal is None:
            return False
        if self.decimal and not 0 <= value <= 9:
            return False  # after the first group, only single spoken digits
        self.decimal += str(value.numerator)
        return True

    def _extends_open_unit(self, value: Fraction, english: bool) -> bool:
        """Whether a unit may follow the open unit.

        A unit may follow a whole hundred ("unnees sau nabbe", 1990). In
        English a single digit may also follow a ten ("twenty five"); Hindi
        has a word for every number below 100, so "das aath" is two numbers.
        """
        if value.denominator != 1 or self.current.denominator != 1:
            return False
        rest = self.current % _HUNDRED
        whole_hundred = self.current >= _HUNDRED and rest == 0
        english_tens = english and self.last_was_english and rest % 10 == 0 and rest >= 20
        return (whole_hundred and value < _HUNDRED) or (english_tens and value < 10)

    def add_scale(self, scale: int) -> bool:
        if self.decimal == "":
            return False
        if self.decimal is not None:  # "teen dashamlav paanch lakh"
            self.current += Fraction(f"0.{self.decimal}")
            self.decimal = None
        # "sava lakh": an offset directly before a scale applies to one.
        if self.offset is not None and not self.add_value(
            Fraction(1), is_digit=False, before_hundred=False
        ):
            return False
        if scale == _HUNDRED:
            if self.current >= _HUNDRED or (self.has_value and not self.open_unit):
                return False
            self.current = (self.current or Fraction(1)) * _HUNDRED
        elif self.last_scale and scale >= self.last_scale:
            if self.current:
                return False
            self.total *= scale  # "das hazaar crore": a larger scale multiplies all
        else:
            self.total += (self.current or Fraction(1)) * scale
            self.current = Fraction(0)
        if scale > _HUNDRED:
            self.last_scale = scale
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
        whole = (self.total + self.current).denominator == 1
        if not self.has_value or self.offset is not None or self.decimal is not None or not whole:
            return False
        self.decimal = ""
        return True

    @property
    def trailing_point(self) -> bool:
        """True if the last token was a decimal point with no digits after it."""
        return self.decimal == ""

    def value(self) -> Fraction | None:
        if not self.has_value:
            return None
        value = self.total + self.current
        if self.decimal:
            value += Fraction(f"0.{self.decimal}")
        return value


def _add_token(builder: _Builder, tokens: list[str], i: int, pack: LanguagePack) -> bool | None:
    """Feed tokens[i] to the builder; None means the token is not part of a number."""
    token = tokens[i]
    nxt = pack.number_words.get(tokens[i + 1]) if i + 1 < len(tokens) else None
    before_hundred = nxt is not None and nxt[0] is WordKind.SCALE and nxt[1] == _HUNDRED
    if _DIGITS.fullmatch(token):
        return builder.add_value(Fraction(token), is_digit=True, before_hundred=before_hundred)
    if token not in pack.number_words:
        return None
    kind, amount = pack.number_words[token]
    if kind in (WordKind.UNIT, WordKind.FRACTION):
        return builder.add_value(
            amount, is_digit=False, before_hundred=before_hundred, english=token.isascii()
        )
    if kind is WordKind.SCALE:
        return builder.add_scale(int(amount))
    if kind is WordKind.OFFSET:
        return builder.add_offset(amount)
    return builder.add_point()


def _span(builder: _Builder, start: int, end: int) -> NumberSpan | None:
    value = builder.value()
    if value is None:
        return None
    if builder.trailing_point:
        end -= 1  # a decimal-point word with nothing after it stays a word
    return NumberSpan(format_value(value), start, end, builder.written_as_digits)


def find_numbers(tokens: list[str], pack: LanguagePack) -> list[NumberSpan]:
    """Return every number in ``tokens``, in order and without overlap.

    Tokens are expected in the form ``normalize`` produces: lower case, NFC,
    folded spellings, ASCII digits and no punctuation except a decimal point.
    """
    spans: list[NumberSpan] = []
    builder, start, i = _Builder(), 0, 0
    while i < len(tokens):
        accepted = _add_token(builder, tokens, i, pack)
        if accepted:
            i += 1
            continue
        if (span := _span(builder, start, i)) is not None:
            spans.append(span)
        if accepted is None or not (builder.has_value or builder.offset is not None):
            i += 1  # the token cannot start a number: skip it
        builder, start = _Builder(), i
    if (span := _span(builder, start, len(tokens))) is not None:
        spans.append(span)
    return spans


def split_ordinals(tokens: list[str], pack: LanguagePack) -> list[str]:
    """Split ordinal number words into number and suffix.

    "pandrahvin" becomes "pandrah vin", matching how "15vin" is split by
    the normaliser, so both are scored as the number 15. Every suffix
    variant after a number is written as the first one listed, so spelling
    differences in the suffix are not counted.
    """
    if not pack.ordinal_suffixes:
        return tokens
    canonical = pack.ordinal_suffixes[0]
    out: list[str] = []
    for token in tokens:
        after_number = bool(out) and (out[-1].isdigit() or out[-1] in pack.number_words)
        if token in pack.ordinal_suffixes and after_number:
            out.append(canonical)
            continue
        for suffix in pack.ordinal_suffixes:
            stem = token.removesuffix(suffix)
            if stem != token and stem in pack.number_words:
                out.extend((stem, canonical))
                break
        else:
            out.append(token)
    return out


def spell_number(value: Fraction, pack: LanguagePack) -> str:
    """Spell a non-negative number in words, the way it is usually read aloud."""
    if not pack.spoken_units:
        raise ValueError(f"{pack.code}: no spoken forms to spell numbers with")
    if value < 0:
        raise ValueError("negative numbers are not supported")
    words = _spell_integer(int(value), pack)
    if value.denominator != 1:
        decimals = format_value(value).split(".")[1]
        words += [pack.spoken_point, *(pack.spoken_units[int(d)] for d in decimals)]
    return " ".join(words)


def _spell_integer(n: int, pack: LanguagePack) -> list[str]:
    units = pack.spoken_units
    if n < _HUNDRED:
        return [units[n]]
    hundred = dict(pack.spoken_scales)[_HUNDRED]
    if pack.years_as_hundreds and 1100 <= n <= 1999:
        rest = [units[n % _HUNDRED]] if n % _HUNDRED else []
        return [units[n // _HUNDRED], hundred, *rest]
    words: list[str] = []
    for scale, word in pack.spoken_scales:
        if n >= scale:
            count, n = divmod(n, scale)
            words += [*_spell_integer(count, pack), word]
    if n:
        words.append(units[n])
    return words


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
