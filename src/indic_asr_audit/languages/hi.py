"""Hindi language pack (Devanagari), with English number words for code-mixed speech."""

import re
from fractions import Fraction

from indic_asr_audit.languages.base import LanguagePack

# Hindi has a distinct word for every number below 100. Common spelling
# variants are listed after the standard form.
_HINDI_UNITS: dict[int, tuple[str, ...]] = {
    0: ("शून्य",),
    1: ("एक",),
    2: ("दो",),
    3: ("तीन",),
    4: ("चार",),
    5: ("पांच",),
    6: ("छह", "छः", "छे"),
    7: ("सात",),
    8: ("आठ",),
    9: ("नौ",),
    10: ("दस",),
    11: ("ग्यारह",),
    12: ("बारह",),
    13: ("तेरह",),
    14: ("चौदह",),
    15: ("पंद्रह", "पन्द्रह"),
    16: ("सोलह",),
    17: ("सत्रह",),
    18: ("अठारह", "अट्ठारह"),
    19: ("उन्नीस",),
    20: ("बीस",),
    21: ("इक्कीस",),
    22: ("बाईस",),
    23: ("तेईस",),
    24: ("चौबीस",),
    25: ("पच्चीस",),
    26: ("छब्बीस",),
    27: ("सत्ताईस",),
    28: ("अट्ठाईस", "अठाईस"),
    29: ("उनतीस",),
    30: ("तीस",),
    31: ("इकतीस", "इकत्तीस"),
    32: ("बत्तीस",),
    33: ("तैंतीस", "तेंतीस"),
    34: ("चौंतीस",),
    35: ("पैंतीस",),
    36: ("छत्तीस",),
    37: ("सैंतीस",),
    38: ("अड़तीस",),
    39: ("उनतालीस", "उन्तालीस"),
    40: ("चालीस",),
    41: ("इकतालीस",),
    42: ("बयालीस",),
    43: ("तैंतालीस", "तेंतालीस"),
    44: ("चवालीस", "चौवालीस"),
    45: ("पैंतालीस",),
    46: ("छियालीस",),
    47: ("सैंतालीस",),
    48: ("अड़तालीस",),
    49: ("उनचास",),
    50: ("पचास",),
    51: ("इक्यावन",),
    52: ("बावन",),
    53: ("तिरेपन", "तिरपन"),
    54: ("चौवन", "चौव्वन"),
    55: ("पचपन",),
    56: ("छप्पन",),
    57: ("सत्तावन",),
    58: ("अट्ठावन", "अठावन"),
    59: ("उनसठ",),
    60: ("साठ",),
    61: ("इकसठ",),
    62: ("बासठ",),
    63: ("तिरसठ",),
    64: ("चौंसठ",),
    65: ("पैंसठ",),
    66: ("छियासठ",),
    67: ("सड़सठ", "सरसठ"),
    68: ("अड़सठ",),
    69: ("उनहत्तर",),
    70: ("सत्तर",),
    71: ("इकहत्तर",),
    72: ("बहत्तर",),
    73: ("तिहत्तर",),
    74: ("चौहत्तर",),
    75: ("पचहत्तर",),
    76: ("छिहत्तर",),
    77: ("सतहत्तर",),
    78: ("अठहत्तर",),
    79: ("उनासी", "उन्यासी"),
    80: ("अस्सी",),
    81: ("इक्यासी",),
    82: ("बयासी",),
    83: ("तिरासी",),
    84: ("चौरासी",),
    85: ("पचासी",),
    86: ("छियासी",),
    87: ("सत्तासी",),
    88: ("अट्ठासी",),
    89: ("नवासी",),
    90: ("नब्बे",),
    91: ("इक्यानबे", "इक्यानवे"),
    92: ("बानबे", "बानवे"),
    93: ("तिरानबे", "तिरानवे"),
    94: ("चौरानबे", "चौरानवे"),
    95: ("पंचानबे", "पंचानवे"),
    96: ("छियानबे", "छियानवे"),
    97: ("सत्तानबे", "सत्तानवे"),
    98: ("अट्ठानबे", "अट्ठानवे"),
    99: ("निन्यानबे", "निन्यानवे"),
}

_ENGLISH_UNITS: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

_NASAL = "\u0902"  # anusvara
_CHANDRABINDU = "\u0901"
_NUKTA = "\u093c"


def _with_variants(word: str) -> set[str]:
    """Add the spellings ASR systems often produce for a number word.

    The nasal sign is often dropped (पैंतीस / पैतीस) and a final long ई is
    often written short (बाईस / बाइस).
    """
    variants = {word, word.replace(_NASAL, "")}
    return variants | {re.sub("ईस$", "इस", v) for v in variants}


HINDI = LanguagePack(
    code="hi",
    name="Hindi",
    native_digits="०१२३४५६७८९",
    units={
        **{
            variant: value
            for value, words in _HINDI_UNITS.items()
            for word in words
            for variant in _with_variants(word)
        },
        **{word: value for value, word in _ENGLISH_UNITS.items()},
    },
    scales={
        "सौ": 100,
        "हज़ार": 1_000,
        "लाख": 100_000,
        "करोड़": 10_000_000,
        "अरब": 1_000_000_000,
        "hundred": 100,
        "thousand": 1_000,
        "lakh": 100_000,
        "crore": 10_000_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    },
    fractions={"डेढ़": Fraction(3, 2), "ढाई": Fraction(5, 2)},
    offsets={"सवा": Fraction(1, 4), "साढ़े": Fraction(1, 2), "पौने": Fraction(-1, 4)},
    decimal_points=frozenset({"दशमलव", "point", "पॉइंट"}),
    # Longest first, so "वीं" is split off before "वी".
    ordinal_suffixes=("वीं", "वां", "वें", "वी", "वा", "वे"),
    symbols={"%": "प्रतिशत"},
    prefix_symbols={"₹": "रुपये"},
    spelling_folds=(
        # The nukta is often dropped in writing and by ASR (ज़्यादा / ज्यादा,
        # पढ़ना / पढना); almost no Hindi word pairs differ by it alone.
        (_NUKTA, ""),
        # Chandrabindu and anusvara are used interchangeably (पाँच / पांच).
        (_CHANDRABINDU, _NASAL),
    ),
)
