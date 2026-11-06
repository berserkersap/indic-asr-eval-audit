import pandas as pd
import pytest

from indic_asr_audit.analysis import rq1_tables
from indic_asr_audit.data import Utterance
from indic_asr_audit.languages import get_language

HINDI = get_language("hi")


def utterance(i: int, text: str, split: str = "test") -> Utterance:
    return Utterance(
        id=f"u{i}",
        audio=f"u{i}.wav",
        text=text,
        sentence_id=f"s{i}",
        gender="",
        split=split,
        has_number=any(c.isdigit() for c in text),
    )


UTTERANCES = [
    utterance(0, "उसने 53 किताबें खरीदीं"),
    utterance(1, "कुल 800,000 सैनिक थे"),
    utterance(2, "वह घर गया"),
    utterance(3, "यह 15वीं सदी की बात है", split="train"),
]
# "digits" writes numbers as digits, "words" says them; both are otherwise perfect.
TRANSCRIPTS = {
    "digits": {
        "u0": "उसने 53 किताबें खरीदीं",
        "u1": "कुल 800000 सैनिक थे",
        "u2": "वह घर गया",
        "u3": "यह 15वीं सदी की बात है",
    },
    "words": {
        "u0": "उसने तिरेपन किताबें खरीदीं",
        "u1": "कुल आठ लाख सैनिक थे",
        "u2": "वह घर गया",
        "u3": "यह पंद्रहवीं सदी की बात है",
    },
}


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return rq1_tables(UTTERANCES, TRANSCRIPTS, HINDI, {})


def wer(tables: dict[str, pd.DataFrame], system: str, level: str) -> float:
    scores = tables["scores"]
    row = scores[
        (scores["system"] == system)
        & (scores["subset"] == "numeric")
        & (scores["level"] == level)
        & (scores["metric"] == "wer")
    ]
    return float(row["estimate"].iloc[0])


def test_spoken_numbers_are_errors_until_the_numbers_level(
    tables: dict[str, pd.DataFrame],
) -> None:
    assert wer(tables, "words", "spelling") > 0
    assert wer(tables, "words", "numbers") == 0
    assert wer(tables, "digits", "numbers") == 0


def test_convention_gap_needs_both_saaras_modes(tables: dict[str, pd.DataFrame]) -> None:
    assert tables["convention"].empty


def test_excluded_split_removes_system_from_subset() -> None:
    scores = rq1_tables(UTTERANCES, TRANSCRIPTS, HINDI, {"words": {"train"}})["scores"]
    assert set(scores[scores["subset"] == "numeric"]["system"]) == {"digits"}
    assert set(scores[scores["subset"] == "held_out_numeric"]["system"]) == {"digits", "words"}
