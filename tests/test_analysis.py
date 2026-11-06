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


def test_convention_gap_is_verbatim_minus_transcribe() -> None:
    transcripts = {
        "saaras-v4-transcribe": TRANSCRIPTS["digits"],
        "saaras-v4-verbatim": TRANSCRIPTS["words"],
    }
    gap = rq1_tables(UTTERANCES, transcripts, HINDI, {})["convention"]
    numeric = gap[gap["subset"] == "numeric"].set_index("level")["wer_gap"]
    assert numeric["spelling"] > 0
    assert numeric["numbers"] == 0


def test_rank_table_lists_flipped_pairs() -> None:
    # "words" says numbers aloud: worse than "digits" until numbers are
    # normalised, then tied with it. "slip" mishears one other word, so it
    # beats "words" at the start and loses to it at the numbers level.
    slip = {**TRANSCRIPTS["digits"], "u0": "उसने 53 किताबें खरीदी"}
    transcripts = {"words": TRANSCRIPTS["words"], "slip": slip}
    ranks = rq1_tables(UTTERANCES, transcripts, HINDI, {})["ranks"]
    row = ranks[(ranks["subset"] == "numeric") & (ranks["versus"] == "whisper vs numbers")]
    assert int(row["n_systems"].iloc[0]) == 2
    assert int(row["pairs_flipped"].iloc[0]) == 1
    assert float(row["kendall_tau"].iloc[0]) < 0


def test_wer_and_numeric_error_can_rank_systems_in_opposite_orders() -> None:
    # "numbers_wrong" gets every word right except the numbers; "words_wrong"
    # gets the numbers right but mishears other words.
    numbers_wrong = {
        "u0": "उसने 35 किताबें खरीदीं",
        "u1": "कुल 900000 सैनिक थे",
        "u2": "वह घर गया",
        "u3": "यह 16वीं सदी की बात है",
    }
    words_wrong = {
        "u0": "उसमें 53 कीताबें खरीदी",
        "u1": "कुल 800000 सेनिक थी",
        "u2": "वो घर गई",
        "u3": "ये 15वीं सदि की बाद हैं",
    }
    tables = rq1_tables(UTTERANCES, {"a": numbers_wrong, "b": words_wrong}, HINDI, {})
    row = tables["wer_vs_numeric"].set_index("subset").loc["numeric"]
    assert row["pairs_reversed"] == 1
    assert row["kendall_tau"] == -1.0


def test_a_reversal_is_significant_only_if_both_intervals_exclude_zero() -> None:
    from indic_asr_audit.analysis import metric_disagreement
    from indic_asr_audit.metrics import Counts

    clusters = [f"s{i}" for i in range(20)]
    # a: best WER, misses every number. b: worst WER, misses none. Both gaps are
    # the same in every sentence, so both intervals exclude zero.
    # c: between them on both metrics, but its WER lead over a flips sign from
    # sentence to sentence, so the a/c WER interval includes zero.
    wer = {
        "a": [Counts(1, 10)] * 20,
        "b": [Counts(3, 10)] * 20,
        "c": [Counts(0 if i % 2 else 3, 10) for i in range(20)],
    }
    numeric = {
        "a": [Counts(1, 1)] * 20,
        "b": [Counts(0, 1)] * 20,
        "c": [Counts(i % 2, 1) for i in range(20)],
    }
    row = metric_disagreement(wer, numeric, clusters)
    assert set(str(row["reversed"]).split()) == {"a/b", "a/c", "b/c"}
    assert set(str(row["significant"]).split()) == {"a/b", "b/c"}
    assert row["significant_reversals"] == 2


def test_diagnostics_count_long_outputs_and_number_word_errors() -> None:
    hyps = {
        "u0": "उसने 53 किताबें खरीदीं " * 3,  # three times the reference: a loop
        "u1": "कुल 8 0 0 सैनिक थे",  # one garbled number: three errors with digits
        "u2": "वह घर गया",
        "u3": "यह 15वीं सदी की बात है",
    }
    tables = rq1_tables(UTTERANCES, {"a": hyps}, HINDI, {})
    row = tables["diagnostics"].set_index("subset").loc["numeric"]
    assert row["long_outputs"] == 1
    assert row["wer_without_long"] < row["wer"]
    # u1: three errors with digits; u0: the loop repeats "53" twice
    assert row["wer_from_number_words"] * row["reference_words"] == pytest.approx(5)
