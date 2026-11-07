from collections import Counter

import pandas as pd
import pytest

from indic_asr_audit.data import Utterance
from indic_asr_audit.languages import get_language
from indic_asr_audit.metrics import missed_values, numeric_counts
from indic_asr_audit.number_audit import (
    audit_tables,
    digit_share,
    miss_kinds,
    missed_by_all,
    number_format_change,
    split_wer_table,
)
from indic_asr_audit.report import (
    miss_kinds_markdown,
    missed_by_all_line,
    number_format_markdown,
    split_wer_markdown,
)

HINDI = get_language("hi")


def _utt(i: int, text: str, split: str = "test", sentence: str | None = None) -> Utterance:
    return Utterance(f"u{i}", f"u{i}.wav", text, sentence or f"s{i}", "", split, True)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("1767 में रेल", "176 में रेल", {"another number": 1}),
        ("करीब 15 मील", "करीब पंद्रा मील", {"a word": 1}),
        ("उसकी 53 साल उम्र", "उसकी साल उम्र", {"dropped": 1}),
        ("उसकी 53 साल उम्र", "उसकी तिरेपन साल उम्र", {}),
        ("उसकी 53 साल उम्र", "", {"dropped": 1}),
        # The second 11 is the one missed; the first is found.
        ("11 चौकों और 11 रन", "11 चौकों और 1 रन", {"another number": 1}),
        ("कुल 53 लोग", "आज कुल बहुत से 35 लोग थे", {"another number": 1}),  # inserted words
    ],
)
def test_miss_kinds(reference: str, hypothesis: str, expected: dict[str, int]) -> None:
    assert miss_kinds(reference, hypothesis, HINDI) == Counter(expected)


def test_missed_values_agree_with_the_numeric_value_error() -> None:
    reference, hypothesis = "53 और 15 और 1767", "तिरेपन और 176"
    errors = numeric_counts(reference, hypothesis, HINDI).errors
    assert sum(missed_values(reference, hypothesis, HINDI).values()) == errors == 2


def test_missed_by_all_keeps_only_numbers_every_system_misses() -> None:
    utts = [_utt(0, "सन 1767 में 53 लोग")]
    transcripts = {"a": {"u0": "सन 176 में लोग"}, "b": {"u0": "सन 1767 में लोग"}}
    table = missed_by_all(utts, transcripts, HINDI)
    assert table[["id", "value"]].values.tolist() == [["u0", "53"]]


def test_missed_by_all_handles_repeats_and_no_common_miss() -> None:
    utts = [_utt(0, "53 और 53 लोग"), _utt(1, "कुल 7 दिन")]
    transcripts = {"a": {"u0": "लोग", "u1": "कुल दिन"}, "b": {"u0": "53 लोग", "u1": "कुल 7 दिन"}}
    table = missed_by_all(utts, transcripts, HINDI)
    assert table[["id", "value"]].values.tolist() == [["u0", "53"]]  # b found one of the two
    assert missed_by_all(utts, {"b": transcripts["b"]}, HINDI).values.tolist() == [
        ["u0", "s0", "53"]
    ]
    assert missed_by_all([utts[1]], {"b": transcripts["b"]}, HINDI).empty


def test_digit_share_counts_digits_and_words() -> None:
    assert digit_share(["53 और तिरेपन", "कुल 7"], HINDI) == (2, 1)


def test_split_wer_table_scores_train_and_held_out_separately() -> None:
    utts = [_utt(0, "53 लोग", "train"), _utt(1, "53 लोग", "dev"), _utt(2, "53 लोग", "test")]
    hyps = {"a": {"u0": "53 लोग", "u1": "53 बच्चे", "u2": "53 लोग"}}
    table = split_wer_table(utts, hyps, HINDI).set_index("splits")
    assert table.loc["train", "wer"] == 0.0
    assert table.loc["dev and test", "wer"] == pytest.approx(0.25)
    assert table.loc["dev and test", "n"] == 2


def test_number_format_change_compares_real_and_synthetic() -> None:
    utts = [_utt(0, "53 लोग")]
    real = {"a": {"u0": "53 लोग"}, "b": {"u0": "53 लोग"}}
    synthetic = {"a": {"u0": "तिरेपन लोग"}}  # "b" was not run on the synthetic set
    table = number_format_change(utts, real, synthetic, HINDI)
    assert table.values.tolist() == [["a", 1.0, 0.0]]


def test_audit_tables_use_held_out_utterances_and_all_splits_for_the_probe() -> None:
    utts = [_utt(0, "53 लोग", "train"), _utt(1, "1767 में", "test")]
    tables = audit_tables(utts, {"a": {"u0": "53 लोग", "u1": "176 में"}}, HINDI)
    assert tables["missed_kinds"].loc[0, "numbers"] == 1  # the train utterance is left out
    assert set(tables["split_wer"]["splits"]) == {"train", "dev and test"}


def test_missed_by_all_line_counts_what_is_left() -> None:
    kinds = pd.DataFrame({"system": ["a", "b"], "numbers": [10, 10], "missed": [3, 5]})
    missed = pd.DataFrame({"id": ["u0", "u0"], "sentence_id": ["s1", "s1"], "value": ["1", "2"]})
    line = missed_by_all_line(missed, kinds)
    assert line.startswith("2 of the 10 numbers are missed by all 2 systems: 1 and 2 (sentence s1)")
    assert line.endswith("of the other 8: `a` 1, `b` 3.")
    assert missed_by_all_line(missed.iloc[:0], kinds).startswith("No reference number")


def test_miss_kinds_markdown_sorts_by_missed_and_notes_the_reference() -> None:
    kinds = pd.DataFrame(
        {
            "system": ["b", "a"],
            "numbers": [10, 10],
            "missed": [5, 3],
            "another number": [1, 1],
            "a word": [2, 1],
            "dropped": [2, 1],
            "other": [0, 0],
        }
    )
    formats = pd.DataFrame(
        {
            "system": ["reference", "a", "b"],
            "as_digits": [9, 5, 1],
            "as_words": [1, 5, 9],
            "digit_share": [0.9, 0.5, 0.1],
        }
    )
    text = miss_kinds_markdown(kinds, formats)
    assert text.index("`a`") < text.index("`b`")
    assert "| `a` | 3 of 10 | 1 | 1 | 1 | 0 | 50% |" in text
    assert text.endswith("as digits (9 of 10 numbers).")


def test_split_wer_markdown_shows_both_groups_with_sizes() -> None:
    split = pd.DataFrame(
        {
            "system": ["a", "a"],
            "splits": ["train", "dev and test"],
            "wer": [0.1, 0.2],
            "low": [0.05, 0.1],
            "high": [0.15, 0.3],
            "n": [40, 8],
        }
    )
    text = split_wer_markdown(split)
    assert "| system | WER on train (40) | WER on dev and test (8) |" in text
    assert "| `a` | 10.0 [5.0, 15.0] | 20.0 [10.0, 30.0] |" in text


def test_number_format_markdown_marks_missing_systems() -> None:
    one = pd.DataFrame(
        {"system": ["a", "b"], "digit_share_real": [0.8, 0.5], "digit_share_synthetic": [0.4, 0.5]}
    )
    two = one.iloc[:1]
    text = number_format_markdown({"x-clean": one, "y-clean": two, "z-clean": pd.DataFrame()})
    assert "| `a` | 80% | 40% | 40% |" in text
    assert "| `b` | 50% | 50% | - |" in text
    assert "z-clean" not in text
