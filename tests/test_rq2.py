from typing import Any

import pandas as pd
import pytest

from indic_asr_audit.data import Utterance
from indic_asr_audit.languages import get_language
from indic_asr_audit.rq2 import real_ceiling, rq2_tables, selection_table

HINDI = get_language("hi")
SENTENCES = ["उसने 53 किताबें खरीदीं", "कुल 800 सैनिक थे", "वह 2 बार घर गया", "यह 15 साल पुरानी बात है"]
UTTERANCES = [
    Utterance(f"u{i}", f"u{i}.wav", text, f"s{i}", "", "test", True)
    for i, text in enumerate(SENTENCES)
]


def with_errors(n_wrong: int) -> dict[str, str]:
    """Correct transcripts, except the first word of the first n sentences."""
    return {
        u.id: ("गलत " + " ".join(u.text.split()[1:])) if i < n_wrong else u.text
        for i, u in enumerate(UTTERANCES)
    }


def row(table: pd.DataFrame, **match: str) -> dict[str, Any]:
    rows = [r for r in table.to_dict("records") if all(r[k] == v for k, v in match.items())]
    assert len(rows) == 1
    return {str(k): v for k, v in rows[0].items()}


def test_identical_rankings_agree_fully() -> None:
    real = {"good": with_errors(0), "mid": with_errors(2), "bad": with_errors(4)}
    synth = {"good": with_errors(1), "mid": with_errors(2), "bad": with_errors(3)}
    agreement = row(rq2_tables(UTTERANCES, real, synth, HINDI, {})["agreement"], subset="numeric")
    assert agreement["kendall_tau"] == 1.0
    assert agreement["pairs_flipped"] == 0


def test_reversed_ranking_is_reported_as_flips() -> None:
    real = {"a": with_errors(0), "b": with_errors(3)}
    synth = {"a": with_errors(3), "b": with_errors(0)}
    tables = rq2_tables(UTTERANCES, real, synth, HINDI, {})
    agreement = row(tables["agreement"], subset="numeric")
    assert agreement["kendall_tau"] == -1.0
    assert agreement["flipped"] == "a/b"
    assert row(tables["gap"], subset="numeric", system="a")["wer_gap"] > 0


def test_identical_results_agree_on_every_resample() -> None:
    real = {"good": with_errors(0), "mid": with_errors(2), "bad": with_errors(4)}
    agreement = row(rq2_tables(UTTERANCES, real, real, HINDI, {})["agreement"], subset="numeric")
    assert (agreement["low"], agreement["high"]) == pytest.approx((1.0, 1.0))


def test_related_systems_get_their_own_agreement_row() -> None:
    real = {"a": with_errors(0), "b": with_errors(2), "mms": with_errors(4)}
    tables = rq2_tables(UTTERANCES, real, real, HINDI, {}, related={"mms"})
    rows = tables["agreement"][tables["agreement"]["subset"] == "numeric"]
    assert list(rows["systems"]) == ["all", "unrelated"]
    assert list(rows["n_systems"]) == [3, 2]


def test_flip_between_near_ties_is_not_significant() -> None:
    real = {"a": with_errors(1), "b": with_errors(2)}
    synth = {"a": with_errors(2), "b": with_errors(1)}
    agreement = row(rq2_tables(UTTERANCES, real, synth, HINDI, {})["agreement"], subset="numeric")
    assert agreement["pairs_flipped"] == 1
    assert agreement["significant_flips"] == 0


def test_real_ceiling_compares_two_recordings_of_each_sentence() -> None:
    first = Utterance("a1", "a1.wav", "उसने 53 किताबें खरीदीं", "s0", "", "test", True)
    second = Utterance("a2", "a2.wav", first.text, "s0", "", "test", True)
    alone = Utterance("b1", "b1.wav", "कुल 800 सैनिक थे", "s1", "", "test", True)
    hyps = {
        "x": {"a1": first.text, "a2": "गलत 53 किताबें खरीदीं", "b1": alone.text},
        "y": {"a1": first.text, "a2": first.text, "b1": alone.text},
    }
    gap = real_ceiling([first, second, alone], hyps, HINDI, {})["gap"]
    assert set(gap["n"]) == {1}
    assert row(gap, subset="numeric", system="x")["wer_gap"] > 0
    assert row(gap, subset="numeric", system="y")["wer_gap"] == 0


def test_selection_table_splits_kept_and_rejected_sentences() -> None:
    real = {"x": with_errors(2)}
    table = selection_table(UTTERANCES, {"u2", "u3"}, real, HINDI, {})
    by_label = {r["sentences"]: r for r in table.to_dict("records")}
    assert by_label["kept"]["wer"] == 0
    assert by_label["rejected"]["wer"] > 0
