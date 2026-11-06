from typing import Any

import pandas as pd

from indic_asr_audit.data import Utterance
from indic_asr_audit.languages import get_language
from indic_asr_audit.rq2 import rq2_tables

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
