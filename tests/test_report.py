from pathlib import Path

import pandas as pd
import pytest

from indic_asr_audit.report import (
    convention_table,
    disagreement_table,
    fill_readme,
    gap_table,
    ranks_summary,
    rejection_line,
    render_rq2,
    wer_table,
)


def test_fill_readme_replaces_only_marked_blocks(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- t -->\nold\n<!-- /t -->\nend\n", encoding="utf-8")
    fill_readme(readme, {"t": "new"})
    assert readme.read_text(encoding="utf-8") == "intro\n<!-- t -->\nnew\n<!-- /t -->\nend\n"


def test_missing_marker_is_an_error(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        fill_readme(readme, {"t": "new"})


def test_wer_table_sorts_by_final_level() -> None:
    rows = []
    for system, base in (("worse", 0.4), ("better", 0.2)):
        for level in ("whisper", "basic", "spelling", "numbers"):
            rows.append(
                {
                    "system": system,
                    "subset": "s",
                    "level": level,
                    "metric": "wer",
                    "estimate": base,
                    "low": base - 0.01,
                    "high": base + 0.01,
                    "n": 10,
                }
            )
    table = wer_table(pd.DataFrame(rows), "s")
    assert table.index("`better`") < table.index("`worse`")
    assert "20.0 [19.0, 21.0]" in table


def test_disagreement_table_separates_significant_reversals() -> None:
    table = pd.DataFrame(
        [
            {
                "subset": "held_out_numeric",
                "n_systems": 3,
                "kendall_tau": 0.33,
                "reversed": "a/m b/m",
                "significant": "a/m",
            },
            {
                "subset": "numeric",
                "n_systems": 2,
                "kendall_tau": 1.0,
                "reversed": float("nan"),
                "significant": float("nan"),
            },
        ]
    )
    text = disagreement_table(table)
    assert "| `a` vs `m` | `b` vs `m` |" in text
    assert "| none | none |" in text


def test_ranks_summary_is_one_sentence_when_nothing_flips() -> None:
    ranks = pd.DataFrame(
        [
            {
                "subset": "numeric",
                "versus": "whisper vs basic",
                "n_systems": 3,
                "kendall_tau": 1.0,
                "pairs_flipped": 0,
            }
        ]
    )
    assert ranks_summary(ranks).startswith("No pair of systems changes order")


def test_write_tables_removes_the_file_of_an_empty_table(tmp_path: Path) -> None:
    from indic_asr_audit.analysis import write_tables

    write_tables({"t": pd.DataFrame([{"a": 1}])}, tmp_path)
    assert (tmp_path / "t.csv").read_bytes() == b"a\n1\n"
    write_tables({"t": pd.DataFrame()}, tmp_path)
    assert not (tmp_path / "t.csv").exists()


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _gap(system: str, real: float, synthetic: float) -> dict[str, object]:
    return {
        "subset": "held_out_numeric",
        "system": system,
        "n": 3,
        "wer_real": real,
        "wer_synthetic": synthetic,
        "wer_gap": synthetic - real,
        "low": 0.0,
        "high": 0.2,
        "numeric_error_real": 0.1,
        "numeric_error_synthetic": 0.3,
    }


def _agreement(systems: str, n: int) -> dict[str, object]:
    return {
        "subset": "held_out_numeric",
        "systems": systems,
        "n_systems": n,
        "kendall_tau": 1.0,
        "low": 0.8,
        "high": 1.0,
        "nan_resamples": 0,
        "pairs_flipped": 0,
        "significant_flips": 0,
        "flipped": float("nan"),
    }


def test_rejection_line_counts_kept_and_left_out() -> None:
    summary = pd.DataFrame(
        [
            {"status": "saved", "reason": "", "sentences": 10},
            {"status": "saved", "reason": "", "sentences": 5},
            {"status": "rejected", "reason": "latin", "sentences": 3},
            {"status": "rejected", "reason": "latin", "sentences": 1},
            {"status": "rejected", "reason": "duration", "sentences": 2},
        ]
    )
    line = rejection_line(summary, "mms-tts")
    assert line.startswith("`mms-tts`: 15 of 21 sentences kept")
    assert "4 with Latin letters" in line
    assert "2 failed the duration check" in line


def test_render_rq2_is_empty_before_rq2_has_run(tmp_path: Path) -> None:
    assert render_rq2(tmp_path, "d") == {}


def _rq2_results(results: Path) -> None:
    """Ceiling, engine "tts" in two conditions, and engine "v" with no selection table."""
    rq2 = results / "rq2"
    _csv(rq2 / "ceiling" / "gap.csv", [_gap("a", 0.2, 0.2), _gap("b", 0.4, 0.4)])
    _csv(rq2 / "ceiling" / "agreement.csv", [_agreement("all", 2)])
    _csv(
        rq2 / "ceiling" / "utterances.csv",
        [{"subset": "held_out_numeric", "system": s, "spearman_rho": 0.5} for s in "ab"],
    )
    for name in ("tts-clean", "tts-telephony", "v-clean"):
        out = rq2 / f"d-{name}"
        control = {**_gap("a", 0.9, 0.9), "subset": "control"}
        _csv(out / "gap.csv", [_gap("b", 0.4, 0.5), _gap("a", 0.2, 0.3), control])
        _csv(out / "agreement.csv", [_agreement("all", 2)])
        if name == "v-clean":
            continue  # write_tables leaves no file for an empty table
        _csv(
            out / "selection.csv",
            [
                {"system": "a", "sentences": "kept", "n": 4, "wer": 0.2},
                {"system": "b", "sentences": "kept", "n": 4, "wer": 0.4},
                {"system": "b", "sentences": "rejected", "n": 1, "wer": 0.6},
            ],
        )
    same = rq2 / "d-tts-clean" / "same_sentences_as_ceiling_"
    _csv(Path(f"{same}gap.csv"), [{**_gap("a", 0.2, 0.3), "n": 2}])
    _csv(Path(f"{same}agreement.csv"), [_agreement("all", 2), _agreement("unrelated", 1)])
    _csv(
        Path(f"{same}utterances.csv"),
        [
            {"subset": "held_out_numeric", "system": "a", "spearman_rho": 0.1},
            {"subset": "held_out_numeric", "system": "b", "spearman_rho": 0.3},
        ],
    )
    summary = results / "synth" / "d-tts-summary.csv"
    _csv(
        summary,
        [{"has_number": True, "split": "test", "status": "saved", "reason": "", "sentences": 4}],
    )


def test_render_rq2_gap_table_covers_every_set_and_one_subset(tmp_path: Path) -> None:
    _rq2_results(tmp_path)
    blocks = render_rq2(tmp_path, "d", agreement_subsets=["held_out_numeric"])
    assert blocks["rq2-size"] == "`tts`: 4 of 4 sentences kept."
    gap = blocks["rq2-gap"]
    assert gap.index("`tts-clean` | `a`") < gap.index("`tts-clean` | `b`")
    assert "`tts-telephony` | `a`" in gap
    assert "`v-clean` | `a`" in gap
    assert "+10.0 [+0.0, +20.0]" in gap
    assert "90.0" not in gap  # the control rows are another subset


def test_render_rq2_agreement_includes_ceiling_and_same_sentences(tmp_path: Path) -> None:
    _rq2_results(tmp_path)
    blocks = render_rq2(tmp_path, "d", agreement_subsets=["held_out_numeric"])
    agreement = blocks["rq2-agreement"]
    assert "| second real recording | 3 | 2 | 1.00 [0.80, 1.00] | none | 0 |" in agreement
    assert "| `tts-clean`, same sentences | 2 | 2 |" in agreement
    assert "| `tts-clean`, same sentences | 2 | 1, none from the TTS family |" in agreement
    assert "| `tts-clean`, all | 3 | 2 |" in agreement
    assert "`tts-telephony`, same sentences" not in agreement  # no file: empty table
    rho = blocks["rq2-rho"]
    assert "| second real recording | 0.50 (0.50 to 0.50) |" in rho
    assert "| `tts-clean` | 0.20 (0.10 to 0.30) |" in rho
    assert "| `tts-telephony` | - |" in rho


def test_render_rq2_selection_has_one_table_per_engine(tmp_path: Path) -> None:
    _rq2_results(tmp_path)
    selection = render_rq2(tmp_path, "d")["rq2-selection"]
    assert selection.count("`tts`") == 2  # clean and telephony share one engine
    assert "`v`" not in selection  # its table was empty
    assert "| `a` | 20.0 (4) | - |" in selection
    assert "| `b` | 40.0 (4) | 60.0 (1) |" in selection


def test_empty_tables_render_as_no_results() -> None:
    assert gap_table({}, "held_out_numeric") == "No results."
    assert gap_table({"s": pd.DataFrame()}, "held_out_numeric") == "No results."


def test_ranks_summary_lists_only_the_rankings_that_change() -> None:
    ranks = pd.DataFrame(
        [
            {
                "subset": subset,
                "versus": "whisper vs basic",
                "n_systems": 3,
                "kendall_tau": tau,
                "pairs_flipped": flips,
                "flipped": pair,
            }
            for subset, tau, flips, pair in (
                ("numeric", 1.0, 0, float("nan")),
                ("control", 0.33, 1, "a/b"),
            )
        ]
    )
    text = ranks_summary(ranks)
    assert text.startswith("Only these rankings change")
    assert "| without numbers | whisper vs basic | 3 | 0.33 | `a` vs `b` |" in text
    assert "all with numbers" not in text


def test_convention_table_has_one_row_per_level() -> None:
    convention = pd.DataFrame(
        [
            {"subset": "control", "level": level, "wer_gap": 0.05, "low": 0.01, "high": 0.1}
            for level in ("whisper", "basic", "spelling", "numbers")
        ]
    )
    text = convention_table(convention)
    assert text.splitlines()[0] == "| level | without numbers |"
    assert "| `numbers` | +5.0 [+1.0, +10.0] |" in text
    assert len(text.splitlines()) == 6


def test_points_never_prints_negative_zero() -> None:
    from indic_asr_audit.report import steps_table

    steps = pd.DataFrame(
        [
            {
                "system": "a",
                "subset": "s",
                "step": step,
                "wer_change": 0.003,
                "low": -0.0004,
                "high": 0.008,
            }
            for step in ("whisper->basic", "basic->spelling", "spelling->numbers")
        ]
    )
    assert "[+0.0, +0.8]" in steps_table(steps, "s")
