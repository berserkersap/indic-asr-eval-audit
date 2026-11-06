from pathlib import Path

import pandas as pd
import pytest

from indic_asr_audit.report import disagreement_table, fill_readme, ranks_summary, wer_table


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
