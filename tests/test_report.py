from pathlib import Path

import pandas as pd
import pytest

from indic_asr_audit.report import fill_readme, wer_table


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
