from pathlib import Path

import pandas as pd

from indic_asr_audit.normalize import Level
from indic_asr_audit.plots import plot_rq1, plot_rq2


def test_plot_rq1_writes_a_figure(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        [
            {
                "metric": "wer",
                "subset": subset,
                "system": system,
                "level": str(level),
                "estimate": 0.3,
                "low": 0.2,
                "high": 0.4,
                "n": 10,
            }
            for subset in ("held_out_numeric", "numeric")
            for system in ("a", "b")
            for level in Level
        ]
    )
    path = tmp_path / "figures" / "rq1.png"
    plot_rq1(scores, path)
    assert path.stat().st_size > 0


def test_plot_rq2_writes_a_figure_even_without_rows(tmp_path: Path) -> None:
    gap = pd.DataFrame(
        {
            "subset": ["held_out_numeric", "held_out_numeric"],
            "system": ["a", "b"],
            "wer_real": [0.2, 0.3],
            "wer_synthetic": [0.25, 0.35],
        }
    )
    plot_rq2(gap, tmp_path / "rq2.png", "engine")
    plot_rq2(gap, tmp_path / "empty.png", "engine", subset="missing")
    assert (tmp_path / "rq2.png").stat().st_size > 0
    assert (tmp_path / "empty.png").stat().st_size > 0
