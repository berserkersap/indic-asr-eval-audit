"""Figures, drawn only from the result tables so they can be redrawn on CPU.

Figures are built with ``Figure`` rather than pyplot, so no GUI backend is
needed and nothing is left open between plots.
"""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from indic_asr_audit.normalize import Level


def plot_rq1(scores: pd.DataFrame, path: Path) -> None:
    """WER of each system at each normalisation level, with 95% intervals.

    One panel per numeric subset: held-out numbers (all systems) and all
    numbers (more utterances, no MMS).
    """
    subsets = ["held_out_numeric", "numeric"]
    levels = list(Level)
    fig = Figure(figsize=(11, 4))
    axes = fig.subplots(1, len(subsets), sharey=True, squeeze=False)[0]
    for ax, subset in zip(axes, subsets, strict=True):
        wer = scores[(scores["metric"] == "wer") & (scores["subset"] == subset)]
        for system, rows in wer.groupby("system"):
            rows = rows.set_index("level").reindex(levels)
            ax.errorbar(
                levels,
                rows["estimate"] * 100,
                yerr=[
                    (rows["estimate"] - rows["low"]) * 100,
                    (rows["high"] - rows["estimate"]) * 100,
                ],
                marker="o",
                capsize=3,
                label=system,
            )
        n = int(wer["n"].max()) if not wer.empty else 0
        ax.set_title(f"{subset} (n={n} utterances)")
        ax.set_xlabel("normalisation level")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("WER (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def plot_rq2(gap: pd.DataFrame, path: Path, title: str, subset: str = "held_out_numeric") -> None:
    """WER on real audio against WER on synthetic audio, one point per system."""
    rows = gap[gap["subset"] == subset]
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.scatter(rows["wer_real"] * 100, rows["wer_synthetic"] * 100)
    for record in rows.to_dict("records"):
        point = (float(record["wer_real"]) * 100, float(record["wer_synthetic"]) * 100)
        ax.annotate(str(record["system"]), point, fontsize=7)
    values = [*rows["wer_real"], *rows["wer_synthetic"]]
    top = max(float(v) for v in values) * 110 if values else 1.0
    ax.plot([0, top], [0, top], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("WER on real speech (%)")
    ax.set_ylabel("WER on synthetic speech (%)")
    ax.set_title(f"{title}, {subset}", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
