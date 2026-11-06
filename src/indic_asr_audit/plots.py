"""Figures, drawn only from the result tables so they can be redrawn on CPU."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from indic_asr_audit.normalize import Level


def plot_rq1(scores: pd.DataFrame, path: Path) -> None:
    """WER of each system at each normalisation level, with 95% intervals.

    One panel per numeric subset: held-out numbers (all systems) and all
    numbers (more utterances, no MMS).
    """
    subsets = ["held_out_numeric", "numeric"]
    levels = list(Level)
    fig, axes = plt.subplots(1, len(subsets), figsize=(11, 4), sharey=True)
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
    plt.close(fig)
