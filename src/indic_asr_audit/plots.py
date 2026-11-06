"""Figures, drawn only from the result tables so they can be redrawn on CPU."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from indic_asr_audit.normalize import Level


def plot_rq1(scores: pd.DataFrame, path: Path, subset: str = "numeric") -> None:
    """WER of each system at each normalisation level, with 95% intervals."""
    wer = scores[(scores.metric == "wer") & (scores.subset == subset)]
    levels = list(Level)
    fig, ax = plt.subplots(figsize=(7, 4))
    for system, rows in wer.groupby("system"):
        rows = rows.set_index("level").loc[levels]
        ax.errorbar(
            levels,
            rows.estimate * 100,
            yerr=[(rows.estimate - rows.low) * 100, (rows.high - rows.estimate) * 100],
            marker="o",
            capsize=3,
            label=system,
        )
    ax.set_xlabel("normalisation level")
    ax.set_ylabel(f"WER (%), {subset} utterances")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
