"""Figures, drawn only from the result tables so they can be redrawn on CPU.

Each figure is written twice, as ``<name>-light.svg`` and ``<name>-dark.svg``,
for GitHub's light and dark themes. Colour follows the vendor (OpenAI, Meta,
Sarvam) so a model keeps its colour across figures, and every series is also
named on the chart, so colour is never the only cue. Figures are built with
``Figure`` rather than pyplot, so no GUI backend is needed.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType

from indic_asr_audit.normalize import Level


@dataclass(frozen=True)
class Theme:
    """Ink and series colours for one background; the background is transparent."""

    name: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    vendors: Mapping[str, str]


# Validated categorical slots 1-3 (blue, orange, aqua), stepped per theme.
THEMES = (
    Theme(
        "light",
        ink="#0b0b0b",
        secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        vendors={"openai": "#2a78d6", "meta": "#eb6834", "sarvam": "#1baf7a"},
    ),
    Theme(
        "dark",
        ink="#f0efec",
        secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        vendors={"openai": "#3987e5", "meta": "#d95926", "sarvam": "#199e70"},
    ),
)

VENDOR_NAMES = {"openai": "OpenAI Whisper", "meta": "Meta MMS", "sarvam": "Sarvam"}
DISPLAY_NAMES = {
    "mms-1b-all": "MMS-1B",
    "saaras-v4-transcribe": "Saaras v4 transcribe",
    "saaras-v4-verbatim": "Saaras v4 verbatim",
    "whisper-large-v3": "Whisper large-v3",
    "whisper-large-v3-turbo": "Whisper large-v3-turbo",
    "whisper-medium": "Whisper medium",
    "whisper-small": "Whisper small",
}
LEVEL_NAMES = {
    Level.WHISPER: "Whisper\nnormaliser",
    Level.BASIC: "basic",
    Level.SPELLING: "+ spelling",
    Level.NUMBERS: "+ numbers",
}
SUBSET_TITLES = {
    "held_out_numeric": "Held-out utterances with numbers",
    "numeric": "All utterances with numbers (no MMS-1B)",
}
RC: dict[RcKeyType, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "svg.fonttype": "none",  # text stays text: small, sharp and searchable
    "svg.hashsalt": "indic-asr-audit",  # stable ids, so redrawing does not churn git
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linewidth": 0.6,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
}


def vendor(system: str) -> str:
    """The vendor whose colour a system takes: whisper, mms or Sarvam models."""
    if system.startswith("whisper"):
        return "openai"
    if system.startswith("mms"):
        return "meta"
    return "sarvam"


def _display(system: str) -> str:
    return DISPLAY_NAMES.get(system, system)


def _spread(values: Sequence[float], gap: float) -> list[float]:
    """Nudge label positions apart by at least ``gap``, keeping their order and centre."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    shift = (sum(values) - sum(placed)) / len(values) if values else 0.0
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k] + shift
    return out


def _axes_style(ax: Axes, theme: Theme) -> None:
    ax.set_facecolor("none")
    ax.grid(color=theme.grid)
    ax.spines["bottom"].set_color(theme.axis)
    ax.tick_params(colors=theme.secondary, labelcolor=theme.secondary)
    ax.xaxis.label.set_color(theme.secondary)
    ax.yaxis.label.set_color(theme.secondary)


def _heading(fig: Figure, theme: Theme, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.985, title, color=theme.ink, fontsize=12, weight="bold", va="top")
    fig.text(0.01, 0.92, subtitle, color=theme.secondary, fontsize=9, va="top")


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True, metadata={"Date": None, "Creator": None})


def _draw(draw: Callable[[Theme], Figure], out_dir: Path, name: str) -> list[Path]:
    paths = []
    with mpl.rc_context(RC):
        for theme in THEMES:
            path = out_dir / f"{name}-{theme.name}.svg"
            _save(draw(theme), path)
            paths.append(path)
    return paths


def wer_vs_numbers(scores: pd.DataFrame, theme: Theme, subset: str = "held_out_numeric") -> Figure:
    """WER (numbers level) against the share of numbers missed, one point per system."""
    rows = scores[scores["subset"] == subset]
    wer = rows[(rows["metric"] == "wer") & (rows["level"] == Level.NUMBERS)].set_index("system")
    missed = rows[rows["metric"] == "numeric_error"].set_index("system")
    systems = [s for s in wer.index if s in missed.index]
    fig = Figure(figsize=(7, 4.8))
    ax = fig.add_axes((0.08, 0.11, 0.9, 0.72))
    _axes_style(ax, theme)
    for system in systems:
        x, y = (wer.loc[system] * 100, missed.loc[system] * 100)
        colour = theme.vendors[vendor(system)]
        ax.errorbar(
            x["estimate"],
            y["estimate"],
            xerr=[[x["estimate"] - x["low"]], [x["high"] - x["estimate"]]],
            yerr=[[y["estimate"] - y["low"]], [y["high"] - y["estimate"]]],
            fmt="none",
            ecolor=colour,
            elinewidth=1,
            alpha=0.45,
        )
        ax.plot(x["estimate"], y["estimate"], "o", ms=8, color=colour, mec="none")
    labels = _spread([float(missed.loc[s, "estimate"]) * 100 for s in systems], gap=4.5)
    for system, label_y in zip(systems, labels, strict=True):
        x_value = float(wer.loc[system, "estimate"]) * 100
        ax.annotate(
            _display(system),
            (x_value, label_y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink,
        )
    for key, label in VENDOR_NAMES.items():
        if any(vendor(s) == key for s in systems):
            ax.plot([], [], "o", ms=7, color=theme.vendors[key], label=label)
    legend = ax.legend(loc="upper left", frameon=False, labelcolor=theme.secondary)
    legend.set_in_layout(False)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("WER after number normalisation (%)")
    ax.set_ylabel("numbers missed (%)")
    n = int(wer["n"].max())
    _heading(
        fig,
        theme,
        "Lowest WER is not fewest missed numbers",
        f"{SUBSET_TITLES.get(subset, subset)} ({n}), FLEURS Hindi; bars: 95% bootstrap intervals",
    )
    return fig


def wer_by_level(scores: pd.DataFrame, theme: Theme) -> Figure:
    """WER of each system at each normalisation level, one panel per numeric subset."""
    subsets = ["held_out_numeric", "numeric"]
    levels = list(Level)
    fig = Figure(figsize=(11, 4.8))
    axes = fig.subplots(1, len(subsets), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.14, wspace=0.12)
    top = float(scores.loc[scores["metric"] == "wer", "estimate"].max()) * 100
    for ax, subset in zip(axes, subsets, strict=True):
        _axes_style(ax, theme)
        wer = scores[(scores["metric"] == "wer") & (scores["subset"] == subset)]
        ends: dict[str, float] = {}
        for system, rows in wer.groupby("system"):
            values = rows.set_index("level").reindex(levels)["estimate"] * 100
            colour = theme.vendors[vendor(str(system))]
            ax.plot(range(len(levels)), values, marker="o", ms=5, color=colour, label=system)
            ends[str(system)] = float(values.iloc[-1])
        for system, label_y in zip(ends, _spread(list(ends.values()), gap=4.0), strict=True):
            ax.annotate(
                _display(system),
                (len(levels) - 1, label_y),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=theme.ink,
            )
        ax.set_xticks(range(len(levels)), [LEVEL_NAMES[level] for level in levels])
        ax.set_xlim(-0.25, len(levels) - 1 + 1.45)
        ax.set_ylim(0, top * 1.05)
        n = int(wer["n"].max()) if not wer.empty else 0
        ax.set_title(f"{SUBSET_TITLES[subset]} ({n})", loc="left", color=theme.ink, fontsize=10)
    axes[0].set_ylabel("WER (%)")
    _heading(
        fig,
        theme,
        "Whisper's normaliser makes Hindi WER look lower; number normalisation closes Saaras's gap",
        "FLEURS Hindi; each level adds one normalisation step; intervals are in the tables",
    )
    return fig


def real_vs_synthetic(
    gaps: Mapping[str, pd.DataFrame], theme: Theme, subset: str = "held_out_numeric"
) -> Figure:
    """WER on real speech and on each engine's audio, one row per system, one panel per condition.

    ``gaps`` maps a test set ("mms-tts-clean") to its gap table.
    """
    engines = {"mms-tts": ("MMS-TTS (Meta)", "meta"), "bulbul": ("Bulbul (Sarvam)", "sarvam")}
    gaps = {n: t for n, t in gaps.items() if n.rpartition("-")[0] in engines and not t.empty}
    conditions = sorted({name.rpartition("-")[2] for name in gaps})
    top = max(float(t[["wer_real", "wer_synthetic"]].max().max()) for t in gaps.values()) * 100
    fig = Figure(figsize=(11, 4.2))
    axes = fig.subplots(1, len(conditions), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(left=0.14, right=0.99, top=0.76, bottom=0.14, wspace=0.08)
    order: list[str] = []
    for ax, condition in zip(axes, conditions, strict=True):
        _axes_style(ax, theme)
        tables = {
            name.rpartition("-")[0]: table[table["subset"] == subset].set_index("system")
            for name, table in gaps.items()
            if name.endswith(f"-{condition}")
        }
        first = next(iter(tables.values()))
        real = first["wer_real"] * 100
        synthetic = {engine: table["wer_synthetic"] * 100 for engine, table in tables.items()}
        order = order or list(real.sort_values(ascending=False).index)
        for y, system in enumerate(order):
            values = [float(real[system])] + [float(s[system]) for s in synthetic.values()]
            ax.plot([min(values), max(values)], [y, y], color=theme.axis, lw=1.5, zorder=1)
            ax.plot(values[0], y, "o", ms=10, mfc="none", mec=theme.ink, mew=2, zorder=3)
            for engine, series in synthetic.items():
                colour = theme.vendors[engines[engine][1]]
                ax.plot(float(series[system]), y, "o", ms=8, color=colour, zorder=2)
        ax.set_yticks(range(len(order)), [_display(s) for s in order])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, top * 1.05)
        ax.set_xlabel("WER (%)")
        n = int(first["n"].max())
        ax.set_title(f"{condition} audio ({n} sentences)", loc="left", color=theme.ink, fontsize=10)
    axes[0].plot([], [], "o", ms=9, mfc="none", mec=theme.ink, mew=2, label="real speech")
    for label, key in engines.values():
        axes[0].plot([], [], "o", ms=8, color=theme.vendors[key], label=label)
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.1),
        ncol=3,
        frameon=False,
        labelcolor=theme.secondary,
        borderaxespad=0,
    )
    _heading(
        fig,
        theme,
        "The TTS engine decides whether synthetic audio looks harder or easier",
        f"{SUBSET_TITLES.get(subset, subset)}, FLEURS Hindi; the same sentences in every row",
    )
    return fig


def draw_figures(rq1_dir: Path, gaps: Mapping[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    """Draw every README figure into ``out_dir`` from the committed result tables."""
    paths: list[Path] = []
    scores_path = rq1_dir / "scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        paths += _draw(lambda t: wer_vs_numbers(scores, t), out_dir, "rq1-wer-vs-numbers")
        paths += _draw(lambda t: wer_by_level(scores, t), out_dir, "rq1-wer-by-level")
    if gaps:
        paths += _draw(lambda t: real_vs_synthetic(gaps, t), out_dir, "rq2-real-vs-synthetic")
    return paths
