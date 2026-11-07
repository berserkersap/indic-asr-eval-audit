"""Figures, drawn only from the result tables so they can be redrawn on CPU.

Each figure is written twice, as ``<name>-light.svg`` and ``<name>-dark.svg``,
for GitHub's light and dark themes. Colour follows the vendor (OpenAI, Meta,
Sarvam), for ASR systems and TTS engines alike, so a vendor keeps its colour
across figures, and every series is also named on the chart, so colour is
never the only cue. Text is drawn as paths in DejaVu Sans, which ships with
matplotlib, so a figure looks and diffs the same whatever fonts the drawing or
viewing machine has. Figures are built with ``Figure`` rather than pyplot, so
no GUI backend is needed.
"""

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
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
    grid: str
    axis: str
    vendors: Mapping[str, str]


# Validated categorical slots 1-3 (blue, orange, green), stepped per theme.
THEMES = (
    Theme(
        "light",
        ink="#0b0b0b",
        secondary="#52514e",
        grid="#e1e0d9",
        axis="#c3c2b7",
        vendors={"openai": "#2a78d6", "meta": "#eb6834", "sarvam": "#1baf7a"},
    ),
    Theme(
        "dark",
        ink="#f0efec",
        secondary="#c3c2b7",
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
    "numeric": "All utterances with numbers",
}
SUBSET_NOTES = {"numeric": ", no MMS-1B"}  # MMS-1B was trained on FLEURS train
VENDOR_PREFIXES = {"whisper": "openai", "mms": "meta", "saaras": "sarvam"}
CONDITION_ORDER = ("clean", "telephony")
ENGINES = {"mms-tts": ("MMS-TTS (Meta)", "meta"), "bulbul": ("Bulbul (Sarvam)", "sarvam")}
RC: dict[RcKeyType, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "svg.fonttype": "path",
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
    """The vendor whose colour a system takes; an unknown system is an error, not a guess."""
    for prefix, name in VENDOR_PREFIXES.items():
        if system.startswith(prefix):
            return name
    raise ValueError(f"no vendor colour for {system}; add it to VENDOR_PREFIXES")


def _display(system: str) -> str:
    return DISPLAY_NAMES.get(system, system)


def _subset_title(subset: str, n: int) -> str:
    """For example "All utterances with numbers (523 utterances, no MMS-1B)"."""
    note = SUBSET_NOTES.get(subset, "")
    return f"{SUBSET_TITLES.get(subset, subset)} ({n} utterances{note})"


def _spread(values: Sequence[float], gap: float, low: float = 0.0) -> list[float]:
    """Label positions at least ``gap`` apart, as close to ``values`` as possible.

    A label that has room stays on its value. Labels that would collide form a
    group, spaced ``gap`` apart and centred on the mean of their values (but
    not below ``low``); groups merge until none overlap.
    """

    def start(group: list[int]) -> float:
        return max(low, fmean(values[i] for i in group) - gap * (len(group) - 1) / 2)

    groups: list[list[int]] = []
    for i in sorted(range(len(values)), key=lambda i: values[i]):
        groups.append([i])
        while len(groups) > 1:
            below, above = groups[-2], groups[-1]
            if start(above) - (start(below) + gap * (len(below) - 1)) >= gap:
                break
            groups[-2:] = [below + above]
    out = [0.0] * len(values)
    for group in groups:
        for k, i in enumerate(group):
            out[i] = start(group) + k * gap
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
    """Write an SVG with LF line ends and no trailing spaces, the same on every OS."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", transparent=True, metadata={"Date": None, "Creator": None})
    lines = (line.rstrip() for line in buffer.getvalue().splitlines())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _draw(draw: Callable[[Theme], Figure | None], out_dir: Path, name: str) -> list[Path]:
    """Write one figure per theme; with no data, remove any stale copies instead."""
    paths = []
    with mpl.rc_context(RC):
        for theme in THEMES:
            fig = draw(theme)
            if fig is None:
                for stale in THEMES:
                    (out_dir / f"{name}-{stale.name}.svg").unlink(missing_ok=True)
                return []
            path = out_dir / f"{name}-{theme.name}.svg"
            _save(fig, path)
            paths.append(path)
    return paths


def wer_vs_numbers(
    scores: pd.DataFrame, theme: Theme, subset: str = "held_out_numeric"
) -> Figure | None:
    """WER (numbers level) against the share of numbers missed, one point per system."""
    rows = scores[scores["subset"] == subset]
    wer = rows[(rows["metric"] == "wer") & (rows["level"] == Level.NUMBERS)].set_index("system")
    missed = rows[rows["metric"] == "numeric_error"].set_index("system")
    systems = [s for s in wer.index if s in missed.index]
    if not systems:
        return None
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
        ax.plot(x["estimate"], y["estimate"], "o", ms=8, color=colour, mec="none", label=system)
    labels = _spread([float(missed.loc[s, "estimate"]) * 100 for s in systems], gap=4.5)
    for system, label_y in zip(systems, labels, strict=True):
        x_value = float(wer.loc[system, "estimate"]) * 100
        ax.annotate(
            _display(system),
            (x_value, label_y),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=theme.ink,
        )
    handles = [
        ax.plot([], [], "o", ms=7, color=theme.vendors[key], label=label)[0]
        for key, label in VENDOR_NAMES.items()
        if any(vendor(s) == key for s in systems)
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, labelcolor=theme.secondary)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("WER after number normalisation (%)")
    ax.set_ylabel("numeric value error (% of numbers missed)")
    _heading(
        fig,
        theme,
        "Among open systems, lowest WER is not fewest missed numbers",
        f"{_subset_title(subset, int(wer['n'].max()))}, FLEURS Hindi; bars: 95% intervals",
    )
    return fig


def wer_by_level(scores: pd.DataFrame, theme: Theme) -> Figure | None:
    """WER of each system at each normalisation level, one panel per numeric subset."""
    subsets = ["held_out_numeric", "numeric"]
    levels = list(Level)
    all_wer = scores[(scores["metric"] == "wer") & scores["subset"].isin(subsets)]
    if all_wer.empty:
        return None
    top = float(all_wer["estimate"].max()) * 100
    fig = Figure(figsize=(11, 4.8))
    axes = fig.subplots(1, len(subsets), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.14, wspace=0.12)
    for ax, subset in zip(axes, subsets, strict=True):
        _axes_style(ax, theme)
        wer = all_wer[all_wer["subset"] == subset]
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
        ax.set_xlim(-0.25, len(levels) - 1 + 1.75)
        ax.set_ylim(0, top * 1.05)
        n = int(wer["n"].max()) if not wer.empty else 0
        ax.set_title(_subset_title(subset, n), loc="left", color=theme.ink, fontsize=10)
    axes[0].set_ylabel("WER (%)")
    _heading(
        fig,
        theme,
        "The stock Whisper normaliser makes the open systems' Hindi WER look lower",
        "FLEURS Hindi. The Whisper normaliser deletes vowel signs; basic, + spelling and "
        "+ numbers each add one step; intervals are in the tables",
    )
    return fig


def real_vs_synthetic(
    gaps: Mapping[str, pd.DataFrame], theme: Theme, subset: str = "held_out_numeric"
) -> Figure | None:
    """WER on real speech and on each engine's audio, one row per system, one panel per condition.

    ``gaps`` maps a test set ("mms-tts-clean") to its gap table. Only systems
    present in every table are drawn, and their real-speech WER must be the same
    in every table: every dot in a row scores the same sentences. Each synthetic
    dot carries the 95% interval of its change from real speech.
    """
    tables = {
        name: table[table["subset"] == subset].set_index("system")
        for name, table in gaps.items()
        if name.rpartition("-")[0] in ENGINES and "subset" in table
    }
    tables = {name: table for name, table in tables.items() if not table.empty}
    if not tables:
        return None
    systems = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    if not systems:
        return None
    real = pd.DataFrame({name: t.loc[systems, "wer_real"] for name, t in tables.items()}) * 100
    if (real.max(axis=1) - real.min(axis=1)).max() > 1e-6:
        raise ValueError("real-speech WER differs between test sets; they cover other sentences")
    order = list(real.iloc[:, 0].sort_values(ascending=False).index)
    # The axis must hold every interval end, which can pass 100% WER.
    top = max(
        max(t.loc[systems, "wer_synthetic"].max(), (t["wer_real"] + t["high"]).loc[systems].max())
        for t in tables.values()
    )
    found = {name.rpartition("-")[2] for name in tables}
    conditions = [c for c in CONDITION_ORDER if c in found] + sorted(found - set(CONDITION_ORDER))
    fig = Figure(figsize=(11, 4.2))
    axes = fig.subplots(1, len(conditions), sharey=True, squeeze=False)[0]
    fig.subplots_adjust(left=0.16, right=0.99, top=0.76, bottom=0.14, wspace=0.08)
    real_marker = {"ms": 10, "mfc": "none", "mec": theme.ink, "mew": 2}
    for ax, condition in zip(axes, conditions, strict=True):
        _axes_style(ax, theme)
        # In legend order, so MMS-TTS is always the upper dot and Bulbul the lower.
        synthetic = {
            engine: tables[f"{engine}-{condition}"]
            for engine in ENGINES
            if f"{engine}-{condition}" in tables
        }
        for y, system in enumerate(order):
            real_wer = float(real.loc[system].iloc[0])
            ax.plot(real_wer, y, "o", zorder=3, label="real", **real_marker)
            # Engines sit just above and below the row so their intervals do not overlap.
            for k, (engine, table) in enumerate(synthetic.items()):
                row = table.loc[system, ["wer_synthetic", "low", "high"]] * 100
                # Two engines at most: the first just above the row, the second below.
                dy = 0.16 * (1 - 2 * k) if len(synthetic) > 1 else 0.0
                colour = theme.vendors[ENGINES[engine][1]]
                interval = [real_wer + float(row["low"]), real_wer + float(row["high"])]
                ax.plot(
                    interval,
                    [y + dy] * 2,
                    color=colour,
                    lw=1.5,
                    alpha=0.45,
                    zorder=1,
                    label="interval",
                )
                ax.plot(float(row["wer_synthetic"]), y + dy, "o", ms=8, color=colour, label=engine)
        ax.set_yticks(range(len(order)), [_display(s) for s in order])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, top * 100 * 1.05)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("WER (%)")
        n = max(int(t["n"].max()) for name, t in tables.items() if name.endswith(f"-{condition}"))
        ax.set_title(f"{condition} audio ({n} sentences)", loc="left", color=theme.ink, fontsize=10)
    handles = [axes[0].plot([], [], "o", label="real speech", **real_marker)[0]]
    handles += [
        axes[0].plot([], [], "o", ms=8, color=theme.vendors[key], label=label)[0]
        for label, key in ENGINES.values()
    ]
    axes[0].legend(
        handles=handles,
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
        "The TTS engine sets whether synthetic audio looks harder or easier",
        f"{SUBSET_TITLES.get(subset, subset)}, FLEURS Hindi; bars: 95% interval of the change "
        "from real speech; colour: TTS vendor",
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
    paths += _draw(lambda t: real_vs_synthetic(gaps, t), out_dir, "rq2-real-vs-synthetic")
    return paths
