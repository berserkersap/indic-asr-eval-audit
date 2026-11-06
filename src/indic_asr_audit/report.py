"""Render result tables as Markdown and place them in the README.

Tables are written between marker comments, so numbers in the README always
come from the result files and are never typed by hand.
"""

import itertools
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from indic_asr_audit.normalize import Level

NEWLINE = chr(10)


def _pct(row: Mapping[Hashable, Any]) -> str:
    """A rate and its interval, in percent."""
    value, low, high = (float(str(row[key])) * 100 for key in ("estimate", "low", "high"))
    return f"{value:.1f} [{low:.1f}, {high:.1f}]"


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return NEWLINE.join([header, rule, *rows])


def wer_table(scores: pd.DataFrame, subset: str) -> str:
    """WER per system and level, plus the numeric value error, sorted by final WER."""
    records = [r for r in scores.to_dict("records") if r["subset"] == subset]
    cell = {(str(r["system"]), str(r["metric"]), str(r["level"])): r for r in records}
    systems = sorted({system for system, metric, _ in cell if metric == "wer"})
    systems.sort(key=lambda s: float(str(cell[(s, "wer", Level.NUMBERS)]["estimate"])))
    table = []
    for system in systems:
        row = {"system": f"`{system}`"}
        row |= {f"WER {level}": _pct(cell[(system, "wer", level)]) for level in Level}
        if (system, "numeric_error", "-") in cell:
            row["numeric value error"] = _pct(cell[(system, "numeric_error", "-")])
        table.append(row)
    return _markdown(pd.DataFrame(table))


def steps_table(steps: pd.DataFrame, subset: str) -> str:
    """Paired WER change for each normalisation step, in percentage points."""
    rows = steps[steps["subset"] == subset].copy()
    rows["change"] = [
        f"{e * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
        for e, lo, hi in zip(rows["wer_change"], rows["low"], rows["high"], strict=True)
    ]
    order = [f"{a}->{b}" for a, b in itertools.pairwise(Level)]
    wide = rows.pivot(index="system", columns="step", values="change")[order].reset_index()
    wide["system"] = [f"`{s}`" for s in wide["system"]]
    return _markdown(wide)


def ranks_table(ranks: pd.DataFrame) -> str:
    """Kendall tau and flipped pairs against the stock Whisper normaliser."""
    table = ranks[["subset", "versus", "n_systems", "kendall_tau", "pairs_flipped"]].copy()
    table["kendall_tau"] = [f"{t:.2f}" for t in table["kendall_tau"]]
    return _markdown(table)


def disagreement_table(table: pd.DataFrame) -> str:
    """Pairs of systems that WER and the numeric value error rank oppositely."""
    shown = table[["subset", "n_systems", "kendall_tau", "pairs_reversed", "significant"]].copy()
    shown["kendall_tau"] = [f"{t:.2f}" for t in shown["kendall_tau"]]
    shown["significant"] = [
        str(v).replace(" ", ", ") if str(v) != "nan" else "" for v in shown["significant"]
    ]
    shown.columns = ["subset", "systems", "Kendall tau", "pairs reversed", "significant reversals"]
    return _markdown(shown)


def render_rq1(results_dir: Path, subset: str = "held_out_numeric") -> dict[str, str]:
    """Markdown blocks for the README, keyed by marker name."""
    scores = pd.read_csv(results_dir / "scores.csv")
    steps = pd.read_csv(results_dir / "level_steps.csv")
    ranks = pd.read_csv(results_dir / "ranks.csv")
    disagreement = pd.read_csv(results_dir / "wer_vs_numeric.csv")
    return {
        "rq1-wer": wer_table(scores, subset),
        "rq1-wer-all-numeric": wer_table(scores, "numeric"),
        "rq1-steps": steps_table(steps, subset),
        "rq1-ranks": ranks_table(ranks),
        "rq1-disagreement": disagreement_table(disagreement),
    }


def fill_readme(readme: Path, blocks: dict[str, str]) -> None:
    """Replace the text between ``<!-- name -->`` and ``<!-- /name -->`` markers."""
    text = readme.read_text(encoding="utf-8")
    for name, block in blocks.items():
        start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
        if start not in text or end not in text:
            raise ValueError(f"README has no {start} ... {end} markers")
        head, rest = text.split(start, 1)
        tail = rest.split(end, 1)[1]
        text = f"{head}{start}{NEWLINE}{block}{NEWLINE}{end}{tail}"
    readme.write_text(text, encoding="utf-8", newline=NEWLINE)
