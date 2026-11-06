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
SUBSET_NAMES = {
    "held_out_numeric": "held-out, with numbers",
    "numeric": "all with numbers (no MMS)",
    "control": "without numbers",
}


def _pct(row: Mapping[Hashable, Any]) -> str:
    """A rate and its interval, in percent."""
    value, low, high = (float(str(row[key])) * 100 for key in ("estimate", "low", "high"))
    return f"{value:.1f} [{low:.1f}, {high:.1f}]"


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return NEWLINE.join([header, rule, *rows])


def _pairs(text: object) -> str:
    """'a/b c/d' from a CSV cell, written as '`a` vs `b`, `c` vs `d`'."""
    if not isinstance(text, str) or not text.strip():
        return "none"
    return ", ".join(" vs ".join(f"`{s}`" for s in pair.split("/")) for pair in text.split())


def wer_table(scores: pd.DataFrame, subset: str) -> str:
    """WER per system and level, plus numeric value errors, sorted by WER at the numbers level."""
    records = [r for r in scores.to_dict("records") if r["subset"] == subset]
    cell = {(str(r["system"]), str(r["metric"]), str(r["level"])): r for r in records}
    systems = sorted({system for system, metric, _ in cell if metric == "wer"})
    systems.sort(key=lambda s: float(str(cell[(s, "wer", Level.NUMBERS)]["estimate"])))
    table = []
    for system in systems:
        row = {"system": f"`{system}`"}
        row |= {f"WER {level}": _pct(cell[(system, "wer", level)]) for level in Level}
        for metric, name in (
            ("numeric_error", "numeric value error"),
            ("numeric_error_lenient", "lenient numeric error"),
        ):
            if (system, metric, "-") in cell:
                row[name] = _pct(cell[(system, metric, "-")])
        table.append(row)
    return _markdown(pd.DataFrame(table))


def size_line(diagnostics: pd.DataFrame, subset: str) -> str:
    """One line with the size of a subset: utterances, sentences and numbers."""
    row = diagnostics[diagnostics["subset"] == subset].iloc[0]
    return (
        f"{row['utterances']} utterances, {row['sentences']} sentences, "
        f"{row['reference_words']} reference words, of which {row['reference_numbers']} "
        "are numbers written as digits."
    )


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
    wide.columns = ["system", *(step.replace("->", " → ") for step in order)]
    return _markdown(wide)


def ranks_summary(ranks: pd.DataFrame) -> str:
    """One sentence if no ranking changes between levels, else a table."""
    if (ranks["pairs_flipped"] == 0).all():
        return (
            "No pair of systems changes order between any two normalisation levels, "
            "in any subset (Kendall tau = 1 throughout)."
        )
    table = ranks[["subset", "versus", "n_systems", "kendall_tau", "pairs_flipped"]].copy()
    table["subset"] = table["subset"].map(SUBSET_NAMES)
    table["kendall_tau"] = [f"{t:.2f}" for t in table["kendall_tau"]]
    table.columns = ["subset", "levels", "systems", "Kendall tau", "pairs that flip"]
    return _markdown(table)


def disagreement_table(table: pd.DataFrame) -> str:
    """Pairs of systems that WER and the numeric value error rank oppositely."""
    rows = []
    for record in table.to_dict("records"):
        reversed_pairs = (
            set(str(record["reversed"]).split()) if isinstance(record["reversed"], str) else set()
        )
        significant = (
            set(str(record["significant"]).split())
            if isinstance(record["significant"], str)
            else set()
        )
        rows.append(
            {
                "subset": SUBSET_NAMES.get(str(record["subset"]), str(record["subset"])),
                "systems": record["n_systems"],
                "Kendall tau": f"{float(record['kendall_tau']):.2f}",
                "reversed, significant": _pairs(" ".join(sorted(significant))),
                "reversed, not significant": _pairs(" ".join(sorted(reversed_pairs - significant))),
            }
        )
    return _markdown(pd.DataFrame(rows))


def diagnostics_table(diagnostics: pd.DataFrame, subset: str) -> str:
    """Long outputs and the share of WER that numbers can explain, per system."""
    rows = diagnostics[diagnostics["subset"] == subset].sort_values("wer")
    table = pd.DataFrame(
        {
            "system": [f"`{s}`" for s in rows["system"]],
            "WER": [f"{w * 100:.1f}" for w in rows["wer"]],
            "outputs over 2× reference length": rows["long_outputs"],
            "WER without them": [f"{w * 100:.1f}" for w in rows["wer_without_long"]],
            "missed numbers": [
                f"{m} of {n}"
                for m, n in zip(rows["missed_numbers"], rows["reference_numbers"], strict=True)
            ],
            "most WER points from numbers": [
                f"{v * 100:.1f}" for v in rows["max_wer_from_numbers"]
            ],
        }
    )
    return _markdown(table)


def render_rq1(results_dir: Path, subset: str = "held_out_numeric") -> dict[str, str]:
    """Markdown blocks for the README, keyed by marker name."""
    scores = pd.read_csv(results_dir / "scores.csv")
    diagnostics = pd.read_csv(results_dir / "diagnostics.csv")
    blocks = {
        "rq1-size": size_line(diagnostics, subset),
        "rq1-wer": wer_table(scores, subset),
        "rq1-wer-all-numeric": wer_table(scores, "numeric"),
        "rq1-steps": steps_table(pd.read_csv(results_dir / "level_steps.csv"), subset),
        "rq1-ranks": ranks_summary(pd.read_csv(results_dir / "ranks.csv")),
        "rq1-diagnostics": diagnostics_table(diagnostics, subset),
    }
    disagreement = results_dir / "wer_vs_numeric.csv"
    if disagreement.exists():
        blocks["rq1-disagreement"] = disagreement_table(pd.read_csv(disagreement))
    return blocks


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
