"""Render result tables as Markdown and place them in the README.

Tables are written between marker comments, so numbers in the README always
come from the result files and are never typed by hand.
"""

import itertools
from collections.abc import Hashable, Mapping, Sequence
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
    if table.empty:
        return "No results."
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
    rate = results_dir / "token_rate.csv"
    if rate.exists():
        r = pd.read_csv(rate).iloc[0]
        blocks["rq1-token-rate"] = (
            f"The references need at most {r['max']:.0f} tokens per second "
            f"(99th percentile {r['p99']:.1f}) with the {r['tokenizer']} tokenizer "
            "(`asr-audit token-rate`), so the limit only cuts off runaway repetition."
        )
    disagreement = results_dir / "wer_vs_numeric.csv"
    if disagreement.exists():
        blocks["rq1-disagreement"] = disagreement_table(pd.read_csv(disagreement))
    return blocks


def _read(path: Path) -> pd.DataFrame:
    """A result table; write_tables leaves no file for an empty one."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _points(estimate: float, low: float, high: float) -> str:
    """A difference of rates and its interval, in percentage points."""
    return f"{estimate * 100:+.1f} [{low * 100:+.1f}, {high * 100:+.1f}]"


def _synthetic_sets(rq2_dir: Path, dataset: str) -> dict[str, Path]:
    """RQ2 result folders for one dataset, keyed by test set ("mms-tts-clean")."""
    prefix = f"{dataset}-"
    return {
        path.name.removeprefix(prefix): path
        for path in sorted(rq2_dir.glob(f"{prefix}*"))
        if path.is_dir()
    }


def _engine_of_set(name: str) -> str:
    """The TTS engine of a test set: "mms-tts" for "mms-tts-clean".

    Conditions ("clean", "telephony") have no hyphen, so no TTS config is needed.
    """
    return name.rpartition("-")[0]


def rejection_line(summary: pd.DataFrame, engine: str) -> str:
    """How many sentences one engine kept, and why it left out the rest."""
    saved = int(summary.loc[summary["status"] == "saved", "sentences"].sum())
    rejected = summary[summary["status"] == "rejected"].groupby("reason")["sentences"].sum()
    reasons = {"latin": "with Latin letters", "duration": "failed the duration check"}
    left_out = ", ".join(f"{n} {reasons.get(str(r), str(r))}" for r, n in rejected.items())
    total = saved + int(rejected.sum())
    return f"`{engine}`: {saved} of {total} sentences kept" + (
        f"; left out: {left_out}." if left_out else "."
    )


def gap_table(gaps: Mapping[str, pd.DataFrame], subset: str) -> str:
    """WER and numeric value error on real and synthetic audio, per test set and system."""
    rows = []
    for name, gap in gaps.items():
        if gap.empty:
            continue
        for r in gap[gap["subset"] == subset].sort_values("wer_real").to_dict("records"):
            rows.append(
                {
                    "test set": f"`{name}`",
                    "system": f"`{r['system']}`",
                    "n": r["n"],
                    "WER real": f"{r['wer_real'] * 100:.1f}",
                    "WER synthetic": f"{r['wer_synthetic'] * 100:.1f}",
                    "change (points)": _points(r["wer_gap"], r["low"], r["high"]),
                    "numeric error real": f"{r['numeric_error_real'] * 100:.1f}",
                    "numeric error synthetic": f"{r['numeric_error_synthetic'] * 100:.1f}",
                }
            )
    return _markdown(pd.DataFrame(rows))


def _agreement_rows(
    label: str, agreement: pd.DataFrame, gap: pd.DataFrame, subset: str
) -> list[dict[str, object]]:
    if agreement.empty or gap.empty or subset not in set(gap["subset"]):
        return []
    n = int(gap.loc[gap["subset"] == subset, "n"].max())
    rows: list[dict[str, object]] = []
    for r in agreement[agreement["subset"] == subset].to_dict("records"):
        systems = str(r["n_systems"])
        if r["systems"] == "unrelated":
            systems += ", none from the TTS family"
        rows.append(
            {
                "subset": SUBSET_NAMES.get(subset, subset),
                "compared with real speech": label,
                "sentences": n,
                "systems": systems,
                "Kendall tau": f"{r['kendall_tau']:.2f} [{r['low']:.2f}, {r['high']:.2f}]",
                "pairs flipped": _pairs(r["flipped"]),
                "significant flips": r["significant_flips"],
            }
        )
    return rows


def agreement_table(rq2_dir: Path, sets: Mapping[str, Path], subsets: Sequence[str]) -> str:
    """Ranking agreement: a second real recording (the ceiling) and each synthetic set."""
    ceiling = rq2_dir / "ceiling"
    rows = []
    for subset in subsets:
        rows += _agreement_rows(
            "second real recording",
            _read(ceiling / "agreement.csv"),
            _read(ceiling / "gap.csv"),
            subset,
        )
        for name, path in sets.items():
            for prefix, scope in (("same_sentences_as_ceiling_", "same sentences"), ("", "all")):
                rows += _agreement_rows(
                    f"`{name}`, {scope}",
                    _read(path / f"{prefix}agreement.csv"),
                    _read(path / f"{prefix}gap.csv"),
                    subset,
                )
    return _markdown(pd.DataFrame(rows))


def rho_table(rq2_dir: Path, sets: Mapping[str, Path], subsets: Sequence[str]) -> str:
    """Median and range over systems of the per-utterance error correlation with real speech."""
    sources = {"second real recording": rq2_dir / "ceiling" / "utterances.csv"} | {
        f"`{name}`": path / "same_sentences_as_ceiling_utterances.csv"
        for name, path in sets.items()
    }
    rows = []
    for label, source in sources.items():
        table = _read(source)
        row: dict[str, str] = {"compared with real speech": label}
        for subset in subsets:
            rho = (
                table.loc[table["subset"] == subset, "spearman_rho"].dropna()
                if not table.empty
                else pd.Series(dtype=float)
            )
            row[SUBSET_NAMES.get(subset, subset)] = (
                f"{rho.median():.2f} ({rho.min():.2f} to {rho.max():.2f})" if len(rho) else "-"
            )
        rows.append(row)
    return _markdown(pd.DataFrame(rows))


def selection_markdown(selections: Mapping[str, pd.DataFrame]) -> str:
    """Real-speech WER on the sentences each engine kept and left out."""
    rows = []
    for engine, table in selections.items():
        if table.empty:
            continue
        for system, group in table.groupby("system", sort=False):
            cells = {str(r["sentences"]): r for r in group.to_dict("records")}
            row = {"engine": f"`{engine}`", "system": f"`{system}`"}
            for label, name in (("kept", "kept"), ("rejected", "left out")):
                r = cells.get(label)
                row[f"WER on {name} (sentences)"] = (
                    f"{r['wer'] * 100:.1f} ({r['n']})" if r is not None else "-"
                )
            rows.append(row)
    return _markdown(pd.DataFrame(rows))


def render_rq2(
    results: Path,
    dataset: str,
    subset: str = "held_out_numeric",
    agreement_subsets: Sequence[str] = ("held_out_numeric", "numeric"),
) -> dict[str, str]:
    """Markdown blocks for the README's RQ2 section; empty if RQ2 has not run."""
    rq2_dir = results / "rq2"
    sets = _synthetic_sets(rq2_dir, dataset)
    if not sets:
        return {}
    engines = list(dict.fromkeys(_engine_of_set(name) for name in sets))
    first_set = {e: next(p for n, p in sets.items() if _engine_of_set(n) == e) for e in engines}
    sizes = []
    for engine in engines:
        summary = results / "synth" / f"{dataset}-{engine}-summary.csv"
        if summary.exists():
            sizes.append(rejection_line(pd.read_csv(summary, keep_default_na=False), engine))
    return {
        "rq2-size": (NEWLINE * 2).join(sizes),
        "rq2-gap": gap_table({n: _read(p / "gap.csv") for n, p in sets.items()}, subset),
        "rq2-agreement": agreement_table(rq2_dir, sets, agreement_subsets),
        "rq2-rho": rho_table(rq2_dir, sets, agreement_subsets),
        "rq2-selection": selection_markdown(
            {e: _read(p / "selection.csv") for e, p in first_set.items()}
        ),
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
