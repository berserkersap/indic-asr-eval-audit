"""Command-line entry point."""

import argparse
import logging
from pathlib import Path

from indic_asr_audit import __version__

DEFAULT_DATA_DIR = Path("data")
DEFAULT_SYSTEMS = Path("configs/systems.toml")
DEFAULT_TTS = Path("configs/tts.toml")
RESULTS = Path("results")


def _manifest_path(dataset: str) -> Path:
    return Path("manifests") / f"{dataset}.jsonl"


def _prepare_fleurs(args: argparse.Namespace) -> None:
    from indic_asr_audit.data import download_fleurs, extract_audio, select_fleurs, write_manifest

    raw_dir = args.data_dir / "raw" / "fleurs"
    download_fleurs(raw_dir, args.lang)
    utterances = select_fleurs(raw_dir, args.lang, n_control=args.n_control, seed=args.seed)
    extract_audio(raw_dir, args.lang, utterances, args.data_dir)
    write_manifest(_manifest_path(f"fleurs_{args.lang}"), utterances)
    numeric = sum(u.has_number for u in utterances)
    print(f"{len(utterances)} utterances ({numeric} with numbers)")


def _transcribe(args: argparse.Namespace) -> None:
    from indic_asr_audit.asr import load_systems, run_transcription
    from indic_asr_audit.data import read_manifest

    utterances = read_manifest(_manifest_path(args.dataset))
    if args.numeric_only:
        utterances = [u for u in utterances if u.has_number]
    systems = load_systems(args.systems)
    for name in args.system:
        out = Path("results") / "transcripts" / args.dataset / f"{name}.jsonl"
        run_transcription(
            systems[name].build(), utterances, args.data_dir, out, batch_size=args.batch_size
        )
        print(f"{name}: {out}")


def _rq1(args: argparse.Namespace) -> None:
    from indic_asr_audit.analysis import rq1_tables, write_tables
    from indic_asr_audit.asr import load_systems, read_transcripts
    from indic_asr_audit.data import read_manifest
    from indic_asr_audit.languages import get_language
    from indic_asr_audit.plots import plot_rq1

    utterances = read_manifest(_manifest_path(args.dataset))
    systems = load_systems(args.systems)
    transcripts = {}
    for name in systems:
        hyps = read_transcripts(Path("results") / "transcripts" / args.dataset / f"{name}.jsonl")
        done = sum(u.id in hyps for u in utterances)
        if done == len(utterances):
            transcripts[name] = hyps
        else:
            print(f"skipping {name}: {done}/{len(utterances)} utterances transcribed")
    if not transcripts:
        raise SystemExit("No system has complete transcripts; run `asr-audit transcribe` first.")
    out_dir = Path("results") / "rq1" / args.dataset
    tables = rq1_tables(
        utterances,
        transcripts,
        get_language(args.lang),
        {name: set(systems[name].exclude_splits) for name in transcripts},
    )
    write_tables(tables, out_dir)
    plot_rq1(tables["scores"], out_dir / "wer_by_level.png")
    print(f"RQ1 for {len(transcripts)} systems written to {out_dir}")


def _synthesize(args: argparse.Namespace) -> None:
    import pandas as pd

    from indic_asr_audit.data import read_manifest, write_manifest
    from indic_asr_audit.languages import get_language
    from indic_asr_audit.synth import load_engine, synthesize_manifest

    utterances = read_manifest(_manifest_path(args.dataset))
    suffix = "-numeric" if args.numeric_only else ""
    if args.numeric_only:
        utterances = [u for u in utterances if u.has_number]
    engine = load_engine(args.tts, args.engine)
    manifests, records = synthesize_manifest(
        engine, get_language(args.lang), utterances, args.data_dir, args.dataset
    )
    base = f"{args.dataset}-{args.engine}{suffix}"
    for condition, synthetic in manifests.items():
        write_manifest(_manifest_path(f"{base}-{condition}"), synthetic)
        print(f"{base}-{condition}: {len(synthetic)} utterances")
    if not records:
        raise SystemExit("Nothing to synthesise: the manifest is empty")
    log = pd.DataFrame(records)
    log["reason"] = [p[0].split(" ")[0] if p else "" for p in log["problems"]]
    summary = log.groupby(["has_number", "split", "status", "reason"]).size().rename("sentences")
    out = Path("results") / "synth" / f"{base}-summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(out, index=False, lineterminator="\n")
    print(summary.to_string())


def _complete(dataset: str, names: list[str], ids: list[str]) -> dict[str, dict[str, str]]:
    from indic_asr_audit.asr import read_transcripts

    found = {}
    for name in names:
        hyps = read_transcripts(Path("results") / "transcripts" / dataset / f"{name}.jsonl")
        done = sum(i in hyps for i in ids)
        if done == len(ids):
            found[name] = hyps
        else:
            print(f"skipping {name} on {dataset}: {done}/{len(ids)} utterances transcribed")
    return found


def _rq2(args: argparse.Namespace) -> None:
    from indic_asr_audit.analysis import write_tables
    from indic_asr_audit.asr import load_systems
    from indic_asr_audit.data import read_manifest
    from indic_asr_audit.languages import get_language
    from indic_asr_audit.plots import plot_rq2
    from indic_asr_audit.rq2 import real_ceiling, repeated_sentences, rq2_tables, selection_table
    from indic_asr_audit.synth import engine_of, related_systems

    pack = get_language(args.lang)
    systems = load_systems(args.systems)
    excluded = {s: set(systems[s].exclude_splits) for s in systems}
    real_utts = read_manifest(_manifest_path(args.dataset))
    real_all = _complete(args.dataset, list(systems), [u.id for u in real_utts])
    ceiling_utts = set(repeated_sentences(real_utts))
    synthetic_sets = {}
    for name in args.synthetic:
        synthetic = read_manifest(_manifest_path(name))
        synth = _complete(name, list(systems), [u.id for u in synthetic])
        if not set(real_all) & set(synth):
            raise SystemExit(f"No system has transcripts for both {args.dataset} and {name}")
        synthetic_sets[name] = (synthetic, synth)
    # The ceiling and the selection table use the systems of the synthetic sets, like for like.
    shared = set.intersection(*(set(synth) for _, synth in synthetic_sets.values()))
    shared_hyps = {s: h for s, h in real_all.items() if s in shared}
    write_tables(real_ceiling(real_utts, shared_hyps, pack, excluded), RESULTS / "rq2" / "ceiling")
    for name, (synthetic, synth) in synthetic_sets.items():
        # synthetic ids are real utterance ids, so real_all has every real transcript needed
        related = related_systems(args.tts, engine_of(args.tts, name, args.dataset))
        tables = rq2_tables(synthetic, real_all, synth, pack, excluded, related)
        same = [u for u in synthetic if u.sentence_id in ceiling_utts]
        tables |= {
            f"same_sentences_as_ceiling_{k}": v
            for k, v in rq2_tables(same, real_all, synth, pack, excluded, related).items()
        }
        kept = {u.id for u in synthetic}
        tables["selection"] = selection_table(real_utts, kept, shared_hyps, pack, excluded)
        out_dir = RESULTS / "rq2" / name
        write_tables(tables, out_dir)
        if not tables["gap"].empty:
            plot_rq2(tables["gap"], out_dir / "real_vs_synthetic.png", title=name)
        print(f"RQ2 for {name} written to {out_dir}")


def _token_rate(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import soundfile as sf
    from transformers import WhisperTokenizer

    from indic_asr_audit.asr import load_systems
    from indic_asr_audit.data import read_manifest

    system = load_systems(args.systems)[args.system]
    tokenizer = WhisperTokenizer.from_pretrained(system.model, revision=system.options["revision"])
    rates = np.array(
        [
            len(tokenizer(u.text, add_special_tokens=False).input_ids)
            / sf.info(args.data_dir / u.audio).duration
            for u in read_manifest(_manifest_path(args.dataset))
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "tokenizer": system.model,
                "utterances": len(rates),
                "median": np.median(rates),
                "p99": np.percentile(rates, 99),
                "max": rates.max(),
            }
        ]
    )
    out = RESULTS / "rq1" / args.dataset / "token_rate.csv"
    summary.round(2).to_csv(out, index=False, lineterminator="\n")
    print(summary.round(2).to_string(index=False))


def _report(args: argparse.Namespace) -> None:
    from indic_asr_audit.report import fill_readme, render_rq1, render_rq2

    results = Path("results")
    blocks = render_rq1(results / "rq1" / args.dataset) | render_rq2(results, args.dataset)
    fill_readme(args.readme, blocks)
    print(f"Updated {args.readme}")


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="asr-audit", description="Audit tools for Indic ASR evaluation."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-fleurs", help="download FLEURS and write a manifest")
    prepare.add_argument("--lang", default="hi_in", help="FLEURS language config")
    prepare.add_argument("--n-control", type=int, default=200, help="utterances without numbers")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.set_defaults(run=_prepare_fleurs)

    transcribe = commands.add_parser("transcribe", help="run ASR systems over a manifest")
    transcribe.add_argument("system", nargs="+", help="names from the systems file")
    transcribe.add_argument("--dataset", default="fleurs_hi_in")
    transcribe.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    transcribe.add_argument("--numeric-only", action="store_true", help="skip control utterances")
    transcribe.add_argument("--batch-size", type=int, default=32)
    transcribe.set_defaults(run=_transcribe)

    synth = commands.add_parser("synthesize", help="speak a manifest's sentences with TTS")
    synth.add_argument("engine", help="name from the TTS file")
    synth.add_argument("--dataset", default="fleurs_hi_in")
    synth.add_argument("--lang", default="hi", help="language pack code")
    synth.add_argument("--tts", type=Path, default=DEFAULT_TTS)
    synth.add_argument("--numeric-only", action="store_true", help="skip control utterances")
    synth.set_defaults(run=_synthesize)

    rq1 = commands.add_parser("rq1", help="score every system at every normalisation level")
    rq1.add_argument("--dataset", default="fleurs_hi_in")
    rq1.add_argument("--lang", default="hi", help="language pack code")
    rq1.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    rq1.set_defaults(run=_rq1)
    rq2 = commands.add_parser("rq2", help="compare synthetic test sets with real speech")
    rq2.add_argument("synthetic", nargs="+", help="synthetic manifest names")
    rq2.add_argument("--dataset", default="fleurs_hi_in", help="the real dataset")
    rq2.add_argument("--lang", default="hi", help="language pack code")
    rq2.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    rq2.add_argument("--tts", type=Path, default=DEFAULT_TTS)
    rq2.set_defaults(run=_rq2)
    rate = commands.add_parser(
        "token-rate", help="Whisper tokens per second of speech in the references"
    )
    rate.add_argument("--system", default="whisper-small", help="system whose tokenizer to use")
    rate.add_argument("--dataset", default="fleurs_hi_in")
    rate.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    rate.set_defaults(run=_token_rate)

    report = commands.add_parser("report", help="fill the README tables from the results")
    report.add_argument("--dataset", default="fleurs_hi_in")
    report.add_argument("--readme", type=Path, default=Path("README.md"))
    report.set_defaults(run=_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.run(args)
