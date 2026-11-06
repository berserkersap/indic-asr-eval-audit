"""Command-line entry point."""

import argparse
import logging
from pathlib import Path

from indic_asr_audit import __version__

DEFAULT_DATA_DIR = Path("data")
DEFAULT_SYSTEMS = Path("configs/systems.toml")


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

    rq1 = commands.add_parser("rq1", help="score every system at every normalisation level")
    rq1.add_argument("--dataset", default="fleurs_hi_in")
    rq1.add_argument("--lang", default="hi", help="language pack code")
    rq1.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    rq1.set_defaults(run=_rq1)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.run(args)
