"""Command-line entry point."""

import argparse

from indic_asr_audit import __version__


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="asr-audit", description="Audit tools for Indic ASR evaluation."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    build_parser().parse_args(argv)
