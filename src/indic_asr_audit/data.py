"""Evaluation manifests: which utterances are scored, and where their audio is.

A manifest is a JSON Lines file with one Utterance per line. Audio paths are
relative to the data directory, so manifests can be committed while the audio
itself is downloaded again by ``prepare_fleurs``.
"""

import csv
import json
import logging
import random
import re
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_SPLITS = ("test", "dev", "train")
FLEURS_SAMPLE_RATE = 16_000
# Whisper decodes 30 s windows and Sarvam's synchronous API rejects longer
# audio, so longer recordings are left out to keep every system comparable.
MAX_SECONDS = 30.0

_HAS_DIGIT = re.compile(r"\d")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Utterance:
    """One scored recording.

    Attributes:
        id: Unique utterance id.
        audio: Audio path relative to the data directory.
        text: Raw reference transcript, with punctuation and digits as written.
        sentence_id: Shared by recordings of the same sentence; the bootstrap
            resamples by this id.
        gender: Speaker gender as given by the dataset, else "".
        split: Source split of the dataset.
        has_number: True if the reference contains a digit.
    """

    id: str
    audio: str
    text: str
    sentence_id: str
    gender: str
    split: str
    has_number: bool


def write_manifest(path: Path, utterances: Iterable[Utterance]) -> None:
    """Write utterances as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for utt in utterances:
            f.write(json.dumps(asdict(utt), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[Utterance]:
    """Read a JSON Lines manifest."""
    with path.open(encoding="utf-8") as f:
        return [Utterance(**json.loads(line)) for line in f if line.strip()]


def _fleurs_rows(raw_dir: Path, lang: str, split: str) -> Iterator[tuple[Utterance, float]]:
    tsv = raw_dir / "data" / lang / f"{split}.tsv"
    with tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentence_id, file_name, raw_text = row[0], row[1], row[2]
            utterance = Utterance(
                id=Path(file_name).stem,
                audio=f"audio/fleurs_{lang}/{file_name}",
                text=raw_text,
                sentence_id=sentence_id,
                gender=row[6].lower() if len(row) > 6 else "",
                split=split,
                has_number=bool(_HAS_DIGIT.search(raw_text)),
            )
            yield utterance, int(row[5]) / FLEURS_SAMPLE_RATE


def select_fleurs(raw_dir: Path, lang: str, n_control: int, seed: int = 0) -> list[Utterance]:
    """Pick every utterance with a number, plus a seeded sample without.

    Numbers are rare in FLEURS, so utterances with numbers come from all
    splits. The split is kept, because some systems (e.g. MMS) were trained
    on FLEURS train; for those, results should be restricted to test and dev.
    Controls come from the test split.
    """
    all_rows = [row for split in FLEURS_SPLITS for row in _fleurs_rows(raw_dir, lang, split)]
    if len({u.id for u, _ in all_rows}) != len(all_rows):
        raise ValueError("FLEURS file names are not unique across splits")
    rows = [u for u, seconds in all_rows if seconds < MAX_SECONDS]
    too_long = [u for u, seconds in all_rows if seconds >= MAX_SECONDS]
    logger.info(
        "Dropped %d recordings of %.0f s or more (%d with numbers)",
        len(too_long),
        MAX_SECONDS,
        sum(u.has_number for u in too_long),
    )
    numeric = [u for u in rows if u.has_number]
    candidates = [u for u in rows if u.split == "test" and not u.has_number]
    control = random.Random(seed).sample(candidates, min(n_control, len(candidates)))
    return sorted(numeric + control, key=lambda u: u.id)


def extract_audio(raw_dir: Path, lang: str, utterances: list[Utterance], data_dir: Path) -> None:
    """Extract only the selected recordings from the FLEURS archives.

    Each file is written under a temporary name and renamed when complete, so
    an interrupted run never leaves a truncated recording behind.
    """
    pending = {Path(u.audio).name: data_dir / u.audio for u in utterances}
    pending = {name: path for name, path in pending.items() if not path.exists()}
    for split in FLEURS_SPLITS:
        if not pending:
            break
        archive = raw_dir / "data" / lang / "audio" / f"{split}.tar.gz"
        with tarfile.open(archive) as tar:
            for member in tar:
                name = Path(member.name).name
                target = pending.get(name)
                source = tar.extractfile(member) if target and member.isfile() else None
                if target is None or source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".part")
                partial.write_bytes(source.read())
                partial.replace(target)
                del pending[name]
                if not pending:
                    break
    missing = list(pending.values())
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings not found, e.g. {missing[0]}")


def download_fleurs(raw_dir: Path, lang: str) -> None:
    """Download FLEURS transcripts and audio archives at a pinned revision."""
    from huggingface_hub import hf_hub_download

    for split in FLEURS_SPLITS:
        for name in (f"data/{lang}/{split}.tsv", f"data/{lang}/audio/{split}.tar.gz"):
            hf_hub_download(
                FLEURS_REPO,
                name,
                repo_type="dataset",
                revision=FLEURS_REVISION,
                local_dir=raw_dir,
            )
