from pathlib import Path

import pytest

from indic_asr_audit.data import Utterance, read_manifest, select_fleurs, write_manifest

# id, file name, raw text, normalised text, characters, samples, gender
ROWS = {
    "test": [
        ("1", "a.wav", "53 साल", "53 साल", "-", "16000", "MALE"),
        ("2", "b.wav", "घर गया", "घर गया", "-", "16000", "FEMALE"),
        ("3", "c.wav", "बहुत लंबा 1", "-", "-", str(16000 * 31), "MALE"),
    ],
    "dev": [("4", "d.wav", "वह आया", "-", "-", "16000", "MALE")],
    "train": [("5", "e.wav", "15वीं सदी", "-", "-", "16000", "FEMALE")],
}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "data" / "hi_in"
    folder.mkdir(parents=True)
    for split, rows in ROWS.items():
        lines = ["\t".join(row) for row in rows]
        (folder / f"{split}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_select_keeps_numbers_from_all_splits_and_controls_from_test(raw_dir: Path) -> None:
    selected = select_fleurs(raw_dir, "hi_in", n_control=5)
    assert {u.id for u in selected} == {"a", "b", "e"}
    assert {u.id: u.has_number for u in selected} == {"a": True, "b": False, "e": True}
    assert next(u for u in selected if u.id == "e").split == "train"


def test_recordings_of_30_seconds_or_more_are_dropped(raw_dir: Path) -> None:
    assert "c" not in {u.id for u in select_fleurs(raw_dir, "hi_in", n_control=5)}


def test_manifest_round_trip(tmp_path: Path) -> None:
    utterance = Utterance("x", "audio/x.wav", "53 साल", "s1", "male", "test", True)
    path = tmp_path / "m.jsonl"
    write_manifest(path, [utterance])
    assert read_manifest(path) == [utterance]
