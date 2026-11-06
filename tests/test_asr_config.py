from pathlib import Path

import pytest

from indic_asr_audit.asr import load_systems, read_transcripts

CONFIG = """
[small]
backend = "whisper"
model = "openai/whisper-small"
language = "hi"

[mms]
backend = "ctc"
model = "facebook/mms-1b-all"
language = "hi"
adapter = "hin"
exclude_splits = ["train"]
"""


def test_load_systems_separates_backend_options(tmp_path: Path) -> None:
    path = tmp_path / "systems.toml"
    path.write_text(CONFIG, encoding="utf-8")
    systems = load_systems(path)
    assert systems["small"].options == {}
    assert systems["mms"].options == {"adapter": "hin"}
    assert systems["mms"].exclude_splits == frozenset({"train"})


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "systems.toml"
    path.write_text('[x]\nbackend = "nope"\nmodel = "m"\nlanguage = "hi"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown backend"):
        load_systems(path)["x"].build()


def test_shipped_config_is_valid() -> None:
    systems = load_systems(Path(__file__).parents[1] / "configs" / "systems.toml")
    assert {s.backend for s in systems.values()} <= {"whisper", "ctc", "sarvam"}


def test_read_transcripts_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_transcripts(tmp_path / "none.jsonl") == {}
