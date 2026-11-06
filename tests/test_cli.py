import json
from pathlib import Path

import pandas as pd
import pytest

from indic_asr_audit import __version__
from indic_asr_audit.cli import main
from indic_asr_audit.data import Utterance, write_manifest


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def _rq2_workspace(root: Path, synthetic: dict[str, list[str]]) -> None:
    """Three systems on real speech; each synthetic set lists the systems it has."""
    (root / "configs").mkdir()
    (root / "configs" / "systems.toml").write_text(
        "".join(f'[{s}]\nbackend = "whisper"\nmodel = "m"\nlanguage = "hi"\n' for s in "abc"),
        encoding="utf-8",
    )
    (root / "configs" / "tts.toml").write_text('[tts]\nengine = "mms"\n', encoding="utf-8")
    texts = ["उसने 53 किताबें खरीदीं", "कुल 800 सैनिक थे", "वह 2 बार घर गया"]
    real = [
        Utterance(f"s{i}t{take}", "", text, f"s{i}", "", "test", True)
        for i, text in enumerate(texts)
        for take in range(2)
    ]
    write_manifest(root / "manifests" / "d.jsonl", real)
    first_takes = [u for u in real if u.id.endswith("t0")]

    def transcripts(dataset: str, system: str, utts: list[Utterance]) -> None:
        path = root / "results" / "transcripts" / dataset / f"{system}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        wrong = {"b": 1, "c": 2}.get(system, 0)  # a is right, b and c get some words wrong
        lines = [
            json.dumps({"id": u.id, "hypothesis": "गलत " * (i < wrong) + u.text})
            for i, u in enumerate(utts)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for system in "abc":
        transcripts("d", system, real)
    for name, systems in synthetic.items():
        write_manifest(root / "manifests" / f"d-{name}.jsonl", first_takes)
        for system in systems:
            transcripts(f"d-{name}", system, first_takes)


def test_rq2_ceiling_uses_only_systems_of_the_synthetic_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rq2_workspace(tmp_path, {"tts-clean": ["a", "b"]})
    monkeypatch.chdir(tmp_path)
    main(["rq2", "d-tts-clean", "--dataset", "d"])
    rq2 = tmp_path / "results" / "rq2"
    assert set(pd.read_csv(rq2 / "ceiling" / "gap.csv")["system"]) == {"a", "b"}
    assert set(pd.read_csv(rq2 / "d-tts-clean" / "selection.csv")["system"]) == {"a", "b"}


def test_rq2_needs_a_system_shared_by_every_synthetic_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rq2_workspace(tmp_path, {"tts-clean": ["a"], "tts-telephony": ["b"]})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="every synthetic set"):
        main(["rq2", "d-tts-clean", "d-tts-telephony", "--dataset", "d"])
