import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from indic_asr_audit.asr import SAMPLE_RATE, Audio
from indic_asr_audit.data import Utterance
from indic_asr_audit.languages import get_language
from indic_asr_audit.synth import (
    CheckLimits,
    check_clip,
    first_voice,
    mu_law,
    speech_seconds,
    synthesize_manifest,
    telephony,
)

HINDI = get_language("hi")


def tone(seconds: float, amplitude: float = 0.3, hz: float = 440) -> Audio:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class FakeEngine:
    """Speaks a tone; the voices listed in ``silent`` produce silence."""

    name = "fake"

    def __init__(self, voices: Sequence[str], silent: Sequence[str] = (), tag: str = "") -> None:
        self.voices = list(voices)
        self.silent = set(silent)
        self.tag = tag
        self.requests: list[tuple[str, str]] = []

    def synthesize(self, text: str, voice: str) -> Audio:
        self.requests.append((text, voice))
        return tone(1.0, amplitude=0.0 if voice in self.silent else 0.3)

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "voices": self.voices, "tag": self.tag}


@pytest.fixture
def corpus(tmp_path: Path) -> list[Utterance]:
    texts = ["उसने 53 किताबें खरीदीं", "उसने 53 किताबें खरीदीं", "वह घर गया", "H5N1 वायरस"]
    sentences = ["0", "0", "1", "2"]
    utterances = []
    for i, (text, sentence) in enumerate(zip(texts, sentences, strict=True)):
        sf.write(tmp_path / f"u{i}.wav", tone(1.2), SAMPLE_RATE)
        utterances.append(Utterance(f"u{i}", f"u{i}.wav", text, sentence, "", "test", True))
    return utterances


def run(engine: FakeEngine, utterances: list[Utterance], data_dir: Path) -> Any:
    return synthesize_manifest(engine, HINDI, utterances, data_dir, "test")


def test_each_sentence_is_spoken_once_with_numbers_as_words(
    tmp_path: Path, corpus: list[Utterance]
) -> None:
    engine = FakeEngine(["a"])
    manifests, _ = run(engine, corpus[:3], tmp_path)
    assert [u.id for u in manifests["clean"]] == ["u0", "u2"]
    assert engine.requests[0][0] == "उसने तिरेपन किताबें खरीदीं"
    assert all((tmp_path / u.audio).exists() for u in manifests["telephony"])


def test_latin_text_is_rejected_without_synthesis(tmp_path: Path, corpus: list[Utterance]) -> None:
    engine = FakeEngine(["a"])
    manifests, records = run(engine, corpus[3:], tmp_path)
    assert engine.requests == []
    assert manifests["clean"] == []
    assert records[0]["problems"] == ["latin text"]


def test_retries_rotate_through_voices(tmp_path: Path, corpus: list[Utterance]) -> None:
    engine = FakeEngine(["a", "b", "c"], silent=["a", "b", "c"])
    _, records = run(engine, corpus[:1], tmp_path)
    first = first_voice("0", 3)
    expected = [engine.voices[(first + k) % 3] for k in range(3)]
    assert [voice for _, voice in engine.requests] == expected
    assert records[0]["status"] == "rejected"
    assert records[0]["attempts"] == 3


def test_rerun_reuses_clips_and_keeps_the_log(tmp_path: Path, corpus: list[Utterance]) -> None:
    run(FakeEngine(["a"]), corpus, tmp_path)
    engine = FakeEngine(["a"])
    _, records = run(engine, corpus, tmp_path)
    assert engine.requests == []
    assert {r["id"]: r["status"] for r in records} == {
        "u0": "saved",
        "u2": "saved",
        "u3": "rejected",
    }
    assert all("spoken" in r for r in records)


def test_changed_settings_use_a_new_folder(tmp_path: Path, corpus: list[Utterance]) -> None:
    first, _ = run(FakeEngine(["a"], tag="v1"), corpus[:1], tmp_path)
    second, _ = run(FakeEngine(["a"], tag="v2"), corpus[:1], tmp_path)
    assert Path(first["clean"][0].audio).parent != Path(second["clean"][0].audio).parent
    settings = Path(tmp_path / first["clean"][0].audio).parents[1] / "settings.json"
    assert json.loads(settings.read_text())["engine"]["tag"] == "v1"


@pytest.mark.parametrize(
    ("audio", "problem"),
    [
        (tone(1.0), None),
        (tone(3.0), "duration ratio"),
        (tone(0.3), "duration ratio"),
        (tone(1.0, amplitude=0.0), "silent"),
        (tone(1.0, amplitude=1.5).clip(-1, 1), "clipped"),
    ],
)
def test_check_clip(audio: Audio, problem: str | None) -> None:
    problems = check_clip(audio, 1.0, CheckLimits())
    if problem is None:
        assert problems == []
    else:
        assert any(problem in p for p in problems)


def test_speech_seconds_ignores_leading_and_trailing_silence() -> None:
    padded = np.concatenate([np.zeros(SAMPLE_RATE), tone(1.0), np.zeros(SAMPLE_RATE)])
    assert speech_seconds(padded.astype(np.float32)) == pytest.approx(1.0, abs=0.04)


def rms(audio: Audio) -> float:
    return float(np.sqrt(np.mean(audio**2)))


def test_telephony_keeps_in_band_speech_and_removes_high_frequencies() -> None:
    in_band, high = tone(1.0, hz=1000), tone(1.0, hz=6000)
    assert rms(telephony(in_band)) == pytest.approx(rms(in_band), rel=0.15)
    assert rms(telephony(high)) < 0.01
    assert len(telephony(in_band)) == len(in_band)


def test_mu_law_has_at_most_255_levels_and_small_error() -> None:
    signal = np.linspace(-1, 1, 10_000, dtype=np.float32)
    decoded = mu_law(signal)
    assert len(np.unique(decoded)) <= 255
    assert float(np.max(np.abs(decoded - signal))) < 0.04


def test_engine_of_picks_the_longest_matching_engine(tmp_path: Path) -> None:
    from indic_asr_audit.synth import engine_of

    config = tmp_path / "tts.toml"
    config.write_text('[mms]\nengine = "mms"\n[mms-tts]\nengine = "mms"\n', encoding="utf-8")
    assert engine_of(config, "fleurs_hi_in-mms-tts-numeric-clean", "fleurs_hi_in") == "mms-tts"
    with pytest.raises(ValueError, match="no engine"):
        engine_of(config, "fleurs_hi_in-other-clean", "fleurs_hi_in")


def test_deleted_clip_is_made_again(tmp_path: Path, corpus: list[Utterance]) -> None:
    manifests, _ = run(FakeEngine(["a"]), corpus[:1], tmp_path)
    (tmp_path / manifests["clean"][0].audio).unlink()
    engine = FakeEngine(["a"])
    run(engine, corpus[:1], tmp_path)
    assert len(engine.requests) == 1
