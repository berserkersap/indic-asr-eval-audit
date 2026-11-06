from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from indic_asr_audit.asr import SAMPLE_RATE, Audio
from indic_asr_audit.data import Utterance
from indic_asr_audit.languages import get_language
from indic_asr_audit.synth import CheckLimits, check_clip, synthesize_manifest, telephony

HINDI = get_language("hi")


def tone(seconds: float, amplitude: float = 0.3) -> Audio:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class FakeEngine:
    """Speaks a tone; the voices listed in ``silent`` produce silence."""

    name = "fake"

    def __init__(self, voices: Sequence[str], silent: Sequence[str] = ()) -> None:
        self.voices = list(voices)
        self.silent = set(silent)
        self.requests: list[tuple[str, str]] = []

    def synthesize(self, text: str, voice: str) -> Audio:
        self.requests.append((text, voice))
        return tone(1.0, amplitude=0.0 if voice in self.silent else 0.3)

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name}


@pytest.fixture
def corpus(tmp_path: Path) -> list[Utterance]:
    utterances = []
    for i, sentence in enumerate(["0", "0", "1"]):
        sf.write(tmp_path / f"u{i}.wav", tone(1.2), SAMPLE_RATE)
        text = "उसने 53 किताबें खरीदीं" if sentence == "0" else "वह घर गया"
        utterances.append(Utterance(f"u{i}", f"u{i}.wav", text, sentence, "male", "test", True))
    return utterances


def test_every_sentence_is_spoken_once_with_numbers_as_words(
    tmp_path: Path, corpus: list[Utterance]
) -> None:
    engine = FakeEngine(["a"])
    manifests = synthesize_manifest(engine, HINDI, corpus, tmp_path, "test")
    assert [u.id for u in manifests["clean"]] == ["u0", "u2"]
    assert engine.requests[0][0] == "उसने तिरेपन किताबें खरीदीं"
    assert all((tmp_path / u.audio).exists() for u in manifests["telephony"])


def test_failed_check_retries_with_the_next_voice(tmp_path: Path, corpus: list[Utterance]) -> None:
    engine = FakeEngine(["a", "b"], silent=["a"])
    manifests = synthesize_manifest(engine, HINDI, corpus[:1], tmp_path, "test")
    assert [voice for _, voice in engine.requests] == ["a", "b"]
    assert len(manifests["clean"]) == 1


def test_clip_is_rejected_when_every_voice_fails(tmp_path: Path, corpus: list[Utterance]) -> None:
    engine = FakeEngine(["a"], silent=["a"])
    manifests = synthesize_manifest(engine, HINDI, corpus[:1], tmp_path, "test")
    assert len(engine.requests) == 3
    assert manifests["clean"] == []


def test_saved_clips_are_reused(tmp_path: Path, corpus: list[Utterance]) -> None:
    synthesize_manifest(FakeEngine(["a"]), HINDI, corpus, tmp_path, "test")
    engine = FakeEngine(["a"])
    synthesize_manifest(engine, HINDI, corpus, tmp_path, "test")
    assert engine.requests == []


@pytest.mark.parametrize(
    ("audio", "real_seconds", "problem"),
    [
        (tone(1.0), 1.0, None),
        (tone(3.0), 1.0, "duration ratio"),
        (tone(0.3), 1.0, "duration ratio"),
        (tone(1.0, amplitude=0.0), 1.0, "silent"),
        (tone(1.0, amplitude=1.5).clip(-1, 1), 1.0, "clipped"),
    ],
)
def test_check_clip(audio: Audio, real_seconds: float, problem: str | None) -> None:
    problems = check_clip(audio, real_seconds, CheckLimits())
    assert (problem is None and not problems) or any(problem in p for p in problems if problem)


def test_telephony_removes_high_frequencies() -> None:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    high = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    out = telephony(high)
    assert len(out) == len(high)
    assert np.sqrt(np.mean(out**2)) < 0.01
