import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf

from indic_asr_audit.asr import (
    Audio,
    SarvamTranscriber,
    call_sarvam,
    read_transcripts,
    run_transcription,
    token_limit,
)
from indic_asr_audit.data import Utterance


class FakeTranscriber:
    """Returns the audio length in samples, so outputs are easy to check."""

    def __init__(self, setting: str = "a") -> None:
        self.setting = setting
        self.calls = 0

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        self.calls += 1
        return [str(len(audio)) for audio in batch]

    def describe(self) -> dict[str, Any]:
        return {"backend": "fake", "setting": self.setting}


@pytest.fixture
def corpus(tmp_path: Path) -> list[Utterance]:
    utterances = []
    for i, seconds in enumerate([0.3, 0.1, 0.2]):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, np.zeros(int(16_000 * seconds), dtype=np.float32), 16_000)
        utterances.append(Utterance(f"u{i}", path.name, "x", f"s{i}", "", "test", False))
    return utterances


def test_run_writes_every_utterance_and_its_settings(
    tmp_path: Path, corpus: list[Utterance]
) -> None:
    out = tmp_path / "out" / "fake.jsonl"
    run_transcription(FakeTranscriber(), corpus, tmp_path, out, batch_size=2)
    assert read_transcripts(out) == {"u0": "4800", "u1": "1600", "u2": "3200"}
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["config"] == {"backend": "fake", "setting": "a"}
    assert meta["transcribed"] == 3


def test_resume_only_transcribes_missing_utterances(
    tmp_path: Path, corpus: list[Utterance]
) -> None:
    out = tmp_path / "fake.jsonl"
    run_transcription(FakeTranscriber(), corpus[:1], tmp_path, out)
    fake = FakeTranscriber()
    run_transcription(fake, corpus, tmp_path, out, batch_size=10)
    assert fake.calls == 1
    assert len(read_transcripts(out)) == 3


def test_complete_run_does_not_rewrite_meta(tmp_path: Path, corpus: list[Utterance]) -> None:
    out = tmp_path / "fake.jsonl"
    run_transcription(FakeTranscriber(), corpus, tmp_path, out)
    before = out.with_suffix(".meta.json").read_text(encoding="utf-8")
    run_transcription(FakeTranscriber(), corpus, tmp_path, out)
    assert out.with_suffix(".meta.json").read_text(encoding="utf-8") == before


def test_resume_with_other_settings_is_refused(tmp_path: Path, corpus: list[Utterance]) -> None:
    out = tmp_path / "fake.jsonl"
    run_transcription(FakeTranscriber("a"), corpus[:1], tmp_path, out)
    with pytest.raises(ValueError, match="other settings"):
        run_transcription(FakeTranscriber("b"), corpus, tmp_path, out)


def test_reading_ignores_a_partial_last_line_without_changing_the_file(tmp_path: Path) -> None:
    out = tmp_path / "fake.jsonl"
    content = b'{"id": "u0", "hypothesis": "ok"}' + bytes([10]) + b'{"id": "u1", "hypo'
    out.write_bytes(content)
    assert read_transcripts(out) == {"u0": "ok"}
    assert out.read_bytes() == content


def test_runner_repairs_a_partial_last_line(tmp_path: Path, corpus: list[Utterance]) -> None:
    out = tmp_path / "fake.jsonl"
    out.write_bytes(b'{"id": "u0", "hypothesis": "4800"}' + bytes([10]) + b'{"id": "u1", "hy')
    run_transcription(FakeTranscriber(), corpus, tmp_path, out)
    assert read_transcripts(out) == {"u0": "4800", "u1": "1600", "u2": "3200"}


def test_meta_from_an_older_runner_is_refused(tmp_path: Path, corpus: list[Utterance]) -> None:
    out = tmp_path / "fake.jsonl"
    out.with_suffix(".meta.json").write_text('{"model": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="predates"):
        run_transcription(FakeTranscriber(), corpus, tmp_path, out)


def test_meta_counts_add_up_over_resumed_runs(tmp_path: Path, corpus: list[Utterance]) -> None:
    out = tmp_path / "fake.jsonl"
    run_transcription(FakeTranscriber(), corpus[:1], tmp_path, out)
    first = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    run_transcription(FakeTranscriber(), corpus, tmp_path, out)
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["transcribed"] == 3
    assert meta["started"] == first["started"]


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0.0, 10), (10.0, 260), (17.2, 440), (30.0, 440)]
)
def test_token_limit(seconds: float, expected: int) -> None:
    assert token_limit(seconds, 25.0) == expected


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(status_code)
        self.status_code = status_code


def _sarvam(responses: list[object]) -> SarvamTranscriber:
    def transcribe(**_: object) -> object:
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    transcriber = SarvamTranscriber.__new__(SarvamTranscriber)
    stub = SimpleNamespace(speech_to_text=SimpleNamespace(transcribe=transcribe))
    transcriber.client = cast(Any, stub)
    transcriber.model, transcriber.mode, transcriber.language = "m", "transcribe", "hi-IN"
    transcriber.max_retries = 3
    return transcriber


AUDIO = np.zeros(1600, dtype=np.float32)


def test_sarvam_empty_transcript_is_empty_string() -> None:
    assert _sarvam([SimpleNamespace(transcript=None)]).transcribe([AUDIO]) == [""]


def test_sarvam_retries_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    responses: list[object] = [_HttpError(429), SimpleNamespace(transcript=" नमस्ते ")]
    assert _sarvam(responses).transcribe([AUDIO]) == ["नमस्ते"]


def test_sarvam_does_not_retry_bad_requests() -> None:
    with pytest.raises(_HttpError):
        _sarvam([_HttpError(400)]).transcribe([AUDIO])


def test_call_sarvam_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", waits.append)
    calls: list[int] = []

    def request() -> None:
        calls.append(1)
        raise _HttpError(503)

    with pytest.raises(_HttpError):
        call_sarvam(request, max_retries=3)
    assert len(calls) == 3
    assert len(waits) == 2
    assert 1 <= waits[0] < 2 <= waits[1] < 3  # exponential backoff with jitter
