"""ASR systems behind one interface, and a resumable runner that caches output.

Heavy imports (torch, transformers, sarvamai) happen inside the backends, so
the rest of the package works without them.
"""

import io
import json
import os
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import gcd
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from indic_asr_audit.data import Utterance

SAMPLE_RATE = 16_000
Audio = NDArray[np.float32]


class Transcriber(Protocol):
    """Anything that turns 16 kHz mono audio into text."""

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        """Return one transcript per input."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return provenance details, such as the resolved model revision."""
        ...


def load_audio(path: Path) -> Audio:
    """Read an audio file as 16 kHz mono float32."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        div = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // div, rate // div)
    return np.asarray(mono, dtype=np.float32)


def _device_and_dtype() -> tuple[str, Any]:
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class WhisperTranscriber:
    """A Whisper checkpoint from Hugging Face, decoded greedily in batches.

    Whisper sometimes repeats a phrase until it reaches its token limit, and
    in a batch that one utterance makes every other wait. The limit is
    therefore set from the audio length: no real transcript needs more than
    ``max_tokens_per_second`` tokens per second of speech.
    """

    def __init__(self, model: str, language: str, max_tokens_per_second: float = 25.0) -> None:
        from huggingface_hub import model_info
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id, self.language = model, language
        self.revision = model_info(model).sha
        self.max_tokens_per_second = max_tokens_per_second
        self.device, dtype = _device_and_dtype()
        self.processor = WhisperProcessor.from_pretrained(model, revision=self.revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, revision=self.revision, dtype=dtype
        )
        self.model.to(self.device).eval()

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        """Return one transcript per input."""
        import torch

        seconds = max(len(audio) for audio in batch) / SAMPLE_RATE
        limit = min(440, int(seconds * self.max_tokens_per_second) + 10)
        features = self.processor(list(batch), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = features.input_features.to(self.device, self.model.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                inputs, language=self.language, task="transcribe", max_new_tokens=limit
            )
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [text.strip() for text in texts]

    def describe(self) -> dict[str, Any]:
        """Return provenance details."""
        return {
            "model": self.model_id,
            "revision": self.revision,
            "language": self.language,
            "decoding": "greedy",
            "max_tokens_per_second": self.max_tokens_per_second,
        }


class CTCTranscriber:
    """A CTC model from Hugging Face, such as MMS with a language adapter."""

    def __init__(
        self, model: str, language: str, adapter: str | None = None, chunk_size: int = 4
    ) -> None:
        from huggingface_hub import model_info
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_id, self.language, self.adapter = model, language, adapter
        self.chunk_size = chunk_size
        self.revision = model_info(model).sha
        self.device, dtype = _device_and_dtype()
        self.processor = AutoProcessor.from_pretrained(model, revision=self.revision)
        self.model = AutoModelForCTC.from_pretrained(model, revision=self.revision, dtype=dtype)
        if adapter:  # MMS: one shared model with a small adapter per language
            self.processor.tokenizer.set_target_lang(adapter)
            self.model.load_adapter(adapter)
        self.model.to(self.device).eval()

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        """Return one transcript per input.

        CTC models read raw waveforms, so memory grows quickly with batch
        size; the batch is decoded in chunks of ``chunk_size``.
        """
        import torch

        texts: list[str] = []
        for start in range(0, len(batch), self.chunk_size):
            chunk = list(batch[start : start + self.chunk_size])
            features = self.processor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = features.input_values.to(self.device, self.model.dtype)
            mask = features.get("attention_mask")
            if mask is not None:
                mask = mask.to(self.device)
            with torch.inference_mode():
                logits = self.model(inputs, attention_mask=mask).logits
            texts.extend(self.processor.batch_decode(logits.argmax(dim=-1)))
        return [text.strip() for text in texts]

    def describe(self) -> dict[str, Any]:
        """Return provenance details."""
        return {"model": self.model_id, "revision": self.revision, "adapter": self.adapter}


class SarvamTranscriber:
    """Sarvam's hosted Saaras model. Needs SARVAM_API_KEY."""

    def __init__(self, model: str, mode: str, language: str, max_retries: int = 5) -> None:
        from sarvamai import SarvamAI

        self.client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        self.model, self.mode, self.language = model, mode, language
        self.max_retries = max_retries

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        """Return one transcript per input, one request each."""
        return [self._one(audio) for audio in batch]

    def _one(self, audio: Audio) -> str:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        for attempt in range(self.max_retries):
            try:
                response = self.client.speech_to_text.transcribe(
                    file=("audio.wav", buffer.getvalue(), "audio/wav"),
                    model=self.model,
                    mode=self.mode,
                    language_code=self.language,
                )
                return str(response.transcript).strip()
            except Exception as error:  # the SDK raises one class per HTTP status
                status = getattr(error, "status_code", None)
                if status not in (429, 500, 502, 503) or attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def describe(self) -> dict[str, Any]:
        """Return provenance details."""
        return {"model": self.model, "mode": self.mode, "language": self.language}


@dataclass(frozen=True)
class SystemConfig:
    """One entry of configs/systems.toml."""

    name: str
    backend: str
    model: str
    language: str
    options: dict[str, Any]
    exclude_splits: frozenset[str] = frozenset()

    def build(self) -> Transcriber:
        """Instantiate the configured transcriber."""
        builders: dict[str, Callable[[], Transcriber]] = {
            "whisper": lambda: WhisperTranscriber(self.model, self.language, **self.options),
            "ctc": lambda: CTCTranscriber(self.model, self.language, **self.options),
            "sarvam": lambda: SarvamTranscriber(self.model, language=self.language, **self.options),
        }
        if self.backend not in builders:
            raise ValueError(f"{self.name}: unknown backend '{self.backend}'")
        return builders[self.backend]()


def load_systems(path: Path) -> dict[str, SystemConfig]:
    """Read system definitions from a TOML file, one table per system."""
    with path.open("rb") as f:
        tables = tomllib.load(f)
    systems = {}
    for name, table in tables.items():
        table = dict(table)
        systems[name] = SystemConfig(
            name=name,
            backend=table.pop("backend"),
            model=table.pop("model"),
            language=table.pop("language"),
            exclude_splits=frozenset(table.pop("exclude_splits", ())),
            options=table,
        )
    return systems


def read_transcripts(path: Path) -> dict[str, str]:
    """Read cached transcripts as {utterance id: hypothesis}."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row["id"]: row["hypothesis"] for row in rows}


def run_transcription(
    transcriber: Transcriber,
    utterances: Sequence[Utterance],
    data_dir: Path,
    out_path: Path,
    batch_size: int = 32,
) -> None:
    """Transcribe utterances not yet in ``out_path``, appending as it goes.

    Output is one JSON line per utterance. A sidecar ``.meta.json`` records
    the system's provenance. Re-running resumes where a previous run stopped.
    Utterances are sorted by length so each batch pads as little as possible.
    """
    import soundfile as sf

    done = read_transcripts(out_path)
    todo = [u for u in utterances if u.id not in done]
    todo.sort(key=lambda u: sf.info(data_dir / u.audio).frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            texts = transcriber.transcribe([load_audio(data_dir / u.audio) for u in batch])
            for utt, text in zip(batch, texts, strict=True):
                f.write(json.dumps({"id": utt.id, "hypothesis": text}, ensure_ascii=False) + "\n")
            f.flush()
    meta = {**transcriber.describe(), "finished": datetime.now(UTC).isoformat(timespec="seconds")}
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
