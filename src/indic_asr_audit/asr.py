"""ASR systems behind one interface, and a resumable runner that caches output.

Heavy imports (torch, transformers, sarvamai) happen inside the backends, so
the rest of the package works without them.
"""

import io
import json
import os
import random
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
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


def package_versions(*names: str) -> dict[str, str]:
    """Installed versions of the named packages; missing ones are left out."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def token_limit(seconds: float, tokens_per_second: float, ceiling: int = 440) -> int:
    """Token limit for audio of a given length; Whisper's own ceiling is 448."""
    return min(ceiling, int(seconds * tokens_per_second) + 10)


class WhisperTranscriber:
    """A Whisper checkpoint from Hugging Face, decoded greedily in batches.

    Whisper sometimes repeats a phrase until it reaches its token limit, and
    in a batch that one utterance makes every other wait. The limit is
    therefore set from the length of the longest clip in the batch, at
    ``max_tokens_per_second``. On the FLEURS Hindi references the real rate
    is at most 19 tokens per second, so the default of 25 only stops runaway
    repetition. Outputs whose length reaches the limit are counted in
    ``capped`` and recorded with the transcripts.
    """

    def __init__(
        self,
        model: str,
        language: str,
        revision: str | None = None,
        max_tokens_per_second: float = 25.0,
    ) -> None:
        from huggingface_hub import model_info
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id, self.language = model, language
        self.revision = revision or model_info(model).sha
        self.max_tokens_per_second = max_tokens_per_second
        self.device, self.dtype = _device_and_dtype()
        self.processor = WhisperProcessor.from_pretrained(model, revision=self.revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, revision=self.revision, dtype=self.dtype, device_map=self.device
        )
        self.model.eval()
        self.capped = 0

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        """Return one transcript per input."""
        import torch

        seconds = max(len(audio) for audio in batch) / SAMPLE_RATE
        limit = token_limit(seconds, self.max_tokens_per_second)
        features = self.processor(list(batch), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = features.input_features.to(self.device, self.model.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                inputs,
                language=self.language,
                task="transcribe",
                do_sample=False,
                num_beams=1,
                max_new_tokens=limit,
            )
        texts = [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        lengths = [
            len(self.processor.tokenizer(t, add_special_tokens=False).input_ids) for t in texts
        ]
        self.capped += sum(n >= limit - 5 for n in lengths)  # a few prompt tokens share the budget
        return texts

    def describe(self) -> dict[str, Any]:
        """Return the settings that determine the output."""
        return {
            "backend": "whisper",
            "model": self.model_id,
            "revision": self.revision,
            "language": self.language,
            "decoding": {"do_sample": False, "num_beams": 1},
            "max_tokens_per_second": self.max_tokens_per_second,
            "device": self.device,
            "dtype": str(self.dtype),
        }


class CTCTranscriber:
    """A CTC model from Hugging Face, such as MMS with a language adapter."""

    def __init__(
        self,
        model: str,
        language: str,
        revision: str | None = None,
        adapter: str | None = None,
        chunk_size: int = 4,
    ) -> None:
        from huggingface_hub import model_info
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_id, self.language, self.adapter = model, language, adapter
        self.chunk_size = chunk_size
        self.revision = revision or model_info(model).sha
        self.device, self.dtype = _device_and_dtype()
        self.processor = AutoProcessor.from_pretrained(model, revision=self.revision)
        self.model = AutoModelForCTC.from_pretrained(
            model, revision=self.revision, dtype=self.dtype, device_map=self.device
        )
        if adapter:  # MMS: one shared model with a small adapter per language
            self.processor.tokenizer.set_target_lang(adapter)
            self.model.load_adapter(adapter)
        self.model.eval()

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
        """Return the settings that determine the output."""
        return {
            "backend": "ctc",
            "model": self.model_id,
            "revision": self.revision,
            "adapter": self.adapter,
            "decoding": "argmax",
            "device": self.device,
            "dtype": str(self.dtype),
        }


SARVAM_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def call_sarvam[T](request: Callable[[], T], max_retries: int) -> T:
    """Make a Sarvam API request, retrying transient failures with backoff.

    Transient means a dropped connection, a timeout, a rate limit or a server
    error; anything else, such as a bad request, is raised at once.
    """
    import httpx  # installed with sarvamai

    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            return request()
        except Exception as error:  # the SDK raises one class per HTTP status
            status = getattr(error, "status_code", None)
            transient = isinstance(error, httpx.TransportError) or status in SARVAM_RETRY_STATUS
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(2**attempt + random.random())
    raise RuntimeError("unreachable: the last attempt returns or raises")


class SarvamTranscriber:
    """Sarvam's hosted Saaras model. Needs SARVAM_API_KEY.

    A hosted model can change without notice, so its cached transcripts, not
    the API, are the reproducible artifact.
    """

    def __init__(self, model: str, mode: str, language: str, max_retries: int = 5) -> None:
        from sarvamai import SarvamAI

        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("Set SARVAM_API_KEY (see .env.example) to run Sarvam systems")
        self.client = SarvamAI(api_subscription_key=key)
        self.model, self.mode, self.language = model, mode, language
        self.max_retries = max_retries

    def transcribe(self, batch: Sequence[Audio]) -> list[str]:
        """Return one transcript per input, one request each."""
        return [self._one(audio) for audio in batch]

    def _one(self, audio: Audio) -> str:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        response = call_sarvam(
            lambda: self.client.speech_to_text.transcribe(
                file=("audio.wav", buffer.getvalue(), "audio/wav"),
                model=self.model,
                mode=self.mode,
                language_code=self.language,
            ),
            self.max_retries,
        )
        return str(response.transcript or "").strip()

    def describe(self) -> dict[str, Any]:
        """Return the settings that determine the output."""
        return {
            "backend": "sarvam",
            "model": self.model,
            "mode": self.mode,
            "language": self.language,
        }


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
    """Read cached transcripts as {utterance id: hypothesis}.

    Read-only. A partial last line, left by a crash or by a run still being
    written, is ignored.
    """
    if not path.exists():
        return {}
    lines = path.read_bytes().split(b"\n")
    rows = [json.loads(line) for line in lines[:-1] if line.strip()]
    return {row["id"]: row["hypothesis"] for row in rows}


def _drop_partial_line(path: Path) -> None:
    """Cut a partial last line so new lines are appended to a clean file."""
    data = path.read_bytes() if path.exists() else b""
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "config" not in meta:
        raise ValueError(f"{path} predates the current runner; delete it and its transcripts")
    return meta


def run_transcription(
    transcriber: Transcriber,
    utterances: Sequence[Utterance],
    data_dir: Path,
    out_path: Path,
    batch_size: int = 32,
) -> None:
    """Transcribe utterances not yet in ``out_path``, appending as it goes.

    Output is one JSON line per utterance. A sidecar ``.meta.json`` holds the
    settings that produced it; resuming with different settings is refused,
    so one file never mixes two configurations. Counts in the sidecar add up
    over resumed runs. Utterances are sorted by duration so each batch pads
    as little as possible.
    """
    import soundfile as sf

    config = transcriber.describe()
    meta_path = out_path.with_suffix(".meta.json")
    previous = _read_meta(meta_path)
    if previous and previous["config"] != config:
        raise ValueError(
            f"{out_path} was made with other settings: {previous['config']} != {config}"
        )

    _drop_partial_line(out_path)
    done = read_transcripts(out_path)
    todo = [u for u in utterances if u.id not in done]
    if not todo:
        return
    todo.sort(key=lambda u: sf.info(data_dir / u.audio).duration)
    meta: dict[str, Any] = {
        "config": config,
        "versions": package_versions("torch", "transformers", "sarvamai", "soundfile", "scipy"),
        "batch_size": batch_size,
        "started": previous.get("started", datetime.now(UTC).isoformat(timespec="seconds")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            texts = transcriber.transcribe([load_audio(data_dir / u.audio) for u in batch])
            for utt, text in zip(batch, texts, strict=True):
                f.write(json.dumps({"id": utt.id, "hypothesis": text}, ensure_ascii=False) + "\n")
            f.flush()
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["transcribed"] = len(read_transcripts(out_path))
    if (capped := getattr(transcriber, "capped", None)) is not None:
        meta["capped_outputs"] = previous.get("capped_outputs", 0) + capped
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
