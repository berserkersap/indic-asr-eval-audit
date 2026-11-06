"""Synthetic test sets: the same sentences as a real test set, spoken by TTS.

Each sentence goes through a small LangGraph workflow:

    verbalize --text ok--> synthesize -> check --ok--> save
        |                     ^            |
        |                     +--retry-----+ (next voice)
        |                                  |
        +--Latin text--> reject <--give up-+

The check uses no ASR model. Filtering clips by how well some ASR system
recognises them would favour that system, which is exactly the comparison
the synthetic set is built to test. Instead, a clip passes when its speech
(silence trimmed) lasts about as long as the real recording's speech, and it
is neither silent nor clipped. Sentences whose spoken text still contains
Latin letters (codes such as "H5N1", units such as "mm") are rejected before
synthesis, because a Hindi voice may skip or misread them and the clip would
then not say the reference.
"""

import base64
import hashlib
import io
import json
import os
import random
import time
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import gcd
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from indic_asr_audit.asr import SAMPLE_RATE, Audio, load_audio, package_versions
from indic_asr_audit.data import Utterance
from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.normalize import has_latin, verbalize

Condition = Literal["clean", "telephony"]
CONDITIONS: tuple[Condition, ...] = ("clean", "telephony")
# Bump when verbalize() changes what is spoken, so old clips are not reused.
VERBALIZER_VERSION = 3


class TTSEngine(Protocol):
    """Anything that speaks text with one of several voices, at 16 kHz."""

    @property
    def name(self) -> str:
        """Short engine name, used in output paths."""
        ...

    @property
    def voices(self) -> Sequence[str]:
        """Voices to rotate through; a retry moves to the next one."""
        ...

    def synthesize(self, text: str, voice: str) -> Audio:
        """Return 16 kHz mono audio for text spoken by voice."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return the settings that determine the output."""
        ...


def _resample(audio: NDArray[Any], rate: int, target: int) -> Audio:
    from scipy.signal import resample_poly

    if rate == target:
        return audio.astype(np.float32)
    div = gcd(rate, target)
    return np.asarray(resample_poly(audio, target // div, rate // div), dtype=np.float32)


class MMSTTS:
    """Meta's MMS VITS model for one language (CC BY-NC 4.0 weights), on CPU.

    It has a single speaker; its "voices" are random seeds, which vary the
    stochastic duration and pitch predictors and so the prosody. Seeded
    output is reproducible on CPU; GPU kernels may differ.
    """

    def __init__(self, model: str, revision: str, seeds: Sequence[int] = (0, 1, 2)) -> None:
        from transformers import AutoTokenizer, VitsModel

        self.name, self.model_id, self.revision = "mms-tts", model, revision
        self.voices = [str(seed) for seed in seeds]
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = VitsModel.from_pretrained(model, revision=revision).eval()

    def synthesize(self, text: str, voice: str) -> Audio:
        """Return 16 kHz mono audio for text spoken with the given seed."""
        import torch

        torch.manual_seed(int(voice))
        with torch.inference_mode():
            wave = self.model(**self.tokenizer(text, return_tensors="pt")).waveform[0]
        return _resample(wave.numpy(), self.model.config.sampling_rate, SAMPLE_RATE)

    def describe(self) -> dict[str, Any]:
        """Return the settings that determine the output."""
        config = self.model.config
        return {
            "engine": self.name,
            "model": self.model_id,
            "revision": self.revision,
            "voices": self.voices,
            "device": "cpu",
            "speaking_rate": config.speaking_rate,
            "noise_scale": config.noise_scale,
            "noise_scale_duration": config.noise_scale_duration,
            "versions": package_versions("torch", "transformers"),
        }


class BulbulTTS:
    """Sarvam's hosted Bulbul model. Needs SARVAM_API_KEY.

    A hosted model can change without notice; the saved clips, not the API,
    are the reproducible artifact.
    """

    RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(
        self, model: str, language: str, voices: Sequence[str], max_retries: int = 5
    ) -> None:
        from sarvamai import SarvamAI

        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("Set SARVAM_API_KEY (see .env.example) to use Bulbul")
        self.client = SarvamAI(api_subscription_key=key)
        self.name, self.model, self.language = "bulbul", model, language
        self.voices = list(voices)
        self.max_retries = max(1, max_retries)

    def _request(self, text: str, voice: str) -> Any:
        for attempt in range(self.max_retries):
            try:
                return self.client.text_to_speech.convert(
                    text=text,
                    language_code=self.language,
                    speaker=voice,
                    model=self.model,
                    speech_sample_rate=SAMPLE_RATE,
                )
            except Exception as error:  # the SDK raises one class per HTTP status
                status = getattr(error, "status_code", None)
                if status not in self.RETRY_STATUS or attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt + random.random())
        raise RuntimeError("unreachable: the last attempt returns or raises")

    def synthesize(self, text: str, voice: str) -> Audio:
        """Return 16 kHz mono audio for text spoken by the given speaker."""
        import soundfile as sf

        chunks = []
        for chunk in self._request(text, voice).audios:  # long text: several WAV files
            audio, rate = sf.read(io.BytesIO(base64.b64decode(chunk)), always_2d=True)
            chunks.append(_resample(audio.mean(axis=1), rate, SAMPLE_RATE))
        return np.concatenate(chunks)

    def describe(self) -> dict[str, Any]:
        """Return the settings that determine the output."""
        return {
            "engine": self.name,
            "model": self.model,
            "language": self.language,
            "voices": self.voices,
            "versions": package_versions("sarvamai"),
        }


def telephony(audio: Audio) -> Audio:
    """Simulate a narrowband phone call: 300-3400 Hz, 8 kHz, G.711-style mu-law.

    The band-pass filter is causal, so it shifts phase a little, as real
    channels do. Quantisation follows the mu-law curve with 8-bit codes but
    is not bit-exact G.711.
    """
    from scipy.signal import butter, sosfilt

    narrow = _resample(audio, SAMPLE_RATE, 8000)
    band = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos"), narrow)
    return _resample(mu_law(band), 8000, SAMPLE_RATE)


def mu_law(audio: NDArray[Any], mu: float = 255.0) -> Audio:
    """Compress to signed 8-bit mu-law codes and expand back."""
    clipped = np.clip(audio, -1.0, 1.0)
    codes = np.round(np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / np.log1p(mu) * 127)
    expanded = np.sign(codes) * np.expm1(np.abs(codes) / 127 * np.log1p(mu)) / mu
    return np.asarray(expanded, dtype=np.float32)


@dataclass(frozen=True)
class CheckLimits:
    """Bounds a synthetic clip must meet, relative to the real recording."""

    min_duration_ratio: float = 0.5
    max_duration_ratio: float = 2.0
    min_rms: float = 1e-3
    max_clipped_fraction: float = 0.001


def speech_seconds(audio: Audio, threshold_db: float = -40.0) -> float:
    """Time from the first to the last 20 ms frame within 40 dB of a loud frame.

    "Loud" is the 95th percentile of frame energy, so a single click does not
    set the reference and trim real speech.
    """
    frame = SAMPLE_RATE // 50
    frames = len(audio) // frame
    if frames == 0:
        return 0.0
    energy = np.sqrt(np.mean(audio[: frames * frame].reshape(frames, frame) ** 2, axis=1))
    reference = float(np.percentile(energy, 95))
    if not reference:
        return 0.0
    loud = np.flatnonzero(energy >= reference * 10 ** (threshold_db / 20))
    return float(loud[-1] - loud[0] + 1) * frame / SAMPLE_RATE


def check_clip(audio: Audio, real_speech_seconds: float, limits: CheckLimits) -> list[str]:
    """Return the problems with a clip; an empty list means it passes."""
    if len(audio) == 0 or float(np.sqrt(np.mean(audio**2))) < limits.min_rms:
        return ["silent"]
    problems = []
    if float(np.mean(np.abs(audio) >= 0.999)) > limits.max_clipped_fraction:
        problems.append("clipped")
    ratio = speech_seconds(audio) / max(real_speech_seconds, 1e-3)
    if not limits.min_duration_ratio <= ratio <= limits.max_duration_ratio:
        problems.append(f"duration ratio {ratio:.2f}")
    return problems


class ClipState(TypedDict, total=False):
    """What the graph knows about one sentence as it moves through the nodes."""

    utterance: Utterance
    real_seconds: float
    spoken: str
    attempt: int
    voice: str
    audio: Audio
    problems: list[str]
    status: Literal["saved", "rejected"]


def first_voice(sentence_id: str, n_voices: int) -> int:
    """Stable, content-neutral voice for a sentence; retries take the next ones."""
    return zlib.crc32(sentence_id.encode()) % n_voices


def _write_wav(path: Path, audio: Audio) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part.wav")
    sf.write(partial, audio, SAMPLE_RATE, subtype="PCM_16")
    partial.replace(path)


def build_graph(
    engine: TTSEngine,
    pack: LanguagePack,
    out_dir: Path,
    conditions: Sequence[Condition] = CONDITIONS,
    limits: CheckLimits | None = None,
    max_attempts: int = 3,
) -> Any:  # a compiled LangGraph graph; its generic type adds nothing here
    """Compile the synthesis workflow for one engine."""
    from langgraph.graph import END, START, StateGraph

    limits = limits or CheckLimits()

    def verbalize_node(state: ClipState) -> ClipState:
        spoken = verbalize(state["utterance"].text, pack)
        problems = ["latin text"] if has_latin(spoken) else []
        return {"spoken": spoken, "attempt": 0, "problems": problems}

    def synthesize_node(state: ClipState) -> ClipState:
        first = first_voice(state["utterance"].sentence_id, len(engine.voices))
        voice = engine.voices[(first + state["attempt"]) % len(engine.voices)]
        return {"voice": voice, "audio": engine.synthesize(state["spoken"], voice)}

    def check_node(state: ClipState) -> ClipState:
        problems = check_clip(state["audio"], state["real_seconds"], limits)
        return {"problems": problems, "attempt": state["attempt"] + 1}

    def after_verbalize(state: ClipState) -> str:
        return "reject" if state["problems"] else "synthesize"

    def after_check(state: ClipState) -> str:
        if not state["problems"]:
            return "save"
        return "synthesize" if state["attempt"] < max_attempts else "reject"

    def save_node(state: ClipState) -> ClipState:
        for condition in conditions:
            audio = state["audio"] if condition == "clean" else telephony(state["audio"])
            _write_wav(out_dir / condition / f"{state['utterance'].id}.wav", audio)
        return {"status": "saved"}

    def reject_node(state: ClipState) -> ClipState:
        return {"status": "rejected", "problems": state["problems"]}

    graph = StateGraph(ClipState)
    graph.add_node("verbalize", verbalize_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("check", check_node)
    graph.add_node("save", save_node)
    graph.add_node("reject", reject_node)
    graph.add_edge(START, "verbalize")
    graph.add_conditional_edges("verbalize", after_verbalize, ["synthesize", "reject"])
    graph.add_edge("synthesize", "check")
    graph.add_conditional_edges("check", after_check, ["save", "synthesize", "reject"])
    graph.add_edge("save", END)
    graph.add_edge("reject", END)
    return graph.compile()


def output_dir(data_dir: Path, dataset: str, settings: dict[str, Any]) -> Path:
    """Folder for one engine's clips, keyed by a hash of every setting that matters.

    Changing the engine, its voices, the check or the verbaliser gives a new
    folder, so clips made with other settings are never reused.
    """
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return data_dir / "audio" / "synth" / dataset / f"{settings['engine']['engine']}-{digest}"


def _read_log(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in (json.loads(line) for line in lines if line.strip())}


def synthesize_manifest(
    engine: TTSEngine,
    pack: LanguagePack,
    utterances: Sequence[Utterance],
    data_dir: Path,
    dataset: str,
    conditions: Sequence[Condition] = CONDITIONS,
    limits: CheckLimits | None = None,
) -> tuple[dict[Condition, list[Utterance]], list[dict[str, Any]]]:
    """Speak every sentence once; return one manifest per condition and the log.

    The synthetic utterance takes the id of the sentence's first recording in
    manifest order, so results pair with that real recording. Each sentence
    is decided once: saved clips and earlier rejections are reused from the
    log, which is appended to as the run goes.
    """
    limits = limits or CheckLimits()
    settings: dict[str, Any] = {
        "engine": engine.describe(),
        "limits": asdict(limits),
        "conditions": list(conditions),
        "verbalizer": VERBALIZER_VERSION,
    }
    out_dir = output_dir(data_dir, dataset, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_path = out_dir / "settings.json"
    if not settings_path.exists():
        created = datetime.now(UTC).isoformat(timespec="seconds")
        settings_path.write_text(json.dumps({**settings, "created": created}, indent=2))
    log_path = out_dir / "synthesis_log.jsonl"
    log = _read_log(log_path)
    graph = build_graph(engine, pack, out_dir, conditions, limits)

    first_take: dict[str, Utterance] = {}
    for utt in utterances:
        first_take.setdefault(utt.sentence_id, utt)
    manifests: dict[Condition, list[Utterance]] = {c: [] for c in conditions}
    for utt in first_take.values():
        record = log.get(utt.id)
        if (
            record
            and record["status"] == "saved"
            and not all((out_dir / c / f"{utt.id}.wav").exists() for c in conditions)
        ):
            record = None  # a saved clip was deleted: make it again
        if record is None:
            real_seconds = speech_seconds(load_audio(data_dir / utt.audio))
            final = graph.invoke({"utterance": utt, "real_seconds": real_seconds})
            log[utt.id] = {
                "id": utt.id,
                "status": final["status"],
                "voice": final.get("voice"),
                "attempts": final["attempt"],
                "problems": final["problems"],
                "spoken": final["spoken"],
                "has_number": utt.has_number,
                "split": utt.split,
            }
            with log_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(log[utt.id], ensure_ascii=False) + "\n")
        if log[utt.id]["status"] == "saved":
            for condition in conditions:
                audio = (out_dir / condition / f"{utt.id}.wav").relative_to(data_dir).as_posix()
                fields = (utt.text, utt.sentence_id, "", utt.split, utt.has_number)
                manifests[condition].append(Utterance(utt.id, audio, *fields))
    return manifests, [log[u.id] for u in first_take.values()]


def load_engine(path: Path, name: str) -> TTSEngine:
    """Build the TTS engine named in a TOML file such as configs/tts.toml."""
    import tomllib

    with path.open("rb") as f:
        options = dict(tomllib.load(f)[name])
    kind = options.pop("engine")
    options.pop("related_systems", None)
    if kind == "mms":
        return MMSTTS(**options)
    if kind == "bulbul":
        return BulbulTTS(**options)
    raise ValueError(f"{name}: unknown engine '{kind}'")


def related_systems(path: Path, name: str) -> set[str]:
    """ASR systems from the same family as a TTS engine, listed in the TOML file."""
    import tomllib

    with path.open("rb") as f:
        return set(tomllib.load(f)[name].get("related_systems", []))


def engine_of(path: Path, manifest: str, dataset: str) -> str:
    """The TTS engine in a synthetic manifest name such as "<dataset>-mms-tts-clean"."""
    import tomllib

    with path.open("rb") as f:
        engines = tomllib.load(f)
    matches = [e for e in engines if manifest.startswith(f"{dataset}-{e}-")]
    if not matches:
        raise ValueError(f"{manifest}: no engine from {path} matches")
    return max(matches, key=len)
