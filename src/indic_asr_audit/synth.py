"""Synthetic test sets: the same sentences as a real test set, spoken by TTS.

Each sentence goes through a small LangGraph workflow:

    verbalize -> synthesize -> check --ok--> save
                    ^            |
                    +--retry-----+ (next voice)
                                 |
                                 +--give up--> reject

The check uses no ASR model. Filtering clips by how well some ASR system
recognises them would favour that system, which is exactly the comparison
the synthetic set is built to test. Instead, a clip passes when its length
is close to the real recording of the same sentence and it is neither silent
nor clipped.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from indic_asr_audit.asr import SAMPLE_RATE, Audio
from indic_asr_audit.data import Utterance
from indic_asr_audit.languages.base import LanguagePack
from indic_asr_audit.normalize import verbalize

Condition = Literal["clean", "telephony"]


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
    """Meta's MMS VITS model for one language (CC BY-NC 4.0 weights).

    It has a single speaker; its "voices" are random seeds, which vary the
    stochastic duration and pitch predictors and so the prosody.
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
        return {"engine": self.name, "model": self.model_id, "revision": self.revision}


class BulbulTTS:
    """Sarvam's hosted Bulbul model. Needs SARVAM_API_KEY."""

    def __init__(self, model: str, language: str, voices: Sequence[str]) -> None:
        from sarvamai import SarvamAI

        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("Set SARVAM_API_KEY (see .env.example) to use Bulbul")
        self.client = SarvamAI(api_subscription_key=key)
        self.name, self.model, self.language = "bulbul", model, language
        self.voices = list(voices)

    def synthesize(self, text: str, voice: str) -> Audio:
        """Return 16 kHz mono audio for text spoken by the given speaker."""
        import base64
        import io

        import soundfile as sf

        response = self.client.text_to_speech.convert(
            text=text,
            language_code=self.language,
            speaker=voice,
            model=self.model,
            speech_sample_rate=SAMPLE_RATE,
        )
        audio, rate = sf.read(io.BytesIO(base64.b64decode("".join(response.audios))))
        mono = np.asarray(audio, dtype=np.float32).reshape(len(audio), -1).mean(axis=1)
        return _resample(mono, rate, SAMPLE_RATE)

    def describe(self) -> dict[str, Any]:
        """Return the settings that determine the output."""
        return {"engine": self.name, "model": self.model, "voices": self.voices}


def telephony(audio: Audio) -> Audio:
    """Simulate a narrowband phone call: 300-3400 Hz, 8 kHz, G.711 mu-law."""
    from scipy.signal import butter, sosfilt

    narrow = _resample(audio, SAMPLE_RATE, 8000)
    band = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos"), narrow)
    mu = 255.0
    clipped = np.clip(band, -1.0, 1.0)
    encoded = np.round(np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / np.log1p(mu) * 127)
    decoded = np.sign(encoded) * np.expm1(np.abs(encoded) / 127 * np.log1p(mu)) / mu
    return _resample(decoded.astype(np.float32), 8000, SAMPLE_RATE)


@dataclass(frozen=True)
class CheckLimits:
    """Bounds a synthetic clip must meet, relative to the real recording."""

    min_duration_ratio: float = 0.5
    max_duration_ratio: float = 2.0
    min_rms: float = 1e-3
    max_clipped_fraction: float = 0.001


def check_clip(audio: Audio, real_seconds: float, limits: CheckLimits) -> list[str]:
    """Return the problems with a clip; an empty list means it passes."""
    problems = []
    ratio = len(audio) / SAMPLE_RATE / real_seconds
    if not limits.min_duration_ratio <= ratio <= limits.max_duration_ratio:
        problems.append(f"duration ratio {ratio:.2f}")
    if len(audio) == 0 or float(np.sqrt(np.mean(audio**2))) < limits.min_rms:
        problems.append("silent")
    elif float(np.mean(np.abs(audio) >= 0.999)) > limits.max_clipped_fraction:
        problems.append("clipped")
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


def build_graph(
    engine: TTSEngine,
    pack: LanguagePack,
    out_dir: Path,
    conditions: Sequence[Condition] = ("clean", "telephony"),
    limits: CheckLimits | None = None,
    max_attempts: int = 3,
) -> Any:
    """Compile the synthesis workflow for one engine."""
    from langgraph.graph import END, START, StateGraph

    limits = limits or CheckLimits()

    def verbalize_node(state: ClipState) -> ClipState:
        return {"spoken": verbalize(state["utterance"].text, pack), "attempt": 0}

    def synthesize_node(state: ClipState) -> ClipState:
        index = (int(state["utterance"].sentence_id) + state["attempt"]) % len(engine.voices)
        voice = engine.voices[index]
        return {"voice": voice, "audio": engine.synthesize(state["spoken"], voice)}

    def check_node(state: ClipState) -> ClipState:
        problems = check_clip(state["audio"], state["real_seconds"], limits)
        return {"problems": problems, "attempt": state["attempt"] + 1}

    def route(state: ClipState) -> str:
        if not state["problems"]:
            return "save"
        return "synthesize" if state["attempt"] < max_attempts else "reject"

    def save_node(state: ClipState) -> ClipState:
        import soundfile as sf

        variants = {"clean": state["audio"], "telephony": telephony(state["audio"])}
        for condition in conditions:
            path = out_dir / condition / f"{state['utterance'].id}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, variants[condition], SAMPLE_RATE, subtype="PCM_16")
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
    graph.add_edge("verbalize", "synthesize")
    graph.add_edge("synthesize", "check")
    graph.add_conditional_edges("check", route, ["save", "synthesize", "reject"])
    graph.add_edge("save", END)
    graph.add_edge("reject", END)
    return graph.compile()


def synthesize_manifest(
    engine: TTSEngine,
    pack: LanguagePack,
    utterances: Sequence[Utterance],
    data_dir: Path,
    dataset: str,
    conditions: Sequence[Condition] = ("clean", "telephony"),
) -> dict[Condition, list[Utterance]]:
    """Speak every sentence once and return one manifest per condition.

    Synthetic utterances keep the real utterance's id, text, sentence and
    split, so every result can be paired with its real counterpart. Each
    sentence is synthesised once, even if FLEURS has two real recordings.
    """
    import soundfile as sf

    out_dir = data_dir / "audio" / "synth" / dataset / engine.name
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(engine, pack, out_dir, conditions)
    log = []
    first_take = {u.sentence_id: u for u in reversed(utterances)}
    manifests: dict[Condition, list[Utterance]] = {c: [] for c in conditions}
    for utt in sorted(first_take.values(), key=lambda u: u.id):
        if all((out_dir / c / f"{utt.id}.wav").exists() for c in conditions):
            status, record = "saved", {"id": utt.id, "status": "saved", "cached": True}
        else:
            real_seconds = sf.info(data_dir / utt.audio).duration
            final = graph.invoke({"utterance": utt, "real_seconds": real_seconds})
            status = final["status"]
            record = {
                "id": utt.id,
                "status": status,
                "voice": final.get("voice"),
                "attempts": final["attempt"],
                "problems": final["problems"],
                "spoken": final["spoken"],
            }
        log.append(record)
        if status == "saved":
            for condition in conditions:
                audio = (out_dir / condition / f"{utt.id}.wav").relative_to(data_dir).as_posix()
                fields = (utt.text, utt.sentence_id, "", utt.split, utt.has_number)
                manifests[condition].append(Utterance(utt.id, audio, *fields))
    (out_dir / "synthesis_log.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in log), encoding="utf-8"
    )
    (out_dir / "engine.json").write_text(json.dumps(engine.describe(), indent=2), encoding="utf-8")
    return manifests


def load_engine(path: Path, name: str) -> TTSEngine:
    """Build the TTS engine named in a TOML file such as configs/tts.toml."""
    import tomllib

    with path.open("rb") as f:
        options = dict(tomllib.load(f)[name])
    kind = options.pop("engine")
    if kind == "mms":
        return MMSTTS(**options)
    if kind == "bulbul":
        return BulbulTTS(**options)
    raise ValueError(f"{name}: unknown engine '{kind}'")
