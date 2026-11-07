import itertools
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from indic_asr_audit.normalize import Level
from indic_asr_audit.plots import (
    RC,
    THEMES,
    Theme,
    _spread,
    draw_figures,
    real_vs_synthetic,
    vendor,
    wer_by_level,
    wer_vs_numbers,
)

SYSTEMS = ["mms-1b-all", "saaras-v4-transcribe", "whisper-large-v3"]


def _scores() -> pd.DataFrame:
    rows = []
    for subset in ("held_out_numeric", "numeric"):
        # MMS is left out of "numeric", as for a system trained on part of it
        for rank, system in enumerate(SYSTEMS[1:] if subset == "numeric" else SYSTEMS):
            for level in Level:
                rows.append(("wer", subset, system, str(level), 0.2 + rank / 10))
            rows.append(("numeric_error", subset, system, "-", 0.1 + rank / 10))
    return pd.DataFrame(
        [
            {"metric": m, "subset": s, "system": y, "level": lv, "estimate": e}
            | {"low": e - 0.05, "high": e + 0.05, "n": 10}
            for m, s, y, lv, e in rows
        ]
    )


def _gaps() -> dict[str, pd.DataFrame]:
    table = pd.DataFrame(
        {
            "subset": "held_out_numeric",
            "system": SYSTEMS,
            "n": 5,
            "wer_real": [0.3, 0.1, 0.2],
            "wer_synthetic": [0.35, 0.12, 0.15],
        }
    )
    return {f"{e}-{c}": table for e in ("mms-tts", "bulbul") for c in ("clean", "telephony")}


def test_vendor_colours_follow_the_model_family() -> None:
    assert [vendor(s) for s in SYSTEMS] == ["meta", "sarvam", "openai"]


def test_every_line_keeps_its_vendor_colour_in_every_panel() -> None:
    # A system missing from one panel must not shift the colours of the others.
    with mpl.rc_context(RC):
        for theme in THEMES:
            fig = wer_by_level(_scores(), theme)
            assert fig is not None
            for ax in fig.axes:
                for line in ax.get_lines():
                    expected = theme.vendors[vendor(str(line.get_label()))]
                    assert to_hex(line.get_color()) == expected


def test_spread_keeps_order_and_minimum_gap() -> None:
    out = _spread([10.0, 10.5, 30.0, 11.0], gap=2.0)
    ranked = sorted(out)
    assert [out.index(v) for v in ranked] == [0, 1, 3, 2]
    assert all(b - a >= 2.0 - 1e-9 for a, b in itertools.pairwise(ranked))


def test_spread_leaves_labels_with_room_on_their_values() -> None:
    assert _spread([50.0, 10.0, 11.0, 12.0, 13.0], gap=4.0)[0] == 50.0
    assert _spread([0.5, 1.0], gap=4.0) == [0.0, 4.0]  # not pushed below the axis


def test_scatter_points_take_their_vendor_colour() -> None:
    with mpl.rc_context(RC):
        for theme in THEMES:
            fig = wer_vs_numbers(_scores(), theme)
            assert fig is not None
            points = [ln for ln in fig.axes[0].get_lines() if ln.get_label() in SYSTEMS]
            assert len(points) == len(SYSTEMS)
            for line in points:
                expected = theme.vendors[vendor(str(line.get_label()))]
                assert to_hex(line.get_color()) == expected


def test_figures_without_data_are_skipped() -> None:
    empty = _scores().iloc[:0]
    with mpl.rc_context(RC):
        assert wer_vs_numbers(_scores(), THEMES[0], subset="missing") is None
        assert wer_by_level(empty, THEMES[0]) is None
        assert real_vs_synthetic({"bulbul-clean": pd.DataFrame()}, THEMES[0]) is None
        only_other = {name: t.assign(subset="control") for name, t in _gaps().items()}
        assert real_vs_synthetic(only_other, THEMES[0]) is None


def test_real_vs_synthetic_draws_only_shared_systems() -> None:
    gaps = _gaps()
    gaps["bulbul-telephony"] = gaps["bulbul-telephony"].iloc[1:]  # drops mms-1b-all
    with mpl.rc_context(RC):
        fig = real_vs_synthetic(gaps, THEMES[0])
    assert fig is not None
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
        "Whisper large-v3",
        "Saaras v4 transcribe",
    ]


def test_real_vs_synthetic_refuses_different_real_speech() -> None:
    gaps = _gaps()
    gaps["bulbul-clean"] = gaps["bulbul-clean"].assign(wer_real=0.9)
    with mpl.rc_context(RC), pytest.raises(ValueError, match="real-speech WER differs"):
        real_vs_synthetic(gaps, THEMES[0])


def test_draw_figures_writes_light_and_dark_svgs(tmp_path: Path) -> None:
    rq1 = tmp_path / "rq1"
    rq1.mkdir()
    _scores().to_csv(rq1 / "scores.csv", index=False)
    paths = draw_figures(rq1, _gaps(), tmp_path / "figures")
    names = sorted(p.name for p in paths)
    assert names == sorted(
        f"{figure}-{theme}.svg"
        for figure in ("rq1-wer-vs-numbers", "rq1-wer-by-level", "rq2-real-vs-synthetic")
        for theme in ("light", "dark")
    )
    first = {p: p.read_bytes() for p in paths}
    draw_figures(rq1, _gaps(), tmp_path / "figures")
    assert all(p.read_bytes() == data for p, data in first.items())  # no churn on redraw


def test_draw_figures_skips_what_has_not_run(tmp_path: Path) -> None:
    assert draw_figures(tmp_path / "missing", {}, tmp_path / "figures") == []


@pytest.mark.parametrize("theme", THEMES, ids=lambda t: t.name)
def test_real_vs_synthetic_ignores_unknown_engines_and_empty_tables(theme: Theme) -> None:
    gaps = _gaps() | {"other-clean": _gaps()["bulbul-clean"], "bulbul-noisy": pd.DataFrame()}
    with mpl.rc_context(RC):
        fig = real_vs_synthetic(gaps, theme)
    assert fig is not None
    assert len(fig.axes) == 2  # clean and telephony
