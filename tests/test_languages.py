import pytest

from indic_asr_audit.languages import LANGUAGES, LanguagePack, WordKind, get_language


def test_unknown_language_lists_supported() -> None:
    with pytest.raises(ValueError, match="Supported: hi"):
        get_language("xx")


@pytest.mark.parametrize("pack", LANGUAGES.values(), ids=list(LANGUAGES))
def test_every_value_below_100_has_a_word(pack: LanguagePack) -> None:
    units = {int(v) for kind, v in pack.number_words.values() if kind is WordKind.UNIT}
    assert set(range(100)) <= units


def test_conflicting_word_is_rejected() -> None:
    pack = LanguagePack(code="xx", name="Test", native_digits="", units={"a": 1}, scales={"a": 100})
    with pytest.raises(ValueError, match="two meanings"):
        _ = pack.number_words


def test_bad_native_digits_are_rejected() -> None:
    with pytest.raises(ValueError, match="10 characters"):
        LanguagePack(code="xx", name="Test", native_digits="012", units={}, scales={})
