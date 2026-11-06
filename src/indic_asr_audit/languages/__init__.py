"""Registry of supported languages."""

from indic_asr_audit.languages.base import LanguagePack, WordKind
from indic_asr_audit.languages.hi import HINDI

LANGUAGES: dict[str, LanguagePack] = {pack.code: pack for pack in (HINDI,)}


def get_language(code: str) -> LanguagePack:
    """Return the language pack for an ISO 639-1 code."""
    try:
        return LANGUAGES[code]
    except KeyError:
        supported = ", ".join(sorted(LANGUAGES))
        raise ValueError(f"No language pack for '{code}'. Supported: {supported}") from None


__all__ = ["LANGUAGES", "LanguagePack", "WordKind", "get_language"]
