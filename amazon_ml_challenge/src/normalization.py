"""
Unicode-safe normalization utilities for business entity resolution.

Important:
- Preserve non-Latin scripts.
- Do not transliterate Indic languages.
- Do not remove Unicode combining marks globally.
- Keep normalization deterministic and lightweight.
"""

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------
# Precompiled regexes
# ---------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")

# Anything that is punctuation or symbol will be converted to a space.
# This is deliberately Unicode-aware through unicodedata.category().
#
# We do this character-by-character rather than using an ASCII regex so
# that non-Latin scripts remain intact.
#
# Example:
#   "Pvt. EFS Print Ventures Ltd."
#       ->
#   "pvt efs print ventures ltd"
# ---------------------------------------------------------------------


def _replace_punctuation_and_symbols(text: str) -> str:
    output = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith("P") or category.startswith("S"):
            output.append(" ")
        else:
            output.append(char)

    return "".join(output)


def normalize_text(value: Optional[str]) -> str:
    """
    General Unicode-safe normalization.

    Operations:
        1. Handle None.
        2. Convert to string.
        3. Unicode NFKC normalization.
        4. Unicode case folding.
        5. Replace punctuation/symbols with spaces.
        6. Collapse repeated whitespace.
        7. Strip leading/trailing whitespace.

    Examples:

        "Pvt. EFS Print Ventures Ltd."
            ->
        "pvt efs print ventures ltd"

        "SCI Ptit Àmicale"
            ->
        "sci ptit àmicale"

        "राम मार्केटिंग प्राइवेट लिमिटेड"
            ->
        "राम मार्केटिंग प्राइवेट लिमिटेड"

    Notice that Indic characters are preserved.
    """

    if value is None:
        return ""

    text = str(value)

    # Unicode compatibility normalization.
    text = unicodedata.normalize("NFKC", text)

    # Unicode-aware lowercase.
    text = text.casefold()

    # Remove punctuation/symbol semantics while preserving letters,
    # numbers, and script-specific marks.
    text = _replace_punctuation_and_symbols(text)

    # Collapse whitespace.
    text = _WHITESPACE_RE.sub(" ", text)

    return text.strip()


def normalize_name(value: Optional[str]) -> str:
    """
    Normalize a business name.

    Currently this intentionally performs only generic normalization.
    Legal-suffix removal will be handled later as a separate feature,
    because suffixes can sometimes be useful evidence.
    """

    return normalize_text(value)


def normalize_address(value: Optional[str]) -> str:
    """
    Normalize a business address.

    Empty or missing addresses become "".
    """

    return normalize_text(value)


def compact_text(value: Optional[str]) -> str:
    """
    Remove whitespace from an already-normalized string.

    Useful for certain blocking keys.

    Example:
        "prime money"
        ->
        "primemoney"
    """

    normalized = normalize_text(value)

    return normalized.replace(" ", "")


def token_set(value: Optional[str]) -> set[str]:
    """
    Return normalized whitespace-separated tokens.
    """

    normalized = normalize_text(value)

    if not normalized:
        return set()

    return set(normalized.split())


def token_list(value: Optional[str]) -> list[str]:
    """
    Return normalized whitespace-separated tokens preserving order.
    """

    normalized = normalize_text(value)

    if not normalized:
        return []

    return normalized.split()


def normalized_length(value: Optional[str]) -> int:
    """
    Length of normalized text.
    """

    return len(normalize_text(value))