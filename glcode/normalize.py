"""Merchant key and file-period rules from TrustGate's dbt macro and CKAN loader.

`normalize_merchant` drops tokens that contain a digit, turns punctuation into
spaces, and keeps the first three remaining tokens. That is the
`normalize_merchant` macro in the TrustGate warehouse.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^A-Z0-9 ]")
_SPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"[0-9]")
_MONTH = {
    name: i
    for i, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def normalize_merchant(value) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "UNKNOWN"
    text = _SPACE.sub(" ", _PUNCT.sub(" ", text.upper())).strip()
    tokens = [tok for tok in text.split(" ") if tok and not _DIGIT.search(tok)]
    return " ".join(tokens[:3]) or "UNKNOWN"


def period_from_url(url: str) -> str | None:
    """'Purchase+Card+-+August+2026.csv' -> '2026-08'."""
    from urllib.parse import unquote_plus

    name = unquote_plus(url.rsplit("/", 1)[-1]).lower()
    match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s_-]*(\d{2,4})", name)
    if not match:
        return None
    year = int(match.group(2))
    if year < 100:
        year += 2000
    return f"{year:04d}-{_MONTH[match.group(1)]:02d}"
