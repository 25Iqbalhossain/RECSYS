"""Utilities for writing FAISS metadata in JSON Lines format.

Each line in the metadata file corresponds to one vector in the FAISS
index and contains:

* ``row_id`` – primary key from the source table.
* ``text`` – the text that was embedded (or a synthesized fallback).
* ``extra`` – optional structured metadata (title, type, language, …).
"""

import json
from typing import Any, Dict, Optional, TextIO


def _build_text_from_extra(extra: Dict[str, Any]) -> str:
    """
    Build a human‑readable fallback ``text`` field from ``extra``.

    When the main ``text`` is empty or missing, we still want every
    vector to have some description for debugging and search UI. This
    helper stitches together a short phrase using common keys like
    ``title``, ``type``, ``language``, ``release_year`` and
    ``category_id``.
    """
    if not extra:
        return ""

    parts: list[str] = []

    title = extra.get("title")
    if title:
        parts.append(str(title))

    ctype = extra.get("type")
    if ctype:
        parts.append(f"Type: {ctype}")

    lang = extra.get("language")
    if lang:
        parts.append(f"Language: {lang}")

    year = extra.get("release_year")
    if year not in (None, ""):
        parts.append(f"Release year: {year}")

    category_id = extra.get("category_id")
    if category_id not in (None, ""):
        parts.append(f"Category ID: {category_id}")

    return " . ".join(parts) if parts else ""


def write_metadata_line(
    f: TextIO,
    row_id: Any,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a single metadata record to a JSON Lines file.

    The resulting line has the shape::

        {
          "row_id": ...,
          "text": "...",
          "extra": { ... }
        }

    If ``text`` is blank or whitespace‑only, a fallback description is
    generated from ``extra`` via :func:`_build_text_from_extra`.
    """
    extra = extra or {}

    if not text or not text.strip():
        auto_text = _build_text_from_extra(extra)
        if auto_text:
            text = auto_text

    record = {
        "row_id": row_id,
        "text": text,
        "extra": extra,
    }

    # ``default=str`` ensures non‑JSON‑native types (datetime, Decimal,
    # etc.) are converted to strings instead of raising an error.
    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

