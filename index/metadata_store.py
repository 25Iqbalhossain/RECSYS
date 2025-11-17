# index/metadata_store.py

import json
from typing import Any, Dict, Optional, TextIO


def _build_text_from_extra(extra: Dict[str, Any]) -> str:
    """
    extra ডিকশনারি থেকে human-readable একটা description বানানোর চেষ্টা করি।
    মূলত fallback হিসেবে ব্যবহার হবে, যদি text খালি থাকে।
    """
    if not extra:
        return ""

    parts = []

    title = extra.get("title")
    if title:
        parts.append(str(title))

    ctype = extra.get("type")
    if ctype:
        parts.append(f"কনটেন্ট টাইপ: {ctype}")

    lang = extra.get("language")
    if lang:
        parts.append(f"ভাষা: {lang}")

    year = extra.get("release_year")
    if year not in (None, ""):
        parts.append(f"রিলিজ বছর: {year}")

    category_id = extra.get("category_id")
    if category_id not in (None, ""):
        parts.append(f"ক্যাটাগরি আইডি: {category_id}")

    # সব মিলিয়ে জোড়া লাগাই
    return " . ".join(parts) if parts else ""


def write_metadata_line(
    f: TextIO,
    row_id: Any,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    একটা record metadata ফাইলে এক লাইনে JSON হিসাবে লেখে।
    JSON Lines format (প্রতিটা লাইনে একটা JSON object)।
    """
    extra = extra or {}

    # যদি text ফাঁকা বা whitespace হয়, চেষ্টা করি extra থেকে একটা description বানাতে
    if not text or not text.strip():
        auto_text = _build_text_from_extra(extra)
        if auto_text:
            text = auto_text

    record = {
        "row_id": row_id,
        "text": text,
        "extra": extra,
    }

    # default=str => যেগুলো JSON জানে না (datetime, Decimal etc.) সেগুলো string এ convert হবে
    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
