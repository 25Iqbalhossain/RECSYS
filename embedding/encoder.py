from __future__ import annotations

import numpy as np
from data.testconfig import hf_embed_batch, TEXT_COLUMNS


class _HFEncoder:
    """Minimal wrapper exposing an `.encode()` method for FAISS search."""

    def encode(self, texts):
        return hf_embed_batch(texts)


_MODEL = _HFEncoder()


def get_model() -> _HFEncoder:
    """Return a singleton encoder compatible with `model.encode([...])`."""
    return _MODEL


def row_to_text(row: dict) -> str:
    """
    Turn a MySQL `contents` row (dict) into a single text string.
    Simple concatenation of known text columns.
    """
    parts = []
    if "name" in row and row["name"]:
        parts.append(str(row["name"]))
    if "age" in row and row["age"] is not None:
        parts.append(str(row["age"]))
    if "gender" in row and row["gender"]:
        parts.append(str(row["gender"]))
    if "city" in row and row["city"]:
        parts.append(str(row["city"]))
    if "favorite_type" in row and row["favorite_type"]:
        parts.append(str(row["favorite_type"]))

    return " . ".join(parts)


def encode_batch(rows):
    """
    rows (list[dict]) -> (embeddings, texts)

    Uses the same HF Inference embedding backend as the FAISS index.
    """
    texts = [row_to_text(r) for r in rows]
    embeddings = hf_embed_batch(texts)  # shape: [batch, EMBEDDING_DIM]
    embeddings = np.array(embeddings, dtype="float32")
    return embeddings, texts

