"""Embedding helpers used across the recommendation system.

All embedding calls are funneled through the Hugging Face Inference
client defined in ``data.testconfig``. That module also exposes a
deterministic local fallback (hash‑based) so that semantic functionality
still works in offline environments where a remote HF endpoint is not
available.

Key conventions
----------------
* ``DIM`` is the global embedding dimension (``EMBEDDING_DIM``).
* Both free‑text and item‑ID embeddings live in the same vector space,
  so cosine similarity between them is meaningful.
* Higher cosine similarity means “more semantically similar” and is
  what the FAISS index stores and returns.
"""

from __future__ import annotations

import numpy as np

from data.testconfig import EMBEDDING_DIM, hf_embed_one

# Public constant: embedding vector dimension used by FAISS and models.
DIM = EMBEDDING_DIM


def embed_text(text: str) -> np.ndarray:
    """
    Embed an arbitrary text string into the shared semantic vector space.

    This is the primary entry point for semantic search queries, titles,
    and descriptions. It delegates to the HF Inference client (or the
    deterministic fallback) and always returns a 1‑D NumPy array of
    length ``DIM``.

    Parameters
    ----------
    text : str
        Input text such as a user query, item title, or description.

    Returns
    -------
    np.ndarray
        1‑D embedding vector with shape ``[DIM]`` and dtype float32.
    """
    return hf_embed_one(text)


def embed_item_id(item_id: str) -> np.ndarray:
    """
    Embed an item identifier into the same semantic vector space.

    The implementation prefixes the ID with ``"ITEM::"`` before
    encoding. This gives each item a *stable* pseudo‑embedding:

    * The same ``item_id`` will always produce the same vector (given
      fixed model weights).
    * Different IDs map to different regions of the space, allowing us
      to compare item‑to‑item similarity via cosine distance.

    Parameters
    ----------
    item_id : str
        Application‑level item identifier (for example ``"123"`` or
        ``"movie_42"``).

    Returns
    -------
    np.ndarray
        1‑D embedding vector with shape ``[DIM]`` and dtype float32.
    """
    return hf_embed_one(f"ITEM::{item_id}")

