# embeddings/models.py
from __future__ import annotations

import numpy as np

# HF embedding client এবং dimension আসছে data/testconfig থেকে
# সেখানেই .env লোড, HF token, InferenceClient সব সেট করা আছে
from data.testconfig import hf_embed_one, EMBEDDING_DIM

# Public constant: embedding vector dimension
DIM = EMBEDDING_DIM


def embed_text(text: str) -> np.ndarray:
    """
    Main public API: টেক্সট থেকে embedding বের করে।
    আগে যেখানে SentenceTransformer ব্যবহার করতে, সেখানেই এটা ব্যবহার করবে।

    Parameters
    ----------
    text : str
        যেকোনো ইনপুট টেক্সট / description

    Returns
    -------
    np.ndarray
        1D embedding vector, shape: [DIM]
    """
    return hf_embed_one(text)


def embed_item_id(item_id: str) -> np.ndarray:
    """
    শুধু item_id থাকলে stable pseudo-embedding বানাতে চাইলে।
    (Same item_id সবসময় একই embedding পাবে, কারণ একই string পাঠাচ্ছি.)

    Parameters
    ----------
    item_id : str
        ডাটাবেজে থাকা আইটেমের আইডি (যেমন "123", "movie_42")

    Returns
    -------
    np.ndarray
        1D embedding vector, shape: [DIM]
    """
    return hf_embed_one(f"ITEM::{item_id}")
