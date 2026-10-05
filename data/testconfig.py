# data/testconfig.py

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List, Union

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ==============================
# Paths & .env
# ==============================

BASE_DIR = Path(__file__).resolve().parent      # .../recsys/data

env_path = BASE_DIR / ".env"
print(f"[DEBUG] loading .env from: {env_path}  exists={env_path.exists()}")

load_dotenv(env_path)

# ==============================
# MySQL config (loaded from .env)
# ==============================

MYSQL_CONFIG = {
    # For security, all connection details are loaded from the
    # .env file in the data/ directory (see `env_path` above).
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "bangla_reco_large"),
}

# nsp_service টেবিল ব্যবহার করব embeddings এর জন্য
TABLE_NAME = os.getenv("MYSQL_TABLE", "nsp_service")

# টেক্সট/description ধরনের কলামগুলো যেগুলো থেকে embedding বানাবে
TEXT_COLUMNS = [
    "name",                 # মূল সার্ভিস নাম (Bangla)
    "name_en",              # ইংরেজি নাম (থাকলে)
    "provider_office_name", # অফিস/প্রোভাইডার নাম
    "place",                # place/অবস্থান টেক্সট
    "type",                 # টাইপ (tinytext)
    "sector",               # সেক্টর
    "keyword",              # keyword ফিল্ড
]

BATCH_SIZE = int(os.getenv("MYSQL_BATCH_SIZE", "500"))

# ==============================
# Hugging Face Inference (REMOTE, with local fallback)
# ==============================

HF_API_TOKEN = (
    os.getenv("HF_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_HUB_TOKEN")
)

print(f"[DEBUG] HF_API_TOKEN found? {'YES' if HF_API_TOKEN else 'NO'}")

if HF_API_TOKEN is None:
    # Offline-friendly: don't crash if token missing; we'll fall back locally.
    print("[WARN] HF_API_TOKEN not set – using local fallback embeddings (no remote HF calls).")

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_DIM = 1024

hf_client: InferenceClient | None = None
if HF_API_TOKEN is not None:
    hf_client = InferenceClient(
        provider="hf-inference",
        api_key=HF_API_TOKEN,
    )


def _fallback_embed_batch(texts: List[str]) -> np.ndarray:
    """
    Deterministic local embedding fallback (no network).

    Instead of random Gaussian vectors (which make cosine scores
    meaningless), we build a simple hashed bag-of-words vector for
    each text:

    - Tokenize on whitespace.
    - For each token, hash into [0, EMBEDDING_DIM) and increment that dimension.

    This keeps similar texts (with overlapping tokens) closer in cosine
    space, so search/recommendation quality is at least lexically
    meaningful when HF Inference is unavailable.

    Shape: [batch, EMBEDDING_DIM].
    """
    if isinstance(texts, str):
        texts = [texts]

    vecs: list[np.ndarray] = []
    for t in texts:
        tokens = str(t).split()
        if not tokens:
            tokens = [str(t)]
        v = np.zeros(EMBEDDING_DIM, dtype="float32")
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % EMBEDDING_DIM
            v[idx] += 1.0
        vecs.append(v)

    if not vecs:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")
    return np.stack(vecs, axis=0)


def hf_embed_batch(texts: Union[str, List[str]]) -> np.ndarray:
    """
    HF Inference batch embedding with fast timeout.
    Falls back to local deterministic embeddings so requests don't hang.
    """
    if isinstance(texts, str):
        texts = [texts]

    if hf_client is None:
        return _fallback_embed_batch(texts)

    try:
        result = hf_client.feature_extraction(
            texts,
            model=EMBEDDING_MODEL_ID,
            timeout=5,  # seconds; keep small so API calls don't block long
        )
        arr = np.array(result, dtype="float32")
    except Exception as e:
        print(f"[WARN] HF embedding call failed ({e}); using local fallback embeddings.")
        return _fallback_embed_batch(texts)

    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    elif arr.ndim == 1:
        arr = arr[None, :]

    return arr


def hf_embed_one(text: str) -> np.ndarray:
    return hf_embed_batch([text])[0]

# ==============================
# FAISS / metadata path
# ==============================

FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index.bin")
METADATA_PATH = str(BASE_DIR / "mysql_content_metadata.jsonl")

print(f"[DEBUG] FAISS_INDEX_PATH = {FAISS_INDEX_PATH}")
print(f"[DEBUG] METADATA_PATH   = {METADATA_PATH}")
print(f"[DEBUG] EMBEDDING_MODEL_ID = {EMBEDDING_MODEL_ID}, DIM = {EMBEDDING_DIM}")
