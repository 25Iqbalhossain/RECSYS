import json

import faiss
import numpy as np

from data.testconfig import FAISS_INDEX_PATH
from embedding.encoder import get_model


def load_index():
    """Load the FAISS index built from MySQL contents embeddings."""
    return faiss.read_index(FAISS_INDEX_PATH)


def load_metadata(metadata_path):
    """Yield JSON records (one per line) from the metadata file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def search(query, k=5, metadata_path=None):
    """
    Run a semantic search against the MySQL-backed FAISS index.

    Steps:
    1. Encode the query with the same embedding model used at index build.
    2. L2-normalize the query vector so inner product == cosine similarity.
    3. Use an IndexFlatIP FAISS index to retrieve top-k neighbors.
    4. Optionally attach metadata from the JSONL file.

    Logging:
    - Prints basic score distribution stats (top score, mean of top-10).
    - Logs a warning if the top score is suspiciously low, which usually
      indicates bad embeddings (e.g., fallback mode or mismatched text).
    """
    model = get_model()
    index = load_index()

    # Encode query and normalize to match the normalized vectors in the index.
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    # For an IndexFlatIP over normalized vectors, the returned values are
    # cosine similarities (higher is better, in [-1, 1]).
    scores, indices = index.search(q_emb, k)

    # Score diagnostics for debugging quality.
    if scores is not None and len(scores) > 0 and len(scores[0]) > 0:
        s_arr = np.asarray(scores[0], dtype="float32")
        top_score = float(s_arr[0])
        top10 = s_arr[: min(10, s_arr.size)]
        mean_top10 = float(top10.mean()) if top10.size else 0.0
        print(
            f"[STATS] mysql_faiss_search: k={k}, top_score={top_score:.4f}, "
            f"mean_top10={mean_top10:.4f}"
        )
        if top_score < 0.1:
            print(
                f"[WARN] mysql_faiss_search: very low top_score={top_score:.4f} "
                f"for query='{query}' (embeddings/index may be misconfigured)"
            )

    # optional: metadata map build
    meta_list = list(load_metadata(metadata_path)) if metadata_path else None

    results = []
    for idx, score in zip(indices[0], scores[0]):
        s_val = float(score)
        # Also expose a distance-style field where smaller is better,
        # derived from cosine similarity.
        item = {
            "faiss_id": int(idx),
            "score": s_val,
            "distance": 1.0 - s_val,
        }
        if meta_list and 0 <= idx < len(meta_list):
            item["metadata"] = meta_list[idx]
        results.append(item)
    return results
