# index/faiss_search.py
import faiss
import numpy as np
from data.testconfig import FAISS_INDEX_PATH
from embedding.encoder import get_model
import json


def load_index():
    return faiss.read_index(FAISS_INDEX_PATH)


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def search(query, k=5, metadata_path=None):
    model = get_model()
    index = load_index()

    # Encode query and normalize to match the normalized vectors in the index.
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    # For an IndexFlatIP over normalized vectors, the returned values are
    # cosine similarities (higher is better, in [-1, 1]).
    scores, indices = index.search(q_emb, k)

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
