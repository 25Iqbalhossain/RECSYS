# scripts/build_index.py
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from data.testconfig import (
    EMBEDDING_MODEL_NAME,
    CONTENT_EMBEDDINGS_PATH,
    CONTENT_IDS_PATH,
    FAISS_INDEX_PATH,
    DATA_DIR,
)
from data.mysql_utils import fetch_contents


META_PATH = DATA_DIR / "content_metadata.jsonl"


def build_index():
    print("🔌 MySQL থেকে ডেটা আনছি...")
    rows = fetch_contents()  # limit চাইলে এখানে parameter দাও
    if not rows:
        print("⚠️ কোনো row পাওয়া যায়নি!")
        return

    # Text বানাবো title + other ফিল্ড দিয়ে
    texts = []
    ids = []
    metas = []

    for r in rows:
        ids.append(r["id"])
        # এখানে তুমি ঠিক করবে কোন ফিল্ড concatenate করবে
        text = f"{r['title']}\n{r['body']}"
        texts.append(text)

        # local metadata store
        metas.append(
            {
                "id": r["id"],
                "title": r["title"],
                "body": r["body"],
                "category": r["category"],
            }
        )

    print(f"✅ মোট {len(texts)} টা row পাওয়া গেছে")

    print("🧠 Embedding model লোড করছি...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("📐 Embedding জেনারেট করছি...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity-এর জন্য ভালো
    )

    # ➜ Local .npy ফাইলে সেভ
    print("💾 Embeddings + IDs সেভ করছি...")
    np.save(CONTENT_EMBEDDINGS_PATH, embeddings)
    np.save(CONTENT_IDS_PATH, np.array(ids, dtype=np.int64))

    # ➜ Metadata JSONL এ সেভ (local থেকে title, body, category পড়ার জন্য)
    print("💾 Metadata JSONL সেভ করছি...")
    with META_PATH.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ➜ FAISS index বানানো
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine-এর জন্য normalize করেছি)
    index.add(embeddings)

    print("💾 FAISS index সেভ করছি...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print("🎉 কাজ শেষ!")


if __name__ == "__main__":
    build_index()
