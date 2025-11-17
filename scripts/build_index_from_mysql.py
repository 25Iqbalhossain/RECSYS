# scripts/build_index_from_mysql.py

import os
import numpy as np

from db.mysql_client import fetch_rows
from index.faiss_builder import create_index, add_embeddings, save_index
from index.metadata_store import write_metadata_line
from data.testconfig import METADATA_PATH, hf_embed_batch  # HF embedding client import

os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)


def row_to_text(row: dict) -> str:
    """
    contents টেবিলের একেকটা row থেকে একটা সুন্দর টেক্সট বানাচ্ছি,
    যেটা embedding-এর জন্যও ব্যবহার করব আর metadata 'text' ফিল্ডেও লিখব।
    """

    # ধরে নিচ্ছি row-তেই এই ফিল্ডগুলো আছে; না থাকলে .get ব্যবহার করো
    title = row.get("title") or ""
    ctype = row.get("type") or ""
    lang = row.get("language") or ""
    year = row.get("release_year") or ""
    category_id = row.get("category_id")

    parts = []

    if title:
        parts.append(str(title))

    if ctype:
        parts.append(f"কনটেন্ট টাইপ: {ctype}")

    if lang:
        parts.append(f"ভাষা: {lang}")

    if year:
        parts.append(f"রিলিজ বছর: {year}")

    if category_id is not None:
        parts.append(f"ক্যাটাগরি আইডি: {category_id}")

    # সব মিলিয়ে এক লাইনের description
    if not parts:
        # fallback, সবই ফাঁকা থাকলে
        return f"কনটেন্ট আইডি {row.get('id', '')}"

    return " . ".join(parts)


def main():
    index = create_index()
    global_id = 0

    with open(METADATA_PATH, "w", encoding="utf-8") as meta_f:
        for rows in fetch_rows():
            # MySQL rows -> টেক্সট description বানাচ্ছি
            texts = [row_to_text(r) for r in rows]

            # HF Inference দিয়ে embedding নিচ্ছি
            embeddings = hf_embed_batch(texts)   # shape: [batch, EMBEDDING_DIM]
            embeddings = np.asarray(embeddings, dtype="float32")

            # FAISS index এ add
            add_embeddings(index, embeddings)

            # metadata file এ এক লাইন করে save
            for row, text in zip(rows, texts):
                write_metadata_line(
                    meta_f,
                    row_id=row.get("id"),   # contents table এর id column
                    text=text,              # ✅ আর খালি না, আমাদের description
                    extra=row,
                )
                global_id += 1

    save_index(index)
    print("✅ Index & metadata ready!")


if __name__ == "__main__":
    main()
