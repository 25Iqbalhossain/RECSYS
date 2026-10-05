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
    Turn an nsp_service row into a single text string for embedding.
    Priority: name + keyword, then other descriptive fields.
    """
    parts = []

    name = row.get("name")
    if name:
        parts.append(str(name))

    keyword = row.get("keyword")
    if keyword:
        parts.append(str(keyword))

    name_en = row.get("name_en")
    if name_en:
        parts.append(str(name_en))

    provider = row.get("provider_office_name")
    if provider:
        parts.append(str(provider))

    place = row.get("place")
    if place:
        parts.append(str(place))

    ctype = row.get("type")
    if ctype:
        parts.append(f"ধরন: {ctype}")

    sector = row.get("sector")
    if sector:
        parts.append(f"খাত: {sector}")

    # Fallback if everything else is empty
    if not parts:
        return str(row.get("id", ""))

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
