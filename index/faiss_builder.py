# index/faiss_builder.py
import faiss
import numpy as np
from data.testconfig import EMBEDDING_DIM, FAISS_INDEX_PATH


def create_index():
    # Use inner product on L2-normalized vectors to approximate cosine similarity.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return index


def add_embeddings(index, embeddings: np.ndarray):
    # Normalize in-place so that inner product corresponds to cosine similarity.
    faiss.normalize_L2(embeddings)
    index.add(embeddings)


def save_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)
