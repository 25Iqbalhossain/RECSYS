# embeddings/models.py
from __future__ import annotations
import numpy as np
#import hashlib

DIM = 64  # keep small for demo
"""
def _hash_to_vec(text: str, dim: int = DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # repeat the 32-byte hash to fill dim
    arr = np.frombuffer((h * ((dim // 32) + 1))[:dim], dtype=np.uint8).astype(np.float32)
    v = arr - arr.mean()
    n = np.linalg.norm(v) + 1e-9
    return v / n

def embed_text(text: str) -> np.ndarray:

    return _hash_to_vec(text, DIM)

def embed_item_id(item_id: str) -> np.ndarray:
   
    return _hash_to_vec(f"ITEM::{item_id}", DIM)
"""
from sentence_transformers import SentenceTransformer


try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    DIM = model.get_sentence_embedding_dimension() 
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please run 'pip install sentence-transformers'")
    model = None
    DIM = 0 

def _hash_to_vec(text: str, dim: int = DIM) -> np.ndarray | None:
   
    if model is None:
        print("Model is not loaded. Cannot create embedding.")
        return None

    embedding = model.encode(text)
    return embedding

def embed_text(text: str) -> np.ndarray | None:
    """Deterministic embedding for demo (no heavy model)."""
  
    return _hash_to_vec(text, DIM)

def embed_item_id(item_id: str) -> np.ndarray | None:
    """If you only have item IDs, this gives a stable pseudo-embedding."""
   
    return _hash_to_vec(f"ITEM::{item_id}", DIM)


