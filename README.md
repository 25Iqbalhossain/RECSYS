# Recommendation System (Semantic + Co-Vis + Reranking)

This project is an end-to-end recommendation API that combines:

- **Semantic similarity** using Hugging Face embeddings + FAISS.
- **Co-visitation** (items frequently seen together) as a robust fallback.
- **Lightweight learning-to-rank** for reranking with diversity and freshness.

The API is implemented with FastAPI in `api_gateway/server.py`.

---

## 1. How Vectors and Distances Work

### Embeddings

- Embeddings are produced via the helpers in `embedding/models.py`.
- The actual embedding backend is configured in `data/testconfig.py`:
  - Uses a Hugging Face Inference endpoint (`intfloat/multilingual-e5-large-instruct`).
  - Falls back to a deterministic local embedding (hash-based) if HF is unavailable.
- **Dimension**: all embeddings have dimension `DIM` (`EMBEDDING_DIM` in `data/testconfig.py`).

Two main helper functions:

- `embed_text(text: str) -> np.ndarray`
  - Used for free-text (queries, titles, descriptions).
- `embed_item_id(item_id: str) -> np.ndarray`
  - Creates a *stable* pseudo-embedding by encoding `"ITEM::<item_id>"`.
  - Same `item_id` → same vector; works in the same space as `embed_text`.

### FAISS Index and Similarity

The FAISS index is built in `index/faiss_builder.py`:

- Uses `faiss.IndexFlatIP(EMBEDDING_DIM)` → **inner product** index.
- All item vectors are **L2-normalized** before adding to the index.

At query time (see `index/faiss_search.py` and `retrieval/vector_store.py`):

1. Encode the query with the same embedding model.
2. Cast to `float32`.
3. L2-normalize the query vector.
4. Call `index.search(q, k)`.

Because both stored vectors and query are unit-length, the inner product
returned by FAISS is **cosine similarity**:

- `score = cos(theta)` in `[-1, 1]` (typically `[0, 1]` for similar items).
- Higher `score` → more similar.

To also expose a distance-style value, code that wraps FAISS (e.g.
`index/faiss_search.py` and `retrieval/vector_store.py`) computes:

- `distance = 1.0 - score`
  - Smaller `distance` → more similar.

This keeps the semantics explicit for any downstream consumer.

### Vector Store (ANN)

`retrieval/vector_store.FaissStore`:

- Loads:
  - FAISS index from `data/artifacts/faiss.index`.
  - Item ID mapping from `data/artifacts/faiss_ids.parquet`.
- `search_vec(v, k)`:
  - Normalizes `v` and runs FAISS search.
  - Returns a list of dicts:

    ```python
    {
        "item_id": "<catalog-id>",
        "score": <cosine_similarity>,
        "distance": 1.0 - score,
    }
    ```

This is the main building block for ANN-based recommendations and offline evaluation.

---

## 2. Co-Visitation (Classical Signal)

`retrieval/covis/covis_services.CoVis`:

- Reads `data/artifacts/covis.parquet` (with `.csv` fallback).
- Each row has:
  - `item_id`, `neighbor`, `count`, and an optional `score`.
- `score` semantics:
  - If a `score` column exists → use it as the association-rule score
    (e.g. confidence / lift).
  - Otherwise → use `count` as the score.
- `topk_for_item(item_id, k)` returns:

  ```python
  [{"item_id": "<neighbor>", "score": <score>}, ...]
  ```

This signal is used as:

- A **fallback** when ANN/embeddings are unavailable.
- An additional feature for the LTR reranker.

---

## 3. Semantic Scores in the API

The main API is in `api_gateway/server.py`. Key ideas:

- The ANN-based endpoints return FAISS `score` directly (cosine similarity).
- Some endpoints also attach a `semantic_score` computed by
  `_annotate_semantic`, which:
  - Takes the query vector (from `embed_item_id` or a blend of recent items).
  - Re-embeds each candidate `item_id`.
  - Computes **exact cosine similarity** between query and candidate.

Conventions:

- ANN paths (pure FAISS):
  - `score` = FAISS cosine similarity.
  - `semantic_score` is set equal to `score`.
- Fallback paths (co-vis / popularity):
  - `score` = co-vis or popularity score.
  - `semantic_score` = exact cosine between the query vector and each result.

This makes it very clear which value is:

- The **retrieval score** (`score`).
- The **semantic similarity** (`semantic_score`).

---

## 4. Endpoints Overview

Quick reference (see also `docs/SYSTEM_OVERVIEW.md`):

- `GET /recs`
  - ANN-first from `context_item_id` or user’s last item.
  - Fallback: co-vis (`CoVis.topk_for_item`) with optional `semantic_score`.
- `GET /recs/auto`
  - Same as `/recs`, but always uses last item from user history.
- `GET /recs/personal`
  - Builds a decayed blend of recent items (`_recent_query_vec`) and runs ANN.
  - Fallback: personalized co-vis.
- `GET /feed`
  - Sections:
    - `for_you`: ANN from recent items.
    - `because_you_watched`: ANN from last item.
    - `trending`: co-vis popularity.
- `GET /recs/semantic`
  - Pure semantic neighbors for a `context_item_id` (ANN, co-vis fallback).
- `GET /search/semantic`
  - Semantic text search over items; returns FAISS cosine similarity scores.
- `GET /recs/ranked`
  - Reranks candidates using the logistic LTR (`ranker/simple_ltr.py`).
- `POST /track`, `POST /track/bulk`
  - Event ingestion into `data/raw_events/events.jsonl`.
- Reload endpoints:
  - `POST /reload_covis`, `POST /reload_faiss`, `POST /reload_ranker`.

---

## 5. Artifacts and Paths

Key files under `data/`:

- Sessions: `data/sessions/sessions.parquet`
- Co-vis table: `data/artifacts/covis.parquet`
- FAISS index: `data/artifacts/faiss.index`
- FAISS IDs: `data/artifacts/faiss_ids.parquet`
- Embeddings: `data/artifacts/item_embeddings.parquet`
- MySQL metadata for FAISS: `data/mysql_content_metadata.jsonl`
- Ranker weights: `data/artifacts/ranker_weights.json`

These are produced by the build scripts in `scripts/` and `processing/`.

---

## 6. Setup and Build Steps

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data and sessions**

   - Load raw events into `data/raw_events/events.jsonl`.
   - Build sessions:

     ```bash
     python processing/sessionization/job.py
     ```

3. **Build co-visitation table**

   ```bash
   python scripts/build_covis.py
   ```

4. **Backfill embeddings**

   ```bash
   python scripts/backfill_embeddings.py
   ```

5. **Build FAISS index**

   ```bash
   python scripts/build_faiss.py
   ```

6. **(Optional) Train reranker**

   ```bash
   python ranker/train_simple.py
   ```

7. **Run the API**

   ```bash
   uvicorn api_gateway.server:app --reload
   ```

8. **Reload artifacts in a running API**

   - `POST /reload_covis`
   - `POST /reload_faiss`
   - `POST /reload_ranker`

---

## 7. Notes and Troubleshooting

- ANN scores look like counts / integers
  - Likely using a fallback path; ensure FAISS index and embeddings are built and reloaded.
- `/search/semantic` returns 503
  - HF model not loaded or failing; check `.env` and `HF_API_TOKEN` in `data/testconfig.py`.
- FAISS index empty (`ntotal = 0`)
  - Rebuild embeddings and FAISS index; check for errors in the build scripts.
- Need to understand a specific distance/score
  - Check `retrieval/vector_store.py` and `index/faiss_search.py` for how `score` and `distance` are computed.

