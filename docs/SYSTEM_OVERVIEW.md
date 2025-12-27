# System Overview

This repo provides an end-to-end recommendation API with two retrieval signals:
co-visitation counts (classical) and semantic similarity using SentenceTransformer + FAISS,
plus a lightweight reranker with diversity and freshness controls.

## Components

- API: `api_gateway/server.py` (FastAPI)
- Co-vis store: `retrieval/covis/covis_services.py`
- Vector store (ANN): `retrieval/vector_store.py`
- Embeddings: `embedding/models.py`
- Catalog: `catalog/store.py` (items with `title/desc/created_ts`)
- Rankers: `ranker/inferences.py` (baseline rules) and `ranker/simple_ltr.py` (logistic), trainer `ranker/train_simple.py`
- Sessionization: `processing/sessionization/job.py`
- Build scripts: `scripts/build_covis.py`, `scripts/backfill_embeddings.py`, `scripts/build_faiss.py`
- Schedulers: `scripts/scheduler_sessionize.py` (30m), `scripts/scheduler_refresh.py` (daily refresh)

## Data Flow

```
/track events                      Catalog items
      |                                   |
      v                                   v
data/raw_events/events.jsonl   <-- catalog ingest
      |
      v
Sessionize (30m gap)
      |
      v
data/sessions/sessions.parquet
      |
      +--> Build co-vis pairs  -> data/artifacts/covis.parquet

Backfill embeddings (SentenceTransformer on title/desc; fallback: ID)
      |
      v
data/artifacts/item_embeddings.parquet
      |
      v
Build FAISS index
      |
      +--> data/artifacts/faiss.index
      +--> data/artifacts/faiss_ids.parquet

Runtime request flow
Client -> API -> (ANN candidates via FAISS) -> optional LTR reranker
      -> diversity (MMR) + freshness -> Response

Fallback to co-vis if ANN unavailable (unless `strict_ann=true`).
```

## Endpoints and Modes

- `GET /recs` (ANN-first)
  - From `context_item_id` or `user_id` last item + ANN neighbors; fallback to co-vis if ANN unavailable.
- `GET /recs/auto` (ANN-first)
  - Uses last item from user history; fallback to popularity.
- `GET /recs/personal` (ANN-first)
  - Blends recent items into a decayed query vector + ANN search; fallback to personalized co-vis.
- `GET /feed`
  - Sections: for_you (ANN), because_you_watched (ANN), trending (co-vis popularity).
- `GET /recs/semantic`
  - Pure ANN neighbors for a given `context_item_id` (co-vis fallback possible).
- `GET /recs/ranked`
  - Reranks candidates using the logistic LTR; supports `diversity=mmr&mmr_lambda=...` and `freshness_boost=true`.
- `GET /search/semantic`
  - Text + ANN item search.
- Reload
  - `POST /reload_covis`, `POST /reload_faiss`, `POST /reload_ranker`.

## Scores

- ANN: `score` is cosine similarity (inner product on L2-normalized vectors), typically in [0, 1].
- Co-vis: `score` is association-rule score (confidence / lift) when available; falls back to `count`.
- Reranked: `rerank_score` is the learned score from the LTR model (0–1 from sigmoid).
- `semantic_score` is included on many results:
  - ANN paths: `semantic_score == score`.
  - Fallback paths: exact cosine computed per item using the appropriate query vector.

## Artifacts

- Sessions: `data/sessions/sessions.parquet`
- Co-vis: `data/artifacts/covis.parquet` (CSV fallback)
- Catalog: `data/catalog/items.csv|.parquet` (see sample `data/catalog/items.sample.csv`)
- Embeddings: `data/artifacts/item_embeddings.parquet`
- FAISS: `data/artifacts/faiss.index`, `data/artifacts/faiss_ids.parquet`
- Ranker weights: `data/artifacts/ranker_weights.json`

## Build/Run Checklist

1) Sessionize: `python processing/sessionization/job.py`
2) Co-vis: `python scripts/build_covis.py`
3) Catalog + embeddings: `python scripts/backfill_embeddings.py`
4) FAISS: `python scripts/build_faiss.py`
5) Train LTR (optional): `python ranker/train_simple.py`
6) Reload in API: `POST /reload_covis`, `POST /reload_faiss`, `POST /reload_ranker`
7) Verify endpoints: `/recs/semantic`, `/recs/ranked`

## Key Internals

- ANN helpers in API:
  - `_ann_from_item(context_item_id, n)`, `_ann_from_recent(user_id, n, ctx_k, alpha)`
  - `_recent_query_vec(user_id, ctx_k, alpha)` builds decayed blend vector
  - `_annotate_semantic(results, v)` adds exact per-item cosine to `semantic_score`
- Co-vis helpers:
  - `_popular_from_covis(k)` aggregates by score if present, else count
  - `_personalized_covis(user_id, n, ctx_k, alpha)` weighted neighbors by recent items
- Reranker:
  - `ranker/train_simple.py` builds features (semantic, covis, popularity, freshness) and trains logistic weights
  - API endpoint `/recs/ranked` scores with weights, then applies optional MMR for diversity

## Troubleshooting

- ANN scores look like counts (big integers)
  - ANN fell back. Build FAISS, reload, or use `strict_ann=true` to fail instead of fallback.
- `/search/semantic` returns 503
  - SentenceTransformer model not loaded. Install `sentence-transformers` and ensure weights are available.
- FAISS index blank
  - Rebuild embeddings; ensure they are valid (see `scripts/backfill_embeddings.py`), then rebuild FAISS; check build summary.
- Reranker not applied
  - Train weights (`python ranker/train_simple.py`) and `POST /reload_ranker`.

