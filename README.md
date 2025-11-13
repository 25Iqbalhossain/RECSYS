# Recommendation-System

API Endpoints (quick reference)
- `GET /recs` — ANN-first from context item or user’s last item; returns SentenceTransformer/FAISS similarity scores. Falls back to co-vis if ANN unavailable.
- `GET /recs/auto?user_id=...` — ANN from last item; co-vis fallback.
- `GET /recs/personal?user_id=...` — ANN from recent items (decayed blend); co-vis fallback.
- `GET /feed?user_id=...` — Sections use ANN (for_you, because_you_watched) and co-vis popularity for trending.
- `GET /recs/semantic?context_item_id=...` — Pure semantic neighbors from an item; co-vis fallback.
- `GET /search/semantic?q=...` — Semantic search over items; returns FAISS cosine similarity scores.
- `POST /track` and `POST /track/bulk` — Ingest events.
- `POST /reload_faiss` — Reload FAISS index and ids without restarting the API.
- `GET /status/ann[?debug=true&context_item_id=...&user_id=...]` — ANN readiness (model loaded, FAISS ntotal) with optional sample outputs.
- `GET /status/covis[?debug=true&item_id=...&k=5]` — Co-Vis table readiness with optional sample neighbors.
  - Debug also includes: top-by-degree, top-by-total-count, and average neighbors per item.

Setup for ANN
- Build item embeddings: `python scripts/backfill_embeddings.py`
- Build FAISS index: `python scripts/build_faiss.py`

Notes
- Scores from ANN endpoints are cosine similarity (higher is better). Co-vis fallbacks return counts.
- All ANN-first endpoints accept `strict_ann=true` to disable co-vis fallback and return HTTP 503 when ANN is unavailable.
