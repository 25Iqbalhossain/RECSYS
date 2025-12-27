from __future__ import annotations

"""
Minimal FastAPI app exposing only the MySQL-backed semantic APIs:

- GET /search/mysql_semantic   -> hybrid (keyword + vector) text search
- GET /recs/mysql_semantic     -> item-id based similar-items recs

The FAISS index and metadata are built by `scripts/build_index_from_mysql.py`
using the Hugging Face embedding backend configured in `data.testconfig`.
Vectors are L2-normalized and searched with inner product, so the FAISS
`score` is cosine similarity.
"""

from fastapi import FastAPI, HTTPException, Query

from data.testconfig import METADATA_PATH
from db.mysql_client import search_candidates_by_keyword
from index.faiss_search import load_metadata, search as mysql_faiss_search

app = FastAPI(title="MySQL Semantic Recs API", version="1.1.0")


def _mysql_text_for_item(row_id: int | str) -> str | None:
    """
    Look up the text field for a given MySQL `contents.id` from the
    JSONL metadata file used to build the FAISS index.

    We prefer to query FAISS using the same text that was embedded
    for this row instead of the raw numeric ID, which is not
    semantically meaningful.
    """
    try:
        for rec in load_metadata(str(METADATA_PATH)):
            if str(rec.get("row_id")) == str(row_id):
                text = rec.get("text") or ""
                return text or None
    except FileNotFoundError:
        return None
    except Exception:
        # Best-effort only; fall back to using the ID as text.
        return None
    return None


# ---- Semantic search over MySQL-backed index (hybrid) ----
@app.get("/search/mysql_semantic")
def search_mysql_semantic(
    q: str = Query(...),
    k: int = 20,
    min_score: float = 0.0,
    only_active: bool = True,
    type_filter: str | None = Query(None, alias="type"),
    language_filter: str | None = Query(None, alias="language"),
    hybrid: bool = True,
    w_vector: float = 1.0,
    w_keyword: float = 1.0,
):
    """
    Semantic text search over the MySQL-backed FAISS index.

    Pipeline:
    1) (Optional) Keyword filter in MySQL (LIKE on title/body) with
       business filters (is_active, type, language) to get candidates.
    2) Vector search in FAISS with the same query text.
    3) Combine vector similarity and a simple keyword score:
       combined_score = w_vector * vector_score + w_keyword * keyword_score.
    4) Filter by `min_score` on vector_score and return top-k by
       combined_score.

    Query params
    -----------
    - q : str
        Free-text query string (same embedding pipeline as index build).
    - k : int
        Number of results to return (1..100) after hybrid filtering.
    - min_score : float
        Minimum cosine similarity to keep (0.0–1.0). Results with
        `vector_score < min_score` are dropped.
    - only_active : bool
        If True, restrict candidates to rows where `is_active = 1`.
    - type : str | None
        Optional `type` filter (e.g., "মভ" for movies, "গন" for songs).
    - language : str | None
        Optional `language` filter (e.g., "বল" for Bangla).
    - hybrid : bool
        If True, apply keyword pre-filtering in MySQL (LIKE) before
        ranking with vectors. If False, use pure vector search.
    - w_vector : float
        Weight for vector (cosine) similarity in the combined score.
    - w_keyword : float
        Weight for keyword match score in the combined score.
    """
    k = max(1, min(k, 100))
    min_score = float(min_score)

    # 1) Keyword + business filters to narrow down candidate rows.
    keyword_scores: dict[str, float] = {}
    candidate_ids: set[str] | None = None
    if hybrid:
        rows = search_candidates_by_keyword(
            q=q,
            max_candidates=max(5 * k, 200),
            type_filter=type_filter,
            language_filter=language_filter,
            only_active=only_active,
        )
        candidate_ids = {str(r["id"]) for r in rows if "id" in r}

        # Simple keyword score: count of query tokens present in title/body.
        tokens = [t for t in str(q).split() if t]
        for r in rows:
            text = f"{r.get('title', '')} {r.get('body', '')}"
            kw = 0.0
            for tok in tokens:
                if tok and tok in text:
                    kw += 1.0
            keyword_scores[str(r["id"])] = kw

    # 2) Vector search on the full index (ANN).
    try:
        raw_results = mysql_faiss_search(
            q,
            k=max(5 * k, 100),
            metadata_path=str(METADATA_PATH),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 3) Combine vector + keyword scores and apply filters.
    results = []
    for r in raw_results:
        vec_score = float(r.get("score", 0.0))
        if vec_score < min_score:
            continue

        meta = r.get("metadata") or {}
        extra = meta.get("extra") or {}
        rid = extra.get("id") or meta.get("row_id") or meta.get("id")
        if rid is None:
            continue
        rid_str = str(rid)

        # If hybrid filtering is enabled, keep only candidates that passed
        # the keyword stage.
        if candidate_ids is not None and rid_str not in candidate_ids:
            continue

        kw_score = float(keyword_scores.get(rid_str, 0.0))
        combined = w_vector * vec_score + w_keyword * kw_score

        results.append(
            {
                "item_id": rid_str,
                "vector_score": vec_score,
                "keyword_score": kw_score,
                "combined_score": combined,
                "metadata": meta,
            }
        )

    # Final ranking by combined score (descending).
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    results = results[:k]

    return {
        "query": q,
        "k": k,
        "min_score": min_score,
        "only_active": only_active,
        "type": type_filter,
        "language": language_filter,
        "hybrid": hybrid,
        "w_vector": w_vector,
        "w_keyword": w_keyword,
        "results": results,
    }


# ---- MySQL-backed semantic recs (item -> similar items) ----
@app.get("/recs/mysql_semantic")
def recs_mysql_semantic(
    context_item_id: int = Query(...),
    n: int = 20,
    min_score: float = 0.0,
    only_active: bool = True,
    type_filter: str | None = Query(None, alias="type"),
    language_filter: str | None = Query(None, alias="language"),
):
    """
    Recommend similar items using the MySQL-backed FAISS index.

    `context_item_id` is the primary key from the MySQL `contents` table.
    The query vector is built from the same text that was embedded for
    this item during index construction.

    Business filters (is_active, type, language) are applied to neighbors
    using the metadata stored alongside the FAISS index.

    Query params
    -----------
    - context_item_id : int
        ID from `contents.id` to find neighbors for.
    - n : int
        Max number of similar items to return (1..100) after filtering.
    - min_score : float
        Minimum cosine similarity to keep (0.0–1.0). Results with
        `score < min_score` are dropped.
    - only_active : bool
        If True, only return neighbors where `is_active = 1` in metadata.
    - type : str | None
        Optional `type` filter for neighbors.
    - language : str | None
        Optional `language` filter for neighbors.
    """
    n = max(1, min(n, 100))
    min_score = float(min_score)
    try:
        # Prefer the same text that was embedded for this row_id when
        # building the FAISS index. Using only the numeric ID as text
        # (e.g. "123") is not semantically meaningful and leads to
        # poor neighbors.
        query_text = _mysql_text_for_item(context_item_id)
        if not query_text:
            # Fallback: use the ID string if metadata is missing.
            query_text = str(context_item_id)

        all_results = mysql_faiss_search(
            query_text,
            k=max(5 * n, 100),
            metadata_path=str(METADATA_PATH),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not all_results:
        raise HTTPException(
            status_code=404,
            detail="No ANN neighbors for this context_item_id",
        )

    out = []
    for r in all_results:
        meta = r.get("metadata") or {}
        extra = meta.get("extra") or {}

        # Prefer contents.id from extra; fall back to row_id / id if needed.
        rid = extra.get("id")
        if rid is None:
            rid = meta.get("row_id") or meta.get("id")

        # Skip if we cannot resolve a meaningful neighbor ID or if it
        # is the same as the query item.
        if rid is None or str(rid) == str(context_item_id):
            continue

        score = float(r.get("score", 0.0))
        if score < min_score:
            continue

        # Business filters: only_active, type, language from metadata.extra.
        if only_active and extra.get("is_active") not in (1, "1", True, "true", "True"):
            continue
        if type_filter is not None and str(extra.get("type")) != str(type_filter):
            continue
        if language_filter is not None and str(extra.get("language")) != str(language_filter):
            continue

        out.append(
            {
                "item_id": str(rid),
                # Cosine-similarity style score exposed by mysql_faiss_search.
                "score": score,
                "metadata": meta,
            }
        )
        if len(out) >= n:
            break

    if not out:
        raise HTTPException(
            status_code=404,
            detail="ANN neighbors found but none met min_score or business filters",
        )

    return {
        "mode": "mysql_semantic",
        "context_item_id": str(context_item_id),
        "min_score": min_score,
        "only_active": only_active,
        "type": type_filter,
        "language": language_filter,
        "results": out,
    }

