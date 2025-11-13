from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator
from datetime import datetime
from pathlib import Path
import json
import pandas as pd

from retrieval.covis.covis_services import CoVis
from ranker.inferences import rank_simple

# --- HYBRID: embeddings + FAISS + fuse ---
import numpy as np
from embedding.models import embed_item_id, embed_text, DIM   # <-- fixed 'embeddings'
from retrieval.vector_store import FaissStore
from candidate_generation.hybrid import fuse_sources

app = FastAPI(title="Recs API (Day 1+)", version="1.3.0")

# ---------- Paths ----------
RAW_DIR = Path("data/raw_events"); RAW_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_PATH = RAW_DIR / "events.jsonl"
SESS_PARQUET = Path("data/sessions/sessions.parquet")

# ---------- State ----------
COVIS = CoVis(path="data/artifacts/covis.parquet")
FAISS = FaissStore()

# ---------- Models ----------
class TrackEvent(BaseModel):
    user_id: str
    item_id: str | None = None
    event_type: str      # 'view' | 'click'
    ts: float | None = None

    @field_validator("event_type")
    @classmethod
    def _check_evt(cls, v: str):
        allowed = {"view", "click"}
        if v not in allowed:
            raise ValueError(f"event_type must be one of {sorted(allowed)}")
        return v

class TrackBulk(BaseModel):
    events: list[TrackEvent]

# ---------- Helpers ----------
def _last_item_for_user(user_id: str) -> str | None:
    if SESS_PARQUET.exists():
        try:
            sess = pd.read_parquet(SESS_PARQUET)
            sub = sess[sess["user_id"].astype(str) == str(user_id)]
            if not sub.empty:
                row = sub.sort_values("end").iloc[-1]
                seq = row["item_seq"]
                if isinstance(seq, list) and seq:
                    return str(seq[-1])
        except Exception:
            pass
    if EVENTS_PATH.exists():
        try:
            df = pd.read_json(EVENTS_PATH, lines=True)
            df = df[(df["user_id"].astype(str) == str(user_id)) &
                    (df["event_type"].isin(["view", "click"]))].copy()
            if df.empty: return None
            df["ts"] = pd.to_datetime(df["ts"], unit="s")
            return str(df.sort_values("ts").iloc[-1]["item_id"])
        except Exception:
            return None
    return None

def _recent_items_for_user(user_id: str, k: int = 10) -> list[str]:
    if SESS_PARQUET.exists():
        try:
            sess = pd.read_parquet(SESS_PARQUET)
            sub = sess[sess["user_id"].astype(str) == str(user_id)]
            if not sub.empty:
                sub = sub.sort_values("end", ascending=False)
                items: list[str] = []
                for seq in sub["item_seq"].tolist():
                    if isinstance(seq, list):
                        items.extend([str(x) for x in seq if x])
                    if len(items) >= k: break
                dedup = []
                for x in items:
                    if not dedup or dedup[-1] != x: dedup.append(x)
                return dedup[:k]
        except Exception:
            pass
    if EVENTS_PATH.exists():
        try:
            df = pd.read_json(EVENTS_PATH, lines=True)
            df = df[(df["user_id"].astype(str) == str(user_id)) &
                    (df["event_type"].isin(["view", "click"]))].copy()
            if df.empty: return []
            df["ts"] = pd.to_datetime(df["ts"], unit="s")
            items = df.sort_values("ts", ascending=False)["item_id"].astype(str).tolist()
            dedup = []
            for x in items:
                if not dedup or dedup[-1] != x: dedup.append(x)
            return dedup[:k]
        except Exception:
            return []
    return []

def _popular_from_covis(k: int = 20):
    try:
        df = getattr(COVIS, "df", None)
        if df is None or df.empty: return []
        agg = df.groupby("item_id")["count"].sum().sort_values(ascending=False).head(k)
        return [{"item_id": str(i), "score": float(s)} for i, s in agg.items()]
    except Exception:
        return []

def _semantic_scores_for_vec(v: np.ndarray | None, k: int) -> dict[str, float]:
    """Return a mapping item_id -> cosine similarity for a given query vector.
    Returns empty mapping if ANN is unavailable.
    """
    try:
        if v is None or getattr(FAISS, "index", None) is None:
            return {}
        v = np.asarray(v, dtype=np.float32)
        ann = FAISS.search_vec(v, k=k)
        return {str(r.get("item_id")): float(r.get("score", 0.0)) for r in ann}
    except Exception:
        return {}

def _annotate_semantic(results: list[dict], v: np.ndarray | None) -> list[dict]:
    """Augment each result with semantic_score computed as exact cosine(v, embed_item_id(item)).
    Returns semantic_score=None if the embedding model isn't available.
    """
    if not results:
        return results
    try:
        if v is None:
            # embedding model unavailable
            return [dict(r, semantic_score=None) for r in results]
        # normalize query vector
        vq = np.asarray(v, dtype=np.float32)
        nq = np.linalg.norm(vq)
        if nq == 0:
            return [dict(r, semantic_score=None) for r in results]
        vq = vq / nq
        cache: dict[str, float | None] = {}
        out = []
        for r in results:
            rr = dict(r)
            iid = str(rr.get("item_id"))
            if iid in cache:
                rr["semantic_score"] = cache[iid]
            else:
                vi = embed_item_id(iid)
                if vi is None:
                    sim = None
                else:
                    vi = np.asarray(vi, dtype=np.float32)
                    ni = np.linalg.norm(vi)
                    sim = float(np.dot(vq, vi / ni)) if ni != 0 else None
                cache[iid] = sim
                rr["semantic_score"] = sim
            out.append(rr)
        return out
    except Exception:
        return [dict(r, semantic_score=None) for r in results]

def _recent_query_vec(user_id: str, ctx_k: int = 5, alpha: float = 0.9) -> np.ndarray | None:
    """Build a decayed blend embedding from the user's recent items."""
    try:
        items = _recent_items_for_user(user_id, k=ctx_k)
        if not items:
            return None
        vecs: list[np.ndarray] = []
        for rank, iid in enumerate(items):
            v = embed_item_id(iid)
            if v is None:
                continue
            w = alpha ** rank
            vecs.append((w * np.asarray(v, dtype=np.float32)))
        if not vecs:
            return None
        q = np.sum(vecs, axis=0).astype(np.float32)
        return q
    except Exception:
        return None

def _ann_from_item(context_item_id: str, n: int = 20):
    """ANN neighbors for a single item using SentenceTransformer embeddings.
    Returns FAISS cosine similarity scores (higher is better).
    """
    try:
        if not context_item_id:
            return []
        v = embed_item_id(context_item_id)
        if v is None or getattr(FAISS, "index", None) is None:
            return []
        v = np.asarray(v, dtype=np.float32)
        ann = FAISS.search_vec(v, k=max(5 * n, 100))
        # drop the query item itself if present
        out = [r for r in ann if str(r.get("item_id")) != str(context_item_id)]
        return out[:n]
    except Exception:
        return []

def _ann_from_recent(user_id: str, n: int = 20, ctx_k: int = 5, alpha: float = 0.9):
    """ANN recommendations from a user’s recent items.
    Build a weighted (decayed) query vector from recent items and search FAISS.
    Returns FAISS cosine similarity scores.
    """
    try:
        q = _recent_query_vec(user_id=user_id, ctx_k=ctx_k, alpha=alpha)
        if q is None or getattr(FAISS, "index", None) is None:
            return []
        # we still need to exclude seen items
        items = _recent_items_for_user(user_id, k=ctx_k)
        seen = set(items)
        ann = FAISS.search_vec(q, k=max(5 * n, 100))
        out = [r for r in ann if str(r.get("item_id")) not in seen]
        return out[:n]
    except Exception:
        return []

def _personalized_covis(user_id: str, n: int = 20, ctx_k: int = 5, alpha: float = 0.7):
    ctx_items = _recent_items_for_user(user_id, k=ctx_k)
    if not ctx_items: return []
    seen = set(ctx_items)
    scores: dict[str, float] = {}
    for rank, item in enumerate(ctx_items):
        w = alpha ** rank
        neighbors = COVIS.topk_for_item(item, k=200)
        for nb in neighbors:
            iid = str(nb["item_id"])
            if iid in seen: continue
            scores[iid] = scores.get(iid, 0.0) + w * float(nb.get("score", 0.0))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"item_id": k, "score": float(v)} for k, v in ranked]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/track")
def track(ev: TrackEvent):
    payload = ev.model_dump()
    payload["ts"] = float(payload.get("ts") or datetime.utcnow().timestamp())
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"ok": True, "written": str(EVENTS_PATH)}

@app.post("/track/bulk")
def track_bulk(body: TrackBulk):
    now = datetime.utcnow().timestamp()
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        for ev in body.events:
            payload = ev.model_dump()
            payload["ts"] = float(payload.get("ts") or now)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"ok": True, "count": len(body.events), "written": str(EVENTS_PATH)}

@app.post("/reload_covis")
def reload_covis():
    try:
        COVIS.reload()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_faiss")
def reload_faiss():
    try:
        FAISS.reload()
        ok = getattr(FAISS, "index", None) is not None and bool(getattr(FAISS, "ids", []))
        ntotal = int(getattr(getattr(FAISS, "index", None), "ntotal", 0) or 0)
        return {"ok": ok, "ntotal": ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Status / Readiness ----
@app.get("/status/ann")
def status_ann(context_item_id: str | None = None, user_id: str | None = None, debug: bool = False, n: int = 5):
    """Report ANN readiness (SentenceTransformer + FAISS) and optional debug samples.

    Query params:
      - context_item_id: optional; if provided with debug=true, returns a sample ANN/fallback for this context
      - user_id: optional; if provided with debug=true, returns a sample personal ANN/fallback
      - debug: include small example results (n items)
    """
    try:
        model_loaded = bool(DIM and DIM > 0)
        faiss_index_loaded = getattr(FAISS, "index", None) is not None
        faiss_ids_count = len(getattr(FAISS, "ids", []) or [])
        faiss_ntotal = int(getattr(getattr(FAISS, "index", None), "ntotal", 0) or 0)
        ann_ready = model_loaded and faiss_index_loaded and faiss_ids_count > 0 and faiss_ntotal > 0

        payload: dict[str, object] = {
            "model_loaded": model_loaded,
            "model_dim": int(DIM or 0),
            "faiss_index_loaded": faiss_index_loaded,
            "faiss_ids_count": faiss_ids_count,
            "faiss_ntotal": faiss_ntotal,
            "ann_ready": ann_ready,
            "endpoints": {
                "recs": {"strategy": "ann_first", "strict_ann": True},
                "recs_auto": {"strategy": "ann_first", "strict_ann": True},
                "recs_personal": {"strategy": "ann_first", "strict_ann": True},
                "recs_semantic": {"strategy": "ann_only", "strict_ann": True},
                "search_semantic": {"strategy": "ann_only"},
            },
        }

        if debug:
            dbg: dict[str, object] = {}
            if context_item_id:
                ann = _ann_from_item(context_item_id, n=n)
                if ann:
                    dbg["recs_semantic"] = {"mode": "semantic", "context_item_id": context_item_id, "results": ann}
                else:
                    cov = rank_simple(COVIS.topk_for_item(context_item_id, k=max(5 * n, 50)), n=n)
                    dbg["recs_semantic"] = {"mode": "covis_fallback", "context_item_id": context_item_id, "results": cov}
            if user_id:
                annp = _ann_from_recent(user_id=user_id, n=n, ctx_k=5, alpha=0.9)
                if annp:
                    dbg["recs_personal"] = {"mode": "personal_ann", "user_id": user_id, "results": annp}
                else:
                    covp = _personalized_covis(user_id=user_id, n=n, ctx_k=5, alpha=0.7)
                    dbg["recs_personal"] = {"mode": ("personal_covis_fallback" if covp else "no_signal"), "user_id": user_id, "results": covp}
            payload["debug"] = dbg

        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/covis")
def status_covis(debug: bool = False, item_id: str | None = None, k: int = 5):
    """Report Co-Vis table readiness and optional sample neighbors.

    Query params:
      - debug: include a small example (top-k neighbors) for the provided item_id or a sample item
      - item_id: optional; if provided with debug=true, returns its neighbors
      - k: number of neighbors to include in debug output
    """
    try:
        df = getattr(COVIS, "df", None)
        idx = getattr(COVIS, "_index", {})
        loaded = df is not None and not getattr(df, "empty", True)
        rows = int(len(df)) if loaded else 0
        items_indexed = int(len(idx)) if idx else 0
        parquet_path = str(getattr(COVIS, "path_parquet", ""))
        csv_path = str(getattr(COVIS, "path_csv", ""))

        # Base stats
        total_neighbors = int(sum((len(g) for g in (idx.values() if idx else [])))) if loaded else 0
        avg_neighbors = (float(total_neighbors) / items_indexed) if items_indexed > 0 else 0.0

        payload: dict[str, object] = {
            "loaded": loaded,
            "rows": rows,
            "items_indexed": items_indexed,
            "total_neighbors": total_neighbors,
            "avg_neighbors_per_item": avg_neighbors,
            "path_parquet": parquet_path,
            "path_csv": csv_path,
        }

        if debug and loaded:
            key = str(item_id) if item_id else (next(iter(idx.keys())) if idx else None)
            if key and key in idx:
                sample = idx[key].head(max(1, k))
                payload["debug"] = {
                    "item_id": key,
                    "neighbors": [
                        {"item_id": str(n), "score": float(c)}
                        for n, c in zip(sample["neighbor"], sample["count"])
                    ],
                }
            else:
                payload["debug"] = {"note": "no index or item not found"}

            # Add top-k items by degree and by total co-vis count
            try:
                deg_list = sorted(((k2, int(len(g2))) for k2, g2 in idx.items()), key=lambda x: x[1], reverse=True)
                cnt_list = sorted(((k2, float(g2["count"].sum())) for k2, g2 in idx.items()), key=lambda x: x[1], reverse=True)
                payload["debug"].update({
                    "top_by_degree": [
                        {"item_id": str(i), "degree": int(d)} for i, d in deg_list[: max(1, k)]
                    ],
                    "top_by_total_count": [
                        {"item_id": str(i), "total_count": float(s)} for i, s in cnt_list[: max(1, k)]
                    ],
                })
            except Exception:
                # keep status minimal if stats computation fails
                pass

        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- ANN-first (SentenceTransformer + FAISS) ----
@app.get("/recs")
def recs(context_item_id: str | None = None, user_id: str | None = None, n: int = 20, strict_ann: bool = False):
    n = max(1, min(n, 100))
    if not context_item_id and user_id:
        context_item_id = _last_item_for_user(user_id)
    if context_item_id:
        ann = _ann_from_item(context_item_id, n=n)
        if ann:
            for r in ann:
                r["semantic_score"] = r.get("score")
            return {
                "mode": "semantic_from_context" if not user_id else "semantic_from_user_last_item",
                "context_item_id": context_item_id,
                "results": ann,
            }
        if strict_ann:
            raise HTTPException(status_code=503, detail="ANN unavailable; strict_ann=true prevents fallback")
        # fallback to co-vis if ANN not available and not strict
        candidates = COVIS.topk_for_item(context_item_id, k=max(5 * n, 50))
        ranked = rank_simple(candidates, n=n)
        # annotate with semantic similarity if possible
        vq = embed_item_id(context_item_id)
        ranked = _annotate_semantic(ranked, vq)
        return {
            "mode": "covis_fallback_from_context" if not user_id else "covis_fallback_from_user_last_item",
            "context_item_id": context_item_id,
            "results": ranked,
            "hint": "ANN empty; using co-vis fallback" if not ranked else None,
        }
    pop = _popular_from_covis(k=n)
    if pop: return {"mode": "popular_fallback", "results": pop}
    raise HTTPException(status_code=404, detail="No context or popularity data available")

@app.get("/recs/auto")
def recs_auto(user_id: str = Query(...), n: int = 20, strict_ann: bool = False):
    n = max(1, min(n, 100))
    ctx = _last_item_for_user(user_id)
    if ctx:
        ann = _ann_from_item(ctx, n=n)
        if ann:
            for r in ann:
                r["semantic_score"] = r.get("score")
            return {"mode": "semantic_from_last_item", "context_item_id": ctx, "results": ann}
        if strict_ann:
            raise HTTPException(status_code=503, detail="ANN unavailable; strict_ann=true prevents fallback")
        cand = COVIS.topk_for_item(ctx, k=max(5*n, 50))
        ranked = rank_simple(cand, n=n)
        ranked = _annotate_semantic(ranked, embed_item_id(ctx))
        return {"mode": "covis_fallback_from_last_item", "context_item_id": ctx, "results": ranked}
    pop = _popular_from_covis(k=n)
    if pop: return {"mode": "popular_fallback", "results": pop}
    raise HTTPException(status_code=404, detail="No context or popularity data available")

@app.get("/recs/personal")
def recs_personal(user_id: str = Query(...), n: int = 20, ctx_k: int = 5, strict_ann: bool = False):
    n = max(1, min(n, 100))
    ann = _ann_from_recent(user_id=user_id, n=n, ctx_k=ctx_k, alpha=0.9)
    if ann:
        for r in ann:
            r["semantic_score"] = r.get("score")
        return {"mode": "personal_ann", "user_id": user_id, "results": ann}
    if strict_ann:
        raise HTTPException(status_code=503, detail="ANN unavailable; strict_ann=true prevents fallback")
    # fallback to covis-based personalization if ANN empty and not strict
    res = _personalized_covis(user_id=user_id, n=n, ctx_k=ctx_k, alpha=0.7)
    if res:
        vq = _recent_query_vec(user_id=user_id, ctx_k=ctx_k, alpha=0.9)
        res = _annotate_semantic(res, vq)
        return {"mode": "personal_covis_fallback", "user_id": user_id, "results": res}
    pop = _popular_from_covis(k=n)
    if pop: return {"mode": "popular_fallback", "user_id": user_id, "results": pop}
    raise HTTPException(status_code=404, detail="No history/popularity signal")

# ---- NEW: Home feed (user_id only) ----
@app.get("/feed")
def home_feed(user_id: str = Query(...), n: int = 20, ctx_k: int = 5):
    n = max(1, min(n, 100))
    for_you = _ann_from_recent(user_id=user_id, n=n, ctx_k=ctx_k, alpha=0.9)
    for r in for_you:
        r["semantic_score"] = r.get("score")
    last_item = _last_item_for_user(user_id)
    by_last = _ann_from_item(last_item, n=n) if last_item else []
    for r in by_last:
        r["semantic_score"] = r.get("score")
    trending = _popular_from_covis(k=n)
    return {"user_id": user_id, "last_item": last_item,
            "sections": {"for_you": for_you, "because_you_watched": by_last, "trending": trending}}

# ---- Semantic search (vector) ----
@app.get("/search/semantic")
def search_semantic(q: str = Query(...), k: int = 20):
    v = embed_text(q)
    if v is None:
        raise HTTPException(status_code=503, detail="ANN unavailable: embedding model not loaded")
    v = v.astype(np.float32)
    ann = FAISS.search_vec(v, k=k)
    for r in ann:
        r["semantic_score"] = r.get("score")
    return {"query": q, "results": ann}

# ---- Hybrid recs (co-vis + ANN) ----
@app.get("/recs/semantic")
def recs_semantic(context_item_id: str = Query(...), n: int = 20, strict_ann: bool = False):
    n = max(1, min(n, 100))
    # Switch to pure ANN (semantic) scores; keep co-vis as fallback
    ann = _ann_from_item(context_item_id, n=n)
    if ann:
        for r in ann:
            r["semantic_score"] = r.get("score")
        return {"mode": "semantic", "context_item_id": context_item_id, "results": ann}
    # fallback
    if strict_ann:
        raise HTTPException(status_code=503, detail="ANN unavailable; strict_ann=true prevents fallback")
    candidates = COVIS.topk_for_item(context_item_id, k=max(5 * n, 50))
    ranked = rank_simple(candidates, n=n)
    ranked = _annotate_semantic(ranked, embed_item_id(context_item_id))
    return {"mode": "covis_fallback", "context_item_id": context_item_id, "results": ranked}

@app.get("/recs/hybrid")
def recs_hybrid(context_item_id: str = Query(...), n: int = 20):
    # Backward-compat: redirect to the new semantic route
    n = max(1, min(n, 100))
    url = f"/recs/semantic?context_item_id={context_item_id}&n={n}"
    return RedirectResponse(url=url, status_code=307)
