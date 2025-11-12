from fastapi import FastAPI, Query, HTTPException
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

app = FastAPI(title="Recs API (Day 1+)", version="1.2.1")

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

# ---- Co-vis only ----
@app.get("/recs")
def recs(context_item_id: str | None = None, user_id: str | None = None, n: int = 20):
    n = max(1, min(n, 100))
    if not context_item_id and user_id:
        context_item_id = _last_item_for_user(user_id)
    if context_item_id:
        candidates = COVIS.topk_for_item(context_item_id, k=max(5 * n, 50))
        ranked = rank_simple(candidates, n=n)
        return {
            "mode": "explicit" if not user_id else "from_user_last_item",
            "context_item_id": context_item_id,
            "results": ranked,
            "hint": "empty means no neighbors in covis table" if not ranked else None,
        }
    pop = _popular_from_covis(k=n)
    if pop: return {"mode": "popular_fallback", "results": pop}
    raise HTTPException(status_code=404, detail="No context or popularity data available")

@app.get("/recs/auto")
def recs_auto(user_id: str = Query(...), n: int = 20):
    n = max(1, min(n, 100))
    ctx = _last_item_for_user(user_id)
    if ctx:
        cand = COVIS.topk_for_item(ctx, k=max(5*n, 50))
        ranked = rank_simple(cand, n=n)
        return {"mode": "last_item", "context_item_id": ctx, "results": ranked}
    pop = _popular_from_covis(k=n)
    if pop: return {"mode": "popular_fallback", "results": pop}
    raise HTTPException(status_code=404, detail="No context or popularity data available")

@app.get("/recs/personal")
def recs_personal(user_id: str = Query(...), n: int = 20, ctx_k: int = 5):
    n = max(1, min(n, 100))
    res = _personalized_covis(user_id=user_id, n=n, ctx_k=ctx_k, alpha=0.7)
    if res: return {"mode": "personal_covis", "user_id": user_id, "results": res}
    pop = _popular_from_covis(k=n)
    if pop: return {"mode": "popular_fallback", "user_id": user_id, "results": pop}
    raise HTTPException(status_code=404, detail="No history/popularity signal")

# ---- NEW: Home feed (user_id only) ----
@app.get("/feed")
def home_feed(user_id: str = Query(...), n: int = 20, ctx_k: int = 5):
    n = max(1, min(n, 100))
    for_you = _personalized_covis(user_id=user_id, n=n, ctx_k=ctx_k, alpha=0.7)
    last_item = _last_item_for_user(user_id)
    by_last = rank_simple(COVIS.topk_for_item(last_item, k=max(5*n, 50)), n=n) if last_item else []
    trending = _popular_from_covis(k=n)
    return {"user_id": user_id, "last_item": last_item,
            "sections": {"for_you": for_you, "because_you_watched": by_last, "trending": trending}}

# ---- Semantic search (vector) ----
@app.get("/search/semantic")
def search_semantic(q: str = Query(...), k: int = 20):
    v = embed_text(q).astype(np.float32)
    ann = FAISS.search_vec(v, k=k)
    return {"query": q, "results": ann}

# ---- Hybrid recs (co-vis + ANN) ----
@app.get("/recs/hybrid")
def recs_hybrid(context_item_id: str = Query(...), n: int = 20):
    n = max(1, min(n, 100))
    cov = COVIS.topk_for_item(context_item_id, k=max(5*n, 100))
    v = embed_item_id(context_item_id)
    ann = FAISS.search_vec(v, k=max(5*n, 100))
    fused = fuse_sources(cov, ann, seq_boost=[context_item_id], topn=5*n)
    ranked = rank_simple(fused, n=n)
    return {"mode": "hybrid", "context_item_id": context_item_id, "results": ranked}
