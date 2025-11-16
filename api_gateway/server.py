from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator
from datetime import datetime
from pathlib import Path
import json
import os, time, random
import runpy
import pandas as pd

from retrieval.covis.covis_services import CoVis
from ranker.inferences import rank_simple

# --- HYBRID: embeddings + FAISS + fuse ---
import numpy as np
from embedding.models import embed_item_id, embed_text, DIM   # <-- fixed 'embeddings'
from retrieval.vector_store import FaissStore
from candidate_generation.hybrid import fuse_sources
from catalog.store import Catalog
from ranker.simple_ltr import load_weights, score_with_weights

app = FastAPI(title="Recs API (Day 1+)", version="1.4.0")

# Absolute project root (repo root)
ROOT = Path(__file__).resolve().parents[1]

# ---------- Paths ----------
RAW_DIR = (ROOT / "data" / "raw_events"); RAW_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_PATH = RAW_DIR / "events.jsonl"
SESS_PARQUET = ROOT / "data" / "sessions" / "sessions.parquet"

# ---------- State ----------
COVIS = CoVis(path="data/artifacts/covis.parquet")
FAISS = FaissStore()
CATALOG = Catalog()
RANKER = load_weights()

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
        col = "score" if "score" in df.columns else "count"
        agg = df.groupby("item_id")[col].sum().sort_values(ascending=False).head(k)
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
    mode = os.environ.get("AUTO_BOOTSTRAP", "1")
    try:
        if mode == "2":
            _maybe_bootstrap_demo(force=True)
        elif mode == "1":
            _maybe_bootstrap_demo(force=False)
    except Exception:
        pass
    return {"ok": True}


# ---------- Auto-bootstrap on app startup ----------
@app.on_event("startup")
def _on_startup_bootstrap():
    """Automatically (re)generate data on app start based on AUTO_BOOTSTRAP.
    AUTO_BOOTSTRAP:
      0 = off, 1 = only if missing (default), 2 = force regenerate
    """
    mode = os.environ.get("AUTO_BOOTSTRAP", "1")
    try:
        if mode == "2":
            _maybe_bootstrap_demo(force=True)
        elif mode == "1":
            _maybe_bootstrap_demo(force=False)
    except Exception:
        # Don't block startup if bootstrap fails
        pass

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

@app.post("/reload_ranker")
def reload_ranker():
    try:
        global RANKER
        RANKER = load_weights()
        return {"ok": RANKER is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Demo bootstrap (sample data) ----------
RAW_DIR = (ROOT / "data" / "raw_events"); RAW_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR = (ROOT / "data" / "catalog"); CAT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = (ROOT / "data" / "artifacts"); ART_DIR.mkdir(parents=True, exist_ok=True)

def _write_sample_catalog(n: int = 50) -> Path:
    path_csv = CAT_DIR / "items.csv"
    if path_csv.exists():
        return path_csv
    rows = []
    now = int(time.time())
    for i in range(1, n + 1):
        iid = f"item_{i}"
        title = f"Sample Item {i}"
        desc = f"Synthetic description for item {i}."
        created_ts = now - random.randint(0, 90) * 86400
        rows.append({"item_id": iid, "title": title, "desc": desc, "created_ts": created_ts})
    pd.DataFrame(rows).to_csv(path_csv, index=False)
    return path_csv

def _write_sample_events(users: int = 8, seq_min: int = 3, seq_max: int = 6) -> Path:
    events_path = RAW_DIR / "events.jsonl"
    if events_path.exists() and events_path.stat().st_size > 0:
        return events_path
    ids = CATALOG.all_ids() or [f"item_{i}" for i in range(1, 51)]
    now = int(time.time()) - 86400
    with open(events_path, "w", encoding="utf-8") as f:
        for u in range(1, users + 1):
            uid = f"U{u}"
            t = now + random.randint(0, 3600)
            for _ in range(random.randint(2, 5)):
                k = random.randint(seq_min, seq_max)
                seq = random.sample(ids, min(k, len(ids)))
                for iid in seq:
                    evt = {"user_id": uid, "item_id": iid, "event_type": "view", "ts": t}
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                    if random.random() < 0.3:
                        evt2 = {"user_id": uid, "item_id": iid, "event_type": "click", "ts": t + 1}
                        f.write(json.dumps(evt2, ensure_ascii=False) + "\n")
                    t += random.randint(5, 60)
                t += random.randint(300, 1800)
    return events_path

def _synthesize_events_from_merged() -> Path | None:
    """If events.jsonl is missing but merged history exists, synthesize events with timestamps.
    Reads data/artifacts/user_history_merged.jsonl where each line is {user_id, history: [item_ids]}.
    Generates view/click events per user with plausible timestamps and writes data/raw_events/events.jsonl.
    """
    merged = ART_DIR / "user_history_merged.jsonl"
    if EVENTS_PATH.exists() and EVENTS_PATH.stat().st_size > 0:
        return EVENTS_PATH
    if not merged.exists():
        return None
    rows = [json.loads(l) for l in merged.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    base = int(time.time()) - 2 * 86400
    with open(EVENTS_PATH, "w", encoding="utf-8") as f:
        for rec in rows:
            uid = str(rec.get("user_id"))
            hist = [str(x) for x in (rec.get("history") or []) if x]
            # assume most-recent-first; replay oldest to newest
            seq = list(reversed(hist))
            t = base + random.randint(0, 6*3600)
            for item in seq:
                evt = {"user_id": uid, "item_id": item, "event_type": "view", "ts": t}
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                if random.random() < 0.25:
                    evt2 = {"user_id": uid, "item_id": item, "event_type": "click", "ts": t + 1}
                    f.write(json.dumps(evt2, ensure_ascii=False) + "\n")
                t += random.randint(5, 90)
            base += random.randint(600, 3600)
    return EVENTS_PATH

def _run_pipeline_once() -> dict:
    # optional sample generators: external history + merge
    try:
        if not ((RAW_DIR / "user_history_external.jsonl").exists()):
            runpy.run_path("data/samples/history_dumy_data.py", run_name="__main__")
        runpy.run_path(str(ROOT / "data" / "samples" / "marge_events_and_histroy.py"), run_name="__main__")
    except Exception:
        # continue if samples not present
        pass
    try:
        from processing.sessionization import job as sess_job
        # ensure we have events; if missing but merged history exists, synthesize
        if not EVENTS_PATH.exists():
            _synthesize_events_from_merged()
        sess_job.run()
    except Exception as e:
        return {"ok": False, "step": "sessionize", "error": str(e)}
    try:
        import scripts.build_covis as bc
        bc.build_assoc(topk=100)
    except Exception as e:
        return {"ok": False, "step": "build_covis", "error": str(e)}
    try:
        import scripts.backfill_embeddings as be
        be.main()
    except Exception as e:
        return {"ok": False, "step": "backfill_embeddings", "error": str(e)}
    try:
        import scripts.build_faiss as bf
        bf.main()
    except Exception as e:
        return {"ok": False, "step": "build_faiss", "error": str(e)}
    # train simple reranker (optional)
    try:
        import ranker.train_simple as ts
        ts.main()
    except Exception:
        # continue even if training fails (e.g., small data)
        pass
    # reload artifacts
    try:
        COVIS.reload(); FAISS.reload()
        global RANKER
        RANKER = load_weights()
    except Exception:
        pass
    return {"ok": True}

def _clear_paths(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

def _maybe_bootstrap_demo(force: bool = False) -> dict:
    if force:
        _clear_paths([
            RAW_DIR / "events.jsonl",
            (ROOT / "data" / "sessions" / "sessions.parquet"),
            ART_DIR / "covis.parquet",
            ART_DIR / "item_embeddings.parquet",
            ART_DIR / "faiss.index",
            ART_DIR / "faiss_ids.parquet",
            (ART_DIR / "ranker_weights.json"),
        ])
    needed = [
        (CAT_DIR / "items.csv").exists(),
        (RAW_DIR / "events.jsonl").exists(),
        Path("data/sessions/sessions.parquet").exists(),
        (ART_DIR / "covis.parquet").exists(),
        (ART_DIR / "item_embeddings.parquet").exists(),
        (ART_DIR / "faiss.index").exists(),
    ]
    if all(needed) and not force:
        return {"ok": True, "skipped": True}
    _write_sample_catalog(); CATALOG.reload(); _write_sample_events()
    return _run_pipeline_once()

@app.post("/bootstrap_demo")
def bootstrap_demo(force: bool = False):
    try:
        return _maybe_bootstrap_demo(force=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/events")
def generate_events(source: str = "synthetic"):
    """Generate data/raw_events/events.jsonl on demand.
    - source=synthetic: writes random sessions using catalog item_ids
    - source=merged: synthesize events from data/artifacts/user_history_merged.jsonl if present
    Returns file path and approximate line count.
    """
    try:
        if source == "merged":
            p = _synthesize_events_from_merged()
            mode = "merged"
        else:
            p = _write_sample_events()
            mode = "synthetic"
        if not p or not p.exists():
            raise HTTPException(status_code=500, detail="Could not generate events.jsonl")
        # count lines
        try:
            n = sum(1 for _ in open(p, "r", encoding="utf-8"))
        except Exception:
            n = None
        return {"ok": True, "mode": mode, "path": str(p), "lines": n}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/paths")
def debug_paths():
    return {
        "root": str(ROOT),
        "raw_dir": str(RAW_DIR),
        "events_path": str(EVENTS_PATH),
        "sessions_path": str(SESS_PARQUET),
        "artifacts_dir": str(ART_DIR),
        "catalog_dir": str(CAT_DIR),
        "exists": {
            "events": EVENTS_PATH.exists(),
            "sessions": SESS_PARQUET.exists(),
            "covis": (ART_DIR / "covis.parquet").exists(),
            "embeddings": (ART_DIR / "item_embeddings.parquet").exists(),
            "faiss_index": (ART_DIR / "faiss.index").exists(),
            "faiss_ids": (ART_DIR / "faiss_ids.parquet").exists(),
            "ranker_weights": (ART_DIR / "ranker_weights.json").exists(),
        },
    }
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

# ---- Ranked Recs (LTR) ----
def _features_for_candidate(context_vec: np.ndarray | None, context_item_id: str | None, item_id: str) -> dict:
    f_sem = 0.0
    if context_vec is not None:
        vi = embed_item_id(item_id)
        if vi is not None:
            vq = context_vec.astype(np.float32)
            vi = vi.astype(np.float32)
            nq = np.linalg.norm(vq) or 1.0
            ni = np.linalg.norm(vi) or 1.0
            f_sem = float(np.dot(vq / nq, vi / ni))
    f_cov = 0.0
    if context_item_id:
        try:
            nbrs = {r["item_id"]: r["score"] for r in COVIS.topk_for_item(context_item_id, k=200)}
            f_cov = float(nbrs.get(str(item_id), 0.0))
        except Exception:
            pass
    f_pop = 0.0
    try:
        dfc = getattr(COVIS, "df", None)
        if dfc is not None and not dfc.empty:
            f_pop = float(dfc[dfc["neighbor"].astype(str) == str(item_id)]["count"].sum())
    except Exception:
        pass
    f_fresh = 0.0
    it = CATALOG.get(item_id) if CATALOG else None
    if it and it.created_ts:
        import math, time
        age_days = max(0.0, (time.time() - float(it.created_ts)) / 86400.0)
        f_fresh = math.exp(-age_days / 30.0)
    return {"semantic": f_sem, "covis": f_cov, "popularity": f_pop, "freshness": f_fresh}

def _apply_mmr(items: list[dict], lambda_mult: float = 0.8, k: int = 20) -> list[dict]:
    if not items:
        return items
    selected = []
    candidates = items[:]
    sims = {}
    def sim(a: dict, b: dict) -> float:
        key = (a["item_id"], b["item_id"]) if a["item_id"] <= b["item_id"] else (b["item_id"], a["item_id"])
        if key in sims:
            return sims[key]
        va = embed_item_id(a["item_id"]) or None
        vb = embed_item_id(b["item_id"]) or None
        if va is None or vb is None:
            s = 0.0
        else:
            va = va.astype(np.float32); vb = vb.astype(np.float32)
            na = np.linalg.norm(va) or 1.0; nb = np.linalg.norm(vb) or 1.0
            s = float(np.dot(va/na, vb/nb))
        sims[key] = s
        return s
    while candidates and len(selected) < k:
        best = None
        best_score = -1e9
        for c in candidates:
            rel = float(c.get("rerank_score") or c.get("score") or 0.0)
            div = 0.0 if not selected else max(sim(c, s) for s in selected)
            mmr = lambda_mult * rel - (1 - lambda_mult) * div
            if mmr > best_score:
                best_score = mmr
                best = c
        selected.append(best)
        candidates.remove(best)
    return selected

@app.get("/recs/ranked")
def recs_ranked(context_item_id: str | None = None, user_id: str | None = None, n: int = 20,
                diversity: str | None = None, mmr_lambda: float = 0.8, freshness_boost: bool = False,
                strict_ann: bool = False):
    n = max(1, min(n, 100))
    if not context_item_id and user_id:
        context_item_id = _last_item_for_user(user_id)
    ann = _ann_from_item(context_item_id, n=5*n) if context_item_id else []
    used_fallback = False
    if not ann:
        if strict_ann:
            raise HTTPException(status_code=503, detail="ANN unavailable; strict_ann=true prevents fallback")
        used_fallback = True
        cov = COVIS.topk_for_item(context_item_id, k=max(5*n, 50)) if context_item_id else _popular_from_covis(k=5*n)
        ann = rank_simple(cov, n=5*n)
    vq = embed_item_id(context_item_id) if context_item_id else None
    feat_names_weights = RANKER
    out = []
    for r in ann:
        iid = str(r.get("item_id"))
        feats = _features_for_candidate(vq, context_item_id, iid)
        base = float(r.get("score", 0.0))
        if freshness_boost:
            base += 0.05 * feats.get("freshness", 0.0)
        rerank = base
        if feat_names_weights:
            names, w, b = feat_names_weights
            rerank = score_with_weights(feats, names, w, b)
        out.append({"item_id": iid, "score": base, "semantic_score": feats.get("semantic"), "rerank_score": rerank})
    if diversity == "mmr":
        out = _apply_mmr(out, lambda_mult=mmr_lambda, k=n)
    out = out[:n]
    return {
        "mode": "ranked_ann" if not used_fallback else "ranked_covis_fallback",
        "context_item_id": context_item_id,
        "results": out,
        "flags": {"diversity": diversity, "mmr_lambda": mmr_lambda, "freshness_boost": freshness_boost},
    }

@app.get("/recs/hybrid")
def recs_hybrid(context_item_id: str = Query(...), n: int = 20):
    # Backward-compat: redirect to the new semantic route
    n = max(1, min(n, 100))
    url = f"/recs/semantic?context_item_id={context_item_id}&n={n}"
    return RedirectResponse(url=url, status_code=307)
