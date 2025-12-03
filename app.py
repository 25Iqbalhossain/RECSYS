# app.py — Neon-only, robust, pgvector if available else float8[] fallback
import json
import math
import re
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI
import psycopg
from dotenv import load_dotenv

load_dotenv()

# Try to import pgvector helpers but tolerate if missing
try:
    from pgvector.psycopg import register_vector, Vector as PgVector
    PGVECTOR_AVAILABLE = True
except Exception:
    register_vector = None
    PgVector = None
    PGVECTOR_AVAILABLE = False

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "local-model")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 768))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("❌ ERROR: DATABASE_URL is not set. Please set Neon connection string in env.")

MYGOV_JSON_PATH = os.environ.get("MYGOV_JSON_PATH", "search_service_dump.json")

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

DB_AVAILABLE = False
VECTOR_EXTENSION = False
USE_PGVECTOR = False

# =========================
# helpers: cleaning / display
# =========================
_WORD_RE = re.compile(r"[\w\u0980-\u09FF]+", flags=re.UNICODE)
_trail_re = re.compile(r"[\s\-\:\,\;\(\)\[\]\/\\]+$")

def clean_for_vocab(s: str) -> str:
    if not s:
        return ""
    s2 = " ".join(_WORD_RE.findall(s.lower()))
    return s2.strip()

def make_display_phrase(orig: str, max_len: int = 80) -> str:
    if not orig:
        return ""
    s = orig.strip()
    s = _trail_re.sub("", s)
    s = re.sub(r"\s*\([^)]{0,120}\)\s*$", "", s).strip()
    s = " ".join(s.split())
    if len(s) > max_len:
        cut = s.rfind(" ", 0, max_len)
        if cut == -1:
            s = s[:max_len].rstrip() + "…"
        else:
            s = s[:cut].rstrip() + "…"
    return s

# -------------------------
# PG Helper
# -------------------------
def get_pg_conn(connect_timeout: int = 5):
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=connect_timeout)

def init_schema_safe():
    """
    Create table:
      id bigserial PRIMARY KEY,
      bn_name text,
      en_name text,
      keywords text,
      profile text,
      embedding float8[] NOT NULL   (or VECTOR if available)
    """
    global DB_AVAILABLE, VECTOR_EXTENSION, USE_PGVECTOR

    try:
        with get_pg_conn() as conn:
            if PGVECTOR_AVAILABLE:
                try:
                    register_vector(conn)
                except Exception:
                    pass

            with conn.cursor() as cur:
                # try enable extension (may be no-op on managed)
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    VECTOR_EXTENSION = True
                except Exception:
                    VECTOR_EXTENSION = False

                if VECTOR_EXTENSION and PGVECTOR_AVAILABLE and PgVector is not None:
                    USE_PGVECTOR = True
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS mygov_services (
                            id BIGSERIAL PRIMARY KEY,
                            doc_id TEXT UNIQUE,
                            bn_name TEXT,
                            en_name TEXT,
                            keywords TEXT,
                            profile TEXT,
                            embedding VECTOR({EMBEDDING_DIM})
                        );
                        """
                    )
                    try:
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS mygov_services_embedding_idx
                            ON mygov_services USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                            """
                        )
                    except Exception:
                        pass
                else:
                    USE_PGVECTOR = False
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS mygov_services (
                            id BIGSERIAL PRIMARY KEY,
                            doc_id TEXT UNIQUE,
                            bn_name TEXT,
                            en_name TEXT,
                            keywords TEXT,
                            profile TEXT,
                            embedding FLOAT8[] NOT NULL
                        );
                        """
                    )
                    cur.execute("CREATE INDEX IF NOT EXISTS mygov_services_text_idx ON mygov_services (bn_name, en_name, keywords);")

        DB_AVAILABLE = True
        print("DB connected. VECTOR_EXTENSION =", VECTOR_EXTENSION, "USE_PGVECTOR =", USE_PGVECTOR)
        return True

    except Exception as e:
        DB_AVAILABLE = False
        VECTOR_EXTENSION = False
        USE_PGVECTOR = False
        print("DB init error:", e)
        return False

# =========================
# Embeddings
# =========================
def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    emb = resp.data[0].embedding
    if len(emb) > EMBEDDING_DIM:
        emb = emb[:EMBEDDING_DIM]
    elif len(emb) < EMBEDDING_DIM:
        emb = emb + [0.0] * (EMBEDDING_DIM - len(emb))
    return list(emb)

def _l2_normalize(vec: List[float]) -> List[float]:
    s = 0.0
    for v in vec:
        s += float(v) * float(v)
    if s == 0.0:
        return [0.0] * len(vec)
    norm = math.sqrt(s)
    return [float(v) / norm for v in vec]

# =========================
# Lexical loading (improved)
# =========================
@st.cache_data
def load_search_documents(path: str):
    docs = []
    try:
        raw = open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        return docs

    if not raw:
        return docs

    rows = []
    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                rows = parsed
            else:
                rows = [parsed]
        except Exception:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    for r in rows:
        bn = (r.get("my_gov_service_name") or r.get("name") or r.get("bn") or "").strip()
        en = (r.get("my_gov_service_name_en") or r.get("name_en") or r.get("en") or "").strip()
        kw = (r.get("my_gov_service_keyword") or r.get("keyword") or r.get("keywords") or "").strip()
        text = " ".join([x for x in [bn, en, kw] if x]).strip()
        if not text:
            continue

        docs.append({
            "bn": bn,
            "en": en,
            "keyword": kw,
            "text": text,
            "bn_clean": clean_for_vocab(bn),
            "en_clean": clean_for_vocab(en),
            "text_clean": clean_for_vocab(text),
            "bn_display": make_display_phrase(bn),
            "en_display": make_display_phrase(en),
        })

    return docs

def build_lexical_models(docs):
    prefix_map_bn = defaultdict(Counter)
    prefix_map_en = defaultdict(Counter)
    bigram_bn = defaultdict(Counter)
    bigram_en = defaultdict(Counter)
    vocab_bn = set()
    vocab_en = set()
    all_bn = set()
    all_en = set()
    allowed_bn_phrases = set()
    allowed_en_phrases = set()

    for d in docs:
        bn = (d.get("bn") or "").strip()
        en = (d.get("en") or "").strip()
        bn_clean = (d.get("bn_clean") or "").strip()
        en_clean = (d.get("en_clean") or "").strip()
        bn_disp = d.get("bn_display") or bn
        en_disp = d.get("en_display") or en

        if bn:
            all_bn.add(bn_disp)
            allowed_bn_phrases.add(bn_disp)
            words = bn_clean.split() if bn_clean else []
            for w in words:
                vocab_bn.add(w)
            for i in range(1, min(3, len(words)) + 1):
                prefix = " ".join(words[:i])
                prefix_map_bn[prefix][bn_disp] += 1
            for w1, w2 in zip(words, words[1:]):
                bigram_bn[w1][w2] += 1

        if en:
            all_en.add(en_disp)
            allowed_en_phrases.add(en_disp)
            words = en_clean.split() if en_clean else []
            for w in words:
                vocab_en.add(w)
            for i in range(1, min(3, len(words)) + 1):
                prefix = " ".join(words[:i])
                prefix_map_en[prefix][en_disp] += 1
            for w1, w2 in zip(words, words[1:]):
                bigram_en[w1][w2] += 1

    prefix_bn_map = {p: [x for x, _ in c.most_common()] for p, c in prefix_map_bn.items()}
    prefix_en_map = {p: [x for x, _ in c.most_common()] for p, c in prefix_map_en.items()}
    next_bn_map = {w: [x for x, _ in c.most_common()] for w, c in bigram_bn.items()}
    next_en_map = {w: [x for x, _ in c.most_common()] for w, c in bigram_en.items()}

    return {
        "prefix_bn": prefix_bn_map,
        "prefix_en": prefix_en_map,
        "all_bn": sorted(all_bn),
        "all_en": sorted(all_en),
        "next_bn": next_bn_map,
        "next_en": next_en_map,
        "vocab_bn": sorted(vocab_bn),
        "vocab_en": sorted(vocab_en),
        "allowed_bn_phrases": allowed_bn_phrases,
        "allowed_en_phrases": allowed_en_phrases,
    }

@st.cache_data
def get_lexical_data(version: int = 3):
    docs = load_search_documents(MYGOV_JSON_PATH)
    return build_lexical_models(docs)

def is_bangla(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in text)

def suggest_queries(query: str, limit: int = 5):
    data = get_lexical_data()
    prefix_bn = data["prefix_bn"]
    prefix_en = data["prefix_en"]
    next_bn = data["next_bn"]
    next_en = data["next_en"]
    vocab_bn = data["vocab_bn"]
    vocab_en = data["vocab_en"]
    allowed_bn_phrases = data["allowed_bn_phrases"]
    allowed_en_phrases = data["allowed_en_phrases"]

    q_orig = (query or "").strip()
    if not q_orig:
        return []
    q = q_orig.lower()

    use_bn = is_bangla(q_orig)
    vocab = set(vocab_bn) if use_bn else set(vocab_en)
    next_dict = next_bn if use_bn else next_en
    prefix_dict = prefix_bn if use_bn else prefix_en
    all_phrases = allowed_bn_phrases if use_bn else allowed_en_phrases

    suggestions = []
    seen = set()

    words = re.findall(r"[\w\u0980-\u09FF]+", q)

    # 1) last-word completion from vocab mapped to allowed phrases
    if words:
        last = words[-1]
        completions = sorted(w for w in vocab if w.startswith(last) and w != last)
        for comp in completions:
            prefix = " ".join(words[:-1] + [comp])
            # exact prefix-based phrases
            if prefix in prefix_dict:
                for ph in prefix_dict[prefix]:
                    if ph not in seen and ph in all_phrases:
                        suggestions.append(ph)
                        seen.add(ph)
                        if len(suggestions) >= limit:
                            return suggestions
            else:
                # fallback: allowed phrases containing the token sequence
                p_sub = " ".join(words[:-1] + [comp])
                for ph in all_phrases:
                    if p_sub in ph.lower() and ph not in seen:
                        suggestions.append(ph)
                        seen.add(ph)
                        if len(suggestions) >= limit:
                            return suggestions

    # 2) bigram next-word -> phrase startswith
    if words:
        last = words[-1]
        if last in next_dict:
            for w2 in next_dict[last]:
                cand = q_orig + " " + w2
                for ph in all_phrases:
                    if ph.lower().startswith(cand.lower()) and ph not in seen:
                        suggestions.append(ph)
                        seen.add(ph)
                        if len(suggestions) >= limit:
                            return suggestions

    # 3) exact prefix completions
    if q in prefix_dict:
        for phrase in prefix_dict[q]:
            if phrase.lower() != q and phrase in all_phrases and phrase not in seen:
                suggestions.append(phrase)
                seen.add(phrase)
                if len(suggestions) >= limit:
                    return suggestions

    # 4) fallback: allowed phrases containing query (rank startswith first)
    if len(suggestions) < limit:
        candidates = []
        ql = q.lower()
        for ph in all_phrases:
            pl = ph.lower()
            if pl.startswith(ql):
                candidates.append((0, len(pl), ph))
            elif ql in pl:
                candidates.append((1, len(pl), ph))
        candidates.sort()
        for _, _, ph in candidates:
            if ph not in seen:
                suggestions.append(ph)
                seen.add(ph)
                if len(suggestions) >= limit:
                    break

    return suggestions[:limit]

# =========================
# Upsert / index dataset
# =========================
def upsert_services(rows: List[Tuple[str, str, str, str, List[float]]]):
    """
    rows: (doc_id, bn, en, kw, embedding)
    Normalize embedding before storing.
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, bn, en, kw, emb in rows:
                try:
                    emb_norm = _l2_normalize(emb)
                    if USE_PGVECTOR and PGVECTOR_AVAILABLE and PgVector is not None:
                        # psycopg + pgvector will accept PgVector as param
                        vec = PgVector(emb_norm)
                        cur.execute(
                            """
                            INSERT INTO mygov_services (doc_id, bn_name, en_name, keywords, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (doc_id) DO UPDATE
                            SET bn_name = EXCLUDED.bn_name,
                                en_name = EXCLUDED.en_name,
                                keywords = EXCLUDED.keywords,
                                embedding = EXCLUDED.embedding;
                            """,
                            (doc_id, bn, en, kw, vec),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO mygov_services (doc_id, bn_name, en_name, keywords, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (doc_id) DO UPDATE
                            SET bn_name = EXCLUDED.bn_name,
                                en_name = EXCLUDED.en_name,
                                keywords = EXCLUDED.keywords,
                                embedding = EXCLUDED.embedding;
                            """,
                            (doc_id, bn, en, kw, emb_norm),
                        )
                except Exception as e:
                    print(f"[!] Upsert error for {doc_id}: {e}")

def prepare_docs(rows: List[Dict]) -> List[Tuple[str, str, str, str, str]]:
    docs = []
    for i, row in enumerate(rows):
        bn = (row.get("my_gov_service_name") or row.get("name") or "").strip()
        en = (row.get("my_gov_service_name_en") or row.get("name_en") or "").strip()
        kw = (row.get("my_gov_service_keyword") or row.get("keyword") or row.get("keywords") or "").strip()
        text = " ".join([x for x in [bn, en, kw] if x]).strip()
        if not text:
            continue
        doc_id = f"row_{i}"
        docs.append((doc_id, bn, en, kw, text))
    return docs

def index_dataset(batch_size: int = 64):
    print("[*] init schema ...")
    init_schema_safe()
    print("[*] loading data ...")
    rows = load_search_documents(MYGOV_JSON_PATH)
    print(f"[*] rows loaded: {len(rows)}")
    # convert to tuples for embedding
    docs_tuples = prepare_docs(rows)
    print(f"[*] docs prepared: {len(docs_tuples)}")
    total = len(docs_tuples)
    for start in range(0, total, batch_size):
        chunk = docs_tuples[start: start + batch_size]
        texts = [t[4] for t in chunk]
        print(f"[*] embedding batch {start}-{start+len(chunk)-1}")
        embeddings = []
        try:
            embeddings = client.embeddings.create(model=EMBEDDING_MODEL, input=texts).data
            # extract embeddings into list[list[float]]
            embs = []
            for item in embeddings:
                e = item.embedding
                if len(e) > EMBEDDING_DIM:
                    e = e[:EMBEDDING_DIM]
                elif len(e) < EMBEDDING_DIM:
                    e = e + [0.0] * (EMBEDDING_DIM - len(e))
                embs.append(list(e))
        except Exception as e:
            print("[!] Embedding API error:", e)
            break

        pg_rows = []
        for (doc_id, bn, en, kw, _), emb in zip(chunk, embs):
            pg_rows.append((doc_id, bn, en, kw, emb))
        upsert_services(pg_rows)
    print("[*] indexing complete")

# =========================
# Search implementation
# =========================
def search_vector(query_emb: List[float], query_text: str, top_k: int = 10):
    q_norm = _l2_normalize(query_emb)
    try:
        conn = get_pg_conn()
    except Exception as e:
        raise RuntimeError("Database not available: could not connect.") from e

    try:
        with conn:
            with conn.cursor() as cur:
                pattern = None
                if query_text:
                    tokens = re.findall(r"[A-Za-z\u0980-\u09FF]{3,}", query_text)
                    if tokens:
                        key = max(tokens, key=len)
                        pattern = f"%{key}%"
                    else:
                        pattern = f"%{query_text}%"

                rows = []
                if USE_PGVECTOR and PGVECTOR_AVAILABLE and PgVector is not None:
                    vec = PgVector(q_norm)
                    if pattern:
                        cur.execute(
                            """
                            SELECT id::text AS doc_id, bn_name, en_name, keywords, profile,
                                   1 - (embedding <=> %s) AS score
                            FROM mygov_services
                            WHERE COALESCE(en_name,'') ILIKE %s OR COALESCE(bn_name,'') ILIKE %s OR COALESCE(keywords,'') ILIKE %s
                            ORDER BY embedding <=> %s
                            LIMIT %s;
                            """,
                            (vec, pattern, pattern, pattern, vec, top_k)
                        )
                        rows = cur.fetchall()
                    if not rows:
                        cur.execute(
                            """
                            SELECT id::text AS doc_id, bn_name, en_name, keywords, profile,
                                   1 - (embedding <=> %s) AS score
                            FROM mygov_services
                            ORDER BY embedding <=> %s
                            LIMIT %s;
                            """,
                            (vec, vec, top_k)
                        )
                        rows = cur.fetchall()
                else:
                    # float8[] normalized stored => cosine = SUM(a*b)
                    if pattern:
                        cur.execute(
                            """
                            SELECT id::text AS doc_id, bn_name, en_name, keywords, profile,
                              (SELECT SUM(a*b)
                               FROM unnest(%s::double precision[]) WITH ORDINALITY AS qa(a, idx)
                               JOIN unnest(embedding) WITH ORDINALITY AS qe(b, idx) USING (idx)
                              )::double precision AS score
                            FROM mygov_services
                            WHERE COALESCE(en_name,'') ILIKE %s OR COALESCE(bn_name,'') ILIKE %s OR COALESCE(keywords,'') ILIKE %s
                            ORDER BY score DESC NULLS LAST
                            LIMIT %s;
                            """,
                            (q_norm, pattern, pattern, pattern, top_k)
                        )
                        rows = cur.fetchall()

                    if not rows:
                        cur.execute(
                            """
                            SELECT id::text AS doc_id, bn_name, en_name, keywords, profile,
                              (SELECT SUM(a*b)
                               FROM unnest(%s::double precision[]) WITH ORDINALITY AS qa(a, idx)
                               JOIN unnest(embedding) WITH ORDINALITY AS qe(b, idx) USING (idx)
                              )::double precision AS score
                            FROM mygov_services
                            ORDER BY score DESC NULLS LAST
                            LIMIT %s;
                            """,
                            (q_norm, top_k)
                        )
                        rows = cur.fetchall()

        # dedupe results by (bn_name,en_name) to avoid repeats
        seen = set()
        unique = []
        for doc_id, bn, en, kw, pf, score in rows:
            key = ((bn or "").strip().lower(), (en or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append((doc_id, bn, en, kw, pf, score))
            if len(unique) >= top_k:
                break
        return unique
    finally:
        try:
            conn.close()
        except Exception:
            pass

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="MyGov Neon Semantic Search", layout="wide")
st.title("MyGov Search (Neon + Embeddings)")

# DB init (store in session)
if "db_init" not in st.session_state:
    st.session_state["db_init"] = init_schema_safe()

if not st.session_state["db_init"]:
    st.error("Neon database is not reachable.")
else:
    st.success("Connected to Neon database.")

# Index button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Index dataset (recreate/update embeddings)"):
        st.info("Indexing started — this may take time for large files. Check console for logs.")
        index_dataset(batch_size=64)
        # clear cached lexical data so suggestions refresh
        get_lexical_data.clear()
        st.success("Indexing complete. Suggestions refreshed.")

with col2:
    st.write("Use the box below to search (suggestions come only from dataset).")

query = st.text_input("Search")

if query:
    suggestions = suggest_queries(query, limit=5)
    if suggestions:
        st.write("Suggestions:")
        for i, s in enumerate(suggestions):
            if st.button(s, key=f"sugg_{i}"):
                st.session_state["query"] = s

    try:
        emb = get_embedding(query)
        rows = search_vector(emb, query, top_k=10)

        if not rows:
            st.info("No results.")
        else:
            for doc_id, bn, en, kw, pf, score in rows:
                title = (bn or "").strip() or (en or "").strip() or "(No title)"
                title = make_display_phrase(title, max_len=120)
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                st.markdown(f"### {title}")
                st.caption(f"doc_id: {doc_id} • score: {score_text}")
                if kw:
                    st.write("Keywords:", kw)
                if pf:
                    st.write(pf)
                st.markdown("---")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Type a query to search.")
