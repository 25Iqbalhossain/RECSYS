import json
import os
import re
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from openai import OpenAI
import psycopg
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "local-model")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 768))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("ERROR: DATABASE_URL is not set. Please set Neon/Aiven connection string in env.")

# NEW JSON FORMAT (name, name_en, keyword)
MYGOV_JSON_PATH = os.environ.get("MYGOV_JSON_PATH", "search_service_dump.json")

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

# =========================
# PG helper (float8[] embeddings)
# =========================

def get_pg_conn(connect_timeout: int = 5):
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=connect_timeout)


def init_schema():
    """
    Mirror app.py: create mygov_services with FLOAT8[] embeddings,
    matching the table populated by txtai-search.py / app.py.
    """
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS mygov_services (
                id          bigserial PRIMARY KEY,
                doc_id      text UNIQUE,
                bn_name     text,
                en_name     text,
                keywords    text,
                profile     text,
                embedding   FLOAT8[] NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS mygov_services_text_idx
            ON mygov_services (bn_name, en_name, keywords);
            """
        )


def pg_vector_search(embedding: List[float], query_text: str, top_k: int = 10):
    """
    Semantic search with lexical boost using FLOAT8[] embeddings,
    matching app.py's float8[] cosine logic.
    """
    # L2-normalize query so dot product ~= cosine
    s = sum(float(v) * float(v) for v in embedding)
    if s > 0:
        norm = s ** 0.5
        q_norm = [float(v) / norm for v in embedding]
    else:
        q_norm = [0.0] * len(embedding)

    # derive lexical pattern (use longest token)
    pattern = None
    if query_text:
        tokens = re.findall(r"[A-Za-z\u0980-\u09FF]{3,}", query_text)
        if tokens:
            key = max(tokens, key=len)
            pattern = f"%{key}%"
        else:
            pattern = f"%{query_text}%"

    with get_pg_conn() as conn, conn.cursor() as cur:
        rows = []

        # 1) lexical + vector filter
        if pattern:
            cur.execute(
                """
                SELECT
                    id::text AS doc_id,
                    bn_name  AS name,
                    en_name  AS name_en,
                    keywords AS keyword,
                    (
                      SELECT SUM(a*b)
                      FROM unnest(%s::double precision[]) WITH ORDINALITY AS qa(a, idx)
                      JOIN unnest(embedding) WITH ORDINALITY AS qe(b, idx) USING (idx)
                    )::double precision AS score
                FROM mygov_services
                WHERE
                    COALESCE(en_name, '') ILIKE %s
                    OR COALESCE(bn_name, '') ILIKE %s
                    OR COALESCE(keywords, '') ILIKE %s
                ORDER BY score DESC NULLS LAST
                LIMIT %s;
                """,
                (q_norm, pattern, pattern, pattern, top_k),
            )
            rows = cur.fetchall()

        # 2) fallback: pure vector dot product ranking
        if not rows:
            cur.execute(
                """
                SELECT
                    id::text AS doc_id,
                    bn_name  AS name,
                    en_name  AS name_en,
                    keywords AS keyword,
                    (
                      SELECT SUM(a*b)
                      FROM unnest(%s::double precision[]) WITH ORDINALITY AS qa(a, idx)
                      JOIN unnest(embedding) WITH ORDINALITY AS qe(b, idx) USING (idx)
                    )::double precision AS score
                FROM mygov_services
                ORDER BY score DESC NULLS LAST
                LIMIT %s;
                """,
                (q_norm, top_k),
            )
            rows = cur.fetchall()

        return rows


# =========================
# EMBEDDING
# =========================

def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    emb = resp.data[0].embedding

    if len(emb) > EMBEDDING_DIM:
        emb = emb[:EMBEDDING_DIM]
    elif len(emb) < EMBEDDING_DIM:
        emb = list(emb) + [0.0] * (EMBEDDING_DIM - len(emb))

    return emb


# =========================
# LOAD NEW JSON FORMAT
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
            s = s[:max_len].rstrip() + ""
        else:
            s = s[:cut].rstrip() + ""
    return s


def load_search_documents(path: str = MYGOV_JSON_PATH):
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


# =========================
# LEXICAL SUGGESTION LAYER
# =========================

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


@lru_cache(maxsize=1)
def get_lexical_data():
    docs = load_search_documents()
    model = build_lexical_models(docs)
    # If build_lexical_models returns the newer dict format (like in app.py),
    # convert it to the tuple structure that suggest_queries currently expects.
    if isinstance(model, dict):
        return (
            model["prefix_bn"],
            model["prefix_en"],
            model["all_bn"],
            model["all_en"],
            model["next_bn"],
            model["next_en"],
            model["vocab_bn"],
            model["vocab_en"],
        )
    return model


def is_bangla(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in text)


def suggest_queries(query: str, limit: int = 5):
    (
        prefix_bn,
        prefix_en,
        all_bn,
        all_en,
        next_bn,
        next_en,
        vocab_bn,
        vocab_en,
    ) = get_lexical_data()

    q = query.strip().lower()
    if not q:
        return []

    use_bn = is_bangla(q)

    vocab = vocab_bn if use_bn else vocab_en
    next_dict = next_bn if use_bn else next_en
    prefix_dict = prefix_bn if use_bn else prefix_en
    all_phrases = all_bn if use_bn else all_en

    suggestions = []
    seen = set()

    words = q.split()

    # 1) last word completion
    if words:
        last = words[-1]
        for w in vocab:
            if w.startswith(last) and w != last:
                suggestion = " ".join(words[:-1] + [w])
                if suggestion not in seen:
                    seen.add(suggestion)
                    suggestions.append(suggestion)
                    if len(suggestions) >= limit:
                        return suggestions

    # 2) bigram next word
    if words:
        last = words[-1]
        if last in next_dict:
            for w2 in next_dict[last]:
                suggestion = query + " " + w2
                if suggestion not in seen:
                    seen.add(suggestion)
                    suggestions.append(suggestion)
                    if len(suggestions) >= limit:
                        return suggestions

    # 3) prefix → phrase
    if q in prefix_dict:
        for phrase in prefix_dict[q]:
            if phrase.lower() != q and phrase not in seen:
                suggestions.append(phrase)
                seen.add(phrase)
                if len(suggestions) >= limit:
                    return suggestions

    return suggestions[:limit]


# =========================
# SCHEMAS
# =========================

class SuggestRequest(BaseModel):
    query: str
    limit: int = 5


class SuggestResponse(BaseModel):
    query: str
    suggestions: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchItem(BaseModel):
    name: Optional[str]
    name_en: Optional[str]
    keyword: Optional[str]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="MyGov Search API (New JSON)")


@app.on_event("startup")
def startup():
    try:
        init_schema()
    except Exception as e:
        print("DB init failed:", e)


@app.post("/suggest", response_model=SuggestResponse)
def suggest_api(body: SuggestRequest):
    suggestions = suggest_queries(body.query, body.limit)
    return SuggestResponse(query=body.query, suggestions=suggestions)


@app.post("/search", response_model=SearchResponse)
def search_api(body: SearchRequest):
    q = body.query.strip()
    if not q:
        return SearchResponse(query=q, results=[])

    emb = get_embedding(q)

    try:
        rows = pg_vector_search(emb, q, body.top_k)
    except Exception as e:
        raise HTTPException(500, f"DB search failed: {e}")

    results: List[SearchItem] = []
    for _doc_id, name, name_en, keyword, _score in rows:
        results.append(
            SearchItem(
                name=(name or None),
                name_en=(name_en or None),
                keyword=(keyword or None),
            )
        )

    return SearchResponse(query=q, results=results)
