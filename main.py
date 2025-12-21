import json
import os
import re
import logging
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openai import OpenAI
import psycopg
from dotenv import load_dotenv

load_dotenv()

# =========================
# LOGGING
# =========================
log = logging.getLogger("mygov-api")
logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "local-model")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 768))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("ERROR: DATABASE_URL is not set. Please set Neon/Aiven connection string in env.")

MYGOV_JSON_PATH = os.environ.get("MYGOV_JSON_PATH", "search_service_dump.json")

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

# =========================
# PG helper (UNCHANGED)
# =========================
def get_pg_conn(connect_timeout: int = 5):
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=connect_timeout)


def init_schema():
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
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
# EMBEDDING (UNCHANGED)
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
# LOAD JSON (service cache source) (UNCHANGED)
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
            s = s[:max_len].rstrip()
        else:
            s = s[:cut].rstrip()
    return s


def load_search_documents(path: str = MYGOV_JSON_PATH):
    docs = []
    try:
        raw = open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        return docs

    if not raw:
        return docs

    rows: List[dict] = []
    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            rows = parsed if isinstance(parsed, list) else [parsed]
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

        docs.append(
            {
                "bn": bn,
                "en": en,
                "keyword": kw,
                "text": text,
                "bn_clean": clean_for_vocab(bn),
                "en_clean": clean_for_vocab(en),
                "text_clean": clean_for_vocab(text),
                "bn_display": make_display_phrase(bn) or bn,
                "en_display": make_display_phrase(en) or en,
            }
        )

    return docs


def is_bangla(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in text)


# =========================
# EXISTING LEXICAL MODELS (kept for BIGRAM CACHE ONLY)
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

    for d in docs:
        bn = (d.get("bn") or "").strip()
        en = (d.get("en") or "").strip()
        bn_clean = (d.get("bn_clean") or "").strip()
        en_clean = (d.get("en_clean") or "").strip()
        bn_disp = d.get("bn_display") or bn
        en_disp = d.get("en_display") or en

        if bn:
            all_bn.add(bn_disp)
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

    return (
        prefix_bn_map,
        prefix_en_map,
        sorted(all_bn),
        sorted(all_en),
        next_bn_map,
        next_en_map,
        sorted(vocab_bn),
        sorted(vocab_en),
    )


@lru_cache(maxsize=1)
def get_lexical_data():
    # Built once from dump; used ONLY for bigram rank-only reorder
    docs = load_search_documents()
    return build_lexical_models(docs)


# =========================
# SUGGESTION SEARCH MATCHING ONLY (FIXED: keyword-based)
# =========================
_SUGG_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z0-9]+", flags=re.UNICODE)


def _is_bangla_token(tok: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in tok)


def _norm_token(tok: str) -> str:
    # English-only case-insensitive; Bangla unchanged
    if _is_bangla_token(tok):
        return tok
    return tok.lower()


def _extract_query_tokens(q: str) -> List[str]:
    q = (q or "").strip()
    toks = _SUGG_TOKEN_RE.findall(q)
    return [_norm_token(t) for t in toks if t]


def _split_keyword_phrases(keyword: str) -> List[str]:
    if not keyword:
        return []
    parts = [p.strip() for p in keyword.split(",") if p.strip()]
    phrases: List[str] = []
    for p in parts:
        phrases.append(p)
        no_paren = re.sub(r"\([^)]*\)", "", p).strip()
        if no_paren and no_paren != p:
            phrases.append(no_paren)

    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _norm_phrase_tokens(phrase: str) -> Tuple[str, List[str]]:
    toks = _SUGG_TOKEN_RE.findall(phrase or "")
    ntoks = [_norm_token(t) for t in toks if t]
    return " ".join(ntoks), ntoks


def _keyword_match_score(query_tokens: List[str], kw_tokens: List[str], kw_norm_str: str) -> int:
    if not query_tokens:
        return 0

    score = 0
    q_str = " ".join(query_tokens)

    # Strong: multi-token substring match
    if len(query_tokens) >= 2 and q_str and q_str in kw_norm_str:
        score += 50

    # Partial token match allowed
    for qt in query_tokens:
        for kt in kw_tokens:
            if kt == qt:
                score += 10
                break
            if kt.startswith(qt) or (qt in kt):
                score += 6
                break

    return score


def _build_suggest_cache(docs: List[dict]) -> dict:
    allowed_bn_phrases = set()
    allowed_en_phrases = set()
    entries: List[dict] = []

    for d in docs:
        bn_disp = (d.get("bn_display") or "").strip()
        en_disp = (d.get("en_display") or "").strip()
        kw = (d.get("keyword") or "").strip()

        if bn_disp:
            allowed_bn_phrases.add(bn_disp)
        if en_disp:
            allowed_en_phrases.add(en_disp)

        if not kw:
            continue

        phrases = _split_keyword_phrases(kw)
        norm_phrases: List[Tuple[str, List[str]]] = []
        for ph in phrases:
            kw_norm_str, kw_toks = _norm_phrase_tokens(ph)
            if kw_toks:
                norm_phrases.append((kw_norm_str, kw_toks))

        if norm_phrases:
            entries.append(
                {
                    "bn_display": bn_disp,
                    "en_display": en_disp,
                    "norm_phrases": norm_phrases,
                }
            )

    return {
        "entries": entries,
        "allowed_bn_phrases": allowed_bn_phrases,
        "allowed_en_phrases": allowed_en_phrases,
    }


@lru_cache(maxsize=1)
def get_suggest_cache():
    docs = load_search_documents()
    cache = _build_suggest_cache(docs)
    log.info(
        "Suggest cache built: entries=%d bn_allowed=%d en_allowed=%d",
        len(cache["entries"]),
        len(cache["allowed_bn_phrases"]),
        len(cache["allowed_en_phrases"]),
    )
    return cache


def suggest_queries(query: str, limit: int = 5) -> List[str]:
    """
    Candidates: keyword match দিয়ে আসে (ONLY suggestion-search matching fix)
    Ranking(optional): bigram cache দিয়ে RANK-ONLY reorder (NO filter)
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return []

    use_bn = is_bangla(q_raw)

    # DEBUG: extracted query tokens
    q_tokens = _extract_query_tokens(q_raw)
    log.info("[SUGGEST DEBUG] query=%r extracted_query_tokens=%s", q_raw, q_tokens)

    if not q_tokens:
        return []

    cache = get_suggest_cache()
    scored: List[Tuple[int, str, List[str]]] = []

    # --- keyword-based matching ONLY ---
    for e in cache["entries"]:
        suggestion = (e["bn_display"] if use_bn else e["en_display"]) or ""
        if not suggestion:
            continue

        best_score = 0
        best_kw_tokens: List[str] = []

        for kw_norm_str, kw_tokens in e["norm_phrases"]:
            sc = _keyword_match_score(q_tokens, kw_tokens, kw_norm_str)
            if sc > best_score:
                best_score = sc
                best_kw_tokens = kw_tokens

        if best_score > 0:
            scored.append((best_score, suggestion, best_kw_tokens))

    # If EMPTY -> must be EMPTY
    if not scored:
        log.info("[SUGGEST DEBUG] no keyword matches -> returning EMPTY")
        return []

    # Initial order by keyword score (matching output)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build candidate list (no filter)
    out: List[str] = []
    seen = set()
    matched_map: Dict[str, List[str]] = {}
    for score, suggestion, matched_kw_tokens in scored:
        if suggestion in seen:
            continue
        seen.add(suggestion)
        out.append(suggestion)
        matched_map[suggestion] = matched_kw_tokens
        log.info(
            "[SUGGEST DEBUG] candidate=%r keyword_score=%d matched_keyword_tokens=%s",
            suggestion, score, matched_kw_tokens
        )

    # =========================
    # BIGRAM RANK-ONLY REORDER (MINIMAL ADD-BACK)
    # (NO FILTERING, NO ADD/REMOVE)
    # =========================
    try:
        (
            _prefix_bn,
            _prefix_en,
            _all_bn,
            _all_en,
            next_bn,
            next_en,
            _vocab_bn,
            _vocab_en,
        ) = get_lexical_data()

        q_norm = clean_for_vocab(q_raw)
        words = q_norm.split()
        if words:
            last = words[-1]
            next_map = next_bn if use_bn else next_en
            pref = next_map.get(last, [])

            def bigram_score(sugg: str) -> int:
                s_norm = clean_for_vocab(sugg)
                s_words = s_norm.split()
                # need at least one token after the query prefix length
                if len(s_words) <= len(words):
                    return 0
                nxt = s_words[len(words)] if len(s_words) > len(words) else ""
                if not nxt:
                    return 0
                try:
                    return max(0, len(pref) - pref.index(nxt))
                except ValueError:
                    return 0

            before = list(out)
            out = sorted(out, key=bigram_score, reverse=True)

            log.info(
                "[SUGGEST DEBUG] bigram_rank_only applied. last_token=%r pref_size=%d",
                last, len(pref)
            )
            if before != out:
                log.info("[SUGGEST DEBUG] order_before=%s", before)
                log.info("[SUGGEST DEBUG] order_after=%s", out)

    except Exception as e:
        log.info("[SUGGEST DEBUG] Bigram rank-only skipped: %s", e)

    # Apply explicit limit only (schema has limit)
    if limit is None or limit <= 0:
        final = out
    else:
        final = out[:limit]

    # DEBUG: show matched keyword tokens for final results
    for s in final:
        log.info("[SUGGEST DEBUG] final=%r matched_keyword_tokens=%s", s, matched_map.get(s, []))

    return final


# =========================
# SCHEMAS (UNCHANGED)
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
app = FastAPI(title="MyGov Search API (Keyword Suggest + Bigram Rank-Only)")

@app.on_event("startup")
def startup():
    # Search schema init (unchanged)
    try:
        init_schema()
    except Exception as e:
        log.warning("DB init failed: %s", e)

    # Warm caches once
    try:
        get_suggest_cache()
        get_lexical_data()
    except Exception as e:
        log.warning("Cache warmup failed: %s", e)


@app.post("/suggest", response_model=SuggestResponse)
def suggest_api(body: SuggestRequest):
    suggestions = suggest_queries(body.query, body.limit)

    # VALIDATION REQUIREMENT: every returned suggestion must exist in cache
    cache = get_suggest_cache()
    allowed = cache["allowed_bn_phrases"] if is_bangla(body.query or "") else cache["allowed_en_phrases"]
    for s in suggestions:
        if s not in allowed:
            log.error("BUG: returned suggestion not in cache. query=%r suggestion=%r", body.query, s)
            raise HTTPException(status_code=500, detail="BUG: suggestion not in service cache")

    # Specific debug note for your example
    if (body.query or "").strip().lower() == "direct visa":
        log.info(
            "[SUGGEST DEBUG] why_failed_before=prefix/name-only matching; "
            "why_succeeds_now=keyword token match; "
            "bigram_effect=rank-only reorder (no filtering)."
        )

    return SuggestResponse(query=body.query, suggestions=suggestions)


@app.post("/search", response_model=SearchResponse)
def search_api(body: SearchRequest):
    # UNCHANGED: document search logic
    q = (body.query or "").strip()
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
