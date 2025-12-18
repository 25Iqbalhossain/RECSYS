# app.py — Neon-only, robust, pgvector if available else float8[] fallback
import json
import math
import re
import os
import time
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

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
    raise RuntimeError("ERROR: DATABASE_URL is not set. Please set Neon connection string in env.")

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


def _safe_key(s: str, max_len: int = 40) -> str:
    if not s:
        return "empty"
    return str(abs(hash(s)))[:max_len]


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _strip_line_comment(line: str) -> str:
    """Remove inline comments starting with # or // when not inside a string."""
    in_str = False
    esc = False
    i = 0
    while i < len(line):
        ch = line[i]
        if esc:
            esc = False
            i += 1
            continue
        if ch == "\\":
            esc = True
            i += 1
            continue
        if ch == '"':
            in_str = not in_str
            i += 1
            continue
        if not in_str:
            if ch == "#":
                return line[:i].rstrip()
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _clean_jsonish_text(raw: str) -> str:
    """Best-effort cleanup for json-ish files."""
    lines = [_strip_line_comment(l) for l in raw.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _extract_json_objects(text: str) -> List[Dict]:
    """
    Extract {...} JSON objects from text that may look like:
      { ... },
      { ... },
    without wrapping [ ].
    """
    objs: List[Dict] = []
    buf: List[str] = []
    depth = 0
    in_str = False
    esc = False

    for ch in text:
        if depth == 0:
            if ch == "{":
                depth = 1
                buf = ["{"]
                in_str = False
                esc = False
            else:
                continue
        else:
            buf.append(ch)

            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue

            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        obj_str = "".join(buf).strip()
                        obj_str = re.sub(r",\s*}$", "}", obj_str)
                        obj_str = re.sub(r",\s*([}\]])", r"\1", obj_str)
                        try:
                            objs.append(json.loads(obj_str))
                        except Exception:
                            pass
                        buf = []

    return objs


# -------------------------
# PG Helper
# -------------------------
def get_pg_conn(connect_timeout: int = 5):
    return psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=connect_timeout)


def ensure_doc_id_schema(cur):
    """Migration for older tables that may not have doc_id."""
    cur.execute("ALTER TABLE mygov_services ADD COLUMN IF NOT EXISTS doc_id TEXT;")
    cur.execute("UPDATE mygov_services SET doc_id = COALESCE(doc_id, 'id_' || id::text);")

    # De-duplicate doc_id collisions
    cur.execute(
        """
        WITH ranked AS (
            SELECT id, doc_id,
                   ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY id) AS rn
            FROM mygov_services
            WHERE doc_id IS NOT NULL
        )
        UPDATE mygov_services m
        SET doc_id = m.doc_id || '_dup_' || m.id::text
        FROM ranked r
        WHERE m.id = r.id AND r.rn > 1;
        """
    )

    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS mygov_services_doc_id_uq
        ON mygov_services (doc_id);
        """
    )


def ensure_profile_schema(cur):
    """Fix for: column 'profile' does not exist."""
    cur.execute("ALTER TABLE mygov_services ADD COLUMN IF NOT EXISTS profile TEXT;")


def init_schema_safe():
    """
    Table:
      id bigserial PRIMARY KEY,
      doc_id text unique,
      bn_name text,
      en_name text,
      keywords text,
      profile text,
      embedding float8[] NOT NULL (or VECTOR if available)
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
                            doc_id TEXT,
                            bn_name TEXT,
                            en_name TEXT,
                            keywords TEXT,
                            profile TEXT,
                            embedding VECTOR({EMBEDDING_DIM})
                        );
                        """
                    )
                else:
                    USE_PGVECTOR = False
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS mygov_services (
                            id BIGSERIAL PRIMARY KEY,
                            doc_id TEXT,
                            bn_name TEXT,
                            en_name TEXT,
                            keywords TEXT,
                            profile TEXT,
                            embedding FLOAT8[] NOT NULL
                        );
                        """
                    )

                ensure_doc_id_schema(cur)
                ensure_profile_schema(cur)

                try:
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS mygov_services_text_idx ON mygov_services (bn_name, en_name, keywords);"
                    )
                except Exception:
                    pass

                if USE_PGVECTOR:
                    try:
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS mygov_services_embedding_idx
                            ON mygov_services USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                            """
                        )
                    except Exception:
                        pass

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
# Lexical loading (Suggestions)
# =========================
@st.cache_data
def load_search_documents(path: str) -> List[Dict]:
    """
    Supports:
      - JSON array: [ {...}, {...} ]
      - JSON object: { ... }
      - JSONL: one object per line
      - Loose objects separated by commas: {...}, {...},
    And your keys:
      name, name_en, keyword
    """
    docs: List[Dict] = []
    try:
        raw = open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        return docs
    if not raw:
        return docs

    raw2 = _clean_jsonish_text(raw)

    rows: List[Dict] = []
    # 1) Valid JSON array/object
    try:
        parsed = json.loads(raw2)
        if isinstance(parsed, list):
            rows = parsed
        elif isinstance(parsed, dict):
            rows = [parsed]
    except Exception:
        rows = []

    # 2) JSONL
    if not rows:
        tmp = []
        ok = True
        for line in raw2.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r",\s*$", "", line)
            try:
                tmp.append(json.loads(line))
            except Exception:
                ok = False
                break
        if ok and tmp:
            rows = tmp

    # 3) Extract objects
    if not rows:
        rows = _extract_json_objects(raw2)

    for r in rows:
        bn = (r.get("my_gov_service_name") or r.get("name") or r.get("bn") or "").strip()
        en = (r.get("my_gov_service_name_en") or r.get("name_en") or r.get("en") or "").strip()
        kw = (r.get("my_gov_service_keyword") or r.get("keyword") or r.get("keywords") or "").strip()
        profile = (r.get("profile") or r.get("description") or r.get("details") or "").strip()

        text = " ".join([x for x in [bn, en, kw] if x]).strip()
        if not text:
            continue

        docs.append(
            {
                "bn": bn,
                "en": en,
                "keyword": kw,
                "profile": profile,
                "text": text,
                "bn_clean": clean_for_vocab(bn),
                "en_clean": clean_for_vocab(en),
                "bn_display": make_display_phrase(bn),
                "en_display": make_display_phrase(en),
                "doc_id": r.get("doc_id"),
            }
        )

    return docs


def build_lexical_models(docs: List[Dict]):
    prefix_map_bn = defaultdict(Counter)
    prefix_map_en = defaultdict(Counter)
    bigram_bn = defaultdict(Counter)
    bigram_en = defaultdict(Counter)
    vocab_bn = set()
    vocab_en = set()

    allowed_bn_phrases = set()
    allowed_en_phrases = set()

    bn_to_en: Dict[str, str] = {}
    en_to_bn: Dict[str, str] = {}

    phrase_freq_bn = Counter()
    phrase_freq_en = Counter()

    for d in docs:
        bn = (d.get("bn") or "").strip()
        en = (d.get("en") or "").strip()
        bn_clean = (d.get("bn_clean") or "").strip()
        en_clean = (d.get("en_clean") or "").strip()
        bn_disp = d.get("bn_display") or bn
        en_disp = d.get("en_display") or en

        if bn_disp:
            allowed_bn_phrases.add(bn_disp)
            phrase_freq_bn[bn_disp] += 1
        if en_disp:
            allowed_en_phrases.add(en_disp)
            phrase_freq_en[en_disp] += 1

        if bn_disp and en_disp:
            bn_to_en.setdefault(bn_disp, en_disp)
            en_to_bn.setdefault(en_disp, bn_disp)

        if bn_clean:
            words = bn_clean.split()
            for w in words:
                vocab_bn.add(w)
            for i in range(1, min(3, len(words)) + 1):
                prefix = " ".join(words[:i])
                prefix_map_bn[prefix][bn_disp] += 1
            for w1, w2 in zip(words, words[1:]):
                bigram_bn[w1][w2] += 1

        if en_clean:
            words = en_clean.split()
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

    popular_bn = [x for x, _ in phrase_freq_bn.most_common(50)]
    popular_en = [x for x, _ in phrase_freq_en.most_common(50)]

    return {
        "prefix_bn": prefix_bn_map,
        "prefix_en": prefix_en_map,
        "next_bn": next_bn_map,
        "next_en": next_en_map,
        "vocab_bn": sorted(vocab_bn),
        "vocab_en": sorted(vocab_en),
        "allowed_bn_phrases": allowed_bn_phrases,
        "allowed_en_phrases": allowed_en_phrases,
        "bn_to_en": bn_to_en,
        "en_to_bn": en_to_bn,
        "popular_bn": popular_bn,
        "popular_en": popular_en,
        "docs_count": len(docs),
    }


@st.cache_data
def get_lexical_data(version: int = 5):
    docs = load_search_documents(MYGOV_JSON_PATH)
    return build_lexical_models(docs)


def is_bangla(text: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in text)


def _suggest_one_lang(query: str, limit: int, use_bn: bool, data: Dict) -> List[str]:
    prefix_dict = data["prefix_bn"] if use_bn else data["prefix_en"]
    next_dict = data["next_bn"] if use_bn else data["next_en"]
    vocab = set(data["vocab_bn"] if use_bn else data["vocab_en"])
    all_phrases = data["allowed_bn_phrases"] if use_bn else data["allowed_en_phrases"]

    q_orig = (query or "").strip()
    if not q_orig:
        return []
    q = q_orig.lower()

    suggestions: List[str] = []
    seen = set()
    words = re.findall(r"[\w\u0980-\u09FF]+", q)

    # 1) last-word completion
    if words:
        last = words[-1]
        completions = sorted(w for w in vocab if w.startswith(last) and w != last)
        for comp in completions:
            prefix = " ".join(words[:-1] + [comp])
            if prefix in prefix_dict:
                for ph in prefix_dict[prefix]:
                    if ph not in seen and ph in all_phrases:
                        suggestions.append(ph)
                        seen.add(ph)
                        if len(suggestions) >= limit:
                            return suggestions

    # 2) bigram next-word
    if words:
        last = words[-1]
        if last in next_dict:
            for w2 in next_dict[last]:
                cand = (q_orig + " " + w2).lower()
                for ph in all_phrases:
                    if ph.lower().startswith(cand) and ph not in seen:
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

    # 4) fallback contains
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


def suggest_queries(query: str, limit: int = 5, mode: str = "Both") -> List[str]:
    data = get_lexical_data()

    # If dataset is empty, no suggestions can be generated
    if int(data.get("docs_count", 0)) == 0:
        return []

    if mode == "Both":
        a = _suggest_one_lang(query, limit=limit, use_bn=True, data=data)
        b = _suggest_one_lang(query, limit=limit, use_bn=False, data=data)
        merged: List[str] = []
        seen = set()
        for x in a + b:
            if x not in seen:
                merged.append(x)
                seen.add(x)
            if len(merged) >= limit:
                break

        # If still empty, fallback to popular phrases (so suggestions never disappear)
        if not merged:
            merged = (data.get("popular_bn", [])[: limit // 2 + 1] + data.get("popular_en", [])[: limit // 2 + 1])[:limit]
        return merged[:limit]

    # Auto
    use_bn = is_bangla(query or "")
    out = _suggest_one_lang(query, limit=limit, use_bn=use_bn, data=data)
    if not out:
        out = (data.get("popular_bn", []) if use_bn else data.get("popular_en", []))[:limit]
    return out[:limit]


def suggestion_display_label(phrase: str, data: Dict) -> str:
    """Show bilingual label when mapping exists."""
    bn_to_en = data.get("bn_to_en", {})
    en_to_bn = data.get("en_to_bn", {})
    if phrase in bn_to_en and bn_to_en[phrase]:
        other = bn_to_en[phrase]
        if other and other != phrase:
            return f"{phrase}  |  {other}"
    if phrase in en_to_bn and en_to_bn[phrase]:
        other = en_to_bn[phrase]
        if other and other != phrase:
            return f"{other}  |  {phrase}"
    return phrase


# =========================
# Upsert / index dataset
# =========================
def upsert_services(rows: List[Tuple[str, str, str, str, str, List[float]]]):
    """
    rows: (doc_id, bn, en, kw, profile, embedding)
    Embeddings are L2-normalized before storing.
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            for doc_id, bn, en, kw, profile, emb in rows:
                try:
                    emb_norm = _l2_normalize(emb)
                    if USE_PGVECTOR and PGVECTOR_AVAILABLE and PgVector is not None:
                        vec = PgVector(emb_norm)
                        cur.execute(
                            """
                            INSERT INTO mygov_services (doc_id, bn_name, en_name, keywords, profile, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (doc_id) DO UPDATE
                            SET bn_name = EXCLUDED.bn_name,
                                en_name = EXCLUDED.en_name,
                                keywords = EXCLUDED.keywords,
                                profile = EXCLUDED.profile,
                                embedding = EXCLUDED.embedding;
                            """,
                            (doc_id, bn, en, kw, profile, vec),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO mygov_services (doc_id, bn_name, en_name, keywords, profile, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (doc_id) DO UPDATE
                            SET bn_name = EXCLUDED.bn_name,
                                en_name = EXCLUDED.en_name,
                                keywords = EXCLUDED.keywords,
                                profile = EXCLUDED.profile,
                                embedding = EXCLUDED.embedding;
                            """,
                            (doc_id, bn, en, kw, profile, emb_norm),
                        )
                except Exception as e:
                    print(f"[!] Upsert error for {doc_id}: {e}")


def prepare_docs(rows: List[Dict]) -> List[Tuple[str, str, str, str, str, str]]:
    """Returns: (doc_id, bn, en, kw, profile, text)"""
    docs: List[Tuple[str, str, str, str, str, str]] = []
    for i, row in enumerate(rows):
        bn = (row.get("my_gov_service_name") or row.get("name") or row.get("bn") or "").strip()
        en = (row.get("my_gov_service_name_en") or row.get("name_en") or row.get("en") or "").strip()
        kw = (row.get("my_gov_service_keyword") or row.get("keyword") or row.get("keywords") or "").strip()
        profile = (row.get("profile") or row.get("description") or row.get("details") or "").strip()

        text = " ".join([x for x in [bn, en, kw] if x]).strip()
        if not text:
            continue

        doc_id = row.get("doc_id") or f"row_{i}"
        docs.append((doc_id, bn, en, kw, profile, text))
    return docs


def index_dataset(batch_size: int = 64):
    print("[*] init schema ...")
    init_schema_safe()
    print("[*] loading data ...")
    rows = load_search_documents(MYGOV_JSON_PATH)
    print(f"[*] rows loaded: {len(rows)}")

    docs_tuples = prepare_docs(rows)
    print(f"[*] docs prepared: {len(docs_tuples)}")

    total = len(docs_tuples)
    for start in range(0, total, batch_size):
        chunk = docs_tuples[start : start + batch_size]
        texts = [t[5] for t in chunk]
        print(f"[*] embedding batch {start}-{start + len(chunk) - 1}")

        try:
            data = client.embeddings.create(model=EMBEDDING_MODEL, input=texts).data
            embs: List[List[float]] = []
            for item in data:
                e = item.embedding
                if len(e) > EMBEDDING_DIM:
                    e = e[:EMBEDDING_DIM]
                elif len(e) < EMBEDDING_DIM:
                    e = e + [0.0] * (EMBEDDING_DIM - len(e))
                embs.append(list(e))
        except Exception as e:
            print("[!] Embedding API error:", e)
            break

        pg_rows: List[Tuple[str, str, str, str, str, List[float]]] = []
        for (doc_id, bn, en, kw, profile, _), emb in zip(chunk, embs):
            pg_rows.append((doc_id, bn, en, kw, profile, emb))

        upsert_services(pg_rows)

    print("[*] indexing complete")


# =========================
# Search implementation
# =========================
def search_vector(
    query_emb: List[float],
    query_text: str,
    top_k: int = 10,
    use_pattern_filter: bool = True,
):
    q_norm = _l2_normalize(query_emb)
    try:
        conn = get_pg_conn()
    except Exception as e:
        raise RuntimeError("Database not available: could not connect.") from e

    try:
        with conn:
            with conn.cursor() as cur:
                pattern = None
                if query_text and use_pattern_filter:
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
                            SELECT COALESCE(doc_id, 'id_' || id::text) AS doc_id,
                                   bn_name, en_name, keywords, profile,
                                   1 - (embedding <=> %s) AS score
                            FROM mygov_services
                            WHERE COALESCE(en_name,'') ILIKE %s
                               OR COALESCE(bn_name,'') ILIKE %s
                               OR COALESCE(keywords,'') ILIKE %s
                            ORDER BY embedding <=> %s
                            LIMIT %s;
                            """,
                            (vec, pattern, pattern, pattern, vec, top_k),
                        )
                        rows = cur.fetchall()

                    if not rows:
                        cur.execute(
                            """
                            SELECT COALESCE(doc_id, 'id_' || id::text) AS doc_id,
                                   bn_name, en_name, keywords, profile,
                                   1 - (embedding <=> %s) AS score
                            FROM mygov_services
                            ORDER BY embedding <=> %s
                            LIMIT %s;
                            """,
                            (vec, vec, top_k),
                        )
                        rows = cur.fetchall()

                else:
                    if pattern:
                        cur.execute(
                            """
                            SELECT COALESCE(doc_id, 'id_' || id::text) AS doc_id,
                                   bn_name, en_name, keywords, profile,
                              (SELECT SUM(a*b)
                               FROM unnest(%s::double precision[]) WITH ORDINALITY AS qa(a, idx)
                               JOIN unnest(embedding) WITH ORDINALITY AS qe(b, idx) USING (idx)
                              )::double precision AS score
                            FROM mygov_services
                            WHERE COALESCE(en_name,'') ILIKE %s
                               OR COALESCE(bn_name,'') ILIKE %s
                               OR COALESCE(keywords,'') ILIKE %s
                            ORDER BY score DESC NULLS LAST
                            LIMIT %s;
                            """,
                            (q_norm, pattern, pattern, pattern, top_k),
                        )
                        rows = cur.fetchall()

                    if not rows:
                        cur.execute(
                            """
                            SELECT COALESCE(doc_id, 'id_' || id::text) AS doc_id,
                                   bn_name, en_name, keywords, profile,
                              (SELECT SUM(a*b)
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

        # Dedupe by (bn_name, en_name)
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
# Evaluation Metrics
# =========================
LABELS = ["Not relevant", "Partially relevant", "Relevant"]
GAIN = {"Not relevant": 0, "Partially relevant": 1, "Relevant": 2}


def _binary_relevant(label: str, count_partial_as_relevant: bool) -> bool:
    if label == "Relevant":
        return True
    if count_partial_as_relevant and label == "Partially relevant":
        return True
    return False


def compute_metrics(
    labels_in_rank_order: List[str],
    total_relevant: Optional[int],
    count_partial_as_relevant: bool = True,
):
    retrieved = len(labels_in_rank_order)
    if retrieved == 0:
        return {"precision": None, "recall": None, "f1": None, "mrr": None, "ndcg": None, "retrieved": 0, "relevant_retrieved": 0}

    relevant_retrieved = sum(1 for l in labels_in_rank_order if _binary_relevant(l, count_partial_as_relevant))
    precision = relevant_retrieved / retrieved if retrieved else None

    recall = None
    if total_relevant is not None and total_relevant > 0:
        recall = min(1.0, relevant_retrieved / float(total_relevant))

    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    mrr = 0.0
    for i, l in enumerate(labels_in_rank_order):
        if _binary_relevant(l, count_partial_as_relevant):
            mrr = 1.0 / float(i + 1)
            break

    gains = [GAIN.get(l, 0) for l in labels_in_rank_order]

    def dcg(vals: List[int]) -> float:
        s = 0.0
        for i, g in enumerate(vals):
            s += float(g) / math.log2(i + 2)
        return s

    dcg_val = dcg(gains)
    ideal = sorted(gains, reverse=True)
    idcg_val = dcg(ideal)
    ndcg = (dcg_val / idcg_val) if idcg_val > 0 else None

    return {
        "retrieved": retrieved,
        "relevant_retrieved": relevant_retrieved,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "ndcg": ndcg,
        "labels": labels_in_rank_order,
    }


# =========================
# Coverage Test Suite (preset)
# =========================
PRESET_SUITE = [
    ("Short query", "passport"),
    ("Long query", "passport renewal fee requirements"),
    ("Misspelling", "passprot renewl"),
    ("Synonym", "driving license"),
    ("Synonym", "vehicle license"),
    ("Mixed", "nid apply"),
    ("Mixed", "btrc application"),
]


def run_coverage_suite(
    suite: List[Tuple[str, str]],
    use_pattern_filter: bool,
    top_k: int,
    emb_cache: Dict[str, List[float]],
):
    results = []
    zero_count = 0
    total_queries = len(suite)
    avg_scores = []
    top_scores = []

    for kind, q in suite:
        q = (q or "").strip()
        if not q:
            continue

        emb = emb_cache.get(q)
        if emb is None:
            emb = get_embedding(q)
            emb_cache[q] = emb

        rows = search_vector(emb, q, top_k=top_k, use_pattern_filter=use_pattern_filter)
        n = len(rows)
        zero = (n == 0)
        if zero:
            zero_count += 1

        scores = [float(r[-1]) for r in rows if isinstance(r[-1], (int, float))]
        avg_score = (sum(scores) / len(scores)) if scores else None
        top1_score = scores[0] if scores else None

        top1_title = ""
        if rows:
            _, bn, en, _, _, _ = rows[0]
            title = (bn or "").strip() or (en or "").strip() or ""
            top1_title = make_display_phrase(title, 60)

        results.append(
            {
                "type": kind,
                "query": q,
                "results": n,
                "zero_result": zero,
                "avg_score": avg_score,
                "top1_score": top1_score,
                "top1_title": top1_title,
            }
        )

        if avg_score is not None:
            avg_scores.append(avg_score)
        if top1_score is not None:
            top_scores.append(top1_score)

    suite_zrr = (zero_count / total_queries) if total_queries else 0.0
    suite_avg_score = (sum(avg_scores) / len(avg_scores)) if avg_scores else None
    suite_avg_top1 = (sum(top_scores) / len(top_scores)) if top_scores else None
    suite_avg_results = (sum(r["results"] for r in results) / len(results)) if results else 0.0

    return {
        "rows": results,
        "suite_zero_result_rate": suite_zrr,
        "suite_avg_score": suite_avg_score,
        "suite_avg_top1_score": suite_avg_top1,
        "suite_avg_results": suite_avg_results,
    }


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="MyGov Neon Semantic Search", layout="wide")
st.title("MyGov Search (Neon + Embeddings)")

# Session state init
st.session_state.setdefault("db_init", False)
st.session_state.setdefault("emb_cache", {})
st.session_state.setdefault("search_log", [])
st.session_state.setdefault("judgements", {})          # qkey -> {doc_id: label}
st.session_state.setdefault("judgements_saved", set()) # qkey set: only then show metrics (fixes "all zeros")
st.session_state.setdefault("last_logged", {"qkey": None, "variant": None, "top_k": None})
st.session_state.setdefault("suggest_shown", Counter())
st.session_state.setdefault("suggest_used", Counter())
st.session_state.setdefault("coverage_report", None)
st.session_state.setdefault("last_suggestion_source_qkey", None)
st.session_state.setdefault("last_suggestion_source_query", None)

# Sidebar containers
with st.sidebar:
    st.header("Quality Dashboard")

    variant = st.selectbox(
        "Search variant",
        ["Hybrid (vector + lexical filter)", "Vector only"],
        index=0,
        key="variant",
    )
    top_k = st.slider("Top K results", 5, 25, 10, 1, key="top_k")

    st.divider()
    st.subheader("Suggestion settings")
    suggest_mode = st.selectbox(
        "Suggestion mode",
        ["Both", "Auto"],
        index=0,
        key="suggest_mode",
        help="Both merges suggestions from Bangla + English names.",
    )

    st.divider()
    st.subheader("Evaluation settings")
    count_partial_as_relevant = st.checkbox(
        "Count 'Partially relevant' as relevant (binary metrics)",
        value=True,
        key="count_partial",
    )
    total_relevant = st.number_input(
        "Total relevant results (for Recall/F1)",
        min_value=0,
        value=0,
        step=1,
        key="total_rel",
    )

    st.divider()
    st.subheader("Query coverage test suite")
    if st.button("Run preset suite", use_container_width=True):
        use_pattern = (variant == "Hybrid (vector + lexical filter)")
        st.session_state["coverage_report"] = run_coverage_suite(
            suite=PRESET_SUITE,
            use_pattern_filter=use_pattern,
            top_k=int(top_k),
            emb_cache=st.session_state["emb_cache"],
        )
        st.toast("Coverage suite completed")

    metrics_box = st.container()
    eval_box = st.container()
    suggest_box = st.container()
    log_box = st.container()
    coverage_box = st.container()

# DB init (includes migration)
if not st.session_state["db_init"]:
    st.session_state["db_init"] = init_schema_safe()

if not st.session_state["db_init"]:
    st.error("Database is not reachable or schema init failed.")
else:
    st.success("Connected to database.")

# Index button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Index dataset (recreate/update embeddings)"):
        st.info("Indexing started. Check the console logs.")
        index_dataset(batch_size=64)
        get_lexical_data.clear()
        st.success("Indexing complete. Suggestions refreshed.")

with col2:
    st.write("Type below to search. Suggestions and evaluation are in the left sidebar.")

# Query input
st.text_input("Search", key="query")
query = (st.session_state.get("query") or "").strip()

rows = []
suggestions = []
lex_data = get_lexical_data()

# Small status to help debugging "no suggestions"
st.caption(f"Suggestion dataset loaded: {int(lex_data.get('docs_count', 0))} documents from {MYGOV_JSON_PATH}")

if query:
    suggestions = suggest_queries(query, limit=5, mode=str(suggest_mode))

    emb = st.session_state["emb_cache"].get(query)
    if emb is None:
        emb = get_embedding(query)
        st.session_state["emb_cache"][query] = emb

    use_pattern = (variant == "Hybrid (vector + lexical filter)")
    rows = search_vector(emb, query, top_k=int(top_k), use_pattern_filter=use_pattern)

    qkey = _safe_key(query)
    last = st.session_state["last_logged"]
    if last["qkey"] != qkey or last["variant"] != variant or last["top_k"] != top_k:
        st.session_state["search_log"].append(
            {
                "ts": time.time(),
                "query": query,
                "qkey": qkey,
                "variant": variant,
                "top_k": int(top_k),
                "results": len(rows),
                "clicked": False,
                "first_click_s": None,
                "zero_result": (len(rows) == 0),
            }
        )
        st.session_state["last_logged"] = {"qkey": qkey, "variant": variant, "top_k": top_k}

    # MAIN: show suggestions too (so you don't miss them in the sidebar)
    if suggestions:
        st.subheader("Suggestions")
        for i, s in enumerate(suggestions):
            label = suggestion_display_label(s, lex_data)
            if st.button(label, key=f"main_sugg_{qkey}_{_safe_key(s)}_{i}"):
                st.session_state["last_suggestion_source_qkey"] = qkey
                st.session_state["last_suggestion_source_query"] = query
                st.session_state["suggest_used"][qkey] = st.session_state["suggest_used"].get(qkey, 0) + 1
                st.session_state["query"] = s
                _rerun()

    # Results
    if not rows:
        st.info("No results.")
    else:
        st.subheader("Results")
        for rank, (doc_id, bn, en, kw, pf, score) in enumerate(rows, start=1):
            bn = (bn or "").strip()
            en = (en or "").strip()

            title = bn or en or "(No title)"
            title = make_display_phrase(title, max_len=120)
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

            st.markdown(f"### {rank}. {title}")
            st.caption(f"doc_id: {doc_id} • score: {score_text}")

            if en:
                st.caption(f"English: {en}")

            if st.button("Click (log CTR / time-to-first-click)", key=f"click_{qkey}_{doc_id}"):
                now = time.time()
                for entry in reversed(st.session_state["search_log"]):
                    if entry["qkey"] == qkey and entry["variant"] == variant and entry["top_k"] == int(top_k):
                        if not entry["clicked"]:
                            entry["clicked"] = True
                            entry["first_click_s"] = max(0.0, now - entry["ts"])
                        break
                st.toast("Click logged")

            if kw:
                st.write("Keywords:", kw)
            if pf:
                st.write(pf)
            st.markdown("---")
else:
    st.info("Type a query to search.")

# =========================
# SIDEBAR PANELS
# =========================
with st.sidebar:
    qkey = _safe_key(query) if query else None

    # Suggestions (sidebar)
    with suggest_box:
        st.subheader("Suggestions")
        if query:
            if suggestions:
                st.session_state["suggest_shown"][qkey] = st.session_state["suggest_shown"].get(qkey, 0) + 1
                for i, s in enumerate(suggestions):
                    label = suggestion_display_label(s, lex_data)
                    if st.button(label, key=f"sugg_btn_{qkey}_{_safe_key(s)}_{i}"):
                        st.session_state["last_suggestion_source_qkey"] = qkey
                        st.session_state["last_suggestion_source_query"] = query
                        st.session_state["suggest_used"][qkey] = st.session_state["suggest_used"].get(qkey, 0) + 1
                        st.session_state["query"] = s
                        _rerun()
            else:
                st.caption("No matching suggestions for this query. Showing popular services instead.")
                popular = (lex_data.get("popular_bn", [])[:3] + lex_data.get("popular_en", [])[:3])[:5]
                for i, s in enumerate(popular):
                    label = suggestion_display_label(s, lex_data)
                    if st.button(label, key=f"popular_btn_{qkey}_{_safe_key(s)}_{i}"):
                        st.session_state["query"] = s
                        _rerun()
        else:
            st.caption("Type a query to see suggestions.")

    # Relevance judgement (only saved on submit; fixes "all zeros")
    with eval_box:
        st.subheader("Relevance judgement")
        if query and rows:
            st.caption("Label the current Top-K results, then click 'Save labels'.")
            doc_ids = [doc_id for (doc_id, *_r) in rows]
            st.session_state["judgements"].setdefault(qkey, {})

            with st.form(f"rel_form_{qkey}"):
                for idx, (doc_id, bn, en, kw, pf, score) in enumerate(rows, start=1):
                    title = make_display_phrase(((bn or "").strip() or (en or "").strip() or doc_id), 55)
                    prev = st.session_state["judgements"][qkey].get(doc_id, "Not relevant")
                    default_idx = LABELS.index(prev) if prev in LABELS else 0

                    st.radio(
                        f"{idx}. {title}",
                        LABELS,
                        index=default_idx,
                        horizontal=True,
                        key=f"lbl_{qkey}_{doc_id}",
                    )

                submitted = st.form_submit_button("Save labels")

            if submitted:
                for doc_id in doc_ids:
                    st.session_state["judgements"][qkey][doc_id] = st.session_state.get(f"lbl_{qkey}_{doc_id}", "Not relevant")
                st.session_state["judgements_saved"].add(qkey)
                st.toast("Labels saved")
        else:
            st.caption("Search to label results.")

    # Metrics (show only after Save labels)
    with metrics_box:
        st.subheader("Metrics")
        if query and rows:
            if qkey not in st.session_state["judgements_saved"]:
                st.caption("Metrics will appear after you click 'Save labels' in Relevance judgement.")
                st.metric("Precision", "N/A")
                st.metric("Recall", "N/A")
                st.metric("MRR", "N/A")
                st.metric("NDCG", "N/A")
                st.metric("F1", "N/A")
            else:
                labels = [st.session_state["judgements"].get(qkey, {}).get(doc_id, "Not relevant") for (doc_id, *_r) in rows]
                total_rel_val = int(total_relevant) if int(total_relevant) > 0 else None
                m = compute_metrics(labels, total_relevant=total_rel_val, count_partial_as_relevant=bool(count_partial_as_relevant))

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Precision", "N/A" if m["precision"] is None else f"{m['precision']:.3f}")
                    st.metric("MRR", "N/A" if m["mrr"] is None else f"{m['mrr']:.3f}")
                with c2:
                    st.metric("Recall", "N/A" if m["recall"] is None else f"{m['recall']:.3f}")
                    st.metric("NDCG", "N/A" if m["ndcg"] is None else f"{m['ndcg']:.3f}")

                st.metric("F1", "N/A" if m["f1"] is None else f"{m['f1']:.3f}")

                dist = Counter(labels)
                chart_rows = [{"label": k, "count": int(dist.get(k, 0))} for k in LABELS]
                st.caption("Label distribution")
                st.bar_chart(chart_rows, x="label", y="count")
        else:
            st.caption("Search + label results to see metrics.")

    # Logs
    with log_box:
        st.subheader("Logs")
        logs = st.session_state.get("search_log", [])
        if logs:
            total = len(logs)
            zero = sum(1 for x in logs if x.get("zero_result"))
            clicked = sum(1 for x in logs if x.get("clicked"))
            ctr = (clicked / total) if total else 0.0
            zrr = (zero / total) if total else 0.0

            ttfcs = [x["first_click_s"] for x in logs if x.get("first_click_s") is not None]
            avg_ttf = (sum(ttfcs) / len(ttfcs)) if ttfcs else None

            st.metric("Total searches", str(total))
            st.metric("Zero result rate", f"{zrr:.2%}")
            st.metric("CTR", f"{ctr:.2%}")
            st.metric("Avg time-to-first-click", "N/A" if avg_ttf is None else f"{avg_ttf:.2f}s")

            shown_total = sum(int(v) for v in st.session_state["suggest_shown"].values())
            used_total = sum(int(v) for v in st.session_state["suggest_used"].values())
            overall_rate = (used_total / shown_total) if shown_total else 0.0
            st.metric("Suggestion click rate (overall)", f"{overall_rate:.2%}")

            if query and qkey:
                s_shown = int(st.session_state["suggest_shown"].get(qkey, 0))
                s_used = int(st.session_state["suggest_used"].get(qkey, 0))
                rate = (s_used / s_shown) if s_shown else 0.0
                st.metric("Suggestion click rate (current query)", f"{rate:.2%}")
        else:
            st.caption("Search activity will appear here.")

    # Coverage suite output
    with coverage_box:
        st.subheader("Coverage suite results")
        rep = st.session_state.get("coverage_report")
        if rep:
            st.metric("Suite zero result rate", f"{rep['suite_zero_result_rate']:.2%}")
            st.metric("Suite avg results/query", f"{rep['suite_avg_results']:.2f}")
            st.metric("Suite avg score", "N/A" if rep["suite_avg_score"] is None else f"{rep['suite_avg_score']:.4f}")
            st.metric("Suite avg Top-1 score", "N/A" if rep["suite_avg_top1_score"] is None else f"{rep['suite_avg_top1_score']:.4f}")
            st.dataframe(rep["rows"], use_container_width=True)
        else:
            st.caption("Run the preset suite to see results here.")
