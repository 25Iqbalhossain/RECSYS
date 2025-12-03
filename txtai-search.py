# txtai_search_aiven_env.py
"""
Aiven-ready indexer + lexical suggestions + semantic search (float8[] fallback).
Reads DB creds and config from environment or a .env file.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not installed — env vars must be set in OS/shell
    pass

from openai import OpenAI
import psycopg

# -------------------------
# Configuration from ENV
# -------------------------
# Prefer DATABASE_URL if provided (like Neon/Aiven URI)
DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or None

# Fallback individual PG settings (used only if DATABASE_URL not provided)
PG_HOST = os.getenv("PG_HOST")
PG_PORT = int(os.getenv("PG_PORT")) if os.getenv("PG_PORT") else None
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")  # keep empty default - .env should set it

# Optional Aiven CA path (if you need verify-ca)
AIVEN_SSLROOT = os.getenv("AIVEN_SSLROOT", "").strip() or None

# OpenAI / embeddings server config
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))

# Data file path (override via env if you want)
DATA_FILE = Path(os.getenv("DATA_FILE", r"C:\Users\hi\OneDrive\Desktop\New folder\search_service_dump.json"))

# -------------------------
# Client init
# -------------------------
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# -------------------------
# Lexical globals
# -------------------------
prefix_suggestions: Dict[str, List[str]] = {}
next_word: Dict[str, List[str]] = {}
vocab_words: List[str] = []


# -------------------------
# Postgres helpers
# -------------------------
def get_pg_conn(connect_timeout: int = 5):
    """
    Create psycopg connection.
    Prefer DATABASE_URL (so query params like sslmode are honored).
    If AIVEN_SSLROOT is provided and using individual PG_* connection, pass sslrootcert.
    """
    kwargs = {"autocommit": True, "connect_timeout": connect_timeout}

    if DATABASE_URL:
        # If DATABASE_URL already contains sslmode and params, pass it directly.
        # If CA path is provided and you want verify-ca, you can still pass sslrootcert here.
        if AIVEN_SSLROOT:
            # pass sslrootcert explicitly (useful if URL uses verify-ca)
            return psycopg.connect(DATABASE_URL, sslrootcert=AIVEN_SSLROOT, sslmode="verify-ca", **kwargs)
        return psycopg.connect(DATABASE_URL, **kwargs)
    else:
        # build connection using individual fields
        if not (PG_HOST and PG_PORT and PG_DB and PG_USER and PG_PASSWORD):
            raise RuntimeError("PG connection info missing: set DATABASE_URL or PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD")
        if AIVEN_SSLROOT:
            return psycopg.connect(
                host=PG_HOST,
                port=PG_PORT,
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASSWORD,
                sslrootcert=AIVEN_SSLROOT,
                sslmode="verify-ca",
                **kwargs,
            )
        return psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            sslmode="require",
            **kwargs,
        )


def init_schema():
    """
    Create table using float8[] embeddings (works on managed providers).
    """
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mygov_services (
                id bigserial PRIMARY KEY,
                bn_name text,
                en_name text,
                keywords text,
                embedding float8[] NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS mygov_services_idx ON mygov_services (bn_name, en_name, keywords);")
    print("[✓] Schema ready / ensured")


# -------------------------
# Embedding utilities
# -------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings via OpenAI-compatible endpoint.
    Ensures result vectors have EMBEDDING_DIM elements (truncate or pad).
    """
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings: List[List[float]] = []
    for item in resp.data:
        emb = item.embedding
        if len(emb) > EMBEDDING_DIM:
            emb = emb[:EMBEDDING_DIM]
        elif len(emb) < EMBEDDING_DIM:
            emb = list(emb) + [0.0] * (EMBEDDING_DIM - len(emb))
        embeddings.append(list(emb))
    return embeddings


# -------------------------
# Data loading + prepare
# -------------------------
def load_mygov_data() -> List[Dict]:
    """
    Robust loader:
      - If file starts with '[' or '{' try json.loads (array or single object).
      - Otherwise attempt NDJSON: parse each non-empty line as JSON.
      - On NDJSON parsing errors, report the offending line(s).
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    text = DATA_FILE.read_text(encoding="utf-8")
    if not text or not text.strip():
        return []

    text = text.strip()

    # Try normal JSON (array or single object)
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as e:
            # Fall through to NDJSON attempt, but log the error
            print(f"[!] json.loads failed (maybe NDJSON). error: {e}")

    # Try NDJSON (one JSON object per line)
    rows: List[Dict] = []
    errors = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            # Save some context for debugging: line number + short snippet + error
            snippet = line if len(line) <= 200 else line[:200] + "..."
            errors.append((i, snippet, str(e)))

    if errors:
        print("[!] Some lines failed to parse as JSON (NDJSON attempt). Example errors:")
        for ln, snippet, err in errors[:10]:
            print(f"  line {ln}: {err}  snippet: {snippet}")
        if rows:
            print(f"[!] Parsed {len(rows)} objects but {len(errors)} lines failed. Returning parsed objects.")
            return rows
        raise RuntimeError(f"Failed to parse JSON file as array/object or NDJSON. {len(errors)} parse errors. Example: {errors[:3]}")

    return rows


def prepare_docs(rows: List[Dict]) -> List[Tuple[str, str, str, str, str]]:
    """
    Return list of tuples: (doc_id, bn_name, en_name, keywords, text_for_embedding)
    Works with the new MyGov JSON format (name, name_en, keyword).
    """
    docs: List[Tuple[str, str, str, str, str]] = []
    for i, row in enumerate(rows):
        # New dump format uses "name", "name_en", "keyword"
        bn_name = (row.get("name") or "").strip()
        en_name = (row.get("name_en") or "").strip()
        keywords = (row.get("keyword") or "").strip()
        text = " ".join(p for p in [bn_name, en_name, keywords] if p).strip()
        if not text:
            continue
        doc_id = f"row_{i}"
        docs.append((doc_id, bn_name, en_name, keywords, text))
    return docs


# -------------------------
# Lexical modeling
# -------------------------
def build_lexical_models(
    docs: List[Dict[str, str]], max_prefix_len: int = 3, max_per_prefix: int = 10
):
    prefix_map: Dict[str, Counter] = defaultdict(Counter)
    bigram_counts: Dict[str, Counter] = defaultdict(Counter)
    vocab = set()

    for d in docs:
        text = d["text"].strip().lower()
        if not text:
            continue
        words = text.split()
        if not words:
            continue
        for w in words:
            vocab.add(w)
        for i in range(1, min(len(words), max_prefix_len) + 1):
            prefix = " ".join(words[:i])
            prefix_map[prefix][text] += 1
        for w1, w2 in zip(words, words[1:]):
            bigram_counts[w1][w2] += 1

    prefix_sug = {p: [ph for ph, _ in c.most_common(max_per_prefix)] for p, c in prefix_map.items()}
    next_w = {w1: [w2 for w2, _ in c.most_common()] for w1, c in bigram_counts.items()}
    vocab_sorted = sorted(vocab)
    return prefix_sug, next_w, vocab_sorted


def suggest_queries(query: str, max_suggestions: int = 5) -> List[str]:
    q_raw = query.strip()
    if not q_raw:
        return []
    q = q_raw.lower()
    suggestions: List[str] = []
    seen = set()
    words_raw = q_raw.split()
    words = q.split()

    # 0) last word completion
    if words:
        last_word = words[-1]
        completions = [w for w in vocab_words if w.startswith(last_word) and w != last_word]
        for w in completions:
            phrase = " ".join(words_raw[:-1] + [w])
            if phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
            if len(suggestions) >= max_suggestions:
                return suggestions

    # 1) bigram next word
    if words:
        last = words[-1]
        if last in next_word:
            for w2 in next_word[last]:
                phrase = q_raw + " " + w2
                if phrase not in seen:
                    seen.add(phrase)
                    suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    return suggestions

    # 2) full prefix -> phrase
    if q in prefix_suggestions and len(suggestions) < max_suggestions:
        for phrase in prefix_suggestions[q]:
            if phrase.lower() == q:
                continue
            if phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
            if len(suggestions) >= max_suggestions:
                return suggestions

    # 3) smaller prefix fallback
    if words and len(suggestions) < max_suggestions:
        for i in range(len(words), 0, -1):
            prefix = " ".join(words[:i])
            if prefix in prefix_suggestions:
                for phrase in prefix_suggestions[prefix]:
                    if phrase.lower() == q:
                        continue
                    if phrase not in seen:
                        seen.add(phrase)
                        suggestions.append(phrase)
                    if len(suggestions) >= max_suggestions:
                        break
            if len(suggestions) >= max_suggestions:
                break

    return suggestions[:max_suggestions]


# -------------------------
# Upsert and indexing
# -------------------------
def upsert_services(rows: List[Tuple[str, str, str, str, List[float]]]):
    with get_pg_conn() as conn, conn.cursor() as cur:
        for doc_id, bn, en, kw, emb in rows:
            try:
                cur.execute(
                    """
                    INSERT INTO mygov_services (bn_name, en_name, keywords, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                    """,
                    (bn, en, kw, emb),
                )
            except Exception as e:
                print(f"[!] Upsert error for doc {doc_id}: {e}")


def index_dataset(batch_size: int = 64):
    global prefix_suggestions, next_word, vocab_words

    print("[*] init schema ...")
    init_schema()

    print("[*] loading data ...")
    rows = load_mygov_data()
    print(f"[*] rows loaded: {len(rows)}")

    docs_tuples = prepare_docs(rows)
    print(f"[*] docs prepared: {len(docs_tuples)}")

    # build lexical models
    lex_docs = [{"id": t[0], "text": t[4]} for t in docs_tuples]
    prefix_suggestions, next_word, vocab_words = build_lexical_models(lex_docs)
    print("[*] lexical models built")

    # batch embedding + upsert
    total = len(docs_tuples)
    for start in range(0, total, batch_size):
        chunk = docs_tuples[start: start + batch_size]
        texts = [t[4] for t in chunk]
        print(f"[*] embedding batch {start}-{start+len(chunk)-1}")
        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            print(f"[!] Embedding API error: {e}")
            break

        pg_rows = []
        for (doc_id, bn, en, kw, _), emb in zip(chunk, embeddings):
            pg_rows.append((doc_id, bn, en, kw, emb))
        upsert_services(pg_rows)
    print("[*] indexing complete")


# -------------------------
# Search implementation
# -------------------------
def search_services(query: str, k: int = 5):
    q_emb = embed_texts([query])[0]

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT bn_name, en_name, keywords,
                (
                  (SELECT SUM(a*b)
                   FROM unnest(%s::double precision[]) WITH ORDINALITY AS qa(a, idx)
                   JOIN unnest(embedding) WITH ORDINALITY AS qb(b, idx) USING (idx)
                  )
                  /
                  ( sqrt((SELECT SUM(x*x) FROM unnest(%s::double precision[]) AS t(x))) *
                    sqrt((SELECT SUM(y*y) FROM unnest(embedding) AS t(y)))
                  )
                )::double precision AS score
            FROM mygov_services
            ORDER BY score DESC
            LIMIT %s;
            """,
            (q_emb, q_emb, k),
        )
        rows = cur.fetchall()
    return rows


# -------------------------
# CLI usage
# -------------------------
if __name__ == "__main__":
    # Index (create table + upsert)
    index_dataset(batch_size=64)

    
