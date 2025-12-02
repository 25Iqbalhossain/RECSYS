import json
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field

from openai import OpenAI
import psycopg
from pgvector.psycopg import register_vector, Vector as PgVector

# =========================
# CONFIG
# =========================

EMBEDDING_MODEL = "local-model"  # same model used when indexing
EMBEDDING_DIM = 768              # same dimension used in pgvector table

PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "mgov"
PG_USER = "postgres"
PG_PASSWORD = "postgres"

MYGOV_JSON_PATH = "mygov_data.json"  # lexical layer-এর জন্য JSON


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)


# =========================
# PG / PGVECTOR
# =========================

def get_pg_conn():
    conn = psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        autocommit=True,
    )
    register_vector(conn)
    return conn


def init_schema():
    """
    extension + table + index আছে কি না নিশ্চিত করি।
    এখানে ডাটা insert করা হচ্ছে না – সেটা আলাদা indexing script দিয়ে করবে।
    """
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS mygov_services (
                id          bigserial PRIMARY KEY,
                doc_id      text UNIQUE,
                bn_name     text,
                en_name     text,
                keywords    text,
                profile     text,
                embedding   vector({EMBEDDING_DIM}) NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS mygov_services_embedding_idx
            ON mygov_services
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )


def pg_vector_search(embedding: List[float], top_k: int = 10):
    """
    pgvector দিয়ে cosine distance-ভিত্তিক search
    NOTE: এখানে embedding-কে PgVector এ কনভার্ট করছি,
    যাতে 'vector <=> vector' অপারেটর ঠিকভাবে কাজ করে।
    """
    vec = PgVector(embedding)

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                doc_id,
                bn_name,
                en_name,
                keywords,
                profile,
                1 - (embedding <=> %s) AS score
            FROM mygov_services
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (vec, vec, top_k),
        )
        rows = cur.fetchall()
    return rows


# =========================
# EMBEDDINGS
# =========================

def get_embedding(text: str) -> List[float]:
    """
    query embedding — indexing এর সময় যে model+dim ব্যবহার করেছো,
    এখানে সেটাই match করতে হবে।
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    emb = resp.data[0].embedding
    # dimension match করা দরকার
    if len(emb) > EMBEDDING_DIM:
        emb = emb[:EMBEDDING_DIM]
    return list(emb)


# =========================
# DATA LOAD (search docs + lexical)
# =========================

def load_search_documents(path: str = MYGOV_JSON_PATH):
    """
    docs:
      - text -> full text (name + keyword + profile) for lexical build
      - bn   -> Bangla service name
      - en   -> English service name
    """
    docs = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                bn = str(row.get("my_gov_service_name", "") or "").strip()
                en = str(row.get("my_gov_service_name_en", "") or "").strip()
                keywords = str(row.get("my_gov_service_keyword", "") or "").strip()
                profile = str(row.get("nsp_profile_name", "") or "").strip()

                parts_full = [bn, en, keywords, profile]
                text = " ".join(p for p in parts_full if p)
                if not text:
                    continue

                docs.append({"bn": bn, "en": en, "text": text})
    except FileNotFoundError:
        pass
    return docs


# =========================
# LEXICAL LAYER (BN + EN)
# =========================

def build_lexical_models(
    docs,
    max_prefix_len: int = 1,
    max_per_prefix: int = 10,
):
    """
    returns:
      prefix_bn: prefix -> [Bangla phrases]
      prefix_en: prefix -> [English phrases]
      all_bn   : list of all unique Bangla names
      all_en   : list of all unique English names
      next_bn  : word  -> [next words]  (bigram)
      next_en  : word  -> [next words]  (bigram)
      vocab_bn : list of all Bangla words
      vocab_en : list of all English words
    """
    prefix_map_bn: Dict[str, Counter] = defaultdict(Counter)
    prefix_map_en: Dict[str, Counter] = defaultdict(Counter)
    bigram_bn: Dict[str, Counter] = defaultdict(Counter)
    bigram_en: Dict[str, Counter] = defaultdict(Counter)

    all_bn = set()
    all_en = set()

    vocab_bn = set()
    vocab_en = set()

    for d in docs:
        bn_phrase = (d["bn"] or "").strip()
        en_phrase = (d["en"] or "").strip()

        # Bangla side
        if bn_phrase:
            all_bn.add(bn_phrase)
            phrase = bn_phrase.lower()
            words = phrase.split()

            for w in words:
                vocab_bn.add(w)

            for i in range(1, min(len(words), max_prefix_len) + 1):
                prefix = " ".join(words[:i])
                prefix_map_bn[prefix][bn_phrase] += 1

            for w1, w2 in zip(words, words[1:]):
                bigram_bn[w1][w2] += 1

        # English side
        if en_phrase:
            all_en.add(en_phrase)
            phrase_en = en_phrase.lower()
            words_en = phrase_en.split()

            for w in words_en:
                vocab_en.add(w)

            for i in range(1, min(len(words_en), max_prefix_len) + 1):
                prefix = " ".join(words_en[:i])
                prefix_map_en[prefix][en_phrase] += 1

            for w1, w2 in zip(words_en, words_en[1:]):
                bigram_en[w1][w2] += 1

    prefix_bn: Dict[str, List[str]] = {}
    for prefix, counter in prefix_map_bn.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_bn[prefix] = phrases

    prefix_en: Dict[str, List[str]] = {}
    for prefix, counter in prefix_map_en.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_en[prefix] = phrases

    next_bn: Dict[str, List[str]] = {}
    for w1, counter in bigram_bn.items():
        next_bn[w1] = [w2 for w2, _ in counter.most_common()]

    next_en: Dict[str, List[str]] = {}
    for w1, counter in bigram_en.items():
        next_en[w1] = [w2 for w2, _ in counter.most_common()]

    all_bn_list = sorted(all_bn)
    all_en_list = sorted(all_en)

    vocab_bn_list = sorted(vocab_bn)
    vocab_en_list = sorted(vocab_en)

    return (
        prefix_bn,
        prefix_en,
        all_bn_list,
        all_en_list,
        next_bn,
        next_en,
        vocab_bn_list,
        vocab_en_list,
    )


@lru_cache(maxsize=1)
def get_lexical_data(path: str = MYGOV_JSON_PATH):
    docs = load_search_documents(path)
    return build_lexical_models(docs)


def is_bangla(text: str) -> bool:
    for ch in text:
        if "\u0980" <= ch <= "\u09FF":
            return True
    return False


def suggest_queries(
    query: str,
    prefix_bn: Dict[str, List[str]],
    prefix_en: Dict[str, List[str]],
    all_bn: List[str],
    all_en: List[str],
    next_bn: Dict[str, List[str]],
    next_en: Dict[str, List[str]],
    vocab_bn: List[str],
    vocab_en: List[str],
    max_suggestions: int = 5,
) -> List[str]:
    q_raw = query.strip()
    if not q_raw:
        return []

    use_bn = is_bangla(q_raw)
    q = q_raw.lower()

    prefix_dict = prefix_bn if use_bn else prefix_en
    all_phrases = all_bn if use_bn else all_en
    next_dict = next_bn if use_bn else next_en
    vocab = vocab_bn if use_bn else vocab_en

    suggestions: List[str] = []
    seen = set()

    words_raw = q_raw.split()
    words = q.split()

    # 0) last word completion
    if words:
        last_word = words[-1]
        completions = [
            w for w in vocab
            if w.startswith(last_word) and w != last_word
        ]
        for w in completions:
            new_tokens = words_raw[:-1] + [w]
            phrase = " ".join(new_tokens)
            if phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
            if len(suggestions) >= max_suggestions:
                return suggestions

    # 1) bigram next word
    if words:
        last = words[-1]
        if last in next_dict:
            for w2 in next_dict[last]:
                phrase = q_raw + " " + w2
                if phrase not in seen:
                    seen.add(phrase)
                    suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    return suggestions

    # 2) full prefix -> phrase
    if q in prefix_dict and len(suggestions) < max_suggestions:
        for phrase in prefix_dict[q]:
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
            if prefix in prefix_dict:
                for phrase in prefix_dict[prefix]:
                    if phrase.lower() == q:
                        continue
                    if phrase not in seen:
                        seen.add(phrase)
                        suggestions.append(phrase)
                    if len(suggestions) >= max_suggestions:
                        break
            if len(suggestions) >= max_suggestions:
                break

    # 4) substring match fallback
    if len(suggestions) < max_suggestions:
        for phrase in all_phrases:
            if q in phrase.lower() and phrase not in seen and phrase.lower() != q:
                seen.add(phrase)
                suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    break

    return suggestions[:max_suggestions]


# =========================
# SCHEMAS
# =========================

class SuggestRequest(BaseModel):
    query: str
    max_suggestions: int = 5


class SuggestResponse(BaseModel):
    query: str
    suggestions: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchItem(BaseModel):
    # pgvector টেবিলের doc_id (string)
    id: str = Field(..., description="Document ID (doc_id from pg)")
    bn: Optional[str] = None
    en: Optional[str] = None
    text: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]


# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="MyGov Search API (pgvector)")


@app.on_event("startup")
def on_startup():
    init_schema()


@app.post("/suggest", response_model=SuggestResponse)
def suggest_api(body: SuggestRequest):
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

    suggestions = suggest_queries(
        body.query,
        prefix_bn,
        prefix_en,
        all_bn,
        all_en,
        next_bn,
        next_en,
        vocab_bn,
        vocab_en,
        max_suggestions=body.max_suggestions,
    )
    return SuggestResponse(query=body.query, suggestions=suggestions)


@app.post("/search", response_model=SearchResponse)
def search_api(body: SearchRequest):
    query_text = body.query.strip()
    if not query_text:
        return SearchResponse(query=body.query, results=[])

    query_embedding = get_embedding(query_text)
    rows = pg_vector_search(query_embedding, top_k=body.top_k)

    results: List[SearchItem] = []

    for doc_id, bn_name, en_name, keywords, profile, score in rows:
        bn_name = (bn_name or "").strip() or None
        en_name = (en_name or "").strip() or None
        keywords = (keywords or "").strip()
        profile = (profile or "").strip()

        full_text_parts = [bn_name or "", en_name or "", keywords, profile]
        full_text = " ".join(p for p in full_text_parts if p)

        score_val: Optional[float] = float(score) if isinstance(score, (int, float)) else None

        results.append(
            SearchItem(
                id=str(doc_id),
                bn=bn_name,
                en=en_name,
                text=full_text,
                score=score_val,
            )
        )

    return SearchResponse(query=body.query, results=results)
