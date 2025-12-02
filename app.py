import json
from collections import defaultdict, Counter
from typing import List, Dict

import streamlit as st
from openai import OpenAI

import psycopg
from pgvector.psycopg import register_vector, Vector as PgVector

# =========================
# CONFIG
# =========================

EMBEDDING_MODEL = "local-model"  # indexing এ যে model ব্যবহার করেছো
EMBEDDING_DIM = 768              # pgvector টেবিলে যে dimension

PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "mgov"
PG_USER = "postgres"
PG_PASSWORD = "postgres"

MYGOV_JSON_PATH = "mygov_data.json"  # lexical layer এর জন্য JSON


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)


# =========================
# PG / PGVECTOR
# =========================

def get_pg_conn():
    """
    প্রতিবার নতুন connection খুলি, কাজ শেষে context manager ওটা বন্ধ করে দেবে।
    """
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
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
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
    pgvector দিয়ে cosine distance search
    embedding কে PgVector এ কনভার্ট করে পাঠাচ্ছি, যাতে <=> operator ঠিকমতো কাজ করে।
    """
    vec = PgVector(embedding)

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
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
    query embedding — indexing এর সময় যে model+dim ব্যবহার করেছো, সেটার সাথেই match থাকতে হবে।
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    emb = resp.data[0].embedding

    # dimension match করি
    if len(emb) > EMBEDDING_DIM:
        emb = emb[:EMBEDDING_DIM]
    elif len(emb) < EMBEDDING_DIM:
        # চাইলে padding করতে পারো; আপাতত যেমন আছে তেমনই রাখছি
        pass

    return list(emb)


# =========================
# LEXICAL DATA LOAD
# =========================

@st.cache_data
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


def build_lexical_models(
    docs, max_prefix_len: int = 1, max_per_prefix: int = 10
):
    """
    returns:
      prefix_bn: prefix -> [Bangla phrases]
      prefix_en: prefix -> [English phrases]
      all_bn   : list of all unique Bangla names
      all_en   : list of all unique English names
      next_bn  : word  -> [next words]  (bigram)
      next_en  : word  -> [next words]  (bigram)
      vocab_bn : list of unique Bangla words
      vocab_en : list of unique English words
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


@st.cache_data
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
    """
    suggestion:
    0) last-word completion (vocab থেকে)
    1) bigram-based next word
    """
    q_raw = query.strip()
    if not q_raw:
        return []

    use_bn = is_bangla(q_raw)
    q = q_raw.lower()

    next_dict = next_bn if use_bn else next_en
    vocab = vocab_bn if use_bn else vocab_en

    suggestions: List[str] = []
    seen = set()

    words_raw = q_raw.split()
    words = q.split()

    # 0) last word completion: "app" -> "application"
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

    # 1) bigram next word: "application" -> "application for"
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

    return suggestions[:max_suggestions]


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="MyGov Search (pgvector)", layout="wide")

st.title("MyGov Semantic Search (pgvector)")
st.write("Type a query below to search your MyGov dataset (Bangla + English).")

# schema ensure
init_schema()

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""


def set_query(new_q: str):
    st.session_state["query_text"] = new_q


query_text = st.text_input("Search query", key="query_text")

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

if query_text:
    # suggestions
    suggestions = suggest_queries(
        query_text,
        prefix_bn,
        prefix_en,
        all_bn,
        all_en,
        next_bn,
        next_en,
        vocab_bn,
        vocab_en,
        max_suggestions=5,
    )

    if suggestions:
        st.markdown("**Suggestions:**")
        # 👉 এখানেই পরিবর্তন: আর columns ব্যবহার করছি না,
        # এক লাইনে একটাঃ vertically stack হবে
        for i, s in enumerate(suggestions):
            st.button(s, key=f"sugg_{i}", on_click=set_query, args=(s,))

    # vector search via pgvector
    search_query = query_text
    query_embedding = get_embedding(search_query)

    try:
        rows = pg_vector_search(query_embedding, top_k=10)
    except Exception as e:
        st.error(f"Search error: {e}")
        rows = []

    if not rows:
        st.info("No results found.")
    else:
        st.markdown(f"### Results for: `{search_query}`")
        for rank, (doc_id, bn_name, en_name, keywords, profile, score) in enumerate(
            rows, start=1
        ):
            bn_name = (bn_name or "").strip()
            en_name = (en_name or "").strip()
            keywords = (keywords or "").strip()
            profile = (profile or "").strip()

            title = bn_name or en_name or "(No name)"
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

            st.markdown(f"#### {title}")
            if en_name and en_name != title:
                st.write(en_name)

            if keywords:
                st.caption(f"Keywords: {keywords}")
            if profile:
                st.caption(f"Profile: {profile}")

            st.caption(f"Rank {rank} • Score {score_text} • doc_id {doc_id}")
            st.markdown("---")
else:
    st.info("Type something to see suggestions and search results.")
