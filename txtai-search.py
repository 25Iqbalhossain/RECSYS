import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from openai import OpenAI
import psycopg
from pgvector.psycopg import register_vector


EMBEDDING_MODEL ="nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBEDDING_DIM = 768                 
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "mgov"
PG_USER = "postgres"
PG_PASSWORD = "postgres"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "dataset" / "mygov_data.json"


DATA_FILE = Path(r"C:\Users\hi\OneDrive\Desktop\New folder\mygov_data.json")

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)




def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    একসাথে একাধিক টেক্সট embed করে list[list[float]] রিটার্ন করবে
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings: List[List[float]] = []
    for item in resp.data:
        emb = item.embedding
    
        if len(emb) > EMBEDDING_DIM:
            emb = emb[:EMBEDDING_DIM]
        elif len(emb) < EMBEDDING_DIM:
         
            pass
        embeddings.append(list(emb))

    return embeddings



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
    pgvector extension + table + index create করে
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


def upsert_services(rows: List[Tuple[str, str, str, str, str, List[float]]]):
    """
    rows: [(doc_id, bn_name, en_name, keywords, profile, embedding), ...]
    """
    with get_pg_conn() as conn, conn.cursor() as cur:
        for doc_id, bn_name, en_name, keywords, profile, emb in rows:
            cur.execute(
                """
                INSERT INTO mygov_services (
                    doc_id, bn_name, en_name, keywords, profile, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                SET bn_name  = EXCLUDED.bn_name,
                    en_name  = EXCLUDED.en_name,
                    keywords = EXCLUDED.keywords,
                    profile  = EXCLUDED.profile,
                    embedding= EXCLUDED.embedding;
                """,
                (doc_id, bn_name, en_name, keywords, profile, emb),
            )




def load_mygov_data() -> List[Dict]:
  
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    text = DATA_FILE.read_text(encoding="utf-8").strip()

 
    if text.startswith("["):
        data = json.loads(text)
        return list(data if isinstance(data, list) else [data])

 
    data: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))
    return data


def build_doc_id(row: Dict) -> str:
    
    profile_id = row.get("nsp_profile_id")
    service_id = row.get("nsp_service_service_id")
    idx = row.get("index")
    return f"{profile_id}_{service_id}_{idx}"


def prepare_docs(rows: List[Dict]) -> List[Tuple[str, str, str, str, str, str]]:
    """
    প্রত্যেক row -> (doc_id, bn_name, en_name, keywords, profile, text_for_embedding)
    """
    docs: List[Tuple[str, str, str, str, str, str]] = []
    for row in rows:
        bn_name = row.get("my_gov_service_name", "") or ""
        en_name = row.get("my_gov_service_name_en", "") or ""
        keywords = row.get("my_gov_service_keyword", "") or ""
        profile = row.get("nsp_profile_name", "") or ""
        doc_id = build_doc_id(row)

     
        text = "\n".join(p for p in [bn_name, en_name, keywords, profile] if p)
        if not text.strip():
            continue

        docs.append((doc_id, bn_name, en_name, keywords, profile, text))
    return docs




prefix_suggestions: Dict[str, List[str]] = {}
next_word: Dict[str, List[str]] = {}
vocab_words: List[str] = []


def build_lexical_models(
    docs: List[Dict[str, str]],
    max_prefix_len: int = 3,
    max_per_prefix: int = 10,
):
    """
    docs: list of {"id": str, "text": str}
    returns:
        prefix_suggestions: prefix(str) -> list of phrases(str)
        next_word:          word(str)   -> list of next words(str)
        vocab_words:        list of all distinct words (lowercase)
    """
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


    prefix_sug: Dict[str, List[str]] = {}
    for prefix, counter in prefix_map.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_sug[prefix] = phrases

    # next_word mapping বানাই
    next_w: Dict[str, List[str]] = {}
    for w1, counter in bigram_counts.items():
        words_sorted = [w2 for w2, _ in counter.most_common()]
        next_w[w1] = words_sorted

    vocab_words_sorted = sorted(vocab)

    return prefix_sug, next_w, vocab_words_sorted


def suggest_queries(query: str, max_suggestions: int = 5) -> List[str]:
    """
    Autocomplete / query suggestion:

    0) word-level completion: "app" -> "application"
    1) bigram-based next word: "application" -> "application for"
    2) prefix-based phrase suggestions
    3) shorter prefix fallback
    """
    raw = query.strip()
    q = raw.lower()
    if not q:
        return []

    suggestions: List[str] = []
    seen = set()

    words_raw = raw.split()
    words = q.split()

    # 0) last word completion
    if words:
        last_word = words[-1]

        completions = [
            w for w in vocab_words
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
        if last in next_word:
            for w2 in next_word[last]:
                phrase = raw + " " + w2
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




def index_dataset(batch_size: int = 64):
    """
    JSON data -> lexical models -> embedding -> pgvector table-এ upsert
    """
    global prefix_suggestions, next_word, vocab_words

    init_schema()
    rows = load_mygov_data()
    print(f"rows: {len(rows)}")

    docs = prepare_docs(rows)
    print(f"docs to embed: {len(docs)}")


    lex_docs = [
        {"id": doc_id, "text": text}
        for (doc_id, bn_name, en_name, keywords, profile, text) in docs
    ]
    prefix_suggestions, next_word, vocab_words = build_lexical_models(lex_docs)
    print("lexical models built ")


    for start in range(0, len(docs), batch_size):
        chunk = docs[start: start + batch_size]
        print(f"embedding {start}-{start + len(chunk) - 1}")
        texts = [c[5] for c in chunk]
        embeddings = embed_texts(texts)

        pg_rows = []
        for (doc_id, bn_name, en_name, keywords, profile, _), emb in zip(
            chunk, embeddings
        ):
            pg_rows.append((doc_id, bn_name, en_name, keywords, profile, emb))

        upsert_services(pg_rows)

    print("indexing done ")




def search_services(query: str, k: int = 5):
    """
    simple vector search using cosine distance
    """

    [emb] = embed_texts([query])

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
            (emb, emb, k),
        )
        rows = cur.fetchall()

    return rows


if __name__ == "__main__":
 
    index_dataset()

