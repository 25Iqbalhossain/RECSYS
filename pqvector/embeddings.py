import json
from pathlib import Path
from typing import List, Dict, Tuple

from openai import OpenAI
import psycopg
from pgvector.psycopg import register_vector

EMBEDDING_DIM = 768  

PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "mgov"
PG_USER = "postgres"
PG_PASSWORD = "postgres"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "dataset" / "mygov_data.json"


EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


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
    docs = []
    for row in rows:
        bn_name = row.get("my_gov_service_name", "") or ""
        en_name = row.get("my_gov_service_name_en", "") or ""
        keywords = row.get("my_gov_service_keyword", "") or ""
        profile = row.get("nsp_profile_name", "") or ""
        doc_id = build_doc_id(row)
        text = "\n".join(p for p in [bn_name, en_name, keywords, profile] if p)
        docs.append((doc_id, bn_name, en_name, keywords, profile, text))
    return docs


def index_dataset(batch_size: int = 64):
    init_schema()
    rows = load_mygov_data()
    print(f"rows: {len(rows)}")

    docs = prepare_docs(rows)

    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        print(f"embedding {start}-{start + len(chunk) - 1}")
        texts = [c[5] for c in chunk]
        embeddings = embed_texts(texts)

        pg_rows = []
        for (doc_id, bn_name, en_name, keywords, profile, _), emb in zip(
            chunk, embeddings
        ):
            pg_rows.append((doc_id, bn_name, en_name, keywords, profile, emb))

        upsert_services(pg_rows)

    print("done")


if __name__ == "__main__":
    index_dataset()