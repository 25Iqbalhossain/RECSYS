import argparse
from typing import List, Tuple

from openai import OpenAI
from embeddings import get_pg_conn, EMBEDDING_MODEL

TOP_K_DEFAULT = 100

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
)


def embed_query(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def search_mygov(
    query: str,
    top_k: int = TOP_K_DEFAULT,
    profile_filter: str | None = None,
) -> list[Tuple]:
    q_emb = embed_query(query)

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL ivfflat.probes = 10;")

        if profile_filter:
            cur.execute(
                """
                SELECT
                    doc_id,
                    bn_name,
                    en_name,
                    keywords,
                    profile,
                    1 - (embedding <-> %s) AS similarity
                FROM mygov_services
                WHERE profile = %s
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (q_emb, profile_filter, q_emb, top_k),
            )
        else:
            cur.execute(
                """
                SELECT
                    doc_id,
                    bn_name,
                    en_name,
                    keywords,
                    profile,
                    1 - (embedding <-> %s) AS similarity
                FROM mygov_services
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (q_emb, q_emb, top_k),
            )

        return cur.fetchall()
    
    
