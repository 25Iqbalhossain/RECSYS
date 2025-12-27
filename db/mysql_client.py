import os
from typing import Any, Dict, List, Tuple

import mysql.connector
from mysql.connector import Error

from data.testconfig import MYSQL_CONFIG

TABLE_NAME = os.getenv("MYSQL_TABLE", "contents")
BATCH_SIZE = int(os.getenv("MYSQL_BATCH_SIZE", "1000"))


def get_connection():
    """
    Create a MySQL connection using MYSQL_CONFIG.
    """
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print("Error while connecting to MySQL:", e)
        raise


def fetch_rows(batch_size: int = BATCH_SIZE):
    """
    Generator: stream all rows from the table in fixed-size batches.

    Useful for offline jobs that need to scan the entire contents table.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    offset = 0
    try:
        while True:
            query = f"SELECT * FROM {TABLE_NAME} LIMIT %s OFFSET %s"
            cursor.execute(query, (batch_size, offset))
            rows = cursor.fetchall()

            if not rows:
                break

            yield rows
            offset += batch_size
    finally:
        cursor.close()
        conn.close()


def search_candidates_by_keyword(
    q: str,
    max_candidates: int = 200,
    type_filter: str | None = None,
    language_filter: str | None = None,
    only_active: bool = True,
) -> List[Dict[str, Any]]:
    """
    Lightweight keyword-based candidate search over the contents table.

    This is intended for *hybrid search*: use SQL LIKE to quickly narrow
    down to rows containing the main query terms, then run vector search
    within those candidates.

    Assumptions:
    - The table has columns: id, title, body, type, language, is_active.
    - We match only on title/body using a simple LIKE '%term%' pattern.
    - For Bangla queries we treat whitespace-separated tokens as terms.

    Returns a list of row dicts (MySQL cursor `dictionary=True` style).
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Pick a simple main term: the longest non-empty token.
    tokens = [t.strip() for t in str(q).split() if t.strip()]
    main_term = tokens[0] if tokens else str(q)

    sql = f"SELECT * FROM {TABLE_NAME} WHERE 1=1"
    params: List[Any] = []

    if only_active:
        sql += " AND is_active = 1"

    if type_filter:
        sql += " AND type = %s"
        params.append(type_filter)

    if language_filter:
        sql += " AND language = %s"
        params.append(language_filter)

    # Basic LIKE filter on title/body. For production you could use
    # FULLTEXT/BM25 instead.
    like_pattern = f"%{main_term}%"
    sql += " AND (title LIKE %s OR body LIKE %s)"
    params.extend([like_pattern, like_pattern])

    sql += " LIMIT %s"
    params.append(max_candidates)

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    return rows
