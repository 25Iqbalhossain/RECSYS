import mysql.connector

from data.testconfig import MYSQL_CONFIG


def fetch_contents(limit: int | None = None):
    """
    Fetch rows from the `contents` table as a list of dicts.
    """
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor(dictionary=True)

    sql = """
        SELECT 
            id,
            title,
            body,
            category
        FROM contents
    """
    if limit:
        sql += " LIMIT %s"
        cursor.execute(sql, (limit,))
    else:
        cursor.execute(sql)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

