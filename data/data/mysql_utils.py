# data/mysql_utils.py
import mysql.connector
from data.testconfig  import MYSQL_CONFIG

def fetch_contents(limit: int | None = None):
    """
    MySQL থেকে প্রয়োজনীয় ফিল্ডগুলো নিয়ে আসবে।
    এখানে উদাহরণ হিসেবে টেবিলের নাম ধরেছি `contents`
    আর কলাম: id, title, body, category (তুমি নিজের মত করে বদলে নেবে)
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
