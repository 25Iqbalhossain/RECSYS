# db/mysql_client.py

import os

import mysql.connector
from mysql.connector import Error

from data.testconfig import MYSQL_CONFIG

TABLE_NAME = os.getenv("MYSQL_TABLE", "contents")
BATCH_SIZE = int(os.getenv("MYSQL_BATCH_SIZE", "1000"))


def get_connection():
    """
    MySQL connection create kore.
    config.MYSQL_CONFIG theke info ney.
    """
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print("Error while connecting to MySQL:", e)
        raise


def fetch_rows(batch_size: int = BATCH_SIZE):
    """
    Generator: protibar ek batch row (list of dict) return korbe.
    Use:
        for rows in fetch_rows():
            ...
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
