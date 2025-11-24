import mysql.connector
import pandas as pd
import json


class MyGovDB:
    def __init__(self, host, user, password, database, port=3306):
        self.cnx = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )

    def get_profile_services(self):
        query = """
            SELECT
                nsp_profile.id AS nsp_profile_id,
                nsp_service.sid AS nsp_service_service_id,
                my_gov_service.name AS my_gov_service_name,
                my_gov_service.name_en AS my_gov_service_name_en,
                my_gov_service.keyword AS my_gov_service_keyword,
                nsp_profile.name as nsp_profile_name
            FROM my_gov_service
            JOIN nsp_service ON my_gov_service.id = nsp_service.sid
            JOIN nsp_profile ON nsp_profile.id = nsp_service.uid;
        """
        cursor = self.cnx.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def close(self):
        self.cnx.close()


if __name__ == "__main__":
    db = MyGovDB(
        host="localhost",
        user="root",
        password="",
        database="mygov",
    )

    dataset = db.get_profile_services()

    df = pd.DataFrame(dataset)
    df.insert(0, "index", range(1, len(df) + 1))

    records = df.to_dict(orient="records")

    with open("mygov_data.json", "w", encoding="utf-8") as f:
        for row in records:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    db.close()

    mygovcsv = df.to_csv("Mygovdata.csv", index=False, encoding="utf-8-sig")


