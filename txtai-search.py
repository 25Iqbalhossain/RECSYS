import json

import openai
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from txtai.pipeline import Similarity



client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def get_embedding(text, prefix="search_document: "):
    response = client.embeddings.create(
        model="local-model",
        input=prefix + text
    )
    return response.data[0].embedding[:256]



db_path = "./lancedb"
db = lancedb.connect(db_path)
table_name = "txtai_lancedb"

class Document(LanceModel):
    id: int = Field()
    text: str = Field()
    vector: Vector(256) = Field()

if table_name not in db.table_names():
    db.create_table(table_name, schema=Document)
table = db.open_table(table_name)



json_path = r"C:\Users\hi\OneDrive\Desktop\New folder\mygov_data.json"

def load_json_documents(path: str):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            uid = int(row["index"])

            parts = [
                str(row.get("my_gov_service_name", "")),
                str(row.get("my_gov_service_name_en", "")),
                str(row.get("my_gov_service_keyword", "")),
                str(row.get("nsp_profile_name", "")),
            ]
            text = " ".join(p for p in parts if p)

            if not text:
                continue

            docs.append({"id": uid, "text": text})
    return docs

documents = load_json_documents(json_path)


data_to_add = []
for doc in documents:
    embedding = get_embedding(doc["text"])
    data_to_add.append(
        Document(id=doc["id"], text=doc["text"], vector=embedding)
    )

if data_to_add:
    table.add(data_to_add)



similarity = Similarity()

query_text = "আর্থিক ক্ষমতা প্রদান (সরকারি কলেজ শাখা কর্তৃক"
query_embedding = get_embedding(query_text, prefix="search_query: ")

results = table.search(query_embedding).limit(5).to_list()

ranked = similarity(query_text, [r["text"] for r in results])

for idx, score in ranked:
    rec = results[idx]
    print(f"Score: {score:.4f}  id={rec['id']}  text={rec['text']}")
