import json
import lancedb
import openai
import streamlit as st
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field


client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def get_embedding(text: str, prefix: str = "search_query: "):
    response = client.embeddings.create(model="local-model", input=prefix + text)
    return response.data[0].embedding[:256]


db_path = "./lancedb"
db = lancedb.connect(db_path)
table_name = "txtai_lancedb"


class Document(LanceModel):
    id: int = Field()
    text: str = Field()
    vector: Vector(256) = Field()


if table_name not in db.table_names():
    table = db.create_table(table_name, schema=Document)
else:
    table = db.open_table(table_name)


@st.cache_data
def load_service_metadata(path: str = "mygov_data.json"):
    data = {}
    try:
        # Allow non-UTF-8 bytes without crashing
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = int(row["index"])
                data[idx] = {
                    "bn": str(row.get("my_gov_service_name", "") or "").strip(),
                    "en": str(row.get("my_gov_service_name_en", "") or "").strip(),
                }
    except FileNotFoundError:
        pass
    return data


service_meta = load_service_metadata()

st.title("txtai Search (MyGov)")
st.write("Type a query below to search your MyGov dataset.")

query_text = st.text_input("Search query")

if query_text:
    query_embedding = get_embedding(query_text)
    results = table.search(query_embedding).limit(10).to_list()

    if not results:
        st.info("No results found.")
    else:
        for rank, rec in enumerate(results, start=1):
            meta = service_meta.get(rec["id"], {})
            bn_name = meta.get("bn", "").strip()
            en_name = meta.get("en", "").strip()

            score = rec.get("_distance")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

            title = bn_name or "(No Bangla name)"
            st.markdown(f"### {title}")

            if en_name:
                st.write(en_name)

            st.caption(f"Rank {rank}  Score {score_text}  ID {rec['id']}")
            st.markdown("---")
else:
    st.info("Enter a query to see search results.")
