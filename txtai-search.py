import json
from collections import defaultdict, Counter

import openai
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from txtai.pipeline import Similarity


# ---------- Embedding client ----------

client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def get_embedding(text, prefix: str = "search_document: "):
    response = client.embeddings.create(
        model="local-model",
        input=prefix + text
    )
    
    return response.data[0].embedding[:256]


# ---------- LanceDB setup ----------

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


# ---------- Load JSON documents ----------

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




def build_lexical_models(docs, max_prefix_len: int = 3, max_per_prefix: int = 10):
    """
    docs: list of {"id": int, "text": str}
    returns:
        prefix_suggestions: prefix(str) -> list of phrases(str)
        next_word: word(str) -> list of next words(str)
    """
    prefix_map: dict[str, Counter] = defaultdict(Counter)
    bigram_counts: dict[str, Counter] = defaultdict(Counter)

    for d in docs:
        text = d["text"].strip().lower()
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        
        for i in range(1, min(len(words), max_prefix_len) + 1):
            prefix = " ".join(words[:i])
            prefix_map[prefix][text] += 1

        # 2) bigram map (word-level next word)
        for w1, w2 in zip(words, words[1:]):
            bigram_counts[w1][w2] += 1

    # prefix -> sorted list of phrases
    prefix_suggestions: dict[str, list[str]] = {}
    for prefix, counter in prefix_map.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_suggestions[prefix] = phrases

    # word -> sorted list of next words
    next_word: dict[str, list[str]] = {}
    for w1, counter in bigram_counts.items():
        words_sorted = [w2 for w2, _ in counter.most_common()]
        next_word[w1] = words_sorted

    return prefix_suggestions, next_word


prefix_suggestions, next_word = build_lexical_models(documents)


def suggest_queries(query: str, max_suggestions: int = 5) -> list[str]:
    """
    Autocomplete / query suggestion:
    - prefix-based phrase suggestions (ফ্রিডম ফাইটার -> ফ্রিডম ফাইটার ভাতা)
    - bigram-based next word suggestions
    """
    q = query.strip().lower()
    if not q:
        return []

    suggestions: list[str] = []

   
    if q in prefix_suggestions:
        suggestions.extend(prefix_suggestions[q])

    
    if len(suggestions) < max_suggestions:
        words = q.split()
        for i in range(len(words), 0, -1):
            prefix = " ".join(words[:i])
            if prefix in prefix_suggestions:
                for phrase in prefix_suggestions[prefix]:
                    if phrase not in suggestions:
                        suggestions.append(phrase)
                    if len(suggestions) >= max_suggestions:
                        break
            if len(suggestions) >= max_suggestions:
                break

    # 3) bigram-based next word: last word থেকে পরের word suggest
    if len(suggestions) < max_suggestions:
        words = q.split()
        if words:
            last = words[-1]
            if last in next_word:
                for w2 in next_word[last]:
                    phrase = q + " " + w2
                    if phrase not in suggestions:
                        suggestions.append(phrase)
                    if len(suggestions) >= max_suggestions:
                        break

    return suggestions[:max_suggestions]




data_to_add = []
for doc in documents:
    embedding = get_embedding(doc["text"])
    data_to_add.append(
        Document(id=doc["id"], text=doc["text"], vector=embedding)
    )

if data_to_add:
    table.add(data_to_add)


# ---------- Search + suggestion demo ----------

similarity = Similarity()

# এখানে query_text change করে test করতে পারো
query_text = "ফ্রিডম ফাইটার"

print("User query:", query_text)
print("\nSuggestions:")
suggests = suggest_queries(query_text, max_suggestions=5)
for s in suggests:
    print("  -", s)


final_query = suggests[0] if suggests else query_text
print("\nUsing for vector search:", final_query)

query_embedding = get_embedding(final_query, prefix="search_query: ")

results = table.search(query_embedding).limit(5).to_list()
ranked = similarity(final_query, [r["text"] for r in results])

print("\nSearch results (re-ranked):")
for idx, score in ranked:
    rec = results[idx]
    print(f"Score: {score:.4f}  id={rec['id']}  text={rec['text']}")
