import json
from collections import defaultdict, Counter

import openai
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from txtai.pipeline import Similarity

client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def get_embedding(text, prefix: str = "search_document: "):
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



def build_lexical_models(docs, max_prefix_len: int = 3, max_per_prefix: int = 10):
    """
    docs: list of {"id": int, "text": str}
    returns:
        prefix_suggestions: prefix(str) -> list of phrases(str)
        next_word:          word(str)   -> list of next words(str)
        vocab_words:        list of all distinct words (lowercase)
    """
    prefix_map: dict[str, Counter] = defaultdict(Counter)
    bigram_counts: dict[str, Counter] = defaultdict(Counter)
    vocab = set()

    for d in docs:
        text = d["text"].strip().lower()
        if not text:
            continue

        words = text.split()
        if not words:
            continue

      
        for w in words:
            vocab.add(w)

    
        for i in range(1, min(len(words), max_prefix_len) + 1):
            prefix = " ".join(words[:i])
            prefix_map[prefix][text] += 1

     
        for w1, w2 in zip(words, words[1:]):
            bigram_counts[w1][w2] += 1

   
    prefix_suggestions: dict[str, list[str]] = {}
    for prefix, counter in prefix_map.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_suggestions[prefix] = phrases

   
    next_word: dict[str, list[str]] = {}
    for w1, counter in bigram_counts.items():
        words_sorted = [w2 for w2, _ in counter.most_common()]
        next_word[w1] = words_sorted

    vocab_words = sorted(vocab)

    return prefix_suggestions, next_word, vocab_words


prefix_suggestions, next_word, vocab_words = build_lexical_models(documents)



def suggest_queries(query: str, max_suggestions: int = 5) -> list[str]:
    """
    Autocomplete / query suggestion:

    0) word-level completion: "app" -> "application"
    1) bigram-based next word: "application" -> "application for"
    2) prefix-based phrase suggestions
    3) shorter prefix fallback
    """
    raw = query.strip()
    q = raw.lower()
    if not q:
        return []

    suggestions: list[str] = []
    seen = set()

    words_raw = raw.split()
    words = q.split()


    if words:
        last_word = words[-1]
     
        completions = [
            w for w in vocab_words
            if w.startswith(last_word) and w != last_word
        ]

        for w in completions:
            new_tokens = words_raw[:-1] + [w]
            phrase = " ".join(new_tokens)
            if phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
            if len(suggestions) >= max_suggestions:
                return suggestions

 
    if words:
        last = words[-1]
        if last in next_word:
            for w2 in next_word[last]:
                phrase = raw + " " + w2
                if phrase not in seen:
                    seen.add(phrase)
                    suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    return suggestions

 
    if q in prefix_suggestions and len(suggestions) < max_suggestions:
        for phrase in prefix_suggestions[q]:
            if phrase.lower() == q:
                continue
            if phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
            if len(suggestions) >= max_suggestions:
                return suggestions

  
    if words and len(suggestions) < max_suggestions:
        for i in range(len(words), 0, -1):
            prefix = " ".join(words[:i])
            if prefix in prefix_suggestions:
                for phrase in prefix_suggestions[prefix]:
                    if phrase.lower() == q:
                        continue
                    if phrase not in seen:
                        seen.add(phrase)
                        suggestions.append(phrase)
                    if len(suggestions) >= max_suggestions:
                        break
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



similarity = Similarity()


query_text = "app"

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


