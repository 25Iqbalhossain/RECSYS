import json
from collections import defaultdict, Counter

import lancedb
import openai
import streamlit as st
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

# ---------- Embedding client ----------

client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def get_embedding(text: str, prefix: str = "search_query: "):
    response = client.embeddings.create(model="local-model", input=prefix + text)
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
    table = db.create_table(table_name, schema=Document)
else:
    table = db.open_table(table_name)


# ---------- Metadata loader (Bangla/English names) ----------

@st.cache_data
def load_service_metadata(path: str = "mygov_data.json"):
    data = {}
    try:
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


# ---------- Documents loader for lexical model & search ----------

@st.cache_data
def load_search_documents(path: str = "mygov_data.json"):
    """
    docs:
      - text -> full text (name + keyword + profile) for embedding search
      - bn   -> Bangla service name
      - en   -> English service name
    """
    docs = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                uid = int(row["index"])

                bn = str(row.get("my_gov_service_name", "") or "").strip()
                en = str(row.get("my_gov_service_name_en", "") or "").strip()
                keywords = str(row.get("my_gov_service_keyword", "") or "").strip()
                profile = str(row.get("nsp_profile_name", "") or "").strip()

                # full text for vector search
                parts_full = [bn, en, keywords, profile]
                text = " ".join(p for p in parts_full if p)
                if not text:
                    continue

                docs.append({"id": uid, "text": text, "bn": bn, "en": en})
    except FileNotFoundError:
        pass
    return docs


# ---------- Lexical models: prefix + bigram (Bangla & English) ----------

def build_lexical_models(
    docs, max_prefix_len: int = 1, max_per_prefix: int = 10
):
    """
    returns:
      prefix_bn: prefix -> [Bangla phrases]
      prefix_en: prefix -> [English phrases]
      all_bn   : list of all unique Bangla names
      all_en   : list of all unique English names
      next_bn  : word  -> [next words]  (bigram)
      next_en  : word  -> [next words]  (bigram)
    """
    prefix_map_bn: dict[str, Counter] = defaultdict(Counter)
    prefix_map_en: dict[str, Counter] = defaultdict(Counter)
    bigram_bn: dict[str, Counter] = defaultdict(Counter)
    bigram_en: dict[str, Counter] = defaultdict(Counter)

    all_bn = set()
    all_en = set()

    for d in docs:
        bn_phrase = (d["bn"] or "").strip()
        en_phrase = (d["en"] or "").strip()

        # Bangla side
        if bn_phrase:
            all_bn.add(bn_phrase)
            phrase = bn_phrase.lower()
            words = phrase.split()

            # prefix শুধু প্রথম শব্দ পর্যন্ত (max_prefix_len=1)
            for i in range(1, min(len(words), max_prefix_len) + 1):
                prefix = " ".join(words[:i])
                prefix_map_bn[prefix][bn_phrase] += 1

            # bigram (word→next_word)
            for w1, w2 in zip(words, words[1:]):
                bigram_bn[w1][w2] += 1

        # English side
        if en_phrase:
            all_en.add(en_phrase)
            phrase_en = en_phrase.lower()
            words_en = phrase_en.split()

            for i in range(1, min(len(words_en), max_prefix_len) + 1):
                prefix = " ".join(words_en[:i])
                prefix_map_en[prefix][en_phrase] += 1

            for w1, w2 in zip(words_en, words_en[1:]):
                bigram_en[w1][w2] += 1

    prefix_bn: dict[str, list[str]] = {}
    for prefix, counter in prefix_map_bn.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_bn[prefix] = phrases

    prefix_en: dict[str, list[str]] = {}
    for prefix, counter in prefix_map_en.items():
        phrases = [p for p, _ in counter.most_common(max_per_prefix)]
        prefix_en[prefix] = phrases

    # bigram → sorted next-word lists
    next_bn: dict[str, list[str]] = {}
    for w1, counter in bigram_bn.items():
        next_bn[w1] = [w2 for w2, _ in counter.most_common()]

    next_en: dict[str, list[str]] = {}
    for w1, counter in bigram_en.items():
        next_en[w1] = [w2 for w2, _ in counter.most_common()]

    all_bn_list = sorted(all_bn)
    all_en_list = sorted(all_en)

    return prefix_bn, prefix_en, all_bn_list, all_en_list, next_bn, next_en


@st.cache_data
def get_lexical_data(path: str = "mygov_data.json"):
    docs = load_search_documents(path)
    return build_lexical_models(docs)


# ---------- Language detection helper ----------

def is_bangla(text: str) -> bool:
    for ch in text:
        if "\u0980" <= ch <= "\u09FF":
            return True
    return False


# ---------- Suggestion function (word-by-word) ----------

def suggest_queries(
    query: str,
    prefix_bn: dict[str, list[str]],
    prefix_en: dict[str, list[str]],
    all_bn: list[str],
    all_en: list[str],
    next_bn: dict[str, list[str]],
    next_en: dict[str, list[str]],
    max_suggestions: int = 5,
) -> list[str]:
    """
    - আগে bigram দিয়ে word-by-word extension:
        "আর্থিক" -> "আর্থিক ক্ষমতা", "আর্থিক সহায়তা", ...
    - কিছু না পেলে prefix + substring fallback
    """
    q_raw = query.strip()
    if not q_raw:
        return []

    use_bn = is_bangla(q_raw)
    q = q_raw.lower()

    prefix_dict = prefix_bn if use_bn else prefix_en
    all_phrases = all_bn if use_bn else all_en
    next_dict = next_bn if use_bn else next_en

    suggestions: list[str] = []
    seen = set()

    words = q.split()

    # 1) bigram-based: last word -> next words
    if words:
        last = words[-1]
        if last in next_dict:
            for w2 in next_dict[last]:
                phrase = q_raw + " " + w2  # original query + next word
                if phrase not in seen:
                    seen.add(phrase)
                    suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    return suggestions

    # 2) prefix: exact prefix match (service name)
    if q in prefix_dict and len(suggestions) < max_suggestions:
        for phrase in prefix_dict[q]:
            if phrase.lower() == q:
                continue
            if phrase not in seen:
                seen.add(phrase)
                suggestions.append(phrase)
            if len(suggestions) >= max_suggestions:
                return suggestions

    # 3) shorter prefixes (right-to-left)
    if words and len(suggestions) < max_suggestions:
        for i in range(len(words), 0, -1):
            prefix = " ".join(words[:i])
            if prefix in prefix_dict:
                for phrase in prefix_dict[prefix]:
                    if phrase.lower() == q:
                        continue
                    if phrase not in seen:
                        seen.add(phrase)
                        suggestions.append(phrase)
                    if len(suggestions) >= max_suggestions:
                        break
            if len(suggestions) >= max_suggestions:
                break


    if len(suggestions) < max_suggestions:
        for phrase in all_phrases:
            if q in phrase.lower() and phrase not in seen and phrase.lower() != q:
                seen.add(phrase)
                suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    break

    return suggestions[:max_suggestions]




st.title("txtai Search (MyGov)")
st.write("Type a query below to search your MyGov dataset.")

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""


def set_query(new_q: str):
    st.session_state["query_text"] = new_q


query_text = st.text_input("Search query", key="query_text")


prefix_bn, prefix_en, all_bn, all_en, next_bn, next_en = get_lexical_data()

if query_text:
 
    suggestions = suggest_queries(
        query_text,
        prefix_bn,
        prefix_en,
        all_bn,
        all_en,
        next_bn,
        next_en,
        max_suggestions=5,
    )

    if suggestions:
        for i, s in enumerate(suggestions):
        
            st.button(s, key=f"sugg_{i}", on_click=set_query, args=(s,))


    search_query = query_text
    query_embedding = get_embedding(search_query)
    results = table.search(query_embedding).limit(10).to_list()

    if not results:
        st.info("No results found.")
    else:
        st.markdown(f"### Results for: `{search_query}`")
        for rank, rec in enumerate(results, start=1):
            meta = service_meta.get(rec["id"], {})
            bn_name = meta.get("bn", "").strip()
            en_name = meta.get("en", "").strip()

            score = rec.get("_distance")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

            title = bn_name or en_name or "(No name)"
            st.markdown(f"#### {title}")

            if en_name and en_name != title:
                st.write(en_name)

            st.caption(f"Rank {rank}  Score {score_text}  ID {rec['id']}")
            st.markdown("---")
else:
    st.info("Type something to see suggestions and search results.")
