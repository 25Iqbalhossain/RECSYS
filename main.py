import json
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Optional
import lancedb
import openai
from fastapi import FastAPI
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, Field



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



@lru_cache(maxsize=1)
def load_service_metadata(path: str = "mygov_data.json"):
    """
    index -> { bn, en }
    """
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

                parts_full = [bn, en, keywords, profile]
                text = " ".join(p for p in parts_full if p)
                if not text:
                    continue

                docs.append({"id": uid, "text": text, "bn": bn, "en": en})
    except FileNotFoundError:
        pass
    return docs


# -------------------------
# Lexical models
# -------------------------
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
      vocab_bn : list of all Bangla words
      vocab_en : list of all English words
    """
    prefix_map_bn: dict[str, Counter] = defaultdict(Counter)
    prefix_map_en: dict[str, Counter] = defaultdict(Counter)
    bigram_bn: dict[str, Counter] = defaultdict(Counter)
    bigram_en: dict[str, Counter] = defaultdict(Counter)

    all_bn = set()
    all_en = set()

    vocab_bn = set()
    vocab_en = set()

    for d in docs:
        bn_phrase = (d["bn"] or "").strip()
        en_phrase = (d["en"] or "").strip()

       
        if bn_phrase:
            all_bn.add(bn_phrase)
            phrase = bn_phrase.lower()
            words = phrase.split()

           
            for w in words:
                vocab_bn.add(w)

         
            for i in range(1, min(len(words), max_prefix_len) + 1):
                prefix = " ".join(words[:i])
                prefix_map_bn[prefix][bn_phrase] += 1

           
            for w1, w2 in zip(words, words[1:]):
                bigram_bn[w1][w2] += 1

     
        if en_phrase:
            all_en.add(en_phrase)
            phrase_en = en_phrase.lower()
            words_en = phrase_en.split()

            for w in words_en:
                vocab_en.add(w)

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

    next_bn: dict[str, list[str]] = {}
    for w1, counter in bigram_bn.items():
        next_bn[w1] = [w2 for w2, _ in counter.most_common()]

    next_en: dict[str, list[str]] = {}
    for w1, counter in bigram_en.items():
        next_en[w1] = [w2 for w2, _ in counter.most_common()]

    all_bn_list = sorted(all_bn)
    all_en_list = sorted(all_en)

    vocab_bn_list = sorted(vocab_bn)
    vocab_en_list = sorted(vocab_en)

    return (
        prefix_bn,
        prefix_en,
        all_bn_list,
        all_en_list,
        next_bn,
        next_en,
        vocab_bn_list,
        vocab_en_list,
    )


@lru_cache(maxsize=1)
def get_lexical_data(path: str = "mygov_data.json"):
    docs = load_search_documents(path)
    return build_lexical_models(docs)


def is_bangla(text: str) -> bool:
    for ch in text:
        if "\u0980" <= ch <= "\u09FF":
            return True
    return False


def suggest_queries(
    query: str,
    prefix_bn: dict[str, list[str]],
    prefix_en: dict[str, list[str]],
    all_bn: list[str],
    all_en: list[str],
    next_bn: dict[str, list[str]],
    next_en: dict[str, list[str]],
    vocab_bn: list[str],
    vocab_en: list[str],
    max_suggestions: int = 5,
) -> list[str]:
    q_raw = query.strip()
    if not q_raw:
        return []

    use_bn = is_bangla(q_raw)
    q = q_raw.lower()

    prefix_dict = prefix_bn if use_bn else prefix_en
    all_phrases = all_bn if use_bn else all_en
    next_dict = next_bn if use_bn else next_en
    vocab = vocab_bn if use_bn else vocab_en

    suggestions: list[str] = []
    seen = set()

    words_raw = q_raw.split()
    words = q.split()


    if words:
        last_word = words[-1]
        completions = [
            w for w in vocab
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

    # --------------------------------
    # STEP 1: bigram-based next word
    # application -> application for
    # --------------------------------
    if words:
        last = words[-1]
        if last in next_dict:
            for w2 in next_dict[last]:
                phrase = q_raw + " " + w2
                if phrase not in seen:
                    seen.add(phrase)
                    suggestions.append(phrase)
                if len(suggestions) >= max_suggestions:
                    return suggestions


    if q in prefix_dict and len(suggestions) < max_suggestions:
        for phrase in prefix_dict[q]:
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



class SuggestRequest(BaseModel):
    query: str
    max_suggestions: int = 5


class SuggestResponse(BaseModel):
    query: str
    suggestions: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchItem(BaseModel):
    id: int
    bn: Optional[str] = None
    en: Optional[str] = None
    text: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]



app = FastAPI(title="MyGov txtai Search API")


@app.post("/suggest", response_model=SuggestResponse)
def suggest_api(body: SuggestRequest):
    (
        prefix_bn,
        prefix_en,
        all_bn,
        all_en,
        next_bn,
        next_en,
        vocab_bn,
        vocab_en,
    ) = get_lexical_data()

    suggestions = suggest_queries(
        body.query,
        prefix_bn,
        prefix_en,
        all_bn,
        all_en,
        next_bn,
        next_en,
        vocab_bn,
        vocab_en,
        max_suggestions=body.max_suggestions,
    )
    return SuggestResponse(query=body.query, suggestions=suggestions)


@app.post("/search", response_model=SearchResponse)
def search_api(body: SearchRequest):
    query_text = body.query.strip()
    if not query_text:
        return SearchResponse(query=body.query, results=[])

    query_embedding = get_embedding(query_text)
    results_raw = table.search(query_embedding).limit(body.top_k).to_list() or []

    results: List[SearchItem] = []
    for rec in results_raw:
        rec_id = rec["id"]
        meta = service_meta.get(rec_id, {})
        bn_name = meta.get("bn", "").strip() or None
        en_name = meta.get("en", "").strip() or None

        score = rec.get("_distance")
        if isinstance(score, (int, float)):
            score_val: Optional[float] = float(score)
        else:
            score_val = None

        results.append(
            SearchItem(
                id=rec_id,
                bn=bn_name,
                en=en_name,
                text=rec.get("text", ""),
                score=score_val,
            )
        )

    return SearchResponse(query=body.query, results=results)
