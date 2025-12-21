#!/usr/bin/env python3
"""
Build lexical cache (Format A) from raw docs list (Format B).

Improvements for better bigram:
- Build token streams from ordered phrases (bn title, en title, keyword phrases, profile sentences)
- Split keywords by comma/semicolon/pipe and build bigrams inside each phrase
- Keep Bangla tokens as-is; lowercase only ASCII
- Add EOS boundary per phrase to avoid cross-phrase noise
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Iterable

TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+", flags=re.UNICODE)
PHRASE_SPLIT_RE = re.compile(r"[,\|;]+")  # for keywords splitting
SENT_SPLIT_RE = re.compile(r"[।\.!\?\n\r]+")  # Bangla danda + English punctuation

EOS_TOKEN = "<eos>"


def contains_bangla(s: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in (s or ""))


def norm_tok(tok: str) -> str:
    return tok if contains_bangla(tok) else tok.lower()


def extract_tokens(text: str) -> List[str]:
    return [norm_tok(t) for t in TOKEN_RE.findall(text or "") if t]


def doc_to_fields(d: Dict[str, Any], i: int) -> Dict[str, str]:
    return {
        "doc_id": str(d.get("doc_id") or d.get("id") or f"row_{i}"),
        "bn": str(d.get("bn") or d.get("my_gov_service_name") or d.get("name") or ""),
        "en": str(d.get("en") or d.get("my_gov_service_name_en") or d.get("name_en") or ""),
        "keywords": str(d.get("keywords") or d.get("my_gov_service_keyword") or d.get("keyword") or ""),
        "profile": str(d.get("profile") or d.get("description") or ""),
    }


def iter_keyword_phrases(keywords: str) -> Iterable[str]:
    """
    Keywords often look like: "Visa on Arrival, Direct Visa, Tourist Visa"
    Split into phrases; keep ordering within each phrase for better bigrams.
    """
    kw = (keywords or "").strip()
    if not kw:
        return
    for p in PHRASE_SPLIT_RE.split(kw):
        p = p.strip()
        if p:
            yield p


def iter_profile_sentences(profile: str) -> Iterable[str]:
    """
    Profile/description can be long; split into sentences to avoid connecting far words.
    """
    pr = (profile or "").strip()
    if not pr:
        return
    for s in SENT_SPLIT_RE.split(pr):
        s = s.strip()
        if s:
            yield s


def build_prefix_map(vocab: List[str]) -> Dict[str, List[int]]:
    pmap: Dict[str, List[int]] = defaultdict(list)
    for tid, tok in enumerate(vocab):
        if not tok:
            continue
        for j in range(1, len(tok) + 1):
            pmap[tok[:j]].append(tid)
    return dict(pmap)


def main(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise SystemExit("Input must be Format B: a JSON list of documents.")

    documents: List[Dict[str, str]] = []
    token_streams: List[List[str]] = []

    empty_phrase_count = 0
    total_phrases = 0

    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            continue

        doc = doc_to_fields(d, i)
        documents.append(doc)

        # Build ordered phrase list (this is the key improvement)
        phrases: List[str] = []

        if doc["bn"].strip():
            phrases.append(doc["bn"].strip())
        if doc["en"].strip():
            phrases.append(doc["en"].strip())

        # keyword phrases
        phrases.extend(list(iter_keyword_phrases(doc["keywords"])))

        # profile sentences (optional but helps)
        phrases.extend(list(iter_profile_sentences(doc["profile"])))

        for ph in phrases:
            total_phrases += 1
            toks = extract_tokens(ph)

            if not toks:
                empty_phrase_count += 1
                continue

            # Add EOS boundary per phrase (prevents cross-phrase noise)
            token_streams.append(toks + [EOS_TOKEN])

    # unigram
    uni_counter = Counter()
    for toks in token_streams:
        uni_counter.update(toks)

    vocab = [t for t, _c in uni_counter.most_common()]
    vocab_to_id = {t: i for i, t in enumerate(vocab)}

    unigram_freq: Dict[int, int] = {vocab_to_id[t]: int(c) for t, c in uni_counter.items()}

    # bigram
    bigram_freq: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    bigram_pairs = 0

    for toks in token_streams:
        if len(toks) < 2:
            continue
        for a, b in zip(toks, toks[1:]):
            ai = vocab_to_id.get(a)
            bi = vocab_to_id.get(b)
            if ai is None or bi is None:
                continue
            bigram_freq[ai][bi] += 1
            bigram_pairs += 1

    bigram_out: Dict[str, Dict[str, int]] = {}
    for ai, row in bigram_freq.items():
        bigram_out[str(ai)] = {str(bi): int(cnt) for bi, cnt in row.items()}

    # prefix map (full coverage)
    prefix_map = build_prefix_map(vocab)

    cache = {
        "vocabulary": vocab,
        "prefix_map": prefix_map,
        "unigram_freq": {str(k): int(v) for k, v in unigram_freq.items()},
        "bigram_freq": bigram_out,
        "documents": documents,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    print("✅ Built cache")
    print(f"- docs: {len(documents)}")
    print(f"- phrases_total: {total_phrases}")
    print(f"- phrases_with_no_tokens: {empty_phrase_count}")
    print(f"- vocab: {len(vocab)}")
    print(f"- prefix_map keys: {len(prefix_map)}")
    print(f"- bigram rows: {len(bigram_out)}")
    print(f"- bigram pairs counted: {bigram_pairs}")
    print(f"Output: {out_path}")

    if len(bigram_out) == 0:
        print("\n⚠️ bigram_freq is EMPTY. Check your data fields or TOKEN_RE.")
        if documents:
            sample = documents[0]
            txt = " ".join([sample.get("bn",""), sample.get("en",""), sample.get("keywords",""), sample.get("profile","")])
            print("Sample text:", txt[:200])
            print("Sample tokens:", extract_tokens(txt)[:30])


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python build_cache.py search_service_dump.json search_service_cache.json")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
