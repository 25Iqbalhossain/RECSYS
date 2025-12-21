"""
Cache-first Gradio realtime lexical suggestion + search + streaming + perf test
STRICT:
- Load search_service_dump.json ONCE at startup
- Suggestions ONLY from cached prefix_map + vocabulary + unigram_freq + bigram_freq
- No rebuilding vocab/bigram/prefix at runtime
- Bangla-safe (no forced lowercasing for Bangla; UTF-8 safe)
- Full prefix coverage (no silent cap); ranking only after retrieval
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr

# -------------------------
# Config
# -------------------------
CACHE_PATH = os.environ.get("MYGOV_JSON_PATH", "search_service_dump.json")
DEFAULT_SUGGEST_TOPK = int(os.environ.get("SUGGEST_TOPK", "20"))
DEFAULT_SEARCH_TOPK = int(os.environ.get("SEARCH_TOPK", "10"))

# Unicode-aware token regex (Bangla + word chars)
TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+", flags=re.UNICODE)


# -------------------------
# Cache schema adapters
# -------------------------
@dataclass(frozen=True)
class SearchDoc:
    doc_id: str
    bn: str
    en: str
    keywords: str
    profile: str


@dataclass(frozen=True)
class Cache:
    # token_id -> token string
    vocab: List[str]

    # prefix(str) -> list[token_id]
    prefix_map: Dict[str, List[int]]

    # token_id -> frequency
    unigram_freq: Dict[int, int]

    # prev_id -> {next_id -> count}
    bigram_freq: Dict[int, Dict[int, int]]

    # docs for search
    docs: List[SearchDoc]


def _as_int_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
    """JSON keys may be strings; convert to int keys."""
    out: Dict[int, Any] = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except Exception:
            # if a key isn't int-like, skip (or keep separately if needed)
            continue
    return out


def load_cache_once(path: str) -> Cache:
    """
    Loads the cache ONCE.
    No recomputation of lexical structures.
    The file must already contain vocabulary/prefix_map/unigram_freq/bigram_freq/docs.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ---- expected keys (adapt if your JSON uses different names)
    vocab = raw["vocabulary"]  # list[str] indexed by token_id
    prefix_map = raw["prefix_map"]  # dict[str -> list[int]]
    unigram_freq = _as_int_keys(raw.get("unigram_freq", {}))
    bigram_raw = raw.get("bigram_freq", {})
    bigram_freq: Dict[int, Dict[int, int]] = {}
    for prev_k, row in (bigram_raw or {}).items():
        try:
            prev_id = int(prev_k)
        except Exception:
            continue
        # row: next_id -> count
        row_int = _as_int_keys(row)
        # ensure counts are ints
        bigram_freq[prev_id] = {int(nk): int(cv) for nk, cv in row_int.items()}

    docs_raw = raw.get("documents") or raw.get("search_index") or []
    docs: List[SearchDoc] = []
    for i, d in enumerate(docs_raw):
        docs.append(
            SearchDoc(
                doc_id=str(d.get("doc_id") or d.get("id") or f"row_{i}"),
                bn=str(d.get("bn") or d.get("my_gov_service_name") or d.get("name") or ""),
                en=str(d.get("en") or d.get("my_gov_service_name_en") or d.get("name_en") or ""),
                keywords=str(d.get("keywords") or d.get("my_gov_service_keyword") or d.get("keyword") or ""),
                profile=str(d.get("profile") or d.get("description") or ""),
            )
        )

    # basic sanity (fail fast)
    if not isinstance(vocab, list) or not vocab:
        raise ValueError("CACHE ERROR: vocabulary missing/empty in search_service_dump.json")
    if not isinstance(prefix_map, dict):
        raise ValueError("CACHE ERROR: prefix_map missing in search_service_dump.json")

    return Cache(
        vocab=vocab,
        prefix_map=prefix_map,
        unigram_freq={int(k): int(v) for k, v in unigram_freq.items()},
        bigram_freq=bigram_freq,
        docs=docs,
    )


# -------------------------
# Bangla-safe normalization
# -------------------------
def contains_bangla(s: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in (s or ""))


def normalize_token_for_lookup(tok: str) -> str:
    """
    CRITICAL:
    - Do NOT lowercase Bangla tokens.
    - English tokens are lowercased (typical cache build behavior).
    If your cache stores English case-sensitively, remove lower() here.
    """
    if contains_bangla(tok):
        return tok
    return tok.lower()


def extract_current_prefix(query: str) -> Tuple[str, Optional[str]]:
    """
    Extract prefix for autocomplete and previous token (for bigram context).

    Rules:
    - If user ends with space: prefix="" and prev_token is last completed token
    - Else: prefix=last token fragment and prev_token is token before it (if any)

    Returns:
      prefix_str, prev_token_str
    """
    q = query or ""
    ends_with_space = bool(re.search(r"\s$", q))

    toks = TOKEN_RE.findall(q)
    if not toks:
        return "", None

    if ends_with_space:
        prev = toks[-1]
        return "", normalize_token_for_lookup(prev)

    # partial last token
    prefix = toks[-1]
    prev = toks[-2] if len(toks) >= 2 else None
    return normalize_token_for_lookup(prefix), (normalize_token_for_lookup(prev) if prev else None)


# -------------------------
# Suggestion ranking (cache-only)
# -------------------------
def score_candidate_token_id(
    cache: Cache,
    prev_token_id: Optional[int],
    cand_token_id: int,
) -> float:
    """
    Ranking ONLY, never filters:
    - If bigram(prev->cand) exists => bigram probability proxy + unigram fallback
    - Else unigram
    """
    uni = float(cache.unigram_freq.get(cand_token_id, 0))

    if prev_token_id is None:
        return uni

    row = cache.bigram_freq.get(prev_token_id)
    if not row:
        return uni

    bi = float(row.get(cand_token_id, 0))
    if bi <= 0:
        return uni

    denom = float(sum(row.values())) or 1.0
    return 1_000_000.0 * (bi / denom) + uni


def suggest_from_cache(
    cache: Cache,
    query: str,
    top_k: int,
    debug: bool,
) -> Tuple[List[Tuple[str, int]], str]:
    """
    Returns:
      choices: list of (display_string, token_id) for dropdown
      debug_markdown
    """
    prefix, prev_tok = extract_current_prefix(query)

    # IMPORTANT: suggestions must come ONLY from prefix_map
    # If prefix == "" (user typed space), we can still show next-word suggestions
    # by using prefix_map with "" only if your cache has it.
    # Otherwise, we rely on bigram of prev token to propose next words (still cache-only).
    prefix_hits: List[int] = []

    if prefix in cache.prefix_map:
        prefix_hits = cache.prefix_map[prefix]
    else:
        prefix_hits = []

    # candidate_count_before_ranking MUST be full retrieved list length
    before = len(prefix_hits)

    # Resolve prev token to token_id if possible (cache-only)
    prev_id: Optional[int] = None
    if prev_tok is not None:
        # Build reverse vocab map ONCE (startup) for O(1). We'll attach on cache later.
        # In this strict file, we build it once globally below.
        prev_id = VOCAB_TO_ID.get(prev_tok)

    candidates: List[Tuple[float, int]] = []
    for tid in prefix_hits:
        # validate id range
        if not isinstance(tid, int) or tid < 0 or tid >= len(cache.vocab):
            continue
        sc = score_candidate_token_id(cache, prev_id, tid)
        candidates.append((sc, tid))

    # Ranking after retrieval
    candidates.sort(key=lambda x: x[0], reverse=True)
    after = len(candidates)

    # Top-20 debug tokens with scores (pre and post are important)
    # NOTE: We do NOT truncate before ranking.
    top20_scored = candidates[:20]

    # ONLY truncation point is here (explicit)
    truncation_note = ""
    if top_k is not None and top_k > 0 and len(candidates) > top_k:
        truncation_note = f"TRUNCATION: output limited to top_k={top_k} (after ranking only)."

    out = candidates if (top_k is None or top_k <= 0) else candidates[: top_k]

    choices: List[Tuple[str, int]] = []
    for sc, tid in out:
        tok = cache.vocab[tid]
        choices.append((tok, tid))

    debug_md = ""
    if debug:
        debug_md = (
            "### Debug (cache autocomplete)\n"
            f"- prefix: `{prefix}`\n"
            f"- prev_token: `{prev_tok}`\n"
            f"- prev_token_id: `{prev_id}`\n"
            f"- total_prefix_hits: **{before}**\n"
            f"- candidates_before_ranking: **{before}**\n"
            f"- candidates_after_ranking: **{after}**\n"
            f"- top20_scored: `{[(cache.vocab[tid], round(sc,3)) for sc, tid in top20_scored]}`\n"
            f"- {truncation_note or 'No truncation applied.'}\n"
        )

    return choices, debug_md


# -------------------------
# Search (cache-only, lightweight lexical)
# -------------------------
def doc_score(query: str, d: SearchDoc) -> float:
    """
    Cache-only search scoring (simple but consistent):
    prefix > word-boundary > contains
    """
    q = (query or "").strip()
    if not q:
        return -1e9

    # DO NOT destroy Bangla; only lower English-ish comparisons
    q_norm = q.lower() if not contains_bangla(q) else q

    bn = d.bn or ""
    en = d.en or ""
    kw = d.keywords or ""

    bn_norm = bn.lower() if not contains_bangla(bn) else bn
    en_norm = en.lower() if not contains_bangla(en) else en
    kw_norm = kw.lower() if not contains_bangla(kw) else kw

    s = 0.0
    if bn_norm.startswith(q_norm) or en_norm.startswith(q_norm):
        s += 100.0
    if re.search(rf"(^|\s){re.escape(q_norm)}", bn_norm) or re.search(rf"(^|\s){re.escape(q_norm)}", en_norm):
        s += 40.0
    if q_norm in bn_norm or q_norm in en_norm:
        s += 20.0
    if q_norm in kw_norm:
        s += 10.0

    return s


def search_cache(cache: Cache, query: str, top_k: int) -> List[Tuple[float, SearchDoc]]:
    q = (query or "").strip()
    if not q:
        return []
    scored: List[Tuple[float, SearchDoc]] = []
    for d in cache.docs:
        s = doc_score(q, d)
        if s > -1e8:
            scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: int(top_k)]


def stream_search_markdown(cache: Cache, query: str, top_k: int, delay_ms: int) -> Iterable[str]:
    q = (query or "").strip()
    if not q:
        yield "Type to search…"
        return

    t0 = time.perf_counter()
    rows = search_cache(cache, q, top_k=top_k)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    if not rows:
        yield f"**No results.**\n\n_Search latency: {lat_ms:.2f} ms_"
        return

    md: List[str] = [f"### Results for: `{q}`", f"_Search latency: {lat_ms:.2f} ms_", ""]
    yield "\n".join(md)

    for i, (s, d) in enumerate(rows, start=1):
        title = (d.bn.strip() or d.en.strip() or "(No title)").strip()
        md.append(f"**{i}. {title}**  ")
        md.append(f"- doc_id: `{d.doc_id}`  ")
        md.append(f"- score: `{s:.3f}`  ")
        if d.en.strip() and d.en.strip() != title:
            md.append(f"- English: {d.en.strip()}  ")
        if d.keywords.strip():
            md.append(f"- Keywords: {d.keywords.strip()}  ")
        if d.profile.strip():
            md.append(f"- {d.profile.strip()}  ")
        md.append("")
        yield "\n".join(md)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)


# -------------------------
# Perf
# -------------------------
def perf_state_init() -> Dict[str, Any]:
    return {"lat_ms": []}


def perf_md(state: Dict[str, Any]) -> str:
    lat = list(state.get("lat_ms", []))
    if not lat:
        return "### Performance\nNo measurements yet."
    lat.sort()
    avg = sum(lat) / len(lat)
    p50 = lat[len(lat) // 2]
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
    return (
        "### Performance (rolling)\n"
        f"- Samples: **{len(lat)}**\n"
        f"- Avg latency (ms): **{avg:.2f}**\n"
        f"- P50 (ms): **{p50:.2f}**\n"
        f"- P95 (ms): **{p95:.2f}**\n"
    )


def run_perf_test(cache: Cache, query: str, n: int, top_k: int) -> str:
    q = (query or "").strip()
    if not q:
        return "Provide a query to benchmark."
    n = int(n)
    if n <= 0:
        return "n must be > 0"

    _ = search_cache(cache, q, top_k=top_k)  # warmup

    t0 = time.perf_counter()
    lat_ms: List[float] = []
    for _i in range(n):
        ts = time.perf_counter()
        _ = search_cache(cache, q, top_k=top_k)
        lat_ms.append((time.perf_counter() - ts) * 1000.0)
    total = time.perf_counter() - t0

    lat_ms.sort()
    avg = sum(lat_ms) / len(lat_ms)
    p50 = lat_ms[len(lat_ms) // 2]
    p95 = lat_ms[max(0, int(len(lat_ms) * 0.95) - 1)]
    tps = (n / total) if total > 0 else 0.0

    print(f"[perf] n={n} avg_ms={avg:.2f} p50_ms={p50:.2f} p95_ms={p95:.2f} tps={tps:.2f}")
    return (
        "### Perf test result\n"
        f"- Requests: **{n}**\n"
        f"- Avg latency (ms): **{avg:.2f}**\n"
        f"- P50 (ms): **{p50:.2f}**\n"
        f"- P95 (ms): **{p95:.2f}**\n"
        f"- Throughput (req/sec): **{tps:.2f}**\n"
    )


# =========================
# Load cache ONCE at startup (ABSOLUTE RULE)
# =========================
CACHE = load_cache_once(CACHE_PATH)

# Build reverse vocab lookup ONCE (startup) for prev-token-id resolution
# NOTE: Must match normalize_token_for_lookup behavior.
VOCAB_TO_ID: Dict[str, int] = {}
for tid, tok in enumerate(CACHE.vocab):
    key = normalize_token_for_lookup(tok)
    # keep first occurrence
    VOCAB_TO_ID.setdefault(key, tid)


# =========================
# Gradio Handlers
# =========================
def ui_update_suggestions(query: str, suggest_top_k: int, debug: bool):
    t0 = time.perf_counter()
    choices, dbg = suggest_from_cache(CACHE, query, top_k=int(suggest_top_k), debug=bool(debug))
    dt = (time.perf_counter() - t0) * 1000.0

    # dropdown wants choices=[(label, value)]
    # we store value as token_id; applying sets query to that token string
    status = f"Suggest latency: {dt:.2f} ms • choices_returned: {len(choices)}"
    return gr.Dropdown(choices=choices, value=None), status, dbg


def ui_apply_token_id_to_query(token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return str(CACHE.vocab[int(token_id)])
    except Exception:
        return ""


def ui_stream_search(query: str, search_top_k: int, delay_ms: int, state: Dict[str, Any]):
    q = (query or "").strip()
    if not q:
        yield ("Type to search…", perf_md(state), state)
        return

    # basic guard to reduce thrash
    if len(q) < 2:
        yield ("Type at least 2 characters…", perf_md(state), state)
        return

    t0 = time.perf_counter()
    first = True
    for md in stream_search_markdown(CACHE, q, top_k=int(search_top_k), delay_ms=int(delay_ms)):
        if first:
            t_first = (time.perf_counter() - t0) * 1000.0
            state["lat_ms"].append(t_first)
            if len(state["lat_ms"]) > 200:
                state["lat_ms"] = state["lat_ms"][-200:]
            first = False
        yield (md, perf_md(state), state)


# =========================
# UI
# =========================
CSS = "#results_md { min-height: 420px; }"

with gr.Blocks(title="Cache-first MyGov Search (Gradio)") as demo:
    perf_state = gr.State(perf_state_init())

    gr.Markdown(
        "## MyGov Search (Cache-first Autocomplete + Streaming Search)\n"
        "- Suggestions come ONLY from `search_service_dump.json` cache\n"
        "- No submit button; updates on keystrokes"
    )

    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="Search", placeholder="Type…", lines=1, autofocus=True)

            with gr.Row():
                suggest_top_k = gr.Slider(1, 500, value=DEFAULT_SUGGEST_TOPK, step=1, label="Autocomplete top_k")
                debug_mode = gr.Checkbox(value=False, label="Debug mode")

            suggestions = gr.Dropdown(label="Suggestions (token)", choices=[], value=None, interactive=True)
            suggestion_status = gr.Markdown(value="Suggestions: 0")
            debug_out = gr.Markdown(value="")

            with gr.Row():
                search_top_k = gr.Slider(3, 25, value=DEFAULT_SEARCH_TOPK, step=1, label="Search Top K")
                stream_delay_ms = gr.Slider(0, 80, value=10, step=5, label="Streaming delay (ms)")

            gr.Markdown("### Performance Testing")
            perf_n = gr.Slider(5, 500, value=100, step=5, label="Requests to run")
            perf_btn = gr.Button("Run perf test (search only)")
            perf_out = gr.Markdown()

        with gr.Column(scale=3):
            results_md = gr.Markdown(value="Type to search…", elem_id="results_md")
            perf_panel = gr.Markdown(value="### Performance\nNo measurements yet.")

    # Realtime suggestions on every keystroke (queue=False)
    query.input(
        fn=ui_update_suggestions,
        inputs=[query, suggest_top_k, debug_mode],
        outputs=[suggestions, suggestion_status, debug_out],
        queue=False,
        trigger_mode="always_last",
    )

    # Apply suggestion -> set query to chosen token
    suggestions.change(
        fn=ui_apply_token_id_to_query,
        inputs=[suggestions],
        outputs=[query],
        queue=False,
    )

    # Streaming search (change reduces thrash)
    query.change(
        fn=ui_stream_search,
        inputs=[query, search_top_k, stream_delay_ms, perf_state],
        outputs=[results_md, perf_panel, perf_state],
        queue=True,
        trigger_mode="always_last",
    )

    perf_btn.click(
        fn=lambda q, n, k: run_perf_test(CACHE, q, int(n), int(k)),
        inputs=[query, perf_n, search_top_k],
        outputs=[perf_out],
        queue=True,
    )

    gr.Markdown(f"**Cache loaded once from:** `{CACHE_PATH}` • **Vocab size:** {len(CACHE.vocab)} • **Docs:** {len(CACHE.docs)}")


if __name__ == "__main__":
    demo.queue(max_size=64).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )
