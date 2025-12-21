from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr

try:
    import psutil
except Exception:
    psutil = None

CACHE_PATH = os.environ.get("MYGOV_JSON_PATH", "search_service_cache.json")
DEFAULT_SUGGEST_TOPK = int(os.environ.get("SUGGEST_TOPK", "20"))
DEFAULT_SEARCH_TOPK = int(os.environ.get("SEARCH_TOPK", "10"))
DEFAULT_SERVICE_SUGGEST_TOPK = int(os.environ.get("SERVICE_SUGGEST_TOPK", "10"))

TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+", flags=re.UNICODE)
KW_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z0-9]+", flags=re.UNICODE)


@dataclass(frozen=True)
class SearchDoc:
    doc_id: str
    bn: str
    en: str
    keywords: str
    profile: str


@dataclass(frozen=True)
class Cache:
    vocab: List[str]
    prefix_map: Dict[str, List[int]]
    unigram_freq: Dict[int, int]
    bigram_freq: Dict[int, Dict[int, int]]
    docs: List[SearchDoc]


def _as_int_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def load_cache_once(path: str) -> Cache:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        # Format B -> token autocomplete impossible by STRICT rules
        docs: List[SearchDoc] = []
        for i, d in enumerate(raw):
            if not isinstance(d, dict):
                continue
            docs.append(
                SearchDoc(
                    doc_id=str(d.get("doc_id") or d.get("id") or f"row_{i}"),
                    bn=str(d.get("bn") or d.get("my_gov_service_name") or d.get("name") or ""),
                    en=str(d.get("en") or d.get("my_gov_service_name_en") or d.get("name_en") or ""),
                    keywords=str(d.get("keywords") or d.get("my_gov_service_keyword") or d.get("keyword") or ""),
                    profile=str(d.get("profile") or d.get("description") or ""),
                )
            )
        return Cache(vocab=[], prefix_map={}, unigram_freq={}, bigram_freq={}, docs=docs)

    if not isinstance(raw, dict):
        raise ValueError("CACHE ERROR: cache must be dict (Format A) or list (Format B)")

    vocab = raw.get("vocabulary", [])
    prefix_map = raw.get("prefix_map", {})
    unigram_freq = _as_int_keys(raw.get("unigram_freq", {}))

    bigram_raw = raw.get("bigram_freq", {})
    bigram_freq: Dict[int, Dict[int, int]] = {}
    for prev_k, row in (bigram_raw or {}).items():
        try:
            prev_id = int(prev_k)
        except Exception:
            continue
        row_int = _as_int_keys(row)
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

    if not isinstance(vocab, list) or not isinstance(prefix_map, dict):
        raise ValueError("CACHE ERROR: invalid cache schema")

    return Cache(
        vocab=vocab,
        prefix_map=prefix_map,
        unigram_freq={int(k): int(v) for k, v in unigram_freq.items()},
        bigram_freq=bigram_freq,
        docs=docs,
    )


def contains_bangla(s: str) -> bool:
    return any("\u0980" <= ch <= "\u09FF" for ch in (s or ""))


def normalize_token_for_lookup(tok: str) -> str:
    return tok if contains_bangla(tok) else tok.lower()


def extract_current_prefix(query: str) -> Tuple[str, Optional[str]]:
    q = query or ""
    ends_with_space = bool(re.search(r"\s$", q))
    toks = TOKEN_RE.findall(q)
    if not toks:
        return "", None
    if ends_with_space:
        prev = toks[-1]
        return "", normalize_token_for_lookup(prev)
    prefix = toks[-1]
    prev = toks[-2] if len(toks) >= 2 else None
    return normalize_token_for_lookup(prefix), (normalize_token_for_lookup(prev) if prev else None)


def score_candidate_token_id(cache: Cache, prev_token_id: Optional[int], cand_token_id: int) -> float:
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


# =========================
# Load cache ONCE
# =========================
APP_START_T = time.time()
CACHE = load_cache_once(CACHE_PATH)

VOCAB_TO_ID: Dict[str, int] = {}
for tid, tok in enumerate(CACHE.vocab or []):
    VOCAB_TO_ID.setdefault(normalize_token_for_lookup(tok), tid)

# Normalize prefix_map keys ONCE at startup
PREFIX_MAP_NORM: Dict[str, List[int]] = {}
for pfx, ids in (CACHE.prefix_map or {}).items():
    if not isinstance(pfx, str):
        continue
    pfx_norm = normalize_token_for_lookup(pfx)
    if pfx_norm not in PREFIX_MAP_NORM:
        PREFIX_MAP_NORM[pfx_norm] = list(ids) if isinstance(ids, list) else []
    else:
        if isinstance(ids, list):
            PREFIX_MAP_NORM[pfx_norm].extend(ids)

for pfx, ids in list(PREFIX_MAP_NORM.items()):
    seen = set()
    out = []
    for tid in ids:
        if isinstance(tid, int) and tid not in seen:
            seen.add(tid)
            out.append(tid)
    PREFIX_MAP_NORM[pfx] = out

TOP_UNIGRAM_IDS: List[int] = []
if CACHE.unigram_freq and CACHE.vocab:
    TOP_UNIGRAM_IDS = sorted(
        (tid for tid in CACHE.unigram_freq.keys() if 0 <= tid < len(CACHE.vocab)),
        key=lambda tid: CACHE.unigram_freq.get(tid, 0),
        reverse=True,
    )


def suggest_from_cache(cache: Cache, query: str, top_k: int, debug: bool) -> Tuple[List[Tuple[str, int]], str]:
    prefix, prev_tok = extract_current_prefix(query)

    if not cache.vocab:
        dbg = (
            "### Debug (token autocomplete)\n"
            "- vocab is empty -> you are using Format B JSON list\n"
            "- build Format A cache (vocabulary/prefix_map/unigram/bigram) offline\n"
            if debug
            else ""
        )
        return [], dbg

    prev_id = VOCAB_TO_ID.get(prev_tok) if prev_tok else None
    before = 0
    candidates: List[Tuple[float, int]] = []

    # After space: next-token suggestion
    if prefix == "":
        if prev_id is not None and cache.bigram_freq.get(prev_id):
            row = cache.bigram_freq[prev_id]
            before = len(row)
            for tid in row.keys():
                if 0 <= tid < len(cache.vocab):
                    candidates.append((score_candidate_token_id(cache, prev_id, tid), tid))
        else:
            before = len(TOP_UNIGRAM_IDS)
            for tid in TOP_UNIGRAM_IDS:
                candidates.append((score_candidate_token_id(cache, None, tid), tid))
    else:
        # Active prefix typing
        hits = PREFIX_MAP_NORM.get(prefix, [])
        before = len(hits)
        for tid in hits:
            if 0 <= tid < len(cache.vocab):
                candidates.append((score_candidate_token_id(cache, prev_id, tid), tid))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out = candidates if (top_k is None or top_k <= 0) else candidates[: int(top_k)]
    choices = [(cache.vocab[tid], tid) for _sc, tid in out]

    dbg = ""
    if debug:
        dbg = (
            "### Debug (token autocomplete)\n"
            f"- prefix: `{prefix}`\n"
            f"- prev_token: `{prev_tok}`\n"
            f"- prev_id: `{prev_id}`\n"
            f"- hits_before_ranking: **{before}**\n"
            f"- returned: **{len(choices)}**\n"
        )

    return choices, dbg


def _kw_norm_token(tok: str) -> str:
    return tok if contains_bangla(tok) else tok.lower()


def _kw_extract_query_tokens(q: str) -> List[str]:
    toks = KW_TOKEN_RE.findall((q or "").strip())
    return [_kw_norm_token(t) for t in toks if t]


def _kw_split_phrases(keyword: str) -> List[str]:
    if not keyword:
        return []
    parts = [p.strip() for p in keyword.split(",") if p.strip()]
    phrases: List[str] = []
    for p in parts:
        phrases.append(p)
        no_paren = re.sub(r"\([^)]*\)", "", p).strip()
        if no_paren and no_paren != p:
            phrases.append(no_paren)

    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _kw_norm_phrase_tokens(phrase: str) -> Tuple[str, List[str]]:
    toks = KW_TOKEN_RE.findall(phrase or "")
    ntoks = [_kw_norm_token(t) for t in toks if t]
    return " ".join(ntoks), ntoks


def _kw_match_score(q_tokens: List[str], kw_tokens: List[str], kw_norm_str: str) -> int:
    if not q_tokens:
        return 0
    score = 0
    q_str = " ".join(q_tokens)
    if len(q_tokens) >= 2 and q_str and q_str in kw_norm_str:
        score += 50
    for qt in q_tokens:
        for kt in kw_tokens:
            if kt == qt:
                score += 10
                break
            if kt.startswith(qt) or (qt in kt):
                score += 6
                break
    return score


def suggest_services_by_keyword(cache: Cache, query: str, top_k: int, debug: bool) -> Tuple[List[Tuple[str, str]], str]:
    q_raw = (query or "").strip()
    if not q_raw:
        return [], ("### Debug (service suggestions)\n- empty query\n" if debug else "")

    use_bn = contains_bangla(q_raw)
    q_tokens = _kw_extract_query_tokens(q_raw)
    if not q_tokens:
        return [], (f"### Debug (service suggestions)\n- tokens empty: `{q_raw}`" if debug else "")

    scored: List[Tuple[int, str, str]] = []
    for d in cache.docs:
        kw = (d.keywords or "").strip()
        if not kw:
            continue
        title = (d.bn.strip() if use_bn and d.bn.strip() else d.en.strip()) or d.bn.strip() or d.en.strip()
        if not title:
            continue
        phrases = _kw_split_phrases(kw)
        best = 0
        for ph in phrases:
            kw_norm_str, kw_tokens = _kw_norm_phrase_tokens(ph)
            best = max(best, _kw_match_score(q_tokens, kw_tokens, kw_norm_str))
        if best > 0:
            scored.append((best, title, d.doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Tuple[str, str]] = []
    seen = set()
    for sc, title, doc_id in scored:
        if title in seen:
            continue
        seen.add(title)
        out.append((title, doc_id))
        if top_k and len(out) >= int(top_k):
            break

    return out, (f"### Debug (service suggestions)\n- returned: {len(out)}" if debug else "")


def doc_score(query: str, d: SearchDoc) -> float:
    q = (query or "").strip()
    if not q:
        return -1e9

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

    return (
        "### Perf test result\n"
        f"- Requests: **{n}**\n"
        f"- Avg latency (ms): **{avg:.2f}**\n"
        f"- P50 (ms): **{p50:.2f}**\n"
        f"- P95 (ms): **{p95:.2f}**\n"
        f"- Throughput (req/sec): **{tps:.2f}**\n"
    )


def ui_update_token_suggestions(query: str, suggest_top_k: int, debug: bool):
    t0 = time.perf_counter()
    choices, dbg = suggest_from_cache(CACHE, query, top_k=int(suggest_top_k), debug=bool(debug))
    dt = (time.perf_counter() - t0) * 1000.0
    status = f"Token suggest latency: {dt:.2f} ms • choices_returned: {len(choices)}"
    return gr.Dropdown(choices=choices, value=None), status, dbg


def ui_update_service_suggestions(query: str, service_top_k: int, debug: bool):
    t0 = time.perf_counter()
    choices, dbg = suggest_services_by_keyword(CACHE, query, top_k=int(service_top_k), debug=bool(debug))
    dt = (time.perf_counter() - t0) * 1000.0
    status = f"Service suggest latency: {dt:.2f} ms • choices_returned: {len(choices)}"
    return gr.Dropdown(choices=choices, value=None), status, dbg


# ✅ UPDATED: token selection should append/replace, not overwrite the whole query
def ui_apply_token_id_to_query(current_query: str, token_id: Optional[int]) -> str:
    q = current_query or ""
    if token_id is None:
        return q
    try:
        tok = str(CACHE.vocab[int(token_id)])
    except Exception:
        return q

    # If ends with whitespace, append
    if re.search(r"\s$", q):
        return q + tok + " "

    # Else replace last token/prefix
    matches = list(TOKEN_RE.finditer(q))
    if not matches:
        return tok + " "
    last = matches[-1]
    new_q = q[: last.start()] + tok + q[last.end() :]
    return new_q + " "


def ui_apply_service_doc_to_query(doc_id: Optional[str]) -> str:
    if not doc_id:
        return ""
    for d in CACHE.docs:
        if d.doc_id == str(doc_id):
            return (d.bn.strip() or d.en.strip() or "").strip()
    return ""


def ui_stream_search(query: str, search_top_k: int, delay_ms: int, state: Dict[str, Any]):
    q = (query or "").strip()
    if not q:
        yield ("Type to search…", perf_md(state), state)
        return
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
# Memory Monitor (Visual)
# =========================
def _bytes_to_mb(x: float) -> float:
    return float(x) / (1024.0 * 1024.0)

def get_mem_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "uptime_s": round(time.time() - APP_START_T, 2),
        "cache_path": CACHE_PATH,
        "vocab_size": len(CACHE.vocab or []),
        "prefix_keys": len(CACHE.prefix_map or {}),
        "bigram_rows": len(CACHE.bigram_freq or {}),
        "docs": len(CACHE.docs or []),
    }

    if psutil is not None:
        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        snap["rss_mb"] = round(_bytes_to_mb(mi.rss), 2)
        snap["vms_mb"] = round(_bytes_to_mb(mi.vms), 2)
    else:
        snap["rss_mb"] = None
        snap["vms_mb"] = None

    return snap

def mem_state_init() -> Dict[str, Any]:
    cur = get_mem_snapshot()
    return {"baseline": cur, "current": cur}

def memory_md(state: Dict[str, Any]) -> str:
    base = state.get("baseline") or {}
    cur = state.get("current") or {}

    lines = ["### Memory Monitor (Professor demo)"]

    if cur.get("rss_mb") is None:
        lines.append("- `psutil` not installed → memory read unavailable.")
        lines.append("- Install: `pip install psutil`")
    else:
        lines.append(f"- Current RSS (MB): **{cur['rss_mb']}**")
        if base.get("rss_mb") is not None:
            delta = round(cur["rss_mb"] - base["rss_mb"], 2)
            lines.append(f"- Baseline RSS (MB): **{base['rss_mb']}**")
            lines.append(f"- Delta RSS (MB): **{delta}** (should stay near 0 if no leak)")

    lines.append("")
    lines.append("**Cache loaded once (invariants):**")
    lines.append(f"- Cache path: `{cur.get('cache_path')}`")
    lines.append(f"- Vocab size: **{cur.get('vocab_size')}**")
    lines.append(f"- Prefix keys: **{cur.get('prefix_keys')}**")
    lines.append(f"- Bigram rows: **{cur.get('bigram_rows')}**")
    lines.append(f"- Docs: **{cur.get('docs')}**")
    lines.append(f"- Uptime (sec): **{cur.get('uptime_s')}**")
    return "\n".join(lines)

def refresh_memory(state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    gc.collect()
    state["current"] = get_mem_snapshot()
    return memory_md(state), state


CSS = """
#results_md { min-height: 520px; }
.small-muted { font-size: 12px; opacity: 0.85; }
.badges { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
.badge {
  display:inline-block;
  padding:6px 10px;
  border-radius:10px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  font-size: 12px;
}
.panel {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 12px;
}
hr { opacity: 0.2; }
"""

with gr.Blocks(title="MyGov Search (Cache-first)", css=CSS) as demo:
    perf_state = gr.State(perf_state_init())
    mem_state = gr.State(mem_state_init())

    token_enabled = bool(CACHE.vocab) and (bool(CACHE.prefix_map) or bool(CACHE.bigram_freq) or bool(CACHE.unigram_freq))

    # ===== Header / Status =====
    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown("## MyGov Search (Cache-first)")
            gr.Markdown(
                "- Token suggestions: prefix_map + unigram + bigram (cache-only)\n"
                "- Service suggestions: documents.keywords (cache-only)\n"
            )
            gr.Markdown(
                f'<div class="badges">'
                f'<span class="badge">Token autocomplete: <b>{token_enabled}</b></span>'
                f'<span class="badge">Cache: <b>{CACHE_PATH}</b></span>'
                f'<span class="badge">Vocab: <b>{len(CACHE.vocab or [])}</b></span>'
                f'<span class="badge">Prefix keys: <b>{len(CACHE.prefix_map or {})}</b></span>'
                f'<span class="badge">Bigram rows: <b>{len(CACHE.bigram_freq or {})}</b></span>'
                f'<span class="badge">Docs: <b>{len(CACHE.docs or [])}</b></span>'
                f'</div>'
            )
        with gr.Column(scale=5):
            gr.Markdown(
                "<div class='panel'>"
                "<b>Tip (demo):</b><br>"
                "1) Type <code>visa</code> → prefix suggestions<br>"
                "2) Type <code>visa </code> (with space) → bigram next-token<br>"
                "3) Run perf test & refresh memory snapshot"
                "</div>"
            )

    gr.Markdown("---")

    # ===== Main layout: Left controls, Right results =====
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tab("Search"):
                query = gr.Textbox(label="Search", placeholder="Type…", lines=1, autofocus=True)

                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        search_top_k = gr.Slider(3, 25, value=DEFAULT_SEARCH_TOPK, step=1, label="Search Top K")
                        stream_delay_ms = gr.Slider(0, 80, value=10, step=5, label="Streaming delay (ms)")
                    with gr.Row():
                        suggest_top_k = gr.Slider(1, 500, value=DEFAULT_SUGGEST_TOPK, step=1, label="Token autocomplete top_k")
                        service_suggest_top_k = gr.Slider(1, 100, value=DEFAULT_SERVICE_SUGGEST_TOPK, step=1, label="Service suggest top_k")
                    debug_mode = gr.Checkbox(value=False, label="Debug mode")

                # small statuses
                token_status = gr.Markdown(value="<span class='small-muted'>Token Suggestions: 0</span>")
                service_status = gr.Markdown(value="<span class='small-muted'>Service Suggestions: 0</span>")

            with gr.Tab("Suggestions"):
                gr.Markdown("### Token Suggestions (prefix / bigram)")
                token_suggestions = gr.Dropdown(
                    label="Token suggestions",
                    choices=[],
                    value=None,
                    interactive=True,
                )

                gr.Markdown("### Service Suggestions (keyword match)")
                service_suggestions = gr.Dropdown(
                    label="Service suggestions",
                    choices=[],
                    value=None,
                    interactive=True,
                )

                with gr.Accordion("Debug output", open=False):
                    token_debug_out = gr.Markdown(value="")
                    service_debug_out = gr.Markdown(value="")

            with gr.Tab("Performance"):
                gr.Markdown("### Perf test (search only)")
                with gr.Row():
                    perf_n = gr.Slider(5, 500, value=100, step=5, label="Requests")
                perf_btn = gr.Button("Run perf test")
                perf_out = gr.Markdown()
                gr.Markdown("### Rolling performance (first-token latency)")
                perf_panel = gr.Markdown(value="### Performance\nNo measurements yet.")

            with gr.Tab("Memory"):
                gr.Markdown("### Memory Testing")
                mem_btn = gr.Button("Refresh memory snapshot")
                mem_out = gr.Markdown(value=memory_md(mem_state_init()))
                gr.Markdown(
                    "<div class='small-muted'>"
                    "Goal: After heavy search/perf, RSS delta should stay small (no leak). "
                    "For accurate RSS, install <code>psutil</code>."
                    "</div>"
                )

        with gr.Column(scale=6):
            results_md = gr.Markdown(value="Type to search…", elem_id="results_md")

    gr.Markdown(
        f"<hr><div class='small-muted'>"
        f"<b>Cache loaded once from:</b> {CACHE_PATH} "
        f"• <b>Vocab:</b> {len(CACHE.vocab or [])} "
        f"• <b>Docs:</b> {len(CACHE.docs)}"
        f"</div>"
    )

    # ===== Events =====

    # realtime token suggestions
    query.input(
        fn=ui_update_token_suggestions,
        inputs=[query, suggest_top_k, debug_mode],
        outputs=[token_suggestions, token_status, token_debug_out],
        queue=False,
        trigger_mode="always_last",
    )

    # realtime service suggestions
    query.input(
        fn=ui_update_service_suggestions,
        inputs=[query, service_suggest_top_k, debug_mode],
        outputs=[service_suggestions, service_status, service_debug_out],
        queue=False,
        trigger_mode="always_last",
    )

    # apply token suggestion -> query (append/replace behavior)
    token_suggestions.change(
        fn=ui_apply_token_id_to_query,
        inputs=[query, token_suggestions],
        outputs=[query],
        queue=False,
    )

    # apply service suggestion -> query
    service_suggestions.change(
        fn=ui_apply_service_doc_to_query,
        inputs=[service_suggestions],
        outputs=[query],
        queue=False,
    )

    # streaming search
    query.change(
        fn=ui_stream_search,
        inputs=[query, search_top_k, stream_delay_ms, perf_state],
        outputs=[results_md, perf_panel, perf_state],
        queue=True,
        trigger_mode="always_last",
    )

    # perf test
    perf_btn.click(
        fn=lambda q, n, k: run_perf_test(CACHE, q, int(n), int(k)),
        inputs=[query, perf_n, search_top_k],
        outputs=[perf_out],
        queue=True,
    )

    # memory refresh
    mem_btn.click(
        fn=refresh_memory,
        inputs=[mem_state],
        outputs=[mem_out, mem_state],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=64).launch(
        server_name="0.0.0.0",
        server_port=7858,
        show_error=True,
        css=CSS,
    )


# http://127.0.0.1:7858
