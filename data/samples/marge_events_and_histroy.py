# scripts/merge_events_and_history.py
"""
Merge two user histories:
  - events: data/raw_events/events.jsonl  (view/click timeline)
  - external: data/raw_events/user_history_external.jsonl  OR data/samples/user_history_external.jsonl

Outputs (under data/artifacts/):
  - user_history_events.jsonl    # derived from events (most-recent-first)
  - user_history_external.jsonl  # normalized copy of external (strings)
  - user_history_merged.jsonl    # merged (external-first, then add from events)
"""

from pathlib import Path
import json
import pandas as pd

RAW_EVENTS = Path("data/raw_events/events.jsonl")
EXT_CANDIDATES = [
    Path("data/raw_events/user_history_external.jsonl"),
    Path("data/samples/user_history_external.jsonl"),
]

ART = Path("data/artifacts"); ART.mkdir(parents=True, exist_ok=True)
MAX_LEN = 200  # cap merged history length

def _load_events_history() -> dict[str, list[str]]:
    if not RAW_EVENTS.exists():
        print(f"[WARN] Missing events file: {RAW_EVENTS}")
        return {}
    rows = [json.loads(l) for l in RAW_EVENTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df = df[df["event_type"].isin(["view","click"])].copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values(["user_id","ts"])

    def pack(seq):
        # most-recent-first, drop consecutive dups; keep strings
        out, last = [], None
        for it in list(seq)[::-1]:
            s = str(it) if it is not None else None
            if s and s != last:
                out.append(s)
                last = s
        return out

    hist_df = df.groupby("user_id")["item_id"].apply(pack).reset_index(name="history")

    # write normalized events history
    with open(ART / "user_history_events.jsonl", "w", encoding="utf-8") as f:
        for _, r in hist_df.iterrows():
            f.write(json.dumps({"user_id": str(r["user_id"]), "history": [str(x) for x in r["history"]]}, ensure_ascii=False) + "\n")

    return {str(r["user_id"]): [str(x) for x in r["history"]] for _, r in hist_df.iterrows()}

def _find_external_history() -> Path | None:
    for p in EXT_CANDIDATES:
        if p.exists():
            return p
    return None

def _load_external_history() -> dict[str, list[str]]:
    p = _find_external_history()
    if p is None:
        print("[WARN] External history not found in expected locations.")
        return {}
    out_norm = ART / "user_history_external.jsonl"
    d: dict[str, list[str]] = {}
    with p.open("r", encoding="utf-8") as fin, out_norm.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            rec = json.loads(line)
            uid = str(rec.get("user_id"))
            hist = [str(x) for x in rec.get("history", [])]
            d[uid] = hist
            fout.write(json.dumps({"user_id": uid, "history": hist}, ensure_ascii=False) + "\n")
    return d

def _merge(ext: dict[str, list[str]], evt: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Merge policy:
      - external-first (authoritative order)
      - append from events any items NOT already in external (preserving events order)
      - keep strings; cap to MAX_LEN
    """
    users = set(ext) | set(evt)
    merged: dict[str, list[str]] = {}
    for u in users:
        base = ext.get(u, [])
        add = [x for x in evt.get(u, []) if x not in base]
        merged[u] = (base + add)[:MAX_LEN]
    return merged

def main():
    evt = _load_events_history()
    ext = _load_external_history()
    merged = _merge(ext, evt)

    out = ART / "user_history_merged.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for u, h in merged.items():
            f.write(json.dumps({"user_id": u, "history": [str(x) for x in h]}, ensure_ascii=False) + "\n")

    print("Wrote:")
    print(" -", ART / "user_history_events.jsonl")
    print(" -", ART / "user_history_external.jsonl")
    print(" -", ART / "user_history_merged.jsonl")

if __name__ == "__main__":
    main()
