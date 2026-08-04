#!/usr/bin/env python3
"""Print few-shot accuracy table. One result file per (model, dataset)."""
import json
import sys
from pathlib import Path
from collections import defaultdict

FEWSHOT = {"gsm8k_fewshot", "math500_fewshot", "aqua_fewshot", "svamp_fewshot", "gpqa_fewshot", "commonsense_qa_fewshot"}
DS = ["gsm8k", "math500", "aqua", "svamp", "gpqa", "commonsense_qa"]

def norm(m):
    if not m: return "?"
    for x in ("https://huggingface.co/", "http://huggingface.co/"):
        if m.startswith(x): m = m[len(x):]
    return m.split("/")[-1] if "/" in m else m

def acc_for(path):
    total = correct = 0
    try:
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                try: d = json.loads(line)
                except: continue
                if "score" not in d: continue
                s = d["score"]
                ok = bool(s[0]) if isinstance(s, list) and s else bool(s)
                total += 1
                if ok: correct += 1
    except Exception:
        pass
    return (correct, total) if total else None

def main():
    db_path = Path(__file__).resolve().parents[1] / "backend" / "job_db.json"
    with open(db_path) as f:
        db = json.load(f)

    # (model, dataset) -> one (path, status)
    best = {}
    for jid, info in db.items():
        if jid.startswith("cot_") or jid.startswith("truncation_"): continue
        r = info.get("request") or {}
        if (r.get("prompt_type") or "") not in FEWSHOT: continue
        rf = info.get("result_file") or ""
        if not rf or not Path(rf).exists(): continue
        m, d = norm(r.get("model") or ""), r.get("dataset") or "?"
        k = (m, d)
        if k not in best or info.get("status") == "DONE":
            best[k] = (rf, info.get("status", ""))

    rows = defaultdict(dict)
    for (m, d), (path, _) in best.items():
        out = acc_for(path)
        if out:
            c, t = out
            rows[m][d] = (c, t, 100.0 * c / t)

    models = sorted(rows.keys())
    print("Few-shot prompt results (accuracy %)\n")
    print("| Model | " + " | ".join(DS) + " |")
    print("|-------|" + "|".join(["----------" for _ in DS]) + "|")
    for m in models:
        buf = [m[:32]]
        for d in DS:
            v = rows[m].get(d)
            buf.append(f" {v[2]:.1f}% " if v else " — ")
        print("| " + " | ".join(buf) + " |")
    print("\nCorrect / Total:")
    for m in models:
        buf = [m[:28].ljust(30)]
        for d in DS:
            v = rows[m].get(d)
            buf.append((f"{v[0]}/{v[1]}" if v else "—").rjust(12))
        print("  " + "".join(buf))
    print(f"\nModels: {len(models)}  |  Datasets: {len(DS)}  |  Cells: {sum(len(rows[m]) for m in models)}")

if __name__ == "__main__":
    main()
