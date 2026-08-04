#!/usr/bin/env python3
"""
Submit HumanEval jobs for all 9 paper models using the reason-then-code prompt
(`humaneval_reason` in evaluation/utils.py).

Defaults: SLURM backend, k=16, prob tracking on, temp=0.7, top_p=0.95.

Usage:
  # Start backend first (./start.sh), then:
  python submit_humaneval_reason_jobs.py

  # Or submit without HTTP (imports runner directly):
  python submit_humaneval_reason_jobs.py --direct

  python submit_humaneval_reason_jobs.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore

REPO = Path(__file__).resolve().parent

MODELS = [
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    "https://huggingface.co/microsoft/Phi-4-reasoning",
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B",
    "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507",
    "https://huggingface.co/google/gemma-7b",
    "https://huggingface.co/microsoft/phi-4",
]

JOB_DEFAULTS = {
    "dataset": "humaneval",
    "prompt": "",
    "prompt_type": "humaneval_reason",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 0,
    "seed": 42,
    "k": 16,
    "eval_method": "pass@k",
    "max_tokens": 4096,
    "backend": "slurm",
    "use_together_api": False,
    "together_api_key": "",
    "together_logprobs": 0,
    "enable_prob_tracking": True,
    "enable_path_vectors": False,
    "max_path_steps": 0,
    "enable_ece_eval": False,
    "ece_runs": 10,
}


def api_base() -> str:
    base = os.environ.get("REASONING_API_BASE", "").strip()
    if base:
        return base.rstrip("/")
    for port in (8000, 8009, 8007):
        url = f"http://localhost:{port}"
        if requests:
            try:
                if requests.get(f"{url}/jobs", timeout=2).status_code < 500:
                    return url
            except Exception:
                pass
    cfg = REPO / "frontend" / "config.js"
    if cfg.exists():
        m = re.search(r"API_BASE\s*=\s*['\"]([^'\"]+)['\"]", cfg.read_text())
        if m:
            return m.group(1).rstrip("/")
    return "http://localhost:8000"


def short_name(model: str) -> str:
    if model.startswith("https://huggingface.co/"):
        return model.rsplit("/", 1)[-1]
    return model.split("/")[-1]


def submit_via_api(base: str, payload: dict) -> dict:
    if requests is None:
        raise RuntimeError("pip install requests")
    r = requests.post(f"{base}/jobs", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def submit_via_runner(payload: dict) -> dict:
    sys.path.insert(0, str(REPO / "backend"))
    from app.runner import launch_job
    from app.schemas import Backend, EvalRequest

    backend = Backend.slurm if payload.get("backend") == "slurm" else Backend.local
    req = EvalRequest(**{**payload, "backend": backend})
    jid = launch_job(req)
    return {"job_id": jid, "slurm_jid": None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit HumanEval reason+comment jobs (9 models).")
    ap.add_argument("--direct", action="store_true", help="Use runner.launch_job (no HTTP)")
    ap.add_argument("--dry-run", action="store_true", help="Print payloads only")
    ap.add_argument("--api-base", default=None, help="Override API URL")
    ap.add_argument("--no-prob-tracking", action="store_true")
    ap.add_argument("--k", type=int, default=16, help="n_sampling / pass@k (default 16)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between submissions")
    args = ap.parse_args()

    base = (args.api_base or api_base()).rstrip("/")
    use_direct = args.direct

    print("HumanEval reason-then-code job batch")
    print(f"  prompt_type: humaneval_reason")
    print(f"  models: {len(MODELS)}")
    print(f"  k={args.k}  prob_tracking={not args.no_prob_tracking}")
    print(f"  submit mode: {'direct runner' if use_direct else f'API {base}'}")
    print("=" * 60)

    ok, fail = 0, 0
    for i, model in enumerate(MODELS, 1):
        payload = {**JOB_DEFAULTS, "model": model, "k": args.k}
        if args.no_prob_tracking:
            payload["enable_prob_tracking"] = False

        label = short_name(model)
        print(f"\n[{i}/{len(MODELS)}] {label}")

        if args.dry_run:
            print(f"  dry-run: {payload}")
            ok += 1
            continue

        try:
            if use_direct:
                out = submit_via_runner(payload)
            else:
                out = submit_via_api(base, payload)
            jid = out.get("job_id", "?")
            sid = out.get("slurm_jid") or out.get("slurm_jid")
            print(f"  ✓ job_id={jid}  slurm={sid or '(pending)'}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {e}")
            fail += 1

        if i < len(MODELS) and args.sleep > 0:
            time.sleep(args.sleep)

    print("\n" + "=" * 60)
    print(f"Done: {ok} submitted, {fail} failed")
    if fail and not use_direct:
        print("Tip: run ./start.sh then retry, or: python submit_humaneval_reason_jobs.py --direct")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
