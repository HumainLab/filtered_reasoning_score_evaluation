#!/usr/bin/env python3
"""
Submit evaluation jobs via the UI backend API.

- Models: All from AVAILABLE_MODELS except DeepSeek 1.5B (and "Link from Hugging Face").
- Datasets: Those with 2-shot prompts only — gsm8k, math500, aqua, svamp, gpqa, commonsense_qa.
- Each dataset uses its corresponding few-shot prompt (e.g. gsm8k -> gsm8k_fewshot).
- Settings: temp=0, seed=42, top_p=1, pass@1 (k=1), max_tokens=2048, backend=slurm.
- Enable probability tracking.

Usage:
  Ensure the backend is running (e.g. ./start.sh). Then:
    python submit_fewshot_slurm_jobs.py

  Avoid QOSMaxSubmitJobPerUserLimit by submitting in batches:
    python submit_fewshot_slurm_jobs.py --batch-size 5 --max-queue 10

  Override API base:
    REASONING_API_BASE=http://localhost:8007 python submit_fewshot_slurm_jobs.py
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing 'requests'. Install with: pip install requests")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Config (matches frontend AVAILABLE_MODELS, excluding DeepSeek 1.5B)
# -----------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-Math-7B",
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    "https://huggingface.co/google/gemma-7b",
    "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    "https://huggingface.co/microsoft/phi-4",
    "https://huggingface.co/microsoft/Phi-4-reasoning",
    "Link from Hugging Face",
]

EXCLUDE_MODELS = {"1.5B", "Link from Hugging Face"}

def is_excluded(model: str) -> bool:
    if model == "Link from Hugging Face":
        return True
    if "1.5B" in model:
        return True
    return False

MODELS = [m for m in AVAILABLE_MODELS if not is_excluded(m)]

# Datasets that have few-shot prompts → (dataset, prompt_type)
DATASET_PROMPT_PAIRS = [
    ("gsm8k", "gsm8k_fewshot"),
    ("math500", "math500_fewshot"),
    ("aqua", "aqua_fewshot"),
    ("svamp", "svamp_fewshot"),
    ("gpqa", "gpqa_fewshot"),
    ("commonsense_qa", "commonsense_qa_fewshot"),
]

# Job request defaults
DEFAULT_PARAMS = {
    "temperature": 0.0,
    "seed": 42,
    "top_p": 1.0,
    "top_k": 0,
    "k": 1,
    "eval_method": "pass@k",
    "max_tokens": 2048,
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


def get_api_base() -> str:
    base = os.environ.get("REASONING_API_BASE", "").strip()
    if base:
        return base.rstrip("/")
    config_path = Path(__file__).resolve().parent / "frontend" / "config.js"
    if config_path.exists():
        raw = config_path.read_text()
        m = re.search(r"API_BASE\s*=\s*['\"]([^'\"]+)['\"]", raw)
        if m:
            return m.group(1).rstrip("/")
    return "http://localhost:8000"


def squeue_count(user: str | None = None) -> int:
    """Return number of user's jobs in squeue (pending + running). -1 if squeue unavailable."""
    u = user or os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "-u", u, "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return -1
        lines = [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
        return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1


def squeue_job_ids(user: str | None = None) -> set[str]:
    """Return set of user's SLURM JOBIDs currently in squeue."""
    u = user or os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "-u", u, "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return set()
        lines = [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
        return set(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


def dedupe_slurm_queue(api_base: str, user: str, session: requests.Session) -> set[tuple[str, str]]:
    """
    Find (model, dataset) pairs with multiple jobs in squeue, cancel duplicates (keep one per pair),
    return set of (model, dataset) that have exactly one job remaining in queue.
    """
    try:
        r = session.get(f"{api_base}/jobs", timeout=30)
        r.raise_for_status()
        data = r.json()
        jobs = data.get("jobs") or []
    except requests.RequestException as e:
        print(f"  Warning: could not fetch /jobs for dedupe: {e}")
        return set()

    in_queue = squeue_job_ids(user)
    # (model, dataset) -> list of slurm_jids in queue
    pair_to_jids: dict[tuple[str, str], list[str]] = {}

    for j in jobs:
        sid = j.get("slurm_jid")
        if not sid:
            continue
        sid = str(sid).strip()
        if sid not in in_queue:
            continue
        req = j.get("request") or {}
        model = (req.get("model") or "").strip()
        dataset = (req.get("dataset") or "").strip()
        if not model or not dataset:
            continue
        key = (model, dataset)
        pair_to_jids.setdefault(key, []).append(sid)

    cancelled = 0
    for (model, dataset), jids in pair_to_jids.items():
        if len(jids) <= 1:
            continue
        jids_sorted = sorted(jids, key=lambda x: (len(x), x))
        keep, drop = jids_sorted[0], jids_sorted[1:]
        for jid in drop:
            try:
                subprocess.run(["scancel", jid], capture_output=True, text=True, timeout=5)
                cancelled += 1
                print(f"  Cancelled duplicate SLURM {jid} ({model} | {dataset})")
            except Exception as e:
                print(f"  Warning: scancel {jid} failed: {e}")

    if cancelled:
        print(f"  Cancelled {cancelled} duplicate job(s).")
        # Re-fetch squeue after cancels (may take a moment)
        time.sleep(2)
        in_queue = squeue_job_ids(user)

    # Return (model, dataset) that still have ≥1 job in queue (we kept one per pair)
    return set(pair_to_jids.keys())


def build_jobs() -> list[dict]:
    jobs = []
    for model in MODELS:
        for dataset, prompt_type in DATASET_PROMPT_PAIRS:
            req = {
                "model": model,
                "dataset": dataset,
                "prompt": "",
                "prompt_type": prompt_type,
                **DEFAULT_PARAMS,
            }
            jobs.append(req)
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit few-shot eval jobs via UI backend (SLURM)."
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Submit at most this many jobs per batch (default: 5)",
    )
    ap.add_argument(
        "--max-queue",
        type=int,
        default=32,
        help="Keep squeue count below this before submitting more (default: 32). Set to your QOS MaxSubmit limit.",
    )
    ap.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds to wait between queue checks when at limit (default: 60)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print jobs that would be submitted; do not submit",
    )
    ap.add_argument(
        "--user",
        type=str,
        default=os.environ.get("USER", ""),
        help="SLURM user for squeue count (default: $USER)",
    )
    args = ap.parse_args()

    api_base = get_api_base()
    session = requests.Session()
    session.headers["Content-Type"] = "application/json"

    all_jobs = build_jobs()
    print(f"API base: {api_base}")
    print(f"Models ({len(MODELS)}): {MODELS}")
    print(f"Datasets×prompts: {DATASET_PROMPT_PAIRS}")
    print(f"Total (model, dataset) pairs: {len(all_jobs)}")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1}] {j['model']} | {j['dataset']} | {j['prompt_type']}")
        return

    print("Deduplicating queue (cancel repeat jobs, keep one per model×dataset) ...")
    already_queued = dedupe_slurm_queue(api_base, args.user, session)
    jobs = [j for j in all_jobs if (j["model"], j["dataset"]) not in already_queued]
    skipped = len(all_jobs) - len(jobs)
    if skipped:
        print(f"Skipping {skipped} (model, dataset) pair(s) already in queue.")
    print(f"Jobs to submit: {len(jobs)}")
    if not jobs:
        print("Nothing to submit. Exiting.")
        sys.exit(0)

    if args.batch_size < len(jobs) or args.max_queue < len(jobs):
        print(f"Batching: batch_size={args.batch_size}, max_queue={args.max_queue}, poll_interval={args.poll_interval}s")
    print()

    ok = 0
    err = 0
    errors: list[tuple[dict, str]] = []
    remaining = list(jobs)
    total = len(remaining)

    while remaining:
        n = squeue_count(args.user)
        if n >= 0:
            allowed = max(0, args.max_queue - n)
            to_submit = min(args.batch_size, allowed, len(remaining))
            if to_submit <= 0:
                print(f"  Queue at limit ({n} >= {args.max_queue}). Waiting {args.poll_interval}s ...")
                time.sleep(args.poll_interval)
                continue
        else:
            to_submit = min(args.batch_size, len(remaining))

        batch = remaining[:to_submit]
        remaining = remaining[to_submit:]

        for job in batch:
            i = total - len(remaining)
            model = job["model"]
            dataset = job["dataset"]
            prompt_type = job["prompt_type"]
            print(f"  [{i}/{total}] {model} | {dataset} | {prompt_type} ... ", end="", flush=True)
            try:
                r = session.post(f"{api_base}/jobs", json=job, timeout=60)
                r.raise_for_status()
                data = r.json()
                jid = data.get("job_id", "?")
                slurm = data.get("slurm_jid", "")
                suf = f" (SLURM {slurm})" if slurm else ""
                print(f"OK → {jid}{suf}")
                ok += 1
            except requests.RequestException as e:
                print(f"FAIL: {e}")
                err += 1
                errors.append((job, str(e)))

        if remaining:
            print(f"  Submitted batch. {len(remaining)} left. Waiting {args.poll_interval}s before next batch ...")
            time.sleep(args.poll_interval)

    print()
    print(f"Submitted: {ok}, Failed: {err}")
    if errors:
        print("\nFailed jobs:")
        for j, msg in errors:
            print(f"  - {j['model']} | {j['dataset']} | {j['prompt_type']}: {msg}")

    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
