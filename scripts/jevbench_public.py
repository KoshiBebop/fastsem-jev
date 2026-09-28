"""Run fastsem-jev once over JevBench's published public JSONL tiers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def write_new(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench-repo", type=Path,
                        default=Path(os.environ.get("JEVBENCH_DIR", project_root.parent / "jevbench")),
                        help="JevBench checkout (default: ../jevbench or JEVBENCH_DIR)")
    parser.add_argument("--output-dir", type=Path,
                        help="new output directory (default: ../results/fastsem-jev-public-l16r25-<UTC timestamp>)")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--retain-ratio", type=float, default=0.25)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Pinned model name or local weights directory")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    args = parser.parse_args()

    repo = args.jevbench_repo.resolve()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = project_root.parent / "results" / f"fastsem-jev-public-l{args.layer}r{args.retain_ratio * 100:g}-{stamp}"
    else:
        out_dir = args.output_dir.resolve()
    if not (repo / "jevbench" / "tasks.py").is_file():
        parser.error("--jevbench-repo must point to a JevBench checkout containing jevbench/tasks.py")
    if out_dir == repo or repo in out_dir.parents:
        parser.error("Keep generated results outside the JevBench checkout")
    if out_dir == project_root or project_root in out_dir.parents:
        parser.error("Keep generated results outside the fastsem-jev checkout")
    if any((out_dir / name).exists() for name in ("results.jsonl", "summary.json")):
        parser.error("Output files already exist; choose a new output directory to preserve prior runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.tasks import dataset_hash, load_jsonl

    data_dir = repo / "datasets" / "public"
    task_files = [data_dir / f"{tier}.jsonl" for tier in ("easy", "original", "hard")]
    if any(not path.is_file() for path in task_files):
        parser.error("Expected public/easy.jsonl, public/original.jsonl, and public/hard.jsonl")
    tasks = [task for path in task_files for task in load_jsonl(str(path))]
    if not tasks or any(task.split != "public" for task in tasks):
        parser.error("The selected files must contain public-split tasks only")

    from fastsem_jev.jevbench_adapter import FastSemLocalAdapter
    adapter = FastSemLocalAdapter(endpoint=args.model, revision=args.revision,
                                  layer=args.layer, retain_ratio=args.retain_ratio)
    adapter.load()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"Loaded {len(tasks)} public tasks; model load completed before timing.", flush=True)

    results_path = out_dir / "results.jsonl"
    records = []
    with results_path.open("x", encoding="utf-8", newline="\n") as results_file:
      for index, task in enumerate(tasks, 1):
        started = time.perf_counter()
        answer = adapter.run(task)
        latency_s = time.perf_counter() - started
        scored = score_task(answer.probs or {}, task)
        record = {
            "task_id": task.id,
            "family": task.family,
            "split": task.split,
            "group": task.group,
            "ok": answer.ok,
            "status": "ok" if answer.ok else "failed",
            "valid": scored["valid"],
            "strict_valid": scored["strict_valid"],
            "renormalized": scored["renormalized"],
            "correct": scored["correct"],
            "predicted": scored.get("predicted"),
            "ordinal_ev": scored.get("ordinal_ev"),
            "probs": scored.get("probs"),
            "probs_as_returned": answer.probs,
            "probs_source": answer.probs_source,
            "model": adapter.model,
            "latency_s": latency_s,
            "usage": answer.usage,
            "cost_usd": None,
            "cost_basis": "self_hosted_gpu_cost_not_estimated",
            "error": answer.error,
        }
        records.append(record)
        results_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        results_file.flush()
        os.fsync(results_file.fileno())
        if index % 10 == 0 or index == len(tasks):
            print(f"{index}/{len(tasks)} completed", flush=True)

    summary = summarize(tasks, records, headline_only=True)
    summary["latency"]["mean_s"] = sum(r["latency_s"] for r in records) / len(records)
    summary["latency"]["total_s"] = sum(r["latency_s"] for r in records)
    summary["dataset_hash"] = dataset_hash(tasks)
    summary["run"] = {
        "method": "fastsem-jev",
        "layer": args.layer,
        "retain_ratio": args.retain_ratio,
        "n_tasks": len(tasks),
        "repetitions_per_task": 1,
        "model_revision": adapter.revision,
        "model_source": adapter.endpoint,
        "adapter": adapter.name,
        "jevbench_git_commit": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip(),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "timing_note": "Model load excluded; latency wraps adapter.run(task), including mapping, inference and result packaging, once per public task.",
        "prompt_note": "JevBench task fields are mapped to fastsem-jev's direct evidence/criterion/options prompt; this is not a JevBench leaderboard submission.",
        "source_tiers": [path.name for path in task_files],
    }
    write_new(out_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n")
    print(json.dumps({"n_planned": summary["n_planned"],
                      "n_attempted": summary["n_attempted"],
                      "n_correct": summary["n_correct"],
                      "accuracy": summary["accuracy"],
                      "latency": summary["latency"],
                      "summary": str(out_dir / "summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
