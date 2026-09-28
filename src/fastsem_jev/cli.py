"""Single-request and one-pass JSONL evaluation CLI."""
import argparse
import json
import math
from pathlib import Path
import sys
import statistics
import time

import torch

from .engine import FastSemJev, MODEL, REVISION
from ._core import validate_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Text to evaluate")
    source.add_argument("--input", type=Path, help="JSON object or JSONL requests")
    parser.add_argument("--question", help="Decision question (with --state)")
    parser.add_argument("--options", nargs="+", help="2-16 choices: label or label=description (with --state)")
    parser.add_argument("--output", type=Path, help="Create-only per-question JSONL output")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--retain-ratio", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.state is not None:
        if not args.question or not args.options:
            parser.error("--state requires --question and --options")
        options = []
        for value in args.options:
            label, separator, description = value.partition("=")
            if not label or (separator and not description):
                parser.error("options must be nonempty labels or label=description")
            options.append({"id": label, "description": description if separator else label})
        rows = [{"id": "inference", "state": args.state,
                 "question": args.question, "options": options}]
    else:
        if args.question is not None or args.options is not None:
            parser.error("--question/--options cannot be combined with --input")
        try:
            content = args.input.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in content.splitlines() if line.strip()] if args.input.suffix == ".jsonl" else [json.loads(content)]
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if not rows:
        parser.error("Input is empty")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            parser.error("Each request must be a JSON object")
        row.setdefault("id", str(i))
        try:
            validate_row(row)
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    if args.output and (args.output.exists() or args.output.with_suffix(".summary.json").exists()):
        parser.error("Output already exists; choose a new path")
    engine = FastSemJev(args.model, args.revision, args.layer, args.retain_ratio)

    def run(row):
        return engine.decide(row["state"], row["question"], row["options"])

    results = []
    stream = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("x", encoding="utf-8")
    try:
        for i, row in enumerate(rows):
            torch.cuda.synchronize(engine.device)
            started = time.perf_counter()
            answer = run(row)
            torch.cuda.synchronize(engine.device)
            seconds = time.perf_counter() - started
            if not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError("Invalid elapsed time")
            result = dict(id=row.get("id", str(i)), method="fastsem", seconds=seconds, **answer)
            if "expected" in row:
                result["expected"] = str(row["expected"])
                result["correct"] = answer["prediction"] == str(row["expected"])
            results.append(result)
            if stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
            else:
                print(json.dumps(result, ensure_ascii=False))
    finally:
        if stream:
            stream.close()
    seconds = [r["seconds"] for r in results]
    labeled = [r for r in results if "correct" in r]
    summary = dict(rows=len(results), timed_calls=len(results), repeats=1,
                   correct=sum(r["correct"] for r in labeled) if labeled else None,
                   accuracy=sum(r["correct"] for r in labeled) / len(labeled) if labeled else None,
                   labeled_rows=len(labeled), mean_s=statistics.fmean(seconds),
                   median_s=statistics.median(seconds), total_s=sum(seconds),
                   configuration=engine.spec, method="fastsem", model=engine.metadata,
                   gpu=torch.cuda.get_device_name(engine.device), cuda=torch.version.cuda)
    if args.output:
        with args.output.with_suffix(".summary.json").open("x", encoding="utf-8") as out:
            json.dump(summary, out, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
