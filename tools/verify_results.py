"""Recompute every reported count and latency from released per-question records."""
from pathlib import Path
import gzip
import json
import math
import statistics


def main():
    root = Path(__file__).resolve().parents[1] / 'results'
    report = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    reference = None
    for key, claimed in report['methods'].items():
        with gzip.open(root / 'rows' / f'{key}.jsonl.gz', 'rt', encoding='utf-8') as source:
            rows = [json.loads(line) for line in source]
        signature = {r['id']: (r['expected'], r['group']) for r in rows}
        assert len(rows) == len(signature) == report['rows'] == 3151, key
        if reference is None:
            reference = signature
        assert signature == reference, key
        times = [r['seconds'] for r in rows]
        assert all(math.isfinite(t) and t > 0 for t in times), key
        correct = sum(r['prediction'] == r['expected'] for r in rows)
        measured = dict(correct=correct, accuracy=correct / len(rows),
                        mean_s=statistics.fmean(times), median_s=statistics.median(times),
                        total_s=sum(times))
        measured['speedup_vs_raw_qwen'] = report['methods']['qwen_generate']['total_s'] / measured['total_s']
        for metric, value in measured.items():
            assert math.isclose(value, claimed[metric], rel_tol=1e-10), (key, metric)
    print(f"Verified {len(report['methods'])} methods × 3151 rows; all counts and latencies match.")


if __name__ == '__main__':
    main()
