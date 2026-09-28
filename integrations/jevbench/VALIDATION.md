# Local adapter validation — 2026-09-28

Real Qwen3.5-4B BF16 weights, l16r25, one RTX 4090, Windows native CUDA.
Model revision metadata: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
weights loaded from the project's existing local directory.
JevBench checkout: `fd54ea7dc02bbe29c6ac8f6e015a54cdcff26805`, with the supplied
registration patch applied. Torch 2.10.0+cu130, Transformers 5.17.0.

All 231 published easy/original/hard tasks were run once per path. Model loading
was excluded; each path kept its first inference, with no extra warmup inference.
The adapter run and direct run were separate passes, not repeated timing trials.

| Path | Correct | Accuracy | Mean (ms) | Median (ms) | Total (s) |
|---|---:|---:|---:|---:|---:|
| Direct engine, identical mapped requests | 180/231 | 77.92% | 103.31 | 62.10 | 23.8653 |
| FastSemLocalAdapter | 180/231 | 77.92% | 107.07 | 70.53 | 24.7331 |

All 231 predictions matched; the maximum absolute probability difference was
exactly 0. Adapter operational success and schema validity were 231/231.
The adapter timer wraps mapping, inference and result packaging. These are
single-pass interface checks, not a new speedup claim against raw Qwen generation.
The published 3,151-task research results were not changed.

The older removed HTTP prototype obtained 185/231 with Noul options in
`yes`, `no` order and Choice options in criteria-dictionary insertion order.
This adapter preserves the standalone public script's canonical `task.labels`
order (Noul: `no`, `yes`; 119 of 139 public Choice items also differ from the
HTTP ordering). The two prompt mappings are
not interchangeable accuracy baselines; no new ordering was selected by test labels.

Six local unit tests passed, including all three mappings, cached loading,
probability pass-through, missing/extra labels, malformed values and inference
exceptions, plus the user CLI tests. The patched JevBench adapter factory was
imported and instantiated successfully, and the registration patch passed the
reverse-application check against the patched checkout. Patched CLI syntax was
checked. The only installed fastsem-jev console entry point remains `fastsem-jev`.

The Windows public run exercised the real adapter and official scoring/summary
modules. The official Linux CLI/Runner/ledger was not executed: its ledger imports
Unix `fcntl`, and this host has no usable native Linux test environment. No Docker
was used. The Linux command is documented in README.md for evaluator integration.

Local full evidence: `D:\jev\results\fastsem-jev-public-l16r25-20260928T151428Z\`
(`results.jsonl`, `summary.json`, `direct-check.jsonl`, `direct-check-summary.json`).
