# fastsem-jev

[English](README.md) · [简体中文](README.zh-CN.md)

Training-free token compression for Qwen3.5-4B.

## Results on our benchmark

All systems were evaluated once per question on the same 3,151-question Qwen3.5-4B workload. Accuracy and end-to-end latency come from that same inference call. Speedup = raw Qwen generation total time across all 3,151 questions ÷ method total time. Raw generation is 1.000×. Mean latency is per question; a single pass does not estimate run-to-run variance.

| System | Track | Correct / 3151 | Accuracy | Mean E2E (ms) | Speedup |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B generation | Generation baseline | 2476 | 78.58% | 505.27 | 1.000× |
| SemIf | Strict same prompt | 2476 | 78.58% | 409.35 | 1.234× |
| Open Alternative | Native prompt migration | 2369 | 75.18% | 405.08 | 1.247× |
| LitJev | Native prompt migration | 2510 | 79.66% | 461.92 | 1.094× |
| SimpleJev | Native prompt migration | 2662 | 84.48% | 1169.06 | 0.432× |
| AnyJev L0 | Native prompt migration | 2348 | 74.52% | 1322.89 | 0.382× |
| jev-local | Native prompt migration | 2033 | 64.52% | 1330.24 | 0.380× |
| reflex | Native prompt migration | 2397 | 76.07% | 477.82 | 1.057× |
| jqv | Native prompt migration | 2242 | 71.15% | 428.58 | 1.179× |
| OneForward | Native prompt migration | 2192 | 69.57% | 435.47 | 1.160× |
| fastsem-jev (l16r25) | Strict same prompt | 2481 | 78.74% | 271.59 | 1.860× |
| l16r17p5 | Strict same prompt | 2486 | 78.90% | 267.19 | 1.891× |

Native-method migration rows port each project's request/readout design to the same frozen Qwen3.5-4B and retain its native prompt. These compare end-to-end migrated implementations; prompt differences remain. SemIf and fastsem-jev use the same direct prompt. Exact upstream code revisions are recorded in results/results.json; per-question records are in results/rows/.

The released default and primary method name `fastsem-jev` means **l16r25**: compress after layer 16 and retain 25% of evidence tokens. `l16r17p5` is a separately measured alternative, not the default.

The 3,151 items combine public Jev questions with locally developed additions. This is a development benchmark, not an official JevBench score or independent blind test. JevBench also evaluates calibration and cost; this table reports accuracy and latency.

### fastsem-jev parameter sweep

| Variant | Layer | Evidence retained | Correct / 3151 | Accuracy | Mean E2E (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| l8r25 | 8 | 25% | 2152 | 68.30% | 203.82 | 2.479× |
| l12r20 | 12 | 20% | 2400 | 76.17% | 231.23 | 2.185× |
| l12r25 | 12 | 25% | 2393 | 75.94% | 241.13 | 2.095× |
| l14r15 | 14 | 15% | 2445 | 77.59% | 333.88 | 1.513× |
| l14r17p5 | 14 | 17.5% | 2446 | 77.63% | 388.42 | 1.301× |
| l14r20 | 14 | 20% | 2442 | 77.50% | 324.06 | 1.559× |
| l15r15 | 15 | 15% | 2438 | 77.37% | 399.58 | 1.265× |
| l15r17p5 | 15 | 17.5% | 2434 | 77.25% | 248.86 | 2.030× |
| l15r20 | 15 | 20% | 2432 | 77.18% | 247.46 | 2.042× |
| l16r12p5 | 16 | 12.5% | 2473 | 78.48% | 258.67 | 1.953× |
| l16r15 | 16 | 15% | 2478 | 78.64% | 266.83 | 1.894× |
| l16r17p5 | 16 | 17.5% | 2486 | 78.90% | 267.19 | 1.891× |
| l16r20 | 16 | 20% | 2483 | 78.80% | 267.95 | 1.886× |
| fastsem-jev | 16 | 25% | 2481 | 78.74% | 271.59 | 1.860× |
| l17r15 | 17 | 15% | 2487 | 78.93% | 274.25 | 1.842× |
| l17r17p5 | 17 | 17.5% | 2484 | 78.83% | 277.03 | 1.824× |
| l17r20 | 17 | 20% | 2489 | 78.99% | 279.46 | 1.808× |

<code>l16rN</code> compresses after layer 16 and retains N percent of evidence tokens. The cosine selector, anchor budget, local-span rule, prompt, model revision and full 32-layer execution are held fixed. <code>l16r25</code> is the original cosine-r25/a64 configuration. <code>l16r17p5</code> is the most balanced variant in this sweep. The existing <code>l16r20</code> measurement was reused. Full metadata: results/results.json.

## Quick start

Requirements: Windows, one visible NVIDIA CUDA GPU, and a driver compatible with CUDA 13. Python 3.12 and dependencies are managed and pinned with uv.

~~~powershell
git clone https://github.com/KoshiBebop/fastsem-jev.git D:\jev\fastsem-jev
Set-Location D:\jev\fastsem-jev
$env:UV_CACHE_DIR = 'D:\jev\.cache\uv'
$env:HF_HOME = 'D:\jev\.cache\huggingface'
$env:HF_HUB_CACHE = 'D:\jev\.cache\huggingface\hub'
uv sync --locked
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run fastsem-jev --state "The parcel arrives tomorrow." --question "Has it been delivered?" --options yes no
~~~

Linux with a CUDA 13 compatible NVIDIA driver:

~~~bash
git clone https://github.com/KoshiBebop/fastsem-jev.git /data/jev/fastsem-jev
cd /data/jev/fastsem-jev
export UV_CACHE_DIR=/data/jev/.cache/uv
export HF_HOME=/data/jev/.cache/huggingface
export HF_HUB_CACHE=/data/jev/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
uv sync --locked
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
uv run fastsem-jev --state "The parcel arrives tomorrow." --question "Has it been delivered?" --options yes no
~~~

The first run downloads Qwen/Qwen3.5-4B from Hugging Face into the configured cache. The default model revision is pinned to the revision used in this evaluation. Each JSONL result includes the option prediction, probabilities, latency and token-retention diagnostics.

Use the single local command `fastsem-jev`: `--state` is the text, `--question` is the criterion, and `--options` lists the allowed answers. Defaults are Qwen3.5-4B and l16r25; no server is needed. JSON decisions go to stdout and the timing summary goes to stderr. A command loads the model once and evaluates each request once; startup and model loading are excluded from the reported decision time, but are part of command wall time.

Optional parameters use the same command:

~~~bash
uv run fastsem-jev --state "The parcel arrives tomorrow." --question "Has it been delivered?" --options "yes=It has arrived." "no=It has not arrived yet." --layer 16 --retain-ratio 0.25
~~~

Use `--model /path/to/Qwen3.5-4B` for existing local weights, or leave it out to use the pinned Hugging Face model. `--options` accepts 2–16 labels or `label=description` entries. For multiple requests, use `--input` to load JSON/JSONL with the same command; model loading is shared across the file. `--output` saves decisions to a new file; otherwise they are printed.

Batch input is JSONL: one request per line with <code>state</code>, <code>question</code>, <code>options</code>, and optional <code>id</code>/<code>expected</code>. Accuracy is calculated when expected labels are present:

~~~powershell
uv run fastsem-jev --input requests.jsonl --output outputs/decisions.jsonl
~~~

### JevBench public-set evaluation

This runs fastsem-jev once on every task in JevBench's published `easy`, `original`, and `hard` public tiers, then scores the predictions with JevBench's own scoring and summary code. It does not access held-out/private tiers and does not submit a leaderboard entry. The adapter maps JevBench's typed rubric to fastsem-jev's direct evidence/criterion/options prompt, so treat this as a local public-set evaluation, not a claim of exact prompt parity with other JevBench adapters. Model loading is outside the per-question timer; each public task is inferred once. Results are written outside both repositories, and existing result files are never overwritten.

Windows PowerShell:

~~~powershell
git clone --filter=blob:none --no-checkout https://github.com/fstandhartinger/jevbench.git D:\jev\jevbench
git -C D:\jev\jevbench sparse-checkout init --cone
git -C D:\jev\jevbench sparse-checkout set jevbench datasets/public
git -C D:\jev\jevbench checkout
$env:UV_CACHE_DIR = 'D:\jev\.cache\uv'
$env:HF_HOME = 'D:\jev\.cache\huggingface'
$env:HF_HUB_CACHE = 'D:\jev\.cache\huggingface\hub'
uv run python scripts/jevbench_public.py
~~~

Linux (CUDA):

~~~bash
git clone --filter=blob:none --no-checkout https://github.com/fstandhartinger/jevbench.git /data/jev/jevbench
git -C /data/jev/jevbench sparse-checkout init --cone
git -C /data/jev/jevbench sparse-checkout set jevbench datasets/public
git -C /data/jev/jevbench checkout
export UV_CACHE_DIR=/data/jev/.cache/uv
export HF_HOME=/data/jev/.cache/huggingface
export HF_HUB_CACHE=/data/jev/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
uv run python scripts/jevbench_public.py
~~~

The command automatically uses a sibling `jevbench` checkout and creates a timestamped result directory under the parent `results/` folder, so it will not overwrite an earlier run. The output contains `results.jsonl` (one prediction, probability distribution, and latency per task) and `summary.json` (JevBench accuracy, calibration, latency, coverage, data hash, and source commit). The default primary configuration is layer 16 / 25% retention (`l16r25`). To test a separately measured alternative, add `--layer 16 --retain-ratio 0.175`; to use a JevBench checkout elsewhere, set `JEVBENCH_DIR` or pass `--jevbench-repo`. The first run downloads the pinned Qwen3.5-4B weights if they are not already cached.

### Method

The model processes the full prompt through its first 16 layers. At that point fastsem-jev ranks tokens inside the evidence value by cosine similarity between each token hidden state and the final decision-token hidden state. It retains the configured evidence fraction, including up to 64 head and tail anchors inside that budget, then prefers contiguous windows of radius two. Instructions, criterion, options and the answer position remain intact. Qwen3.5 attention masks are rebuilt using original position indices. All 32 transformer layers execute; final option-letter logits are read directly, with no autoregressive decoding. The method uses frozen weights, no fitted calibration and no length-based routing.

## License

See LICENSE and NOTICE.md. Qwen model weights retain their upstream license and are downloaded separately. The repository does not include benchmark question text or weights. Released per-question rows contain IDs, scoring labels, predictions and elapsed time.
