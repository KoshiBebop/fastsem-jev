# JevBench local adapter / 本地适配器

The user-facing command remains `uv run fastsem-jev`. This integration lets
JevBench call the same engine in-process, without HTTP or a subprocess per item.
默认仍使用 `l16r25`；日常使用入口不变。本目录面向评测集成维护者。

## Contract

`fastsem_jev.jevbench_adapter.FastSemLocalAdapter` provides `load()`,
`build_request(task)`, `run(task) -> DecisionResult`, and `reserve_estimate(task)`.
The constructor accepts JevBench's common adapter arguments. `endpoint` is a
local model directory or Hugging Face model ID, not a URL; `model` is its display
name. `revision` defaults to the pinned Qwen3.5-4B revision. Direct constructor
calls can override `layer` and `retain_ratio`; the CLI registration uses l16r25.

- Model loading is cached. Warm-load before timing; a cold `run()` includes loading.
- Choice follows `task.labels`, with descriptions `label: criterion`.
- Noul follows `task.labels` (normally `no`, `yes`), mapping `false`/`true` rubric text.
- Score requires consecutive string indices `0` through `k-1` in level order.
- Only state, instructions and options reach the engine; expected labels do not.
- Returned label sets and numeric values are checked. Probabilities are neither
  filled nor normalized; JevBench's scorer owns probability-sum tolerance.
- `probs_source="native"`, input-token usage and zero output tokens are reported.
- Budget reservation is zero; unpriced compute is not reported as a free service.

The mapping preserves the existing standalone public-set script's label order.
It differs from the removed HTTP prototype's yes/no ordering for Noul and
its criteria-dictionary insertion order for Choice (119 of 139 public Choice items).
These prompt orderings must not be treated as identical accuracy comparisons.

## Registration patch

`fastsem-local.patch` targets JevBench commit
`fd54ea7dc02bbe29c6ac8f6e015a54cdcff26805`. It registers the optional adapter
lazily, so other JevBench adapters do not require fastsem-jev to be installed.
It is a supplied integration patch, not an upstream-merged feature.

From the fastsem-jev checkout, with JevBench cloned into the sibling directory:

```bash
git -C ../jevbench checkout fd54ea7dc02bbe29c6ac8f6e015a54cdcff26805
git -C ../jevbench apply --check ../fastsem-jev/integrations/jevbench/fastsem-local.patch
git -C ../jevbench apply ../fastsem-jev/integrations/jevbench/fastsem-local.patch
uv sync --locked
```

Use a clean JevBench checkout for this pinned integration. Do not reapply the
patch to an already-patched checkout.

## Linux: official runner

From the fastsem-jev checkout after applying the patch:

```bash
export PYTHONPATH="$PWD/../jevbench"
export JEVBENCH_WARM_LOAD=1
RUN="../results/fastsem-local-$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m jevbench.cli run \
  --adapter fastsem_local --endpoint /data/jev/models/Qwen3.5-4B \
  --tasks ../jevbench/datasets/public/easy.jsonl,../jevbench/datasets/public/original.jsonl,../jevbench/datasets/public/hard.jsonl \
  --results "$RUN/results.jsonl" --raw-dir "$RUN/raw" \
  --ledger "$RUN/ledger.jsonl" --manifest "$RUN/manifest.json" \
  --reserve-usd 0 --cap-usd 0
```

The official runner measures wall time around `run(task)`, including request
mapping, tokenization, inference and response packaging. Warm model loading is
outside that timer. Run serially on a single visible CUDA GPU.

## Windows / Linux: public validation

The standalone script now uses this exact adapter plus JevBench scoring and
summary modules. No registration patch is needed for this path:

```powershell
uv run python scripts/jevbench_public.py --jevbench-repo D:\jev\jevbench --model D:\jev\models\Qwen3.5-4B
```

On Linux, substitute the corresponding `/data/jev/...` paths. This path works
on Windows because it does not import the official runner's Unix `fcntl` ledger.
It is not a test of the official runner/ledger itself.

Adapter tests (from fastsem-jev, PowerShell):

```powershell
$env:PYTHONPATH = 'D:\jev\jevbench'
uv run python -m unittest discover -s tests -v
```

Linux equivalent: `PYTHONPATH=../jevbench uv run python -m unittest discover -s tests -v`.
