# fastsem-jev：面向 Qwen3.5-4B 的免训练 token 压缩

[English](README.md) · [简体中文](README.zh-CN.md)

使用免训练的 token 压缩加速 Qwen3.5-4B 决策推理。

## 自有benchmark结果

所有方法在相同的 Qwen3.5-4B、3,151 题上各推理一次；准确率和端到端耗时来自同一次调用。加速比 = 原始 Qwen 生成的全题总耗时 ÷ 对应方法的全题总耗时。原始生成固定为 1.000×。均值按每题计算；单遍测试不估算重复运行波动。

| 方法 | 轨道 | 正确数 / 3151 | 准确率 | E2E均值 (ms) | 加速比 |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B generation | 原始Qwen生成基线 | 2476 | 78.58% | 505.27 | 1.000× |
| SemIf | 严格同提示词 | 2476 | 78.58% | 409.35 | 1.234× |
| Open Alternative | 原生提示词迁移 | 2369 | 75.18% | 405.08 | 1.247× |
| LitJev | 原生提示词迁移 | 2510 | 79.66% | 461.92 | 1.094× |
| SimpleJev | 原生提示词迁移 | 2662 | 84.48% | 1169.06 | 0.432× |
| AnyJev L0 | 原生提示词迁移 | 2348 | 74.52% | 1322.89 | 0.382× |
| jev-local | 原生提示词迁移 | 2033 | 64.52% | 1330.24 | 0.380× |
| reflex | 原生提示词迁移 | 2397 | 76.07% | 477.82 | 1.057× |
| jqv | 原生提示词迁移 | 2242 | 71.15% | 428.58 | 1.179× |
| OneForward | 原生提示词迁移 | 2192 | 69.57% | 435.47 | 1.160× |
| fastsem-jev (l16r25) | 严格同提示词 | 2481 | 78.74% | 271.59 | 1.860× |
| l16r17p5 | 严格同提示词 | 2486 | 78.90% | 267.19 | 1.891× |

开源方法迁移行把各项目的请求/读出逻辑移植到相同的冻结 Qwen3.5-4B，并保留各自原生提示词。这是迁移后端到端表现对比，提示词差异仍然存在。SemIf 与 fastsem-jev 使用相同 direct 提示词。上游代码 revision 见 results/results.json；逐题记录见 results/rows/。

3,151 题包括 Jev 公开题及本地构建的扩展题。这是开发用的自有benchmark，不是官方 JevBench 成绩，也不是独立盲测。JevBench 还评价校准度和成本；本表报告准确率与延迟。

### fastsem-jev 参数实验

| 参数版本 | 压缩层 | 证据保留率 | 正确数 / 3151 | 准确率 | E2E均值 (ms) | 加速比 |
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

<code>l16rN</code> 表示第 16 层后进行压缩、保留 N% 证据 token。余弦选择器、锚点预算、局部连续窗口、提示词、模型 revision 和完整 32 层执行均保持一致。<code>l16r25</code> 是原始 cosine-r25/a64 配置；<code>l16r17p5</code> 是本轮中准确率和速度较均衡的参数。<code>l16r20</code> 直接复用了既有测量。完整元数据见 results/results.json。

## 快速开始

需要 Windows、单张可见的 NVIDIA CUDA GPU，以及兼容 CUDA 13 的驱动。使用 uv 管理 Python 3.12 和锁定依赖。

~~~powershell
git clone https://github.com/KoshiBebop/fastsem-jev.git D:\jev\fastsem-jev
Set-Location D:\jev\fastsem-jev
$env:UV_CACHE_DIR = 'D:\jev\.cache\uv'
$env:HF_HOME = 'D:\jev\.cache\huggingface'
$env:HF_HUB_CACHE = 'D:\jev\.cache\huggingface\hub'
uv sync --locked
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run fastsem-jev --input examples/request.json --output outputs/example.jsonl
~~~

首次运行会将 Qwen/Qwen3.5-4B 下载到配置的 Hugging Face 缓存；默认使用本实验对应的固定模型 revision。逐条 JSONL 输出含选项预测、概率、耗时及 token 保留诊断。

批量输入使用 JSONL，每行一个含 <code>state</code>、<code>question</code>、<code>options</code> 的请求，可选 <code>id</code> 与 <code>expected</code>。有 expected 标签时会计算准确率：

~~~powershell
uv run fastsem-jev --input requests.jsonl --output outputs/decisions.jsonl --method fastsem
uv run fastsem-jev --input requests.jsonl --method semif
uv run fastsem-jev --input requests.jsonl --method qwen_generate
~~~

Python 调用：

~~~python
from fastsem_jev import FastSemJev

engine = FastSemJev(layer=16, retain_ratio=0.175)
answer = engine.decide(
    state="包裹昨天离开仓库。",
    question="包裹送达了吗？",
    options=[{"id": "yes", "description": "包裹已经送达。"},
             {"id": "no", "description": "包裹尚未送达。"}],
)
print(answer["prediction"], answer["probabilities"])
~~~

### 方法简介

模型先对完整提示词执行前 16 层。fastsem-jev 按证据 value 中每个 token 的隐状态与最终决策 token 隐状态之间的余弦相似度排序，保留配置比例的证据 token；最多 64 个首尾锚点包含在此预算内，然后优先保留半径为 2 的连续窗口。指令、criterion、选项和答案位置保留。重建 Qwen3.5 注意力掩码时沿用原始位置索引。模型仍执行全部 32 层，最终直接读取选项字母 logits，不做自回归解码。权重冻结，不拟合校准参数，也不按输入长度切换路径。

## 许可证与致谢

许可证和上游归属见 LICENSE 与 NOTICE.md。Qwen 权重按上游许可单独下载。本仓库不包含基准题目原文或模型权重。逐题记录仅含 ID、评分标签、预测和耗时。
