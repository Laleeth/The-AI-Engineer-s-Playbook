# Quantization

Storing numbers with less precision so the model takes less memory and runs faster.

The reason it matters operationally: memory is the binding constraint in inference, and
quantization is the cheapest way to relax it. A model that doesn't fit on one GPU is a
completely different cost structure from one that does.

---

## The idea

Model weights are numbers. By default they're stored at 16 bits each. You can store them
at 8 bits, or 4, and lose some precision.

```
FP32   32 bits   4 bytes   full precision, rarely used for inference
BF16   16 bits   2 bytes   the common default
FP8     8 bits   1 byte    half the memory
INT4    4 bits   0.5 byte  quarter the memory
```

The size follows directly:

```python
def model_size_gb(params_billion, bits_per_weight):
    return params_billion * 1e9 * bits_per_weight / 8 / 1e9

for bits, name in [(16, "BF16"), (8, "FP8"), (4, "INT4")]:
    print(f"{name:5} 8B model: {model_size_gb(8, bits):5.1f} GB   "
          f"70B model: {model_size_gb(70, bits):5.1f} GB")

# BF16  8B model:  16.0 GB   70B model: 140.0 GB
# FP8   8B model:   8.0 GB   70B model:  70.0 GB
# INT4  8B model:   4.0 GB   70B model:  35.0 GB
```

That table is the whole business case. A 70B model at BF16 needs multiple GPUs. At INT4 it
fits on one 40 GB card. Multi-GPU serving means communication overhead, more complex
deployment, and worse economics — so fitting on one card is often worth real accuracy.

---

## Three things you get

**1. Less memory for weights.** Obvious, and it's what lets bigger models fit.

**2. More room for KV cache.** This is the one people miss. Smaller weights leave more
memory for the [KV cache](kv-cache.md), which raises concurrency, which raises throughput:

```python
def concurrency_gain(gpu_gb, params_b, kv_per_token_kb, avg_context):
    for bits, name in [(16, "BF16"), (8, "FP8"), (4, "INT4")]:
        weights = params_b * 1e9 * bits / 8 / 1e9
        free_gb = gpu_gb - weights
        per_request_gb = avg_context * kv_per_token_kb / 1e6
        print(f"{name:5} weights {weights:5.1f} GB, "
              f"~{int(free_gb / per_request_gb):4d} concurrent requests")

concurrency_gain(gpu_gb=80, params_b=8, kv_per_token_kb=128, avg_context=2500)
# BF16  weights  16.0 GB, ~ 200 concurrent requests
# FP8   weights   8.0 GB, ~ 225 concurrent requests
# INT4  weights   4.0 GB, ~ 237 concurrent requests
```

Note the gain is modest here because KV cache already dominates the memory. For a 70B
model, where weights dominate, the effect is much larger.

**3. Faster decode.** Decode is memory-bandwidth-bound — the GPU reads the whole model per
token. Half the weights means half the reading, which means faster generation.

---

## Where the precision goes

Quantization maps a range of real numbers onto a small set of representable values.

```python
import numpy as np

def quantize_int8(weights):
    """Simple symmetric quantization: scale into [-127, 127]."""
    scale = np.abs(weights).max() / 127
    quantized = np.round(weights / scale).astype(np.int8)
    return quantized, scale

def dequantize_int8(quantized, scale):
    return quantized.astype(np.float32) * scale
```

The error is the rounding. Two things determine how much it hurts:

**Outliers.** If one weight is far larger than the rest, the scale is set by it and every
other weight loses precision. Real models have outlier weights, and handling them is most
of what distinguishes good quantization methods from naive ones.

**Granularity.** One scale for the whole model is crude. One per layer is better. One per
row or per small group of weights is better still, at the cost of storing more scales.

```python
def quantize_per_group(weights, group_size=128):
    """A scale per group of weights — much less damage from outliers."""
    groups = weights.reshape(-1, group_size)
    scales = np.abs(groups).max(axis=1, keepdims=True) / 127
    quantized = np.round(groups / scales).astype(np.int8)
    return quantized, scales
```

---

## Calibrated or not

**Uncalibrated (post-training, data-free).** Just convert the numbers. Instant, no data
needed. Works surprisingly well down to 8 bits.

**Calibrated.** Run a few hundred representative samples through the model, observe the
actual range of activations, and choose scales accordingly. Better accuracy, especially at
4 bits. Costs an hour and a sample of data.

**Quantization-aware training.** Train with quantization simulated in the loop. Best
accuracy, and much more expensive — you need the training pipeline.

For most teams: **uncalibrated FP8 is nearly free and safe. INT4 usually wants
calibration.**

Important: your calibration data should look like your production traffic. Calibrating on
generic web text and serving legal documents means the scales are tuned for the wrong
distribution.

---

## What it costs you in quality

Rough expectations, and they vary by model and task:

| Precision | Typical quality impact |
|---|---|
| BF16 | Baseline |
| FP8 | Very small — often within noise |
| INT8 | Small, usually acceptable |
| INT4 | Noticeable but often usable, especially with calibration |
| Below 4 bits | Significant, and task-dependent |

**Two rules that matter more than the table:**

**Measure on your own hard cases, not on average.** Quantization tends to hurt most where
the model was already marginal — long-context tasks, multi-step reasoning, rare languages,
edge cases. An average benchmark score can look fine while your hardest 10% degrades
badly.

```python
async def quantization_impact(cases, full_model, quantized_model, scorer):
    """Segment by difficulty — the aggregate hides where it hurts."""
    by_difficulty = defaultdict(lambda: {"full": [], "quant": []})
    for case in cases:
        d = case.segments["difficulty"]
        by_difficulty[d]["full"].append(await scorer(case, await full_model(case)))
        by_difficulty[d]["quant"].append(await scorer(case, await quantized_model(case)))
    return {
        d: {"full": mean(v["full"]), "quant": mean(v["quant"]),
            "gap": mean(v["full"]) - mean(v["quant"])}
        for d, v in by_difficulty.items()
    }
```

A typical result:

```
difficulty    full    quant    gap
easy          0.94    0.94   +0.00
medium        0.88    0.86   +0.02
hard          0.71    0.58   +0.13     ← this is what you're buying
```

**A bigger quantized model often beats a smaller full-precision one.** If you can fit a
70B model at INT4 or an 8B model at BF16 in the same memory, the 70B is usually better
despite the precision loss. Test both — this comparison is the one worth running and it's
frequently skipped.

---

## Quantizing the KV cache separately

You can quantize weights and the KV cache independently, and they have different
trade-offs.

FP8 KV cache is production-ready for many deployments. As of the
[vLLM analysis (April 2026)](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html),
using `e4m3`:

- ~50% KV cache memory reduction versus BF16
- 94–98% of baseline performance on long-context evaluations at 1M tokens
- 1–2 points at most on reasoning benchmarks
- ~5–15% throughput gains under realistic load
- Break-even around **7,000 tokens** of context

Their guidance: enable above ~7k context and for decode-heavy memory-bound workloads; skip
for short contexts, and for models where uncalibrated accuracy falls below 95%.

The reason this is often a better first move than weight quantization: for long-context
workloads the KV cache dominates memory, so halving it buys more concurrency than halving
the weights does.

---

## Practical decisions

**When to quantize weights:**
- The model doesn't fit on one GPU and would otherwise need multi-GPU serving
- You're memory-constrained and want more concurrency
- Decode latency matters and you're bandwidth-bound

**When not to:**
- You're already comfortable on memory and the task is accuracy-sensitive
- Your hard cases degrade measurably (measure first)
- You're on an API — this is a self-hosting concern

**A reasonable default sequence:**

1. Start at BF16. Establish the quality baseline.
2. Try FP8 weights, uncalibrated. Measure. Usually nearly free.
3. Try FP8 KV cache if contexts are long. Measure.
4. Only go to INT4 if you need the memory, and calibrate.
5. Compare "bigger model, more quantized" against "smaller model, less quantized."

**Always keep the unquantized baseline** so you can attribute a quality change to the
quantization rather than to something else.

---

## Interview questions

**1. "What does quantization buy you?"**

Three things: smaller weights (bigger models fit), more room for KV cache (higher
concurrency), and faster decode (less memory traffic per token). The second is the one
people forget and it's often the largest practical gain.

**2. "How much quality do you lose?"**

FP8 is usually within noise; INT4 is noticeable but often usable with calibration. Then the
important part: measure on your hard cases, because quantization degrades most where the
model was already marginal, and an average score hides that.

**3. "70B at INT4 or 8B at BF16, same memory?"**

Usually the bigger quantized model wins, but it's an empirical question and the comparison
is worth running. Saying "test both" here is the right answer.

**4. "What's calibration?"**

Running representative samples through to observe the actual activation ranges and set
scales accordingly, rather than guessing from the weights alone. Matters most at 4 bits.
And the calibration data should match your production distribution.

**5. "Would you quantize the KV cache or the weights first?"**

Depends where memory is going. For long-context workloads the KV cache dominates, so FP8
KV cache buys more concurrency. For short contexts with a large model, weights dominate.
Compute the split before choosing.

**6. "Why do outliers matter?"**

A single large weight sets the scale for its group, so every other weight in that group
loses precision. Finer granularity — per-group rather than per-tensor scales — limits the
damage, which is why good quantization methods handle outliers specially.

---

## What to remember

- Quantization buys memory, concurrency, and decode speed.
- The concurrency gain from freeing KV cache space is the underrated one.
- FP8 is usually nearly free. INT4 usually wants calibration.
- Measure on hard cases — the average hides where it degrades.
- A bigger quantized model often beats a smaller full-precision one. Test it.
- KV cache and weights quantize independently; work out which dominates your memory.
- Calibration data must resemble production traffic.
- Always keep the unquantized baseline for attribution.

---

**Next:** [speculative-decoding.md](speculative-decoding.md) — generating several tokens
per forward pass.

---

Sources:
- [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)
