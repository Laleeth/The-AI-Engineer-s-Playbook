# Serving Stacks

A serving stack is the thing between your HTTP endpoint and the GPU. It holds the model
weights, decides which requests run together, manages the KV cache, and streams tokens
back.

You can serve a model without one — load the weights, call `generate()`, return the text.
That works, and it is roughly twenty to fifty times more expensive per token than it needs
to be. This file is about what the gap consists of, because "we'll just run it ourselves"
is a common answer in interviews and the follow-up is always *what does the serving layer
actually do for you?*

---

## The naive server, and what it costs

Here is inference without a serving stack:

```python
# One request at a time. Correct, and catastrophically inefficient.
def handle(request):
    tokens = tokenize(request.prompt)
    output = model.generate(tokens, max_new_tokens=request.max_tokens)
    return detokenize(output)
```

Three things are wrong, and they compound:

**The GPU is idle most of the time.** Decoding is memory-bandwidth-bound: the GPU reads
the entire model weights out of memory to produce *one* token for *one* sequence (see
[prefill-vs-decode.md](../01-llm-internals/prefill-vs-decode.md)). Doing that for one
request uses a few percent of the available compute. The arithmetic units sit waiting on
memory.

**The KV cache is rebuilt and thrown away.** Every request allocates its cache, fills it,
and frees it. Shared prefixes — your system prompt, the few-shot examples, the tool
definitions — are recomputed for every request.

**Memory is allocated for the worst case.** Without paged memory management you reserve
`max_tokens` of KV cache per sequence up front, because you don't know how long the output
will be. A request that generates 40 tokens holds a 4,000-token reservation.

The fix for all three is batching, and batching is most of what a serving stack is.

---

## What the stack actually does

| Job | Why you don't want to write it |
|---|---|
| **Continuous batching** | Sequences join and leave the running batch every step. Most of the throughput win. [Its own file.](continuous-batching.md) |
| **Paged KV cache** | Allocates cache in fixed blocks instead of contiguous per-sequence reservations, so memory isn't fragmented and reservations aren't worst-case. |
| **Prefix caching** | Detects a shared prompt prefix across requests and reuses its KV blocks. Free, and large — see [positional-encoding.md](../01-llm-internals/positional-encoding.md) for why the prefix must be byte-identical. |
| **Scheduling and admission** | Decides which queued requests enter the batch, and rejects work it can't serve rather than accepting everything and degrading. |
| **Preemption** | When memory runs out mid-generation, evicts a sequence and recomputes it later instead of crashing. |
| **Quantized execution** | Runs the model at reduced precision with kernels that are actually faster, not just smaller. See [quantization.md](../01-llm-internals/quantization.md). |
| **Streaming** | Token-by-token delivery over SSE or websockets, with cancellation that actually stops the work. |
| **Multi-GPU sharding** | Splitting a model that doesn't fit on one card. [Its own file.](multi-gpu.md) |

Named implementations change faster than this repo could track, and the specific names
are the least interesting part of the answer. What matters in an interview is that you can
say *what the layer is for* and *which of these features your workload actually needs*.

---

## The one number that separates serving stacks

Throughput per GPU at your latency target. Not peak throughput, and not latency at batch
size 1 — both are easy to win and neither is what you pay for.

```python
def output_tokens_per_gpu_hour(tokens_per_second):
    return tokens_per_second * 3600

def cost_per_million_output_tokens(tokens_per_second, gpu_hourly=2.00):
    return gpu_hourly / (output_tokens_per_gpu_hour(tokens_per_second) / 1e6)

cost_per_million_output_tokens(2_000)   # ≈ $0.28 per 1M output tokens
cost_per_million_output_tokens(400)     # ≈ $1.39 per 1M output tokens
```

At 2,000 output tokens/sec per GPU and $2/GPU-hour, output costs about **$0.28 per
million tokens**. At 400 tokens/sec — a plausible number for a poorly batched
deployment — the same model on the same hardware costs about **$1.39 per million**, which
is no longer meaningfully below the API price it was supposed to beat.

**Same weights, same GPU, same output. Five times the cost.** That difference is the
serving stack, and it is why "self-hosting is cheaper" is a claim about your serving
efficiency rather than about hardware.

---

## When you don't need one

A serving stack is infrastructure, and infrastructure has an owner and a pager. Skip it
when:

- **You're calling an API.** The provider runs one. This entire section is about the case
  where you've decided not to. See
  [api-vs-open-model.md](../02-model-selection/api-vs-open-model.md) for whether you
  should have.
- **Throughput is genuinely tiny.** A batch job of a few thousand documents per day, with
  no latency requirement, runs fine on a naive loop overnight. Efficiency only matters
  when you're buying capacity to serve it.
- **The model is small enough that CPU works.** An embedding model or a small classifier
  may not need a GPU at all, and a GPU-shaped serving stack around a CPU-shaped model is
  pure operational cost.

The awkward middle — enough traffic that efficiency matters, not enough that anyone owns
the infrastructure — is where most self-hosting projects get into trouble. That's a
staffing observation, not a technical one, and it belongs in your answer.

---

## Deciding what you need

Work down this list and stop when the answer is "no":

1. **Does the model fit on one GPU, weights plus KV cache?** If yes, skip
   [multi-gpu.md](multi-gpu.md) entirely — sharding a model that fits costs you
   throughput and adds a failure mode.
2. **Do requests share a long prefix?** If yes, prefix caching is the highest-value
   feature and you should measure your hit rate.
3. **Is traffic spiky?** If yes, read [autoscaling.md](autoscaling.md) before choosing
   anything — cold-start time constrains your architecture more than throughput does.
4. **Is the workload latency-sensitive or throughput-sensitive?** Batch jobs and chat
   want opposite scheduler settings, and running both through one deployment means one of
   them loses. Separate deployments are usually the right answer.

---

## What to monitor

- **Output tokens/sec per GPU**, which is your unit cost expressed in engineering terms.
- **Running batch size** — the average number of sequences generating concurrently. If
  it's near 1 under load, batching isn't working and nothing else matters.
- **Queue depth and time-in-queue**, separately from execution latency. A p95 that looks
  fine while requests wait 4 seconds to be admitted is measuring the wrong thing.
- **KV cache utilization**, and the **preemption/recompute rate** that follows from it.
- **Prefix cache hit rate**, if you rely on it. It drops silently — one timestamp in the
  wrong place is enough.

---

## Interview questions

**"Why is your self-hosted deployment more expensive per token than the API you left?"**

Because the API provider's serving efficiency is better than yours. Ask for output
tokens/sec per GPU and the average running batch size. A batch size near 1 means requests
aren't overlapping — check whether the scheduler is admitting them, whether the KV cache
has room, and whether one long-context request is monopolizing memory. The unit economics
of self-hosting are a serving-efficiency question, not a hardware question.

**"You have 40% headroom on GPU compute but latency is bad. What's happening?"**

Compute utilization is close to meaningless during decode, which is memory-bandwidth-bound
— you can be at 100% of achievable throughput while a compute-utilization dashboard reads
40%. Look at memory bandwidth utilization, running batch size, and time-in-queue instead.
If time-in-queue dominates, you have an admission problem, not a speed problem.

**"When would you not use a serving stack?"**

When you're calling an API, when throughput is small enough that efficiency doesn't pay
for the operational cost, or when the model doesn't need a GPU. Also worth saying out
loud: when nobody will own it. A serving deployment with no owner degrades quietly, and
the first sign is usually the bill.

---

## What to remember

- The serving stack is the difference between paying for the GPU and using it. Roughly a
  5× swing in cost per token on identical hardware.
- Continuous batching, paged KV cache and prefix caching are the three features that
  produce most of that difference.
- Throughput per GPU *at your latency target* is the number that matters. Peak throughput
  and single-request latency are both easy to win and neither one is billed.
- Compute utilization is a misleading signal during decode. Batch size, queue depth and
  memory bandwidth tell you the truth.
