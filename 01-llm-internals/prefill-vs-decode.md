# Prefill vs. Decode

Generating a response has two phases with completely different performance
characteristics. They're bottlenecked by different hardware resources, they scale
differently, and they compete for the same GPU.

This is the file that explains a claim made throughout this repo: **input tokens drive
cost, output tokens drive latency.**

---

## The two phases

### Prefill — reading the prompt

The model processes your entire prompt at once. All tokens go through together, in
parallel.

```
prompt: [The][ quick][ brown][ fox][ jumps]
         ↓     ↓       ↓       ↓      ↓
        all processed in one pass
         ↓
        first output token
```

**Parallel.** **Compute-bound.** The GPU is doing large matrix multiplications and its
arithmetic units are the limit.

Time is roughly linear in prompt length: a 2,000-token prompt takes about twice as long as
1,000 tokens.

### Decode — writing the response

One token at a time. Each token depends on the one before it, so there's no way to
parallelize within a request.

```
                    → [The] → [fox] → [jumped] → [over] → ...
one at a time, each needs the previous
```

**Sequential.** **Memory-bandwidth-bound.** For each token, the GPU must read the entire
model's weights from memory to do a comparatively small amount of arithmetic. The
bottleneck is moving weights, not computing with them.

---

## Why this asymmetry matters

Some illustrative numbers to show the shape:

```python
def phase_timings(input_tokens, output_tokens,
                  prefill_tokens_per_sec=20_000,
                  decode_tokens_per_sec=50):
    """decode rate is PER SEQUENCE — it falls as batch size grows."""
    prefill_s = input_tokens / prefill_tokens_per_sec
    decode_s = output_tokens / decode_tokens_per_sec
    return {
        "prefill_ms": round(prefill_s * 1000),
        "decode_ms": round(decode_s * 1000),
        "decode_share": round(decode_s / (prefill_s + decode_s), 2),
    }

phase_timings(2000, 500)
# {'prefill_ms': 100, 'decode_ms': 10000, 'decode_share': 0.99}
```

Reading 2,000 tokens: ~100ms. Writing 500 tokens: ~10 seconds.

Per token that's 0.05ms to read versus 20ms to write — **decode is roughly 400× slower per
token** with these figures. The exact ratio depends on the model and batch size, but the
order of magnitude is the point: reading is nearly free, writing is not.

### So why does input dominate cost?

Because there's usually far more of it, and prefill is efficient.

```python
def cost_split(input_tokens, output_tokens, price_in, price_out):
    cost_in = input_tokens / 1e6 * price_in
    cost_out = output_tokens / 1e6 * price_out
    total = cost_in + cost_out
    return {"input_share": round(cost_in / total, 2),
            "output_share": round(cost_out / total, 2)}

cost_split(3000, 300, price_in=0.50, price_out=1.50)
# {'input_share': 0.77, 'output_share': 0.23}
```

A typical RAG request sends 3,000 tokens of retrieved context and produces a 300-token
answer. Even though output is priced 3× higher per token, input is 77% of the cost because
there's 10× more of it.

**The two rules that follow:**

> **Bill too high?** Look at input tokens first — prompt caching, reranking to send fewer
> chunks, trimming the system prompt.
>
> **Too slow?** Look at output tokens first — shorter responses, streaming, structured
> output.

Optimizing the wrong one is the most common wasted week in this work.

---

## Time to first token vs. total time

Two latency metrics, and conflating them causes a lot of unnecessary engineering.

```
request ──[prefill]──> first token ──[decode]──────> last token
          ~100ms                     ~10s

          └── TTFT ──┘
          └────────── total completion time ────────┘
```

**Time to first token (TTFT)** is dominated by prefill. It's what a user perceives as
responsiveness.

**Total completion time** is dominated by decode. It's what matters for batch jobs and for
anything a machine consumes.

If your product **streams**, TTFT is the number that matters. Users read while the rest
generates, and a 10-second total feels fine.

If it doesn't stream, users stare at a spinner for the full 10 seconds and it feels broken.

**This is why "our p95 latency is 5 seconds and the requirement is 2" often has a free
answer.** Ask whether it streams. Ask which metric the requirement refers to. Very often
the problem dissolves without any capacity change — see
[../02-model-selection/cost-quality-latency.md](../02-model-selection/cost-quality-latency.md).

---

## They compete for the same GPU

Here's where it gets operationally awkward.

Prefill is compute-heavy and bursty. Decode is memory-bandwidth-heavy and steady. On a
shared GPU they interfere:

**Head-of-line blocking.** A request arrives with a 30,000-token prompt. Prefilling it
monopolises the GPU. Every request currently decoding stalls, and their per-token latency
spikes.

The standard fix is **chunked prefill** — process long prompts in pieces, interleaving
decode steps between chunks. Latency for the long request rises slightly; latency for
everyone else stops depending on it.

```
without chunked prefill:
  [────── 30k prefill ──────][decode][decode][decode]
   everything else waits

with chunked prefill:
  [prefill][decode][prefill][decode][prefill][decode]
   long prompt makes progress, others keep flowing
```

If your p99 latency is much worse than your p50 and you have variable prompt lengths, this
is the first thing to check.

**Some deployments separate the phases entirely**, running prefill and decode on different
machines, so each can be sized and scheduled for its own bottleneck. It adds complexity —
the KV cache has to move between them — but it removes the interference and lets you scale
the two independently. Worth knowing exists; not something to reach for early.

---

## Batching helps decode much more than prefill

Because decode is memory-bandwidth-bound, the GPU reads the model weights once and can
apply them to many sequences at once. Going from 1 to 32 concurrent sequences barely
increases the time per step.

```python
def decode_throughput(batch_size, per_step_ms=20):
    """Per-step time is roughly flat as batch grows; throughput scales."""
    return {
        "tokens_per_sec_total": batch_size * 1000 / per_step_ms,
        "tokens_per_sec_per_request": 1000 / per_step_ms,
    }

decode_throughput(1)    # 50 total, 50 per request
decode_throughput(32)   # 1600 total, 50 per request
```

That's an enormous throughput gain for free — which is why **continuous batching** (adding
and removing sequences from the batch as they arrive and finish, rather than waiting for a
fixed batch) is the single most important serving feature.

The catch: batch size is limited by KV cache memory (see [kv-cache.md](kv-cache.md)). So
the chain runs:

```
KV cache size → batch size → decode throughput → GPUs needed
```

Prefill benefits far less from batching, because it's already using the GPU's arithmetic
units efficiently with a single long sequence.

---

## Sizing a deployment

Put it together. Given a target load, work out both phases separately:

```python
def capacity_estimate(requests_per_sec, input_tokens, output_tokens,
                      prefill_per_gpu=20_000, decode_per_gpu=2_000):
    """decode_per_gpu is AGGREGATE across the batch, not per sequence."""
    prefill_load = requests_per_sec * input_tokens
    decode_load = requests_per_sec * output_tokens

    return {
        "prefill_tokens_per_sec": prefill_load,
        "decode_tokens_per_sec": decode_load,
        "gpus_for_prefill": prefill_load / prefill_per_gpu,
        "gpus_for_decode": decode_load / decode_per_gpu,
    }

capacity_estimate(500, 2000, 500)
# prefill: 1,000,000 tokens/s → ~50 GPUs
# decode:    250,000 tokens/s → ~125 GPUs
```

**Note the inversion.** Prefill handles 4× more tokens but needs fewer GPUs, because it's
far more efficient per token. Decode dominates the GPU count.

That inversion is the single most useful intuition in inference capacity planning, and
it's the thing candidates most often get backwards.

---

## Optimizing each phase

**Prefill (cost, TTFT):**
- Prompt caching — the biggest lever if you have a shared prefix
- Reranking to send fewer retrieved chunks
- Trimming accumulated system-prompt rules
- Scoping context to the task

**Decode (latency, throughput):**
- Shorter outputs — structured output, concise instructions, sensible `max_tokens`
- Continuous batching
- Larger batches (bounded by KV cache)
- Quantisation — smaller weights mean less memory traffic per token
- [Speculative decoding](speculative-decoding.md) — generate several tokens per pass

A warning on `max_tokens`: setting it too low causes truncation, which fails validation
downstream, which triggers retries — and retries re-send the whole input, which is where
your cost is. There's a documented case of someone lowering `max_tokens` to save money and
tripling the bill. Set it from the actual distribution of output lengths.

---

## Interview questions

**1. "Explain prefill and decode."**

Prefill processes the prompt in parallel and is compute-bound; decode generates one token
at a time and is memory-bandwidth-bound. Then give the consequence: input drives cost,
output drives latency.

**2. "Your p95 latency is 5 seconds, the requirement is 2. What do you do?"**

First ask whether that's TTFT or total, and whether the product streams. That often ends
the problem. If not: shorter outputs, prompt caching to cut prefill, smaller model for the
fast path.

**3. "You need 500 requests/second at 2,000 in and 500 out. What's the bottleneck?"**

Do the arithmetic: 1M prefill tokens/s and 250k decode tokens/s. Decode needs more GPUs
despite fewer tokens, because it's memory-bandwidth-bound. Then check whether the latency
requirement is even achievable — 500 sequential tokens can't be generated in 2 seconds at
typical per-sequence rates.

**4. "Your p99 is 8× your p50 with variable prompt lengths. Why?"**

Likely head-of-line blocking from long prefills stalling in-flight decodes. Fix with
chunked prefill, and consider bucketing by sequence length.

**5. "Why does batching help so much?"**

Decode is memory-bandwidth-bound — the GPU reads the weights once and applies them across
the whole batch, so per-step time is roughly flat while throughput scales. Batch size is
capped by KV cache memory.

**6. "Cost is too high. Where do you look first?"**

Input tokens, because they're usually 70–80% of the bill. Prompt caching and sending fewer
retrieved chunks. Not "use a smaller model," which is the riskiest lever and usually the
first one people reach for.

---

## What to remember

- Prefill: parallel, compute-bound, fast per token. Decode: sequential,
  memory-bandwidth-bound, hundreds of times slower per token.
- **Input tokens drive cost. Output tokens drive latency.** Check which you're fixing.
- TTFT is dominated by prefill; total time by decode. Ask which the requirement means.
- Streaming changes the metric that matters and often dissolves a latency problem.
- Decode needs more GPUs than prefill despite fewer tokens. This inversion surprises
  people.
- Chunked prefill prevents long prompts stalling everyone's decode.
- Continuous batching is the most important serving feature; KV cache caps it.
- Setting `max_tokens` too low costs more, not less.

---

**Next:** [quantization.md](quantization.md) — making the model smaller.
