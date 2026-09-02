# KV Cache

The KV cache is the single most important internal detail for anyone running inference.
It decides how many requests fit on a GPU, which decides your throughput, which decides
your GPU count, which decides your bill.

If you learn one thing from this section, learn this file.

---

## Why it exists

When generating text, the model produces one token at a time. Each new token attends to
every previous token — which means it needs their key and value vectors.

Without caching, generating token 500 means recomputing K and V for tokens 1–499. Then
token 501 recomputes 1–500. You'd redo the same work over and over, and generation would
be quadratic in output length.

So you cache them:

```python
def generate_with_cache(model, prompt_tokens, max_new_tokens):
    # Prefill: process the whole prompt at once, keep K and V
    logits, cache = model.forward(prompt_tokens, cache=None)
    tokens = [sample(logits[-1])]

    # Decode: one token at a time, reusing the cache
    for _ in range(max_new_tokens - 1):
        logits, cache = model.forward(tokens[-1:], cache=cache)
        tokens.append(sample(logits[-1]))
        if tokens[-1] == EOS:
            break

    return tokens
```

That's the whole idea. Now the expensive part.

---

## How big it gets

```python
def kv_cache_bytes(seq_len, layers, kv_heads, head_dim, bytes_per_number=2):
    """2x because we store both K and V."""
    return 2 * seq_len * layers * kv_heads * head_dim * bytes_per_number
```

A worked example — a mid-sized model with grouped-query attention:

```python
# 32 layers, 8 KV heads, head_dim 128, fp16 (2 bytes)
per_token = kv_cache_bytes(1, 32, 8, 128)
# = 2 * 1 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KB per token
```

**128 KB per token.** Now scale it:

| Context length | KV cache per request |
|---|---|
| 1,000 tokens | 128 MB |
| 4,000 tokens | 512 MB |
| 32,000 tokens | 4 GB |
| 128,000 tokens | 16 GB |

One request with a 128k context can consume 16 GB. On an 80 GB GPU that's a fifth of the
card for a single conversation.

---

## The number that decides everything

How many requests fit on one GPU:

```python
def max_concurrent_requests(gpu_memory_gb, model_size_gb, avg_context_tokens,
                            layers, kv_heads, head_dim, bytes_per_number=2):
    available = (gpu_memory_gb - model_size_gb) * 1e9
    per_request = kv_cache_bytes(avg_context_tokens, layers, kv_heads,
                                 head_dim, bytes_per_number)
    return int(available / per_request)


# 80 GB GPU, 26 GB of weights, 2,500-token average context
max_concurrent_requests(80, 26, 2500, layers=32, kv_heads=8, head_dim=128)
# ≈ 168 concurrent requests
```

That number is your **batch size ceiling**. And batch size is what determines throughput,
because GPUs are far more efficient processing many sequences together than one at a time.

So the chain is:

```
KV cache size  →  concurrency  →  batch size  →  throughput  →  GPUs needed  →  cost
```

This is why the attention variant matters so much. From
[attention.md](attention.md): a model using MHA instead of GQA has a 4× larger cache,
which means roughly a quarter of the concurrency, which means roughly four times the GPUs
for the same throughput.

**When evaluating models for self-hosting, check the KV head count. It can matter more to
your costs than the parameter count.**

---

## What goes wrong

### Memory pressure and preemption

When the cache fills, the serving stack has to do something. Usually it **preempts** —
evicts a sequence and recomputes it later.

This doesn't degrade gracefully. Throughput collapses non-linearly as the system spends
its time recomputing rather than generating. You see latency spike while GPU utilisation
looks high.

Prevent it rather than handling it:

- Cap maximum context length at admission, not inside the model server
- Cap concurrency below the theoretical maximum
- Monitor cache utilisation and preemption rate as first-class metrics

### Long sequences starve short ones

One 128k-token request occupies the memory of 50 short ones. Mixing them in the same pool
means a handful of long requests can push everything else out.

Fix: separate pools or admission limits by length. Bucketing by sequence length also
reduces padding waste in batching.

### Fragmentation

Naive allocation reserves contiguous memory for the maximum possible length of each
sequence. Most sequences finish much earlier, so most of that reservation is wasted.

**Paged attention** solves this by allocating the cache in fixed-size blocks, like virtual
memory pages, so sequences use only what they need. This typically increases usable
concurrency substantially. Any serious serving stack does this — if yours doesn't, that's
your first fix.

---

## Prefix caching

The optimisation with the biggest cost impact for most applications.

If many requests share a prefix — a system prompt, tool definitions, policy text — the K
and V vectors for that prefix are identical every time. Compute them once, reuse them.

```
request 1:  [system prompt 1200 tokens][user query A]
request 2:  [system prompt 1200 tokens][user query B]
                     ↑ identical — compute once
```

The savings are large because that prefix is often most of your input tokens, and input
tokens are usually most of your bill.

**Two rules, and violating either kills it:**

**1. The prefix must be exactly identical.** Byte-for-byte. One different character and
nothing after it can be reused (positions shift — see
[positional-encoding.md](positional-encoding.md)).

**2. Stable content must come first.**

```
[system prompt]        ← never changes      ┐
[tool definitions]     ← rarely changes     │  cacheable
[policy text]          ← rarely changes     ┘
[retrieved documents]  ← varies
[conversation history] ← varies
[user message]         ← varies
```

Put a timestamp, request ID, or user name at the top and you have no cache at all.

### The incident this causes

This is a real and recurring production failure:

> Someone edits a word in the system prompt through a management UI. Every request's
> prefix hash changes. Cache hit rate drops from 87% to 4%. Latency triples, cost triples.
> No code was deployed, so nobody looks at deploys.

Two defences:

**Monitor cache hit rate as a first-class SLI**, and alert on a drop. It moves immediately
and points straight at the cause, where a latency alert only tells you something is wrong.

**Treat prompt edits as deploys** — review, canary, rollback. A UI that lets someone change
production behaviour with no gate is the actual root cause; the cache is just the
mechanism.

---

## Quantising the cache

The cache can be stored at lower precision than the model weights. FP8 KV cache is
production-ready for many deployments and roughly halves cache memory, which roughly
doubles concurrency.

As of the [vLLM FP8 KV-cache analysis (April 2026)](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html),
using the `e4m3` format:

- KV cache memory reduced ~50% versus BF16
- Long-context evaluations recover 94–98% of baseline at 1M tokens
- Reasoning benchmarks show 1–2 points of degradation at most
- Throughput gains of roughly 5–15% under realistic load
- Break-even around **7,000 tokens** of context

Their practical guidance: enable it for contexts above ~7k tokens and decode-heavy
memory-bound workloads; skip it for short contexts, and for models where uncalibrated
accuracy drops below 95%.

The general principle holds regardless of format specifics: **measure the recall or
accuracy cost on your own hard cases, not on the average.** Quantisation tends to hurt most
where things were already marginal.

---

## What to monitor

```python
KV_CACHE_METRICS = [
    "kv_cache_utilization",      # approaching 1.0 means preemption is coming
    "preemption_rate",           # should be ~0; non-zero means you're over capacity
    "prefix_cache_hit_rate",     # a drop is a cost and latency incident
    "avg_sequence_length",       # creeping up = capacity shrinking
    "max_sequence_length",       # one huge request can starve everything
    "batch_size",                # the throughput driver
]
```

The first two tell you whether you're about to fall off a cliff. The third catches the
prompt-edit incident. All three are cheap and most teams have none of them.

---

## Interview questions

**1. "What is the KV cache and why does it matter?"**

It stores key and value vectors for previous tokens so generation doesn't recompute them.
It matters because its size limits concurrency, which limits batch size, which determines
throughput and therefore GPU count. Give the chain — that's the answer that shows you've
sized a fleet.

**2. "Compute the KV cache for a 32-layer model, 8 KV heads, head_dim 128, at 4,000
tokens."**

`2 × 4000 × 32 × 8 × 128 × 2 bytes` = 512 MB. Be willing to do it out loud; the
willingness matters more than the precision.

**3. "Your throughput collapsed but GPU utilisation is high. What's happening?"**

Likely KV cache exhaustion causing preemption — the GPU is busy recomputing evicted
sequences rather than generating. Check cache utilisation and preemption rate. Fix by
capping context length and concurrency at admission.

**4. "How do you double your serving capacity without buying GPUs?"**

Several: prefix caching if you have shared prompts, FP8 KV cache (roughly halves cache
memory), paged attention if you're not using it, shorter contexts, or a model with fewer
KV heads. Note that some are free and some cost accuracy.

**5. "Why did editing a system prompt triple our costs with no deploy?"**

Prefix cache invalidation. The cached prefix must be byte-identical, so an edit near the
top invalidates everything. Then say what you'd monitor and how you'd gate prompt changes.

**6. "One customer sends 100k-token requests. What happens to everyone else?"**

Each of those consumes the cache space of dozens of normal requests, so concurrency drops
and everyone's latency rises. Fix with per-length admission limits or separate pools.

---

## What to remember

- KV cache size → concurrency → batch size → throughput → GPU count → cost. Know the
  chain.
- `2 × seq_len × layers × kv_heads × head_dim × bytes`. Be able to compute it.
- GQA versus MHA can change cache size several-fold, which changes your GPU count.
- Cache exhaustion causes preemption, and throughput collapses non-linearly.
- Cap context length and concurrency at admission, not in the model server.
- Prefix caching needs a byte-identical prefix. Stable content first, volatile last.
- Monitor prefix cache hit rate — a drop is a cost incident with no deploy.
- FP8 KV cache roughly halves memory; measure the accuracy cost on hard cases.

---

**Next:** [prefill-vs-decode.md](prefill-vs-decode.md) — why input drives cost and output
drives latency.

---

Sources:
- [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)
