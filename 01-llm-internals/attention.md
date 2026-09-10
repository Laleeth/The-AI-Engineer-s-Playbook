# Attention

Attention is how a model decides which parts of the input matter when producing each
token. It's the mechanism behind most of what you care about operationally: why long
contexts are expensive, why memory limits your concurrency, and why some architectures
serve far more cheaply than others.

This file covers what it does and — more importantly for this repo — **what it costs
you**.

---

## What it does

For each token, the model asks: *which other tokens should I pay attention to?*

Three vectors per token:

- **Query (Q)** — what this token is looking for
- **Key (K)** — what this token offers
- **Value (V)** — what this token contributes if attended to

You compare each query against every key, turn the comparison into weights, and take a
weighted sum of the values.

```
attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

Reading it left to right:

- `Q·Kᵀ` — how well does each query match each key? Gives an n×n matrix for n tokens.
- `/ √d` — divide by the square root of the head dimension, so the values don't get large
  enough to make softmax saturate.
- `softmax` — turn scores into weights that sum to 1.
- `· V` — take the weighted average of the values.

```python
import numpy as np

def attention(Q, K, V, mask=None):
    """Q, K, V: (seq_len, head_dim)"""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)          # (seq, seq)

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)   # can't see the future

    weights = softmax(scores, axis=-1)     # rows sum to 1
    return weights @ V                     # (seq, head_dim)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)        # stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

**Causal masking** is the `mask` argument. When generating text, token 5 must not see
tokens 6 onwards — otherwise the model would be reading the answer. The mask sets those
scores to negative infinity, which softmax turns into zero.

---

## The cost that matters

Here's the operational consequence, which is the reason this file exists.

`Q·Kᵀ` produces an **n×n** matrix for n tokens. That's quadratic:

| Context length | Score matrix entries | Relative |
|---|---|---|
| 1,000 | 1,000,000 | 1× |
| 4,000 | 16,000,000 | 16× |
| 32,000 | 1,024,000,000 | 1,024× |
| 128,000 | 16,384,000,000 | 16,384× |

Double the context, quadruple the attention compute.

```python
def attention_cost_ratio(len_a, len_b):
    """How much more attention compute for a longer context?"""
    return (len_b / len_a) ** 2

attention_cost_ratio(1_000, 4_000)     # 16.0
attention_cost_ratio(4_000, 128_000)   # 1024.0
```

**But — and this matters — quadratic attention is usually not what dominates your bill.**
For typical context lengths, the linear costs (the feed-forward layers, which scale with
sequence length not its square) are larger. Attention becomes dominant at long contexts.

The crossover depends on the model, but the practical point is: don't assume doubling your
context quadruples your cost. Measure. It's somewhere between 2× and 4× depending on where
you are on the curve.

---

## Multi-head attention

Rather than one attention operation, models run several in parallel — "heads" — each with
its own Q, K, V projections. Different heads learn different relationships: one might
track syntax, another long-range references.

```python
def multi_head(x, W_q, W_k, W_v, W_o, n_heads):
    seq, d_model = x.shape
    head_dim = d_model // n_heads

    Q = (x @ W_q).reshape(seq, n_heads, head_dim)
    K = (x @ W_k).reshape(seq, n_heads, head_dim)
    V = (x @ W_v).reshape(seq, n_heads, head_dim)

    heads = [attention(Q[:, h], K[:, h], V[:, h], causal_mask(seq))
             for h in range(n_heads)]

    return np.concatenate(heads, axis=-1) @ W_o
```

Total compute is roughly the same as one big head — you split the dimension rather than
multiplying the work. What multi-head buys you is representational, not computational.

---

## MHA, GQA, MQA — and why you should care

This is the part with the biggest operational consequence, and it's about **KV cache
memory**, not accuracy.

When generating, you cache the K and V vectors for every previous token so you don't
recompute them (see [kv-cache.md](kv-cache.md)). How many K and V heads you have decides
how big that cache is.

```
Multi-Head Attention (MHA)
  n query heads, n key heads, n value heads
  Best quality, biggest cache

Grouped-Query Attention (GQA)
  n query heads, g key/value heads (g < n), grouped
  Nearly MHA quality, much smaller cache        ← the common choice

Multi-Query Attention (MQA)
  n query heads, 1 key head, 1 value head
  Smallest cache, some quality cost
```

The memory difference is large:

```python
def kv_cache_bytes_per_token(layers, kv_heads, head_dim, bytes_per_number=2):
    """2x for both K and V."""
    return 2 * layers * kv_heads * head_dim * bytes_per_number

# A 32-layer model, head_dim 128, fp16:
mha = kv_cache_bytes_per_token(32, 32, 128)   # 32 KV heads
gqa = kv_cache_bytes_per_token(32,  8, 128)   #  8 KV heads
mqa = kv_cache_bytes_per_token(32,  1, 128)   #  1 KV head

# mha = 524,288 bytes/token  (512 KB)
# gqa = 131,072 bytes/token  (128 KB)   — 4x smaller
# mqa =  16,384 bytes/token  ( 16 KB)   — 32x smaller
```

**Why this decides your infrastructure:** KV cache memory limits how many requests you can
run concurrently. Concurrency limits your batch size. Batch size decides your throughput
per GPU. So the attention variant a model uses directly determines how many GPUs you need.

GQA is the common middle ground precisely because it captures most of the memory saving
with little quality loss. When you're evaluating models for self-hosting, **check the KV
head count** — it can matter more to your costs than the parameter count.

---

## Flash Attention

A change in *how* attention is computed, not what it computes. The results are the same.

The problem it solves: the naive implementation writes that n×n score matrix to GPU
memory, then reads it back. For long sequences that's a lot of memory traffic, and memory
bandwidth is the bottleneck on modern GPUs, not arithmetic.

Flash Attention computes attention in tiles that fit in fast on-chip memory, never
materializing the full matrix. Same output, substantially faster, and much less memory.

**What you need to know operationally:**

- It's mathematically equivalent — not an approximation. No quality trade-off.
- It's why long contexts became practical.
- Any serious serving stack uses it. If yours doesn't, that's the first thing to fix.
- It doesn't change the quadratic *compute*, just the memory traffic and the constant
  factor.

---

## Attention over long contexts

Two practical effects worth knowing, because they show up as product problems.

**Position matters.** Models attend better to the beginning and end of a context than the
middle. Information buried in the middle of a long prompt can be effectively invisible.

Consequence for RAG: putting 20 retrieved chunks in a prompt doesn't mean the model reads
all 20. This is part of why [reranking](../03-rag/reranking.md) and sending fewer, better
chunks often *improves* quality rather than trading it away.

**More context is not free quality.** Adding marginally-relevant text gives the model more
opportunities to attend to the wrong thing. There's a real optimum, and it's usually lower
than people assume — quality typically plateaus around 5–8 retrieved chunks.

Both of these are why "just use a model with a million-token context window" doesn't solve
retrieval. It makes the problem invisible rather than fixing it.

---

## Interview questions

**1. "Explain attention."**

Q, K, V; each token asks which others matter; weighted sum. Then — and this is what
separates a good answer — go to the operational consequence: it's quadratic in sequence
length, and the KV cache it requires is what limits your serving concurrency.

**2. "Why is long context expensive?"**

Attention is O(n²) in sequence length, and the KV cache grows linearly with it. Then add
the honest nuance: quadratic attention isn't always the dominant cost at typical lengths —
the linear terms often are — so measure rather than assuming 2× context means 4× cost.

**3. "What's GQA and why does it matter to you?"**

Fewer key/value heads than query heads. It shrinks the KV cache several-fold, which raises
how many requests fit on a GPU, which raises throughput, which lowers your GPU count. It's
an infrastructure decision disguised as an architecture detail.

**4. "Does Flash Attention change the output?"**

No — it's mathematically equivalent, just computed in tiles that avoid writing the full
score matrix to memory. It's a memory-bandwidth optimization, and it's why long contexts
became practical.

**5. "Your RAG answers got worse when you increased top-k from 5 to 20. Why?"**

More context isn't more quality. Models attend less well to the middle of long contexts,
and extra marginally-relevant chunks give the model more chances to latch onto the wrong
thing. Plus you're paying more for it.

---

## What to remember

- Attention: each token asks which others matter, and takes a weighted sum.
- It's quadratic in sequence length — but the linear costs often dominate at typical
  lengths, so measure.
- Multi-head splits the work rather than multiplying it.
- **GQA vs. MHA changes KV cache size several-fold**, which decides your serving
  concurrency and therefore your GPU count.
- Flash Attention is exact, not approximate — a memory optimization.
- Models attend worse to the middle of long contexts, which is why fewer better chunks
  beats more chunks.

---

**Next:** [tokenization.md](tokenization.md) — the layer below, and where a surprising
number of bugs live.
