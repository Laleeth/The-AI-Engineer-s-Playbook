# Positional Encoding

Attention has no built-in sense of order. Without positional information, "the dog bit the
man" and "the man bit the dog" look identical to it — same tokens, same set.

Positional encoding is how order gets in. The operational reason to care: **it's what
decides whether a model works beyond the context length it was trained on.**

---

## Why attention needs help

Look at the attention computation again:

```
softmax(Q·Kᵀ / √d) · V
```

Every token is compared with every other token. Nothing in that operation knows which came
first. Shuffle the input and you get the same set of comparisons.

So position has to be injected somewhere.

---

## Approaches, briefly

### Learned positional embeddings

Give each position its own learned vector and add it to the token embedding.

```python
def add_learned_positions(token_embeddings, position_embeddings):
    seq_len = token_embeddings.shape[0]
    return token_embeddings + position_embeddings[:seq_len]
```

Simple, works well. **Hard limit:** if you trained positions up to 2,048, position 2,049
has no embedding. The model cannot process longer inputs at all.

### Sinusoidal encoding

Fixed sine and cosine waves at different frequencies. No parameters to learn, and
mathematically defined for any position — so it extends beyond training length in
principle, though quality degrades.

### Rotary Position Embedding (RoPE)

The common approach in modern models, and the one worth understanding because its
properties explain a lot of behavior you'll encounter.

Instead of *adding* position information, RoPE **rotates** the query and key vectors by an
angle proportional to their position.

```python
import numpy as np

def rope(x, positions, base=10000.0):
    """Rotate pairs of dimensions by an angle proportional to position."""
    d = x.shape[-1]
    # Different dimension pairs rotate at different frequencies
    freqs = 1.0 / (base ** (np.arange(0, d, 2) / d))
    angles = positions[:, None] * freqs[None, :]

    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]

    rotated = np.empty_like(x)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    return rotated
```

**The property that makes it work:** when you take the dot product of a rotated query at
position *i* and a rotated key at position *j*, the result depends on `i − j` — the
*relative* distance — not on the absolute positions.

That means the model learns "how far apart are these?" rather than "where exactly is
this?", which generalizes much better.

---

## Why this matters operationally

### Context length is not a free parameter

A model trained on 8,000-token contexts doesn't automatically work at 32,000. The
positional patterns at position 30,000 are outside anything it saw during training, and
quality degrades — often badly, and often without any error.

This is why "extended context" versions of models exist as separate releases. Extending
context is real work, not a config change.

**Practical consequence:** if a provider offers a longer-context variant, don't assume
quality is uniform across that window. Test at the lengths you'll actually use. A model
advertised at 128k tokens may perform substantially worse at 100k than at 8k on your task.

### Context extension techniques exist, with trade-offs

RoPE can be stretched to longer contexts by adjusting the frequencies — scaling them so
positions are compressed into the range the model was trained on. Several variations of
this idea are in common use.

What you need to know:

- Extension usually costs some quality at short contexts while gaining long-context
  capability.
- Extended models often need some additional fine-tuning to recover.
- **An extended context window is a capability, not a guarantee.** Measure at length.

### The "lost in the middle" effect

Models attend better to the start and end of a context than the middle. Information buried
in the middle of a long prompt is measurably less likely to be used.

This has a direct RAG consequence: putting 20 retrieved chunks in a prompt doesn't mean
all 20 are read. It's part of why sending 5 good chunks beats 20 mediocre ones — see
[../03-rag/reranking.md](../03-rag/reranking.md).

If you must send many chunks, consider ordering by relevance with the best at the
beginning *and* end, rather than in a straight ranked list.

```python
def order_for_attention(chunks):
    """Put the strongest chunks where the model attends best: the edges."""
    ranked = sorted(chunks, key=lambda c: -c.score)
    front, back = [], []
    for i, chunk in enumerate(ranked):
        (front if i % 2 == 0 else back).append(chunk)
    return front + list(reversed(back))
```

Whether this helps is model-dependent — test it rather than assuming. But it costs nothing
to try, and the underlying effect is real.

---

## Positions and the KV cache

One detail that matters for serving: **position is baked into the cached keys.**

With RoPE, the key vectors stored in the KV cache have already been rotated by their
position. That means a cached prefix is only valid at the position it was cached at.

Consequence for [prefix caching](kv-cache.md): the cache works on a *prefix*, and it only
works if the prefix starts at position 0 and is byte-identical. Change one token near the
beginning and every subsequent position shifts, so nothing after it can be reused.

This is the mechanism behind a common production incident: someone edits the top of a
system prompt, the prefix cache invalidates for every request, and latency and cost jump
several-fold with no code deploy. The fix is structural — put stable content first,
volatile content last:

```
[system prompt]        ← never changes     ┐
[tool definitions]     ← rarely changes    │  cacheable prefix
[policy text]          ← rarely changes    ┘
[retrieved documents]  ← varies per request
[conversation history] ← varies
[user message]         ← varies
```

Put a timestamp or request ID at the top and you have destroyed your cache.

---

## Interview questions

**1. "Why does a transformer need positional encoding?"**

Attention is permutation-invariant — it compares every token with every other and has no
notion of order. Position has to be injected explicitly.

**2. "What's RoPE and why is it used?"**

It rotates queries and keys by an angle proportional to position, so their dot product
depends on relative distance rather than absolute position. That generalizes better than
learned absolute positions and extends more gracefully beyond training length.

**3. "Can I use a model beyond its trained context length?"**

Not safely by default — positions beyond training are out of distribution and quality
degrades, often silently. Extension techniques exist but usually trade some short-context
quality and need fine-tuning. Measure at the lengths you'll actually use.

**4. "Why did editing our system prompt triple our latency?"**

Prefix caching works on an exact prefix, and positions are baked into cached keys. Editing
near the top shifts every subsequent position and invalidates the whole cache. Fix by
ordering the prompt stable-first, and by treating prompt edits as deploys with a canary.

**5. "Your RAG system ignores information in the middle of long contexts. What do you
do?"**

That's the lost-in-the-middle effect. Reduce the number of chunks via reranking rather than
adding more, and consider ordering the strongest chunks at the edges. The general principle
is that a bigger context window doesn't fix retrieval, it hides the problem.

---

## What to remember

- Attention has no sense of order; position is injected explicitly.
- RoPE rotates queries and keys so attention depends on *relative* distance.
- A model's context limit is a real capability boundary, not a config value — quality
  degrades beyond training length, often silently.
- Test long-context models at the lengths you'll actually use.
- Models attend worse to the middle of long contexts. Fewer, better chunks beats more.
- Position is baked into cached keys, so prefix caching needs an exact, stable prefix.
- Put stable content first in your prompts, volatile content last.

---

**Next:** [kv-cache.md](kv-cache.md) — the memory structure that decides your GPU count.
