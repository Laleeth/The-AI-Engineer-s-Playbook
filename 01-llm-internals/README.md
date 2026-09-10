# 01 — LLM Internals

How models work underneath, with a strict filter: **only the parts that change what you
build.**

This is not a deep-learning course. There's no backpropagation, no training theory, no
derivations. Every file exists because something in it decides an operational number —
your GPU count, your bill, your latency, your context budget.

## The files

| File | The operational consequence |
|---|---|
| [attention.md](attention.md) | Why long context costs more, and why GQA vs. MHA changes your GPU count |
| [tokenization.md](tokenization.md) | Why non-English users cost 2–3× more and get less context |
| [positional-encoding.md](positional-encoding.md) | Why context limits are real, and why editing a prompt's top breaks caching |
| [kv-cache.md](kv-cache.md) | **The one that decides your infrastructure.** Read this one. |
| [prefill-vs-decode.md](prefill-vs-decode.md) | Why input drives cost and output drives latency |
| [quantization.md](quantization.md) | Fitting bigger models, and buying concurrency |
| [speculative-decoding.md](speculative-decoding.md) | Free latency at low batch sizes, worse throughput at high ones |

## If you read one file

[kv-cache.md](kv-cache.md). It contains the chain that connects an architecture detail to
your budget:

```
KV cache size → concurrency → batch size → throughput → GPUs → cost
```

Being able to walk that chain, with arithmetic, is what separates someone who has sized an
inference fleet from someone who has read about one.

## The short version

**Attention is quadratic in sequence length** — but the linear costs often dominate at
typical lengths, so measure rather than assuming double context means quadruple cost.

**GQA vs. MHA changes KV cache size several-fold.** When evaluating a model for
self-hosting, check the KV head count. It can matter more to your costs than the parameter
count.

**Tokenization is not uniform across languages.** Non-Latin scripts fragment into far more
tokens per character, so those users cost more *and* — under a global token budget — get
structurally less context. That looks like a model quality problem and is a budgeting
problem.

**Prefill is parallel and compute-bound. Decode is sequential and
memory-bandwidth-bound**, hundreds of times slower per token. Hence: input tokens drive cost,
output tokens drive latency. Optimizing the wrong one is the most common wasted week.

**Decode needs more GPUs than prefill despite processing fewer tokens.** This inversion
surprises people and it's the core intuition in capacity planning.

**Prefix caching needs a byte-identical prefix.** Positions are baked into cached keys, so
editing the top of a system prompt invalidates everything after it. This is a real
recurring incident: cost and latency triple with no deploy.

**Models attend worse to the middle of long contexts.** Which is why sending 5 good
retrieved chunks beats 20 mediocre ones — and why a bigger context window doesn't fix
retrieval, it hides the problem.

**Speculative decoding produces identical output.** It's pure latency gain, not a quality
trade-off — but it helps at low batch sizes and can hurt at high ones.

## Arithmetic worth being able to do out loud

```python
# KV cache per token
2 * layers * kv_heads * head_dim * bytes_per_number

# How many requests fit on a GPU
(gpu_memory - model_weights) / (kv_cache_per_token * avg_context)

# Model size at a given precision
params * bits / 8

# Attention cost ratio between two context lengths
(len_b / len_a) ** 2

# Vector index size
n_vectors * dimensions * bytes_per_number
```

Interviewers notice when you're willing to compute rather than guess. The precision
matters less than the willingness.

## Related

- [02-model-selection/cost-quality-latency.md](../02-model-selection/cost-quality-latency.md)
  — applies the prefill/decode split to real cost decisions
- [02-model-selection/api-vs-open-model.md](../02-model-selection/api-vs-open-model.md) —
  the self-hosting arithmetic these numbers feed
- [12-senior-scenarios/scaling.md](../12-senior-scenarios/scaling.md) — capacity planning
  as interview scenarios
- [12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md)
  — the prefix-cache incident, worked through
