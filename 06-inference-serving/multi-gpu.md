# Multi-GPU Serving

Sharding a model across GPUs is the answer to exactly one question: *the weights and their
KV cache don't fit on one card.* It is not a throughput technique. Applied to a model that
already fits, it makes things slower and adds a failure mode.

This file is mostly about knowing when you need it, because "just add GPUs" is a
reflex answer that a good interviewer will push on.

---

## First, does it fit?

```python
def fits_on_one_gpu(params_b, bits, gpu_gb, kv_bytes_per_token,
                    target_concurrency, avg_context):
    weights_gb = params_b * 1e9 * bits / 8 / 1e9
    kv_gb = target_concurrency * avg_context * kv_bytes_per_token / 1e9
    return {
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
        "total_gb": weights_gb + kv_gb,
        "fits": weights_gb + kv_gb < gpu_gb * 0.92,   # leave room for activations
    }

# A 70B model at BF16 on an 80 GB card
fits_on_one_gpu(70, 16, 80, 131_072, target_concurrency=64, avg_context=2_500)
# weights_gb 140.0 → doesn't fit before you even count the cache

# The same model at INT4
fits_on_one_gpu(70, 4, 80, 131_072, target_concurrency=64, avg_context=2_500)
# weights_gb 35.0, kv_gb ≈ 21.0, total ≈ 56.0 → fits, with room to serve
```

Two lessons in one calculation. A 70B model at BF16 needs 140 GB of weights and cannot
run on one 80 GB card at any batch size. The same model at INT4 needs 35 GB and leaves 45
GB for KV cache, which at 2,500-token contexts is a few hundred concurrent sequences.

**Quantization and sharding solve the same problem, and quantization is usually the
cheaper answer.** Before designing a multi-GPU deployment, check whether reduced precision
removes the need for it — and check what it costs in quality on your own evaluation set,
because that is the trade you're making.

The second lesson is that the KV cache term is not a rounding error. Weights fitting is
necessary, not sufficient; you need room to serve a useful batch, or you have an
expensive way to run at batch size 2.

---

## Tensor parallelism

Split every layer across GPUs. Each card holds a slice of each weight matrix, computes its
part, and the results are combined.

- **What it costs:** an all-reduce — a collective communication where every GPU exchanges
  and sums partial results — at *every layer*, on *every token*. For a 32-layer model
  generating 500 tokens that's 16,000 collective operations per request.
- **What it needs:** a fast interconnect between the GPUs. On a high-bandwidth link inside
  one node this is fine. Over ordinary networking between nodes it is not, and this is the
  hard rule of tensor parallelism: **keep it inside one machine.**
- **What it buys:** the model fits, and latency per token improves somewhat, because the
  per-layer work is divided.

Efficiency is sub-linear. Two GPUs in tensor parallel give noticeably less than twice the
throughput of one, because the communication is pure overhead. Expect to pay 10–30%
depending on model shape and interconnect, and measure it rather than trusting a number
from a blog post about different hardware.

---

## Pipeline parallelism

Split the model by layers instead. GPU 0 holds layers 1–8, GPU 1 holds 9–16, and so on. A
request flows through them in sequence.

- **What it costs:** a bubble. While GPU 0 works on a request, GPUs 1–3 have nothing to do
  unless other requests are in flight behind it. Keeping the pipeline full requires enough
  concurrent requests to fill every stage.
- **What it needs:** far less bandwidth than tensor parallelism — activations are passed
  once per stage boundary, not summed every layer. It tolerates ordinary networking, so it
  can cross machines.
- **What it buys:** the model fits, across nodes if necessary. It does not improve
  single-request latency at all; the request still passes through every layer in order.

The rule of thumb that follows: **tensor parallelism within a node, pipeline parallelism
across nodes**, and only reach for the second when a model genuinely doesn't fit in one
machine.

---

## What sharding costs you operationally

The throughput tax is the part people cite. The operational costs are the part that bites.

**Failure granularity gets worse.** With one GPU per replica, losing a GPU loses one
replica. With eight-way tensor parallelism, losing one GPU takes down the whole
eight-GPU replica — you've made your failure unit eight times more expensive, and the
mean time between failures across a fleet scales with GPU count.

**Scheduling gets harder.** The scheduler must place eight GPUs on one machine, together,
with the right interconnect. Partial availability is useless. In a shared cluster this
turns into queueing for whole nodes, and your effective capacity drops in a way that
doesn't show in a GPU count.

**Cold start gets slower.** More weight bytes to load, more processes to coordinate, and
the replica isn't serving until all of them are ready. See
[autoscaling.md](autoscaling.md) — this pushes a five-minute cold start toward ten.

**Scaling granularity gets coarser.** You add capacity eight GPUs at a time. At small
fleet sizes that's a large step, and it means you're usually over-provisioned by up to
one full replica.

---

## The decision, in order

1. **Does it fit at reduced precision?** Try INT8, then INT4. Measure quality on your
   evaluation set, not on a benchmark. If it fits and quality holds, stop here.
2. **Do you need this model?** A smaller model with good retrieval frequently beats a
   larger one with poor retrieval, at a fraction of the serving cost — see
   [small-vs-large-model.md](../02-model-selection/small-vs-large-model.md). This is the
   question people skip.
3. **Does it fit with tensor parallelism inside one node?** Use the smallest degree that
   fits — 2 before 4, 4 before 8. Every doubling costs communication overhead and failure
   granularity.
4. **Only then, pipeline across nodes.** And check the arithmetic against just calling an
   API for this model, because at this point the operational cost is substantial and the
   comparison has probably moved.

---

## What to monitor

- **Throughput per GPU**, not per replica. Per-replica numbers hide the sharding tax
  completely, which is how deployments end up over-sharded.
- **Collective communication time** as a share of step time. Rising share means the
  interconnect is the bottleneck, and adding GPUs will make it worse.
- **Pipeline bubble time**, if you're using pipeline parallelism. Low concurrency means an
  idle pipeline.
- **Replica failure rate**, and whether single-GPU faults are taking down whole replicas.
- **Placement failures and time-to-schedule** in a shared cluster. Whole-node requirements
  turn into invisible capacity loss.

---

## Interview questions

**"You need to serve a 70B model. Walk me through it."**

Start with memory: 140 GB at BF16 doesn't fit an 80 GB card, so either quantize or shard.
At INT4 it's 35 GB and fits with plenty of room for KV cache, so I'd try that first and
validate quality on our own eval set. If quality doesn't hold, tensor parallelism inside a
single node — smallest degree that fits — and I'd expect to give up 10–30% throughput per
GPU to communication. I'd also ask whether we need 70B, because a smaller model with
better retrieval is often the cheaper answer to the same product requirement.

**"Would sharding across two GPUs double your throughput?"**

No. Sharding is for fitting, not throughput. Two GPUs in tensor parallel give less than
2× because of the all-reduce at every layer on every token, and you've also made your
failure unit twice as large. If the model already fits on one GPU, two independent
replicas beat one two-way sharded replica on every axis: throughput, failure isolation,
and scaling granularity.

**"Your eight-way sharded deployment has worse availability than the single-GPU one it
replaced. Why?"**

Because a replica now dies when any of its eight GPUs dies. Hardware failure rate per
replica went up roughly eight-fold, and each failure takes out eight GPUs' worth of
capacity while it recovers — with a longer cold start. If the model didn't need eight-way
sharding, that's the finding. If it did, the fix is more replicas and faster recovery, not
more sharding.

---

## What to remember

- Sharding solves fitting, not throughput. If the model fits, don't shard.
- Check quantization first. 70B goes from 140 GB to 35 GB between BF16 and INT4, which
  turns a multi-GPU problem into a single-GPU one.
- Weights fitting isn't enough — you need KV cache room for a useful batch size.
- Tensor parallelism inside a node, pipeline parallelism across nodes. Tensor parallelism
  over ordinary networking is a mistake you only make once.
- The real costs are operational: bigger failure units, coarser scaling, slower cold
  starts, harder scheduling.
