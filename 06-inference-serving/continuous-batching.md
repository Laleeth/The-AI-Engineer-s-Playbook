# Continuous Batching

Continuous batching is the single change that makes self-hosted inference economically
viable. It is also the thing people mean, without knowing it, when they say a serving
stack is "faster."

The idea takes one sentence: **sequences join and leave the batch at every decoding step,
instead of the batch being fixed for its lifetime.** Everything below is why that sentence
is worth several times your GPU bill.

---

## Why batching helps at all

Decoding is memory-bandwidth-bound. To produce one token, the GPU reads the entire set of
model weights out of high-bandwidth memory. It does that whether it is generating for one
sequence or for sixty.

```python
# Reading 26 GB of weights per decoding step, at roughly 3 TB/s of memory bandwidth
weights_gb = 26
bandwidth_gb_per_s = 3_000

seconds_per_step = weights_gb / bandwidth_gb_per_s   # ≈ 0.0087 s

# That step produces one token per sequence in the batch:
tokens_per_second_at_batch = {
    1:  1 / seconds_per_step,      # ≈ 115 tokens/sec total
    32: 32 / seconds_per_step,     # ≈ 3,692 tokens/sec total
}
```

The weight read is amortized across the batch. Going from batch 1 to batch 32 costs
almost nothing extra in time per step and produces 32× the tokens. This is why batch size
is the throughput lever, and why a deployment running at batch 1 is burning roughly
97% of what it's paying for.

It is also why the gain stops. Past some batch size the per-step time starts rising —
attention work grows with the number of sequences and their context lengths, and
eventually you become compute-bound or you run out of KV cache. The curve flattens, then
bends down. Find yours by measuring; there is no universal number, and anyone who quotes
one hasn't measured their own.

---

## Static batching, and the hole in it

The obvious implementation collects requests, runs them together, and returns when they're
all done.

```python
def static_batch_step(requests):
    """Every sequence in the batch finishes when the LONGEST one finishes."""
    batch = [tokenize(r.prompt) for r in requests]
    return model.generate_batch(batch, max_new_tokens=max(r.max_tokens for r in requests))
```

The problem is that output lengths vary enormously and you don't know them in advance. One
request generates 30 tokens; another generates 800. With static batching, the 30-token
request occupies its slot for all 800 steps, producing padding.

Worked example, with a batch of 8 and realistic output lengths:

```python
outputs = [30, 45, 60, 80, 120, 200, 400, 800]

steps_run = max(outputs) * len(outputs)   # 800 * 8   = 6,400 slot-steps
useful     = sum(outputs)                 #            = 1,735 useful steps

utilization = useful / steps_run          # ≈ 0.27
```

**27%.** Nearly three quarters of the GPU-time in that batch produces nothing. And the
distribution above is not adversarial — a mix of short answers and long ones is what
normal chat traffic looks like. The more variance in output length, the worse static
batching gets, and output length variance in production is high.

There is a second cost that doesn't show up in that number: the 30-token request also
*waits* for all 800 steps before its response is returned. Static batching converts
output-length variance into latency for everybody.

---

## What continuous batching changes

At each decoding step the scheduler re-forms the batch:

```python
def continuous_batch_loop(scheduler):
    """One iteration per token, not one iteration per request."""
    while True:
        # Sequences that hit EOS or their token limit leave immediately
        scheduler.evict_finished()

        # Their KV blocks are freed, so queued requests can be admitted now
        scheduler.admit_from_queue()

        # Everything currently running advances by exactly one token
        scheduler.step()
```

The 30-token request leaves after 30 steps. Its KV cache blocks are freed. A queued
request takes its place on step 31. Nothing waits for the 800-token request except the
800-token request.

In the example above, utilization goes from 27% to essentially 100% of the slots, bounded
now by KV cache memory rather than by the longest sequence. The reported throughput gains
of continuous batching over static — often quoted around 2–4× on chat-shaped traffic —
come almost entirely from removing that padding. **The gain is proportional to your
output-length variance**, which is a useful thing to say in an interview because it tells
you when the technique won't help: uniform output lengths, such as a fixed-format
classification task, get very little from it.

---

## Prefill interrupts decode

Continuous batching creates a scheduling problem that didn't exist before. Admitting a new
request means running its prefill — processing its entire prompt — and prefill is
compute-heavy.

```python
# A 2,000-token prompt at 20,000 tokens/sec of prefill throughput
prefill_seconds = 2_000 / 20_000        # 0.1 s

# Meanwhile, every sequence already decoding produces nothing for 100 ms.
# At a typical 50 tokens/sec per sequence, that's ~5 tokens of stall each.
```

Every user already streaming sees a 100 ms hitch each time a long prompt is admitted. With
enough arrivals, this shows up as visible stutter in the token stream — smooth output,
pause, smooth output — and it is a common and confusing production complaint because the
p95 of total latency looks fine.

Two mitigations, and both are trade-offs rather than fixes:

- **Chunked prefill.** Split the prompt into pieces and interleave them with decoding
  steps. Stalls become short and frequent instead of long and occasional. Time to first
  token for the arriving request gets slightly worse; smoothness for everyone else gets
  much better.
- **Separate prefill and decode fleets.** Run prompt processing on one pool and token
  generation on another, shipping the KV cache between them. This removes the
  interference completely and adds a network transfer of the cache, plus a great deal of
  operational complexity. Worth it at large scale, over-engineering below it. See
  [capacity-planning.md](capacity-planning.md), where the two phases already want
  different GPU counts.

---

## Where the batch size actually comes from

You do not set batch size directly in a continuous-batching system. It is an *outcome* of
how much KV cache memory is free:

```python
def max_concurrent_sequences(gpu_gb, weights_gb, kv_bytes_per_token, avg_context):
    free_bytes = (gpu_gb - weights_gb) * 1e9
    return free_bytes / (kv_bytes_per_token * avg_context)

# 80 GB card, 26 GB of weights, 128 KB/token, 2,500-token average context
max_concurrent_sequences(80, 26, 131_072, 2_500)    # ≈ 165 sequences
```

This is the same arithmetic as [kv-cache.md](../01-llm-internals/kv-cache.md), and it is
the reason that file says to read it. Your achievable batch size — and therefore your
throughput, and therefore your GPU count — falls out of KV cache size per token and
average context length.

Which produces the counter-intuitive result worth carrying into an interview: **shortening
your prompts increases throughput more than it decreases prefill cost.** Halving average
context doubles the number of sequences that fit, which roughly doubles the batch size.
Trimming retrieved context from 20 chunks to 5 (see
[reranking.md](../03-rag/reranking.md)) buys throughput on top of the token saving.

---

## What goes wrong

**Long-context requests starve short ones.** One 100k-token request consumes the KV budget
of forty 2,500-token requests. Admit a few and your batch size collapses. Fixes: a
separate lane for long-context traffic with its own memory budget, or an admission policy
that caps how much of the cache long requests may hold in aggregate.

**Preemption thrashing.** When memory fills, the scheduler evicts a running sequence and
recomputes it later. Under sustained pressure it can evict and recompute the same
sequences repeatedly, burning throughput on repeated prefill. The signal is a rising
recompute rate with flat or falling goodput. Fix admission, not eviction — reject or queue
at the door instead of over-admitting and sorting it out later.

**Fairness has to be deliberate.** First-come-first-served under load means a burst from
one tenant fills the batch and everyone else queues behind it. This is the same problem
as [rate-limits.md](../07-production/rate-limits.md), one layer down, and it needs the
same answer: per-tenant lanes with reserved floors.

---

## What to monitor

- **Running batch size**, averaged over time. The single best indicator of serving health.
- **KV cache utilization**, and **preemption/recompute rate** alongside it. Recompute rate
  rising is the early warning; throughput falling is the late one.
- **Time-in-queue**, separate from generation time. These have different fixes and adding
  them together hides both.
- **Inter-token latency distribution**, not just the mean. Prefill interference shows up as
  a fat tail here and nowhere else.
- **Admission rejections**, if you have admission control. Zero rejections under heavy load
  means you don't have admission control, you have a queue.

---

## Interview questions

**"Throughput dropped 40% this week. No deploy. What do you look at?"**

Batch size first, then what's constraining it. Falling batch size with rising KV
utilization means context lengths grew — a longer system prompt, more retrieved chunks, a
change in traffic mix. Falling batch size with *low* KV utilization means requests aren't
arriving in the batch: check admission, queueing, and whether one tenant's long requests
are occupying the cache. A deploy is not required for any of this, which is the point of
the question.

**"Users say output is smooth then pauses then smooth. Latency percentiles look normal."**

Prefill interrupting decode. Total latency percentiles hide it because the stalls are
short relative to the whole response. Measure inter-token latency and correlate its tail
with prompt-admission events. Chunked prefill is the usual mitigation; separating prefill
and decode fleets is the heavyweight one.

**"Would continuous batching help a workload that classifies documents into one of five
labels?"**

Barely. The gain comes from output-length variance, and every response there is a few
tokens long. Batching still helps enormously — amortizing the weight read — but the
*continuous* part adds little over static batching. Knowing which half of the technique is
doing the work is the answer being looked for.

---

## What to remember

- Batching amortizes the weight read. That's why throughput scales with batch size at
  almost no cost in step time, until you run out of memory or become compute-bound.
- Static batching wastes GPU-time in proportion to output-length variance. A realistic
  chat batch can be under 30% useful.
- Continuous batching re-forms the batch every step, so finished sequences free their
  memory immediately. The win is roughly proportional to that same variance.
- Batch size is an outcome of free KV cache, not a setting. Shorter contexts mean bigger
  batches mean more throughput.
- Admitting a prompt stalls everyone who is decoding. That's visible as stutter and
  invisible in end-to-end percentiles.
