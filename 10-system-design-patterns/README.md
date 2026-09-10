# 10 — System Design Patterns

The shapes that recur across AI products, as reference rather than as scenarios. Where
[12-senior-scenarios](../12-senior-scenarios/) gives you a situation and asks you to
reason, this section names the structure and says when each one is right.

Plain language throughout.

> **Most of these decisions get made by accident.** Someone writes an HTTP handler that
> calls a model, and the delivery shape is set. Someone says "a human will review it," and
> a queue nobody drains becomes the safety property. The patterns here are the versions
> of those decisions made on purpose.

## The files

| File | What it covers |
|---|---|
| [request-response-vs-async.md](request-response-vs-async.md) | Sync, streaming, or a job — and the timeout mismatch that produces 504s next to success logs |
| [human-in-the-loop.md](human-in-the-loop.md) | Review as a budget, and where a person actually goes |
| [batch-vs-realtime.md](batch-vs-realtime.md) | **The cheapest lever most teams leave alone.** Roughly 5× on identical work. |
| [multi-region.md](multi-region.md) | Three different projects sharing one name |
| [platform-pattern.md](platform-pattern.md) | The thin waist, and why thick platforms get bypassed |

## Read in this order

File order is fine. If you're designing something new,
[request-response-vs-async.md](request-response-vs-async.md) first — the delivery shape
constrains everything after it.

If you're trying to cut a bill, [batch-vs-realtime.md](batch-vs-realtime.md) is the
highest-value file here and the one most likely to apply to a system you already have.

If someone has asked for multi-region, read
[multi-region.md](multi-region.md) before agreeing to anything.

## The short version

**Pick the delivery shape per surface, not per product.** Ask who is waiting and for what.
A person watching minutes of work wants progress streamed over an async job, which is
neither of the obvious answers.

**Timeouts must decrease inward from the client.** When a proxy gives up before the
application does, the client sees a 504 while your logs show success and you pay for
tokens nobody reads.

**Streaming fixes perceived latency, not throughput.** It costs clean error handling,
correct cancellation, and any validation that needs the whole response.

**Human review is a budget.** Four reviewers at three minutes an item covers about 1% of
50,000 daily items. That fraction is the design constraint — you're not choosing whether
to review, you're choosing which 1%.

**Route escalation by stakes, not by model confidence.** Self-reported confidence doesn't
track correctness, and it's worst exactly when the model is confidently wrong.

**Specify what happens while the human decides.** User-visible state, a timeout, and an
explicit default — which must be "reject" for anything irreversible.

**Batch is roughly 5× cheaper than real-time for identical work.** Full batches, no
latency headroom, interruptible capacity, and the idle hours you already pay for. The test
is whether the answer depends on something that only exists at request time.

**Staleness is change rate × interval**, and it's a product decision. A nightly job over a
corpus that turns over daily is wrong half the time.

**Network is about 4% of a multi-second response.** Latency rarely justifies multi-region.
Time to first token and multi-turn flows are the exceptions worth checking.

**Data residency covers embeddings, prompts, logs and eval data** — not just documents. It
means a full stack per region, and enforcement that refuses unlabeled requests rather than
defaulting them.

**A platform should own what must be uniform and nothing else.** Credentials, cost
attribution, quota, logging, PII policy. Not prompts, model choice, or evals. A platform
that can block a team gets bypassed, and partial coverage is no coverage.

## The arithmetic worth doing out loud

```python
# Timeouts, decreasing inward from the client's budget
[client_timeout * (0.9 ** hop) for hop in range(hops)]

# What share of volume can humans actually review?
reviewers * (productive_hours * 60 / minutes_per_item) / daily_volume

# What fraction of precomputed values are stale at serving time?
min(1.0, change_rate_per_day * recompute_interval_hours / 24)

# Is the network actually your latency problem?
network_ms / (network_ms + retrieval_ms + generation_ms)
```

## Related

- [06-inference-serving/](../06-inference-serving/) — the serving layer these shapes sit
  on, and where the batch cost advantage comes from
- [07-production/fallbacks.md](../07-production/fallbacks.md) — degradation, which covers
  more failure modes than multi-region does
- [07-production/caching.md](../07-production/caching.md) — the middle ground between
  batch and real-time, and its key-scope hazard
- [09-data-pipelines/](../09-data-pipelines/) — the precomputation half of most AI systems
- [08-ai-security/authorization.md](../08-ai-security/authorization.md) — the fail-closed
  rule these patterns inherit
- [12-senior-scenarios/architecture-tradeoffs.md](../12-senior-scenarios/architecture-tradeoffs.md)
  — these choices as interview scenarios, with the constraints changing round by round
