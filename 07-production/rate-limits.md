# Rate Limits

> **This file is about living inside a quota.** The implementation — token buckets,
> multi-tenant fairness, distributed limiting, adaptive limits — is in
> [../11-coding-rounds/rate-limiter.md](../11-coding-rounds/rate-limiter.md) with tested
> code.

The key idea, and the one most teams learn the expensive way:

> **Provider quota is a resource with a supply chain, not an environmental constraint.**
> It has a lead time, a commercial relationship, and a forecast. Treat it like capacity
> planning, not like weather.

---

## The limits you're actually subject to

Providers usually enforce several at once, and the one that bites is rarely the one you're
watching.

```
requests per minute
input tokens per minute
output tokens per minute
concurrent requests
tokens per day / month
```

You can be at 78% of your request limit and 108% of your input-token limit. Watching
requests per minute would tell you everything is fine.

**LLM rate limits are usually token-based, and token counts vary by two orders of magnitude
across requests.** A limiter that counts requests will either waste most of your quota or
blow through it.

```python
def headroom(usage, limits):
    """Every limit, separately. The binding one is rarely the obvious one."""
    return {name: usage[name] / limits[name] for name in limits}

# {'requests': 0.78, 'input_tokens': 1.08, 'output_tokens': 0.41}
#                                    ↑ this is the one killing you
```

---

## Discovering your limit by getting 429s is the failure

Every 429 costs you:

- A request against your request-per-minute quota (yes, rejections count)
- A full network round trip of latency
- A retry, which may synchronize with every other client's retry
- Nothing gained

**Client-side admission control costs microseconds and gives you control over *which*
request waits.** That's the whole argument for it.

```python
lease = await limiter.acquire(est_input=2_400, est_output=500)
try:
    response = await provider.complete(prompt)
finally:
    lease.settle(response.input_tokens, response.output_tokens)
```

### The estimation problem, and the pattern that solves it

You have to reserve capacity *before* the response exists, but output length is unknown
until it does.

The asymmetry decides the design:

- **Under-estimating** → 429s → wasted quota, latency, retry amplification. Expensive.
- **Over-estimating** → under-utilization. Cheap, and mostly recoverable.

So: **reserve pessimistically, refund quickly.**

```python
# Reserve max_tokens (safe), then return the difference the moment you know
lease = await limiter.acquire(est_input=2_400, est_output=1_024)
response = await provider.complete(prompt, max_tokens=1024)
lease.settle(response.input_tokens, response.output_tokens)   # refunds ~900
```

Without refunds, reserving 1,024 output tokens when you typically use 180 caps you at
roughly a fifth of your real quota. With them, you get safety and utilization.

Calibrate the estimate from observed output lengths — use a high percentile like p90, not
the mean, because of that same asymmetry.

---

## Headroom is a leading indicator nobody watches

By the time you see 429s, you've been at 100% for a while.

```python
HEADROOM_ALERTS = [
    (0.70, "warning",  "start the quota increase conversation"),
    (0.85, "high",     "shed batch work, escalate commercially"),
    (0.95, "critical", "you are about to be rate limited"),
]
```

Alert at 70%, not at 100%. **Provider quota increases typically take weeks**, and that's a
commercial process with humans in it — a support ticket, an account manager, sometimes a
contract change.

Knowing at 70% turns a crisis into a scheduled conversation. Knowing at 100% means you
spend three weeks throttled.

---

## Separate lanes, one pool

The most common operational failure with quotas: a bulk job consumes everything and
interactive users sit behind it.

```python
class Priority(IntEnum):
    INTERACTIVE = 0     # a human is waiting
    STANDARD    = 1
    BATCH       = 2     # nobody is waiting
```

Admit preferentially by priority, but use **one shared pool with weights**, not separate
pools:

- Separate pools waste capacity — interactive sits idle while batch queues.
- A naive "batch only when interactive is empty" policy starves batch forever on a busy
  service.

Explicit weights (say 4:1) prevent starvation in both directions.

### The bulk-backfill arithmetic

A customer needs 400,000 documents processed. Peak-hour capacity is already tight. Panic?

Do the arithmetic before deciding:

```python
def overnight_capacity(spare_tokens_per_min, hours, tokens_per_document):
    total = spare_tokens_per_min * 60 * hours
    return total / tokens_per_document

overnight_capacity(spare_tokens_per_min=3_000_000, hours=10, tokens_per_document=4_000)
# 450,000 documents in one overnight window
```

You have a 4M tokens/minute budget for 1,440 minutes a day. The problem is almost never
total volume — it's **distribution**. Scheduling the backfill into off-peak hours often
solves it entirely, with no quota increase and no capacity purchase.

That reframe — "this is a scheduling problem, not a capacity problem" — is worth reaching
for before escalating.

---

## What changes with 20 servers

A per-process limiter multiplied by 20 processes is not a limit. Three approaches:

| Approach | Accuracy | Latency cost | Fails how |
|---|---|---|---|
| Static split (limit ÷ N) | Poor — strands idle capacity | None | Nothing |
| Central coordination (Redis) | Excellent | ~1ms per request | **Redis is now in your critical path** |
| Local + periodic reconciliation | Good | None | Converges slowly |

The practical answer is usually **hybrid**: local admission with a safety margin, Redis for
the shared portion, and — critically — a local fallback if Redis is unavailable.

```python
async def acquire(self, est_in, est_out):
    if not self._redis_healthy:
        # Degrade to a conservative local share. Under-utilize, but stay up.
        return await self._local_fallback.acquire(est_in, est_out)
    ...
```

> **A rate limiter must never be the reason your service is down.** If the coordination
> layer fails, fall back to conservative local limits and keep serving. The code that
> handles failure must not itself have a hard dependency that fails.

---

## Adapting to a limit you can't see

Your effective limit often isn't the published one. It varies by model, by time of day, and
by whatever the provider's capacity planning is doing. Teams routinely see 429s at 70% of
the documented rate.

The answer is congestion control — the same shape TCP uses:

- **On success:** increase the admitted rate slowly (additive).
- **On a 429:** decrease it sharply (multiplicative, e.g. ×0.7).

The asymmetry is deliberate: probing upward should be cautious, backing off decisive,
because exceeding the limit is more expensive than running slightly under it.

Two refinements worth having:

**A cooldown after a decrease**, so a burst of 429s from one incident doesn't apply the
reduction five times and collapse your rate.

**Latency as a secondary signal.** Rising latency often precedes 429s. Backing off on it
means slowing down *before* you're rejected.

And log the discovered limit — if it consistently settles at 60% of your documented quota,
that's real information for the provider conversation.

---

## Your own customers need limits too

If you expose an AI feature, you're the provider now, and the same problems apply
downstream.

Things to bound per customer:

```python
PER_TENANT_LIMITS = {
    "requests_per_minute": 60,
    "tokens_per_day":      2_000_000,
    "concurrent_requests": 5,
    "max_queue_depth":     1_000,      # backpressure, not unbounded queueing
}
```

**Bound the queue.** An unbounded queue turns a rate problem into a memory problem, and
lets one tenant occupy every slot even if fair scheduling stops them consuming quota.
Rejecting with a clear `Retry-After` is a better experience than a request that waits four
minutes and then times out.

**Return proper headers** so clients can behave well:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 12
X-RateLimit-Reset: 1735689600
Retry-After: 8
```

Clients that can see their headroom don't need to discover it by failing.

---

## What to monitor

```python
RATE_LIMIT_METRICS = [
    "headroom_requests", "headroom_input_tokens", "headroom_output_tokens",
    "429_rate",                    # should be ~0 with client-side limiting
    "queue_depth_by_lane",
    "wait_time_p50_p95_by_lane",
    "token_estimation_error",      # estimated vs actual — drift breaks admission
    "rejected_requests",
    "secondary_provider_share",
]
```

Two worth explaining:

**429 rate should be near zero.** Any 429 means your model of the limit is wrong — either
your estimator drifted or the provider changed something. It's a signal, not a normal
operating condition.

**Token estimation error** silently degrades everything. If your estimates run 15% low, you
admit too much and get 429s despite having a limiter. Reconcile estimates against actual
usage continuously.

---

## Interview questions

**1. "You're getting 429s. What do you do?"**

Check *which* limit is binding — usually tokens, not requests. Add client-side admission
control so you stop discovering the limit by failing. Then reduce token consumption
(prompt caching, reranking to send fewer chunks) and start the quota-increase conversation,
because that takes weeks.

**2. "Concurrency limits or rate limits?"**

Different things, and you usually need both. Concurrency bounds simultaneous requests and
protects *your* memory and connections. Rate limits bound requests per unit time and
protect the provider's quota. With fast responses, low concurrency can still exceed a rate
limit.

**3. "How do you reserve capacity when you don't know the output length?"**

Reserve pessimistically at `max_tokens`, then refund the difference on settlement. The
asymmetry justifies it: under-estimating causes 429s, over-estimating just costs
utilization, and refunds recover most of that.

**4. "A customer needs 400,000 documents processed and you're near your limit."**

Do the off-peak arithmetic first. You have the full daily budget across 1,440 minutes;
the problem is usually distribution, not volume. Schedule it overnight in a batch lane
with weighted admission so interactive traffic is unaffected.

**5. "Twenty application servers. What changes?"**

A per-process limiter isn't a limit. Options: static split (wasteful), central coordination
(accurate, adds a dependency to your failure-handling path), or hybrid with local fallback.
Whatever you pick, the limiter must not take down the service when its coordination layer
fails.

**6. "You're at 40% utilization and still getting 429s."**

Several possibilities: you're measuring the wrong limit; your token estimates are low;
another team shares the account; the effective limit is below the documented one; or your
usage is bursty and the provider measures over a shorter window than you do.

---

## What to remember

- Provider quota is a managed resource with a lead time, not weather.
- Check every limit separately — the binding one is usually tokens, not requests.
- Client-side admission beats discovering the limit through 429s.
- Reserve pessimistically, refund fast. That's what makes token-aware limiting usable.
- Alert on headroom at 70%. Quota increases take weeks.
- One pool with weighted lanes, not separate pools.
- Bulk work is usually a scheduling problem, not a capacity problem. Do the off-peak math.
- Bound your own queues; unbounded queueing converts one problem into two.
- The limiter must never be the reason you're down — always have a local fallback.
- 429 rate should be ~0. Any 429 means your model of the limit is wrong.

---

**Next:** [caching.md](caching.md) — the biggest cost lever, and the easiest one to get
dangerously wrong.
