# Coding Round: Rate Limiting for AI APIs

Rate limiting for LLM APIs differs from classic API rate limiting in one decisive way:
**the cost of a request is not known until you've estimated it, and it varies by two
orders of magnitude.** A 200-token request and a 30,000-token request both count as "one
request," but only one of them will exhaust your token-per-minute quota.

Everything interesting in this problem follows from that.

Four problems:

1. [A token-aware client-side limiter](#problem-1-token-aware-rate-limiting)
2. [Multi-tenant fairness](#problem-2-fairness-across-tenants)
3. [Distributed limiting across 20 servers](#problem-3-twenty-application-servers)
4. [Adapting to a limit you can't see](#problem-4-adaptive-limiting)

Python 3.11+, standard library; Redis for the distributed variant.

---

## Problem 1: Token-Aware Rate Limiting

## Scenario

Your provider enforces three simultaneous limits:

```
8,000    requests per minute
4,000,000 input tokens per minute
800,000   output tokens per minute
```

Your service currently discovers these limits by receiving 429s. Each 429 wastes a
request against the request-per-minute limit, triggers a retry, and adds latency. During
business hours you're getting thousands of them.

Build a client-side limiter that admits work against all three limits, so you approach
the ceiling without hitting it.

## Requirements

- Enforce requests/minute, input tokens/minute, and output tokens/minute together.
- Support bursts up to the limit, then smooth.
- Async: callers `await` admission rather than polling or sleeping.
- Fair FIFO ordering — no starvation of large requests.
- Reconcile estimated tokens against actual usage after the response.
- Expose headroom for monitoring.

## Starter Code

```python
class RateLimiter:
    def __init__(self, requests_per_min: int, input_tokens_per_min: int,
                 output_tokens_per_min: int): ...

    async def acquire(self, est_input: int, est_output: int) -> "Lease": ...

    def headroom(self) -> dict[str, float]: ...
```

## Expected Behavior

```python
limiter = RateLimiter(8_000, 4_000_000, 800_000)

lease = await limiter.acquire(est_input=2_400, est_output=500)
try:
    response = await provider.complete(prompt)
finally:
    lease.settle(actual_input=response.input_tokens,
                 actual_output=response.output_tokens)

# Properties:
#  - never exceeds any of the three limits over any 60-second window
#  - a request needing more tokens than remain waits, it doesn't fail
#  - a request larger than the whole limit fails fast with a clear error
#  - headroom() reports each limit's utilization for alerting
```

---

<details>
<summary>💡 Reveal a hint</summary>

**Token bucket, not fixed window.** A fixed window (reset a counter every minute) allows
double the limit across a boundary: 8,000 requests at 11:59:59 and 8,000 more at
12:00:01. A token bucket refills continuously and has no boundary to exploit.

**Three buckets, all of which must admit.** A request needs a request-token *and*
input-tokens *and* output-tokens. Acquiring from one and waiting on another is where
deadlocks and unfairness creep in — decide against all three atomically, then deduct
from all three.

**Estimation is the hard part.** You must estimate output tokens *before* the model
generates them. Options:
- Use `max_tokens` as the estimate. Safe (never under-estimates) but wasteful — most
  responses are far shorter, so you under-utilize badly.
- Use a historical p90 per request type. Much better utilization, occasionally under-
  estimates.
- Reserve `max_tokens`, then **refund the difference** once the actual count is known.
  Best of both: never exceeds the limit, and reclaims the unused reservation quickly.

The reserve-and-refund pattern is the answer, and it's the design idea this problem is
really testing.

**Waiters need a queue, not a sleep loop.** `while not allowed: await sleep(0.1)` starves
large requests indefinitely — small requests keep slipping in ahead. Use a FIFO of
waiters, each woken when capacity might be available, and check the head of the queue
first.
</details>

---

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""Token-aware, multi-dimensional rate limiting for LLM APIs.

Three simultaneous limits, reserve-and-refund accounting, and FIFO fairness so
large requests are not starved by a stream of small ones.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("ratelimit")


# --------------------------------------------------------------------------
# Bucket
# --------------------------------------------------------------------------

class TokenBucket:
    """Continuously-refilling bucket.

    Chosen over a fixed window because a fixed window permits 2x the limit
    across a boundary, which is exactly when traffic is bursty.
    """

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(refill_per_second)
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    def peek(self, now: float | None = None) -> float:
        self._refill(now or time.monotonic())
        return self._tokens

    def can_take(self, amount: float, now: float) -> bool:
        self._refill(now)
        return self._tokens >= amount

    def take(self, amount: float, now: float) -> None:
        self._refill(now)
        self._tokens -= amount           # may go negative only via refund logic

    def give_back(self, amount: float, now: float) -> None:
        """Return an over-reservation. Capped at capacity so refunds can't
        manufacture burst capacity that never existed."""
        self._refill(now)
        self._tokens = min(self.capacity, self._tokens + amount)

    def seconds_until(self, amount: float, now: float) -> float:
        self._refill(now)
        if self._tokens >= amount:
            return 0.0
        if self.rate <= 0:
            return float("inf")
        return (amount - self._tokens) / self.rate

    @property
    def utilization(self) -> float:
        return 1.0 - (self.peek() / self.capacity if self.capacity else 0.0)


# --------------------------------------------------------------------------
# Lease
# --------------------------------------------------------------------------

@dataclass
class Lease:
    """A granted reservation. The caller MUST settle it, ideally in a finally
    block: an unsettled lease leaks reserved capacity until it expires."""
    limiter: "RateLimiter"
    est_input: int
    est_output: int
    granted_at: float
    settled: bool = False

    def settle(self, actual_input: int, actual_output: int) -> None:
        if self.settled:
            return
        self.settled = True
        self.limiter._settle(self, actual_input, actual_output)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        if not self.settled:
            # Nothing reported: assume the estimate was correct rather than
            # refunding, which would over-admit after a crash.
            self.settle(self.est_input, self.est_output)


@dataclass(order=True)
class _Waiter:
    seq: int
    event: asyncio.Event = field(compare=False)
    est_input: int = field(compare=False, default=0)
    est_output: int = field(compare=False, default=0)


# --------------------------------------------------------------------------
# Limiter
# --------------------------------------------------------------------------

class RateLimiter:
    def __init__(
        self,
        requests_per_min: int,
        input_tokens_per_min: int,
        output_tokens_per_min: int,
        *,
        safety_margin: float = 0.90,
    ) -> None:
        """safety_margin leaves headroom for estimation error, clock skew and
        the fact that the provider's window may not align with ours. 0.90 is a
        reasonable default; raise it once your estimator is well calibrated."""
        self.requests = TokenBucket(
            requests_per_min * safety_margin, requests_per_min * safety_margin / 60
        )
        self.input_tokens = TokenBucket(
            input_tokens_per_min * safety_margin,
            input_tokens_per_min * safety_margin / 60,
        )
        self.output_tokens = TokenBucket(
            output_tokens_per_min * safety_margin,
            output_tokens_per_min * safety_margin / 60,
        )
        self._lock = asyncio.Lock()
        self._waiters: list[_Waiter] = []
        self._seq = 0
        self._stats = {"admitted": 0, "waited": 0, "rejected": 0,
                       "total_wait_s": 0.0, "refunded_tokens": 0}

    # ---------------- admission ----------------

    async def acquire(self, est_input: int, est_output: int,
                      *, timeout: float | None = None) -> Lease:
        """Wait until all three limits can accommodate this request.

        Reserves the ESTIMATED cost. The caller settles with actuals, and any
        over-reservation is refunded immediately — which is what lets us use a
        conservative estimate without destroying utilization.
        """
        if est_input > self.input_tokens.capacity or \
           est_output > self.output_tokens.capacity:
            # Larger than the whole per-minute budget: waiting cannot help.
            self._stats["rejected"] += 1
            raise ValueError(
                f"request needs {est_input}+{est_output} tokens, exceeding the "
                f"per-minute capacity of "
                f"{self.input_tokens.capacity:.0f}/{self.output_tokens.capacity:.0f}"
            )

        started = time.monotonic()
        deadline = started + timeout if timeout else None

        async with self._lock:
            self._seq += 1
            waiter = _Waiter(self._seq, asyncio.Event(), est_input, est_output)
            self._waiters.append(waiter)
            granted = self._try_grant_head(time.monotonic())

        if not granted:
            self._stats["waited"] += 1
            try:
                if deadline:
                    await asyncio.wait_for(
                        waiter.event.wait(), timeout=deadline - time.monotonic()
                    )
                else:
                    await waiter.event.wait()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                async with self._lock:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
                    # A cancelled waiter may have been holding up the queue;
                    # give the next one a chance immediately.
                    self._try_grant_head(time.monotonic())
                raise

        self._stats["admitted"] += 1
        self._stats["total_wait_s"] += time.monotonic() - started
        return Lease(self, est_input, est_output, time.monotonic())

    def _try_grant_head(self, now: float) -> bool:
        """Grant to waiters in FIFO order, stopping at the first that can't fit.

        Stopping (rather than skipping to a smaller request that fits) is what
        prevents starvation: a large request would otherwise never be admitted
        while small ones keep arriving. The cost is head-of-line blocking, which
        is the right trade for fairness here.
        """
        granted_any = False
        while self._waiters:
            head = self._waiters[0]
            if not (self.requests.can_take(1, now)
                    and self.input_tokens.can_take(head.est_input, now)
                    and self.output_tokens.can_take(head.est_output, now)):
                break
            self.requests.take(1, now)
            self.input_tokens.take(head.est_input, now)
            self.output_tokens.take(head.est_output, now)
            self._waiters.pop(0)
            head.event.set()
            granted_any = True
        if self._waiters:
            asyncio.get_running_loop().call_later(
                self._next_wake_delay(now), self._wake
            )
        return granted_any

    def _next_wake_delay(self, now: float) -> float:
        head = self._waiters[0]
        return max(0.02, min(
            self.requests.seconds_until(1, now),
            self.input_tokens.seconds_until(head.est_input, now),
            self.output_tokens.seconds_until(head.est_output, now),
        ))

    def _wake(self) -> None:
        # call_later runs outside the lock; scheduling a task keeps the
        # bucket mutation serialized.
        asyncio.create_task(self._wake_async())

    async def _wake_async(self) -> None:
        async with self._lock:
            self._try_grant_head(time.monotonic())

    # ---------------- settlement ----------------

    def _settle(self, lease: Lease, actual_input: int, actual_output: int) -> None:
        """Reconcile the reservation against reality.

        Over-reservation (the common case, because output is estimated at or
        near max_tokens) is refunded so the capacity is usable immediately.
        Under-reservation is charged, which may push a bucket negative — correct,
        because we genuinely consumed that capacity and subsequent requests
        should wait for it to refill.
        """
        now = time.monotonic()
        in_delta = lease.est_input - actual_input
        out_delta = lease.est_output - actual_output

        if in_delta > 0:
            self.input_tokens.give_back(in_delta, now)
        elif in_delta < 0:
            self.input_tokens.take(-in_delta, now)

        if out_delta > 0:
            self.output_tokens.give_back(out_delta, now)
            self._stats["refunded_tokens"] += int(out_delta)
        elif out_delta < 0:
            self.output_tokens.take(-out_delta, now)

        asyncio.create_task(self._wake_async())

    # ---------------- observability ----------------

    def headroom(self) -> dict[str, float]:
        """Utilization per limit, 0.0 (idle) to 1.0 (exhausted).

        Alert at 0.70 and 0.85. Discovering a limit by receiving 429s means you
        were not watching the number that predicted them.
        """
        return {
            "requests": self.requests.utilization,
            "input_tokens": self.input_tokens.utilization,
            "output_tokens": self.output_tokens.utilization,
            "queue_depth": float(len(self._waiters)),
        }

    def stats(self) -> dict:
        s = dict(self._stats)
        s["mean_wait_s"] = (
            s["total_wait_s"] / s["admitted"] if s["admitted"] else 0.0
        )
        return s


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

class TokenEstimator:
    """Estimate input and output tokens before a call.

    Input is countable exactly with the model's tokenizer. Output is not, so we
    track observed lengths per request type and use a high percentile: the cost
    of over-estimating is under-utilization (recovered by refunds), while the
    cost of under-estimating is a 429.
    """

    def __init__(self, default_output: int = 500, percentile: float = 0.90) -> None:
        self.default_output = default_output
        self.percentile = percentile
        self._observed: dict[str, list[int]] = {}

    def estimate_input(self, text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            return len(tokenizer.encode(text))
        return max(1, len(text) // 4 + 16)      # +16 for message scaffolding

    def estimate_output(self, request_type: str, max_tokens: int) -> int:
        history = self._observed.get(request_type)
        if not history or len(history) < 30:
            return max_tokens                   # conservative until calibrated
        ordered = sorted(history)
        idx = min(len(ordered) - 1, int(len(ordered) * self.percentile))
        return min(max_tokens, ordered[idx])

    def observe(self, request_type: str, actual_output: int) -> None:
        history = self._observed.setdefault(request_type, [])
        history.append(actual_output)
        if len(history) > 1000:
            del history[:500]                   # bounded, recency-weighted
```

**Usage:**

```python
limiter = RateLimiter(8_000, 4_000_000, 800_000)
estimator = TokenEstimator()

async def call(prompt: str, request_type: str, max_tokens: int = 1024):
    est_in = estimator.estimate_input(prompt, tokenizer)
    est_out = estimator.estimate_output(request_type, max_tokens)

    lease = await limiter.acquire(est_in, est_out, timeout=30.0)
    try:
        response = await provider.complete(prompt, max_tokens=max_tokens)
        estimator.observe(request_type, response.output_tokens)
        return response
    finally:
        lease.settle(
            getattr(response, "input_tokens", est_in),
            getattr(response, "output_tokens", est_out),
        )
```

</details>

---

<details>
<summary>📖 Reveal the explanation</summary>

### The decisions

**Token bucket over fixed window.** A fixed window permits a burst of `2 × limit` around
a boundary — 8,000 requests in the last second of one minute and 8,000 in the first
second of the next. Providers implement rolling or leaky-bucket windows, so a client that
uses a fixed window will get 429s while believing it is compliant.

**Three buckets checked atomically.** A request must satisfy all three limits. Taking
from one and then blocking on another leaves capacity reserved for a request that isn't
running, which under load produces a slow deadlock as every waiter holds a partial
reservation. The `can_take` × 3 then `take` × 3 pattern under a single lock avoids it.

**Reserve-and-refund is the central idea.** Output length is unknowable before
generation, so you must reserve pessimistically or risk 429s. Reserving `max_tokens` and
refunding the unused portion at settlement gives you both safety and utilization: if
requests reserve 1,024 and use 180 on average, ~82% of the reservation is returned within
a couple of seconds. Without refunds, a 1,024-token reservation on every request would
cap you at roughly a fifth of your actual quota.

**FIFO with head-of-line blocking is deliberate.** `_try_grant_head` stops at the first
waiter that doesn't fit rather than skipping ahead to smaller requests. Skipping would
improve throughput and starve large requests forever — a 30,000-token request would never
run while 500-token requests keep arriving. Fairness is worth the throughput cost here,
and the choice should be stated explicitly rather than made by accident.

**Cancellation removes the waiter and re-runs the grant loop.** A cancelled waiter at the
head of the queue would otherwise block everyone behind it until the next scheduled wake.

**The safety margin exists because the estimate is wrong and the clocks differ.** Your
minute and the provider's minute do not align; your token counting may differ from theirs
by a few percent. Running at 90% of the stated limit costs a little throughput and
prevents a class of 429s that is expensive to debug.

**`Lease.__exit__` settles with the estimate, not with a refund.** If the caller crashes
without reporting actuals, assuming the estimate was right is the conservative choice.
Refunding would over-admit; charging `max_tokens` would under-admit. Assuming the estimate
is neutral.

**Estimation improves itself.** The `TokenEstimator` starts at `max_tokens` (safe, low
utilization) and moves to a p90 of observed lengths once it has data. Using p90 rather
than the mean is deliberate: under-estimating causes 429s, over-estimating is refunded.
The asymmetry of the costs should drive the choice of percentile, and that reasoning is
what an interviewer is listening for.

### Complexity

- `acquire`: `O(1)` amortized when capacity is available; `O(w)` in the grant loop when
  waking, where `w` is the number of waiters that can be admitted.
- Memory: `O(waiters)` plus a bounded per-request-type history in the estimator.
- The single lock serializes admission. At tens of thousands of requests/minute this is
  not a bottleneck; at millions it would need sharding.

</details>

---

## Production Improvements

- **Distributed coordination** (Problem 3). A per-process limiter multiplied by 20
  processes is not a limit.
- **Priority lanes.** Interactive requests should be admitted ahead of batch work. A
  single FIFO is fair between equals; it is not correct when requests have different
  urgency.
- **Provider header reconciliation.** Many providers return remaining-quota headers.
  Correcting your local buckets against them is the difference between an estimate and a
  measurement.
- **Per-model buckets.** Limits are usually per-model, not per-account. One bucket per
  model, plus an account-level bucket if one exists.
- **Reject rather than queue past a depth.** An unbounded queue converts a rate-limit
  problem into a memory and latency problem; requests wait 4 minutes and then time out
  anyway.
- **Metrics**: utilization per limit, queue depth, wait-time percentiles, estimation
  error distribution, refund volume, and 429s received (which should be near zero — any
  429 means the limiter's model of reality is wrong).

## Edge Cases

| Case | Correct behavior |
|---|---|
| Request larger than the per-minute limit | Fails fast with a clear message; waiting cannot help |
| Caller never settles the lease | Capacity leaks; mitigate with a settlement timeout sweeper |
| Actual tokens exceed the estimate | Bucket goes negative; later requests wait longer — correct |
| All waiters cancelled | Grant loop re-runs, no wedged queue |
| `requests_per_min=0` | `seconds_until` returns infinity; admission never happens (arguably should raise at construction) |
| Clock skew / NTP adjustment | `time.monotonic()` is immune |
| Burst after an idle period | Bucket is full, so a full burst is admitted — this is intended token-bucket behavior |
| Two limits exhausted, one free | Waits for all three; no partial admission |
| Estimator has no history | Falls back to `max_tokens`, safe but conservative |

## Follow-up Questions

1. **Twenty application servers.** What changes? (Problem 3.)
2. **Your token estimate is 15% low.** What happens, and how do you detect and fix it?
3. **Some requests are interactive and some are batch.** Redesign admission.
4. **The provider changes your limits with no notice.** How resilient are you?
5. **A single tenant floods the queue.** What happens to everyone else? (Problem 2.)
6. **You're at 40% utilization but still getting 429s.** Give three explanations.
7. **How do you test this** without waiting 60 seconds per test?

## Senior-Level Discussion

**Concurrency limits and rate limits are different things and both are usually needed.**
Concurrency bounds simultaneous in-flight requests, which protects *your* memory,
connections and thread pools. Rate limits bound requests per unit time, which protects
*the provider's* quota. With 100ms responses, a concurrency of 10 produces 6,000
requests/minute; with 10-second responses, the same concurrency produces 60. Managing the
wrong one is the most common mistake in this area.

**Client-side limiting is strictly better than reacting to 429s.** A 429 consumes a
request against your request-per-minute quota, adds a full round-trip of latency,
triggers retry logic, and — if your retries lack jitter — synchronizes with every other
client's retries. Admitting locally costs microseconds and gives you control over *which*
request waits.

**The estimation asymmetry drives the design.** Under-estimating causes 429s (expensive:
wasted quota, latency, retry amplification). Over-estimating causes under-utilization
(cheap, and largely recovered by refunds). So estimate high and refund fast. Candidates
who propose using the mean output length have not thought about which error is worse.

**Utilization is a leading indicator and nobody watches it.** By the time you see 429s,
you have already been at 100% for a while. `headroom()` at 0.70 and 0.85 gives you weeks
of warning — and provider quota increases take weeks (see
`../12-senior-scenarios/scaling.md#5-the-provider-rate-limit-wall`). This is the
monitoring point that turns a recurring crisis into a planned capacity conversation.

---

## Problem 2: Fairness Across Tenants

## Scenario

Your platform serves 340 tenants through one provider account. One tenant starts a bulk
import and submits 400,000 requests. Your FIFO queue admits them in order, and every
other tenant's interactive requests sit behind 400,000 items.

## Requirements

Add per-tenant fairness. No tenant should be able to monopolize the shared quota, and
capacity unused by idle tenants should not be wasted.

<details>
<summary>💡 Reveal a hint</summary>

FIFO is fair *between requests* and deeply unfair *between tenants*: a tenant that
submits more work gets more service, which is exactly backwards.

Two families of solution:

- **Hard per-tenant quotas.** Simple, predictable, and wasteful: 340 tenants each capped
  at 1/340th of capacity means an idle majority strands most of your quota.
- **Weighted fair queueing.** Each tenant has a virtual clock; you always serve the
  tenant with the smallest virtual finish time. Idle tenants' capacity flows to active
  ones automatically, and a heavy tenant can use the whole quota when nobody else wants
  it — but yields immediately when they do. This is the property you want.

The refinement that makes WFQ practical: **cap how far a tenant's virtual clock can lag**
behind the global clock. Otherwise a tenant that was idle for an hour accumulates enormous
credit and monopolizes the queue when it returns.

Then add a second dimension: **priority classes within a tenant**, so a tenant's batch
work doesn't block their own interactive work.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""Weighted fair queueing across tenants, with priority classes within a tenant."""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    INTERACTIVE = 0        # a human is waiting
    STANDARD = 1
    BATCH = 2              # nobody is waiting


@dataclass(order=True)
class _Request:
    virtual_finish: float           # WFQ ordering key
    priority: Priority
    seq: int
    tenant: str = field(compare=False, default="")
    est_input: int = field(compare=False, default=0)
    est_output: int = field(compare=False, default=0)
    event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)


class FairRateLimiter(RateLimiter):
    """Weighted fair queueing over the shared token buckets.

    The virtual clock: each tenant has a `virtual_time` that advances by
    cost/weight for every request it submits. We always serve the request with
    the smallest virtual finish time, so a tenant consuming heavily advances its
    own clock and naturally yields to tenants that have consumed less.
    """

    def __init__(self, *args, default_weight: float = 1.0,
                 max_lag_seconds: float = 5.0, max_queue_per_tenant: int = 10_000,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights: dict[str, float] = {}
        self.default_weight = default_weight
        self.max_lag = max_lag_seconds
        self.max_queue_per_tenant = max_queue_per_tenant
        self._virtual_time: dict[str, float] = {}
        self._queue: list[_Request] = []          # heap
        self._queued_per_tenant: dict[str, int] = {}
        self._global_virtual_time = 0.0
        self._seq = 0

    def set_weight(self, tenant: str, weight: float) -> None:
        """Higher weight = larger share. Use for paid tiers or SLAs."""
        self.weights[tenant] = weight

    async def acquire_for(
        self,
        tenant: str,
        est_input: int,
        est_output: int,
        *,
        priority: Priority = Priority.STANDARD,
        timeout: float | None = None,
    ) -> Lease:
        if est_input > self.input_tokens.capacity or \
           est_output > self.output_tokens.capacity:
            raise ValueError("request exceeds per-minute capacity")

        async with self._lock:
            queued = self._queued_per_tenant.get(tenant, 0)
            if queued >= self.max_queue_per_tenant:
                # Bounded queue per tenant: an unbounded queue turns a rate
                # problem into a memory problem, and lets one tenant occupy all
                # the queue slots even if WFQ prevents them consuming quota.
                raise TenantQueueFull(
                    f"tenant {tenant} has {queued} queued requests"
                )

            weight = self.weights.get(tenant, self.default_weight)
            cost = est_input + est_output * 4      # output is scarcer; weight it

            # Cap how far behind the global clock a returning tenant can be,
            # so idleness doesn't bank unlimited credit.
            vt = max(
                self._virtual_time.get(tenant, self._global_virtual_time),
                self._global_virtual_time - self.max_lag,
            )
            virtual_finish = vt + cost / weight
            self._virtual_time[tenant] = virtual_finish

            self._seq += 1
            req = _Request(virtual_finish, priority, self._seq,
                           tenant, est_input, est_output)
            heapq.heappush(self._queue, req)
            self._queued_per_tenant[tenant] = queued + 1
            self._pump(time.monotonic())

        if not req.event.is_set():
            try:
                if timeout:
                    await asyncio.wait_for(req.event.wait(), timeout=timeout)
                else:
                    await req.event.wait()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                async with self._lock:
                    self._cancel(req)
                raise

        return Lease(self, est_input, est_output, time.monotonic())

    def _pump(self, now: float) -> None:
        """Admit from the head of the heap while capacity allows.

        The heap is ordered by (virtual_finish, priority, seq), so among
        requests with similar virtual time, interactive beats batch.
        """
        while self._queue:
            head = self._queue[0]
            if head.event.is_set():                  # cancelled
                heapq.heappop(self._queue)
                continue
            if not (self.requests.can_take(1, now)
                    and self.input_tokens.can_take(head.est_input, now)
                    and self.output_tokens.can_take(head.est_output, now)):
                break
            heapq.heappop(self._queue)
            self.requests.take(1, now)
            self.input_tokens.take(head.est_input, now)
            self.output_tokens.take(head.est_output, now)
            self._global_virtual_time = max(self._global_virtual_time,
                                            head.virtual_finish)
            self._queued_per_tenant[head.tenant] -= 1
            head.event.set()

    def _cancel(self, req: _Request) -> None:
        req.event.set()                              # tombstone; popped lazily
        self._queued_per_tenant[req.tenant] = max(
            0, self._queued_per_tenant.get(req.tenant, 1) - 1
        )
        self._pump(time.monotonic())

    def tenant_stats(self) -> dict[str, dict]:
        return {
            t: {"queued": n, "virtual_time": self._virtual_time.get(t, 0.0),
                "weight": self.weights.get(t, self.default_weight)}
            for t, n in self._queued_per_tenant.items() if n
        }


class TenantQueueFull(Exception):
    """Backpressure signal: the tenant should slow down or retry later."""
```

**Why this behaves correctly:**

- The bulk-import tenant's virtual clock advances rapidly as it consumes, so its next
  request always sorts *behind* tenants that have consumed less. It gets service, just
  not priority.
- When other tenants are idle, the heavy tenant's requests are the only ones in the heap
  and it uses the full quota — no capacity is stranded.
- `max_lag` prevents a tenant that was quiet for an hour from arriving with an hour of
  accumulated credit and monopolizing the queue.
- `max_queue_per_tenant` bounds memory and provides an explicit backpressure signal
  (`TenantQueueFull`), which a client can use to slow its submission rate.
- Priority within the sort key means a tenant's interactive request beats their own batch
  request at similar virtual time.
</details>

## Follow-up Questions

1. A tenant hits `TenantQueueFull`. What should the client do, and what should you tell
   them?
2. How do you set weights? Should a paying tenant get 10× a free tenant?
3. Your heap has 400,000 entries. Is that a problem? What would you do?
4. A tenant's requests are all enormous. WFQ gives them fewer requests but the same token
   share. Is that fair?
5. How would you expose per-tenant fairness metrics to customers?

---

## Problem 3: Twenty Application Servers

## Scenario

The limiter works beautifully in one process. You run 20. Each process independently
admits up to the full limit, so you're issuing 20× the allowed rate and receiving 429s
constantly.

## Requirements

Make it work across processes. Discuss the trade-offs; there is no free answer.

<details>
<summary>💡 Reveal a hint</summary>

Three approaches, in increasing order of accuracy and cost:

1. **Static partitioning.** Each process gets `limit / 20`. Zero coordination, zero
   latency. Wastes capacity when load is uneven — and it always is — and breaks when the
   process count changes.

2. **Centralized coordination (Redis).** A shared token bucket in Redis, mutated with a
   Lua script for atomicity. Accurate and fair. Costs a network round trip (~1ms) on
   every admission and puts Redis in the critical path of every request — including
   during the incident where Redis is the thing that's struggling.

3. **Local buckets with periodic reconciliation.** Each process keeps a local bucket
   sized to its share, and a background task periodically redistributes unused capacity
   based on observed demand. No per-request coordination; approximate; converges within
   a few seconds.

The right answer is usually **hybrid**: local admission with a small safety margin, plus
Redis-based coordination for the shared portion, plus a local fallback if Redis is
unavailable. And in all cases: **never let the rate limiter's dependency take down the
service it protects** — if Redis is down, degrade to conservative local limits rather
than failing requests.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""Distributed token bucket in Redis, with a local fallback.

The Lua script makes check-and-consume atomic. Doing it with GET/SET or even
INCR is a race: two processes both read 100 remaining, both take 80, and you
have consumed 160 of a 100 budget.
"""

# --- Lua: atomic multi-dimensional token bucket -------------------------
# KEYS[1..3]  = request / input-token / output-token bucket keys
# ARGV[1]     = now (ms)
# ARGV[2..4]  = capacities
# ARGV[5..7]  = refill rates per ms
# ARGV[8..10] = amounts requested
# ARGV[11]    = TTL seconds
#
# Returns {1, 0} on admission, or {0, wait_ms} when it must wait.
LUA_ACQUIRE = """
local now = tonumber(ARGV[1])
local ttl  = tonumber(ARGV[11])
local wait = 0
local ok = true
local tokens = {}

for i = 1, 3 do
  local cap  = tonumber(ARGV[1 + i])
  local rate = tonumber(ARGV[4 + i])
  local need = tonumber(ARGV[7 + i])

  local state = redis.call('HMGET', KEYS[i], 'tokens', 'ts')
  local t  = tonumber(state[1]) or cap
  local ts = tonumber(state[2]) or now

  t = math.min(cap, t + (now - ts) * rate)
  tokens[i] = t

  if t < need then
    ok = false
    local w = (need - t) / rate
    if w > wait then wait = w end
  end
end

if not ok then
  -- Persist the refill even when refusing, so the next caller sees fresh state.
  for i = 1, 3 do
    redis.call('HSET', KEYS[i], 'tokens', tokens[i], 'ts', now)
    redis.call('EXPIRE', KEYS[i], ttl)
  end
  return {0, math.ceil(wait)}
end

for i = 1, 3 do
  local need = tonumber(ARGV[7 + i])
  redis.call('HSET', KEYS[i], 'tokens', tokens[i] - need, 'ts', now)
  redis.call('EXPIRE', KEYS[i], ttl)
end
return {1, 0}
"""

LUA_REFUND = """
local now = tonumber(ARGV[1])
for i = 1, 3 do
  local cap    = tonumber(ARGV[1 + i])
  local amount = tonumber(ARGV[4 + i])
  if amount > 0 then
    local t = tonumber(redis.call('HGET', KEYS[i], 'tokens')) or cap
    redis.call('HSET', KEYS[i], 'tokens', math.min(cap, t + amount), 'ts', now)
  end
end
return 1
"""


class DistributedRateLimiter:
    """Redis-coordinated limiting with a conservative local fallback.

    Availability principle: a rate limiter must never be the reason a service
    is down. If Redis is unreachable we fall back to a local bucket sized at
    limit/expected_instances — under-utilizing the quota but continuing to
    serve, which is the right failure mode.
    """

    def __init__(self, redis, key_prefix: str, *, requests_per_min: int,
                 input_tokens_per_min: int, output_tokens_per_min: int,
                 instances_hint: int = 20, safety_margin: float = 0.90) -> None:
        self.redis = redis
        self.keys = [
            f"{key_prefix}:req", f"{key_prefix}:in", f"{key_prefix}:out",
        ]
        self.caps = [requests_per_min * safety_margin,
                     input_tokens_per_min * safety_margin,
                     output_tokens_per_min * safety_margin]
        self.rates_per_ms = [c / 60_000 for c in self.caps]
        self._acquire_sha = None
        self._refund_sha = None

        # Local fallback: this instance's conservative share.
        self._fallback = RateLimiter(
            int(requests_per_min / instances_hint),
            int(input_tokens_per_min / instances_hint),
            int(output_tokens_per_min / instances_hint),
        )
        self._redis_healthy = True

    async def _ensure_scripts(self):
        if self._acquire_sha is None:
            self._acquire_sha = await self.redis.script_load(LUA_ACQUIRE)
            self._refund_sha = await self.redis.script_load(LUA_REFUND)

    async def acquire(self, est_input: int, est_output: int,
                      *, timeout: float = 30.0) -> Lease:
        deadline = time.monotonic() + timeout

        while True:
            if not self._redis_healthy:
                return await self._fallback.acquire(
                    est_input, est_output,
                    timeout=max(0.0, deadline - time.monotonic()),
                )

            try:
                await self._ensure_scripts()
                now_ms = int(time.time() * 1000)
                ok, wait_ms = await self.redis.evalsha(
                    self._acquire_sha, 3, *self.keys,
                    now_ms, *self.caps, *self.rates_per_ms,
                    1, est_input, est_output, 120,
                )
            except Exception:                                # noqa: BLE001
                log.exception("rate limiter: redis unavailable, using local limits")
                self._redis_healthy = False
                asyncio.create_task(self._probe_redis())
                continue

            if ok:
                return Lease(self, est_input, est_output, time.monotonic())

            wait = min(wait_ms / 1000, max(0.0, deadline - time.monotonic()))
            if wait <= 0:
                raise TimeoutError("rate limit wait exceeded the deadline")
            # Jitter so 20 instances don't retry in lockstep and stampede Redis.
            await asyncio.sleep(wait * random.uniform(0.9, 1.3))

    async def _probe_redis(self) -> None:
        while not self._redis_healthy:
            await asyncio.sleep(5)
            try:
                await self.redis.ping()
                self._redis_healthy = True
                log.info("rate limiter: redis recovered")
            except Exception:                                # noqa: BLE001
                pass

    def _settle(self, lease: Lease, actual_input: int, actual_output: int) -> None:
        refunds = [0, max(0, lease.est_input - actual_input),
                   max(0, lease.est_output - actual_output)]
        if any(refunds):
            asyncio.create_task(self._refund(refunds))

    async def _refund(self, refunds) -> None:
        try:
            await self._ensure_scripts()
            await self.redis.evalsha(
                self._refund_sha, 3, *self.keys,
                int(time.time() * 1000), *self.caps, *refunds,
            )
        except Exception:                                    # noqa: BLE001
            pass    # a lost refund costs a little utilization, not correctness
```

**Trade-off comparison:**

| Approach | Accuracy | Latency added | Failure mode | Use when |
|---|---|---|---|---|
| Static partition | Poor (strands idle capacity) | 0 | None | Uniform load, fixed fleet |
| Redis coordination | Excellent | ~1ms/request | Redis down = must degrade | Accuracy matters, Redis already present |
| Local + reconciliation | Good | 0 | Converges slowly | High QPS, latency-sensitive |
| Hybrid (above) | Excellent, degrades gracefully | ~1ms | Falls back to local | Production default |
</details>

## Follow-up Questions

1. Redis adds 1ms to every request. At 2,000 req/s across the fleet, what does that cost
   and is it acceptable?
2. Redis fails over and loses state. What happens in the next 60 seconds?
3. Your fleet autoscales from 20 to 60 instances. What breaks in each approach?
4. How would you shard the Redis keys if one key becomes a hot spot?
5. Could you do this without Redis at all? What would you use?

---

## Problem 4: Adaptive Limiting

## Scenario

You do not actually know your limits. The provider publishes numbers, but the effective
limit varies by model, by time of day, and by whatever their capacity planning is doing.
You get 429s at 70% of the published rate on Tuesday afternoons.

## Requirements

Discover the real limit from the system's behavior and adapt.

<details>
<summary>💡 Reveal a hint</summary>

This is congestion control. The same problem TCP solves, with the same shape of answer:
**additive increase, multiplicative decrease (AIMD)**.

- On success, increase the admitted rate slowly (additively).
- On a 429, decrease it sharply (multiplicatively — e.g. ×0.7).

The asymmetry is the point: probing upward should be cautious and backing off should be
decisive, because the cost of exceeding the limit (429s, retries, amplification) exceeds
the cost of running slightly below it.

Refinements worth mentioning:
- **Floor and ceiling** so it can't collapse to zero or run away.
- **Cooldown after a decrease** so a burst of 429s from one incident doesn't multiply the
  decrease five times.
- **Use latency as a secondary signal.** Rising latency often precedes 429s and gives you
  earlier warning.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""AIMD congestion control for an unknown, drifting rate limit."""

import time


class AdaptiveLimit:
    def __init__(self, initial: float, *, floor: float, ceiling: float,
                 increase_per_second: float = 2.0, decrease_factor: float = 0.7,
                 cooldown_seconds: float = 5.0) -> None:
        self.limit = float(initial)
        self.floor = float(floor)
        self.ceiling = float(ceiling)
        self.increase = increase_per_second
        self.decrease = decrease_factor
        self.cooldown = cooldown_seconds
        self._last_increase = time.monotonic()
        self._last_decrease = 0.0

    def on_success(self) -> None:
        """Additive increase: probe upward gently, and only while healthy."""
        now = time.monotonic()
        if now - self._last_decrease < self.cooldown:
            return                                  # still recovering
        elapsed = now - self._last_increase
        if elapsed >= 1.0:
            self.limit = min(self.ceiling, self.limit + self.increase * elapsed)
            self._last_increase = now

    def on_rate_limited(self) -> None:
        """Multiplicative decrease: back off decisively.

        The cooldown prevents a burst of 429s arriving together from applying
        the decrease five times and collapsing the limit.
        """
        now = time.monotonic()
        if now - self._last_decrease < self.cooldown:
            return
        self.limit = max(self.floor, self.limit * self.decrease)
        self._last_decrease = now
        self._last_increase = now
        log.warning("rate limit reduced to %.1f after 429", self.limit)

    def on_latency_signal(self, p95_ms: float, baseline_ms: float) -> None:
        """Secondary signal: sustained latency inflation often precedes 429s
        and lets us back off before we are rejected."""
        if p95_ms > baseline_ms * 2.0:
            now = time.monotonic()
            if now - self._last_decrease >= self.cooldown:
                self.limit = max(self.floor, self.limit * 0.9)
                self._last_decrease = now
```

**Integration:** the adaptive limit drives the bucket's refill rate, which is recomputed
whenever the limit changes. Start at the published limit (or 80% of it), let it find the
real ceiling, and log the discovered value — it's genuinely useful information for
capacity planning and for the conversation with your provider.
</details>

## Follow-up Questions

1. Your adaptive limiter settles at 60% of the published limit. Is that right, or is
   something else wrong?
2. How does this interact with the circuit breaker from `retry-and-backoff.md`?
3. Twenty instances each running AIMD independently — do they converge or oscillate?
4. How do you distinguish "we hit our rate limit" from "the provider is degraded"?
5. What would you log so a human can understand the limiter's behavior a week later?

## Senior-Level Discussion

**AIMD converges to fairness among independent clients**, which is why TCP uses it and
why 20 instances running it independently do converge rather than oscillate — provided
each backs off on its own 429s. That's a genuinely useful property and worth knowing.

**The signal you back off on determines what you're actually controlling.** Backing off
on 429s controls your rate against the provider's quota. Backing off on latency controls
against the provider's *capacity*, which is a different and often more useful thing —
you'd rather slow down before you're rejected. Using both is standard practice in
congestion control and unusual in application code.

**The discovered limit is valuable data.** If AIMD consistently settles at 70% of your
published quota, that's either a bug in your estimation or a real discrepancy worth
raising with the provider. Either way, a limiter that reports what it found is more
useful than one that silently copes.

**And the whole stack composes.** Client-side limiting prevents 429s; retry with jitter
handles the ones that slip through; the retry budget caps amplification during partial
degradation; the circuit breaker stops everything during total failure; adaptive limiting
tunes the whole thing to a limit nobody told you. Being able to describe how these four
mechanisms divide the failure space is a strong senior answer in its own right.
