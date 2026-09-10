# Retries

> **This file is the operational policy.** The implementation — decorator, jitter
> functions, circuit breaker, retry budget — is in
> [../11-coding-rounds/retry-and-backoff.md](../11-coding-rounds/retry-and-backoff.md)
> with tested code. Read that one if you need to write it.

The one idea underneath everything here:

> **Retries are load amplification.** They add work precisely when a system is least able
> to handle it.

Good retry design is therefore mostly about deciding when to **stop**.

---

## Three failure regimes

Most systems implement retries for the first regime, nothing for the second, and a circuit
breaker for the third. The gap in the middle is where long incidents live.

| Regime | Symptom | What handles it |
|---|---|---|
| **Isolated failure** | 0.5% errors, random | Per-request retry with backoff |
| **Partial degradation** | 20% errors, sustained | **Retry budget** |
| **Total failure** | 95% errors | Circuit breaker |

### Why the middle one matters

A dependency degrades to a 20% error rate — not enough to trip a circuit breaker set at
50%. With 3 retries per failure:

```
2,000 req/s normal
+ (2,000 × 0.20 × 2 extra attempts)
= 2,800 req/s hitting an already-struggling service
```

A 40% load increase. It degrades further to 35% failures, which drives 3,400 req/s, which
degrades it further. **Retries create a feedback loop that a threshold-based breaker never
reaches.**

The fix is a **retry budget**: cap retries as a fraction of total requests (say 10%)
across the whole service. Under normal error rates it never binds. Under partial
degradation it caps amplification at 1.1× instead of 3×, giving the dependency room to
recover.

---

## What to retry, and what never to

Retry an **allowlist**. Never "everything except a denylist" — that retries your own
programming errors three times before surfacing them.

| Retryable | Not retryable |
|---|---|
| 408 timeout | 400 malformed request |
| 409 conflict | 401 / 403 auth |
| 429 rate limited | 404 not found |
| 500, 502, 503, 504 | 422 validation |
| Connection errors | 501 not implemented |
| Read timeouts | `TypeError`, `KeyError` — your bugs |

Two special cases worth calling out:

**429 is not 503.** A 503 means the service is struggling — back off and hope. A 429 means
*you specifically* exceeded a limit. If you're getting them regularly, retrying treats the
symptom; the cure is a client-side rate limiter. See [rate-limits.md](rate-limits.md).

**Schema violations are permanent.** If a response failed JSON validation, retrying the
identical request usually produces identical invalid JSON. A repair loop that *changes
something* is a different operation from a retry.

---

## Four things that must bound a retry

**1. A deadline, not just an attempt count.**

Three attempts with a 30-second timeout each is a 90-second worst case nobody asked for.
Compute the deadline once and clamp every attempt and every sleep to what remains. Callers
care about elapsed time.

**2. Jitter.**

Without it, N clients that failed at the same instant retry at the same instant, forever,
in waves. **Full jitter** — a random delay in `[0, exponential]` — decorrelates them
completely. `exponential ± 10%` does not; the wave survives.

**3. `Retry-After` when the provider sends it.**

Their header is information; your backoff curve is a guess. But cap it — a
`Retry-After: 600` inside a request handler must become a fast failure, not a ten-minute
sleep holding a worker.

**4. Cancellation.**

`CancelledError` must propagate immediately and never be retried. The caller is gone;
continuing spends money for nobody. Swallowing it also makes tasks uncancellable, which
turns a graceful shutdown into a hang.

---

## Retries and non-idempotent operations

The question behind "should I retry?" is usually "is this safe to repeat?"

- A completion request: usually safe. You get a different answer, which is fine.
- A tool call that sends an email: **not safe.**
- A request that **timed out**: the dangerous case. You don't know whether the server
  processed it.

For anything with side effects, retry only with an idempotency key:

```python
def call_key(request_id, operation, args):
    blob = json.dumps({"op": operation, "args": args}, sort_keys=True)
    return f"{request_id}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"
```

The server (or your own dedupe layer) treats a repeat of the same key as a no-op returning
the original result. Without this, a timeout-then-retry can double-charge a customer.

---

## Retry at one layer only

The composition problem, and it's how partial outages become total ones.

```
service A retries 3× → service B retries 3× → service C
                                              sees 9× load
```

Every layer behaved reasonably. The result is catastrophic. Add a load balancer with its
own retries and it's worse.

**Retry at the outermost layer that has enough context to decide**, and make the inner
layers fail fast. Then audit what you actually have:

```python
# Things that might be retrying without you knowing:
#   - your code
#   - the provider SDK (check its defaults — many retry by default)
#   - your HTTP client
#   - a service mesh or load balancer
#   - the caller's code
```

The provider SDK is the one people miss. If it retries 3× internally and your decorator
retries 3× around it, your "3 attempts" is really 9.

A useful diagnostic: compare your request count against the provider's reported request
count. A gap means someone is retrying that you didn't count.

---

## Circuit breakers: when the right number of retries is zero

When a dependency is definitively down, retrying is a bet you've already lost — and it
costs you your own availability.

The failure mode without one:

```
provider returns 95% errors
  → every request waits for its full retry window
  → connection pool exhausted
  → your health checks fail
  → requests that could have been served from cache also fail
  → you are now down too
```

A circuit breaker fails fast instead: no waiting, no held connections, no exhausted pools.

**Two design points that decide whether it works:**

**Trip on a failure *rate* over a window, with a minimum request volume.** "5 consecutive
failures" trips on noise for a low-traffic service and takes forever on a high-traffic
one. And 2 failures out of 2 requests is not 100% failure, it's insufficient data.

**Bound the probing when it half-opens.** Otherwise the entire backlog rushes a recovering
service and re-opens the circuit immediately.

**Order matters:** the breaker wraps the retry loop, not the other way round. Inside, a
single request's failures could trip a global circuit, and retries would keep firing
against an open one.

```
retry budget  →  circuit breaker  →  retry loop  →  the call
```

---

## Failing fast is only useful if there's somewhere to fail to

A breaker that turns a 30-second timeout into an instant error has improved your resource
usage and not your user's experience.

Design the fallback at the same time as the breaker:

```python
async def complete_with_fallback(prompt):
    try:
        return await primary(prompt)
    except CircuitOpen:
        if cached := await cache.get(prompt):
            return cached
        if secondary_available():
            return await secondary(prompt)
        return degraded_response()      # honest, immediate
```

See [fallbacks.md](fallbacks.md).

---

## Recovery is its own incident

When the dependency comes back, if you do nothing:

- Queued work drains all at once
- Every client's breaker closes simultaneously
- Every user who's been refreshing hits at once

You either re-trip the provider's rate limits or overwhelm yourself. **Many incidents have
two outages: the original and the recovery.**

Ramp: half-open with small probe volume, drain queued work at a controlled rate,
prioritize interactive over batch, and jitter everything.

---

## What to monitor

```python
RETRY_METRICS = [
    "attempts_per_request",        # THE metric — invisible in error rate
    "retry_rate_by_reason",        # 429 vs 503 vs timeout
    "retry_budget_exhausted",      # early warning of partial degradation
    "circuit_state",               # closed / open / half_open
    "deadline_exceeded_vs_attempts_exhausted",   # ran out of time vs tries
    "provider_request_count_delta",  # gap = someone else is retrying
]
```

**Attempts per request is the one that matters.** Retries that eventually succeed do not
appear in your error rate at all. A pipeline at 3.4 attempts instead of 1.06 has tripled
your bill while every dashboard looks healthy.

**Retry budget exhaustion** is a distinct, alertable signal that fires during partial
degradation — well before the error rate is high enough to trip a breaker.

---

## Never retry a deterministic failure

The amplifier that turned one agent incident into 4,200 iterations overnight: a scheduler
retried runs that had hit their iteration limit. That run will hit the limit again,
identically, at full cost.

```python
def should_retry_job(result):
    return result.stop_reason in {"transient_error"}
    # NOT: max_iterations, no_progress, budget_exceeded, validation_failed
    # Those recur identically. Send them to a human.
```

The general rule: **retry outcomes that might differ next time. Everything else goes to a
queue a person looks at.**

---

## Interview questions

**1. "Why shouldn't every error be retried?"**

Four costs: wasted quota, delayed errors the caller needed immediately, load amplification
on a struggling dependency, and duplicate side effects for non-idempotent operations. The
third is the one that turns an incident into a big incident.

**2. "Walk me through a retry storm."**

A calls B calls C. C degrades. B retries 3×, so C sees 3× load. A retries 3×, so C sees
9×. C fails completely. Every layer behaved reasonably; the composition is catastrophic.
Fix: retry at one layer, use retry budgets, add circuit breakers.

**3. "How do you handle 429 versus 503?"**

503 means back off and hope. 429 means you specifically exceeded a limit — honor
`Retry-After`, and treat regular 429s as a sign you need client-side rate limiting rather
than more retries.

**4. "Your request timed out. Do you retry?"**

Only if the operation is idempotent or you have an idempotency key. A timeout means you
don't know whether it was processed, so retrying a non-idempotent operation may
double-execute it.

**5. "What's a retry budget and why isn't a circuit breaker enough?"**

A cap on retries as a fraction of total requests. Circuit breakers handle total failure;
budgets handle partial degradation, where the error rate is too low to trip a breaker but
retries create a feedback loop. Most systems have neither.

**6. "What happens when the dependency recovers?"**

A thundering herd — queued work, closing breakers, and refreshing users all at once. Ramp
recovery with half-open probing, controlled queue drain, and jitter.

---

## What to remember

- Retries amplify load exactly when a system can least handle it.
- Retry an allowlist, never everything-except-a-denylist.
- Bound by a deadline, not an attempt count.
- Full jitter, or your retries synchronize into waves.
- Retry at one layer. Audit what else is retrying, especially the provider SDK.
- Non-idempotent operations need an idempotency key, or don't retry them.
- Circuit breaker wraps the retry loop, not the reverse.
- Failing fast needs somewhere to fail to.
- Recovery is its own incident. Ramp it.
- Never retry a deterministic failure — the scheduler is an amplifier.
- Watch attempts per request. Error rate cannot see retries that succeed.

---

**Next:** [fallbacks.md](fallbacks.md) — where to fail to.
