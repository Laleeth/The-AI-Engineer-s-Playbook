# Coding Round: Retry and Backoff

Everyone can write a retry loop. The interview is about the decisions inside it: what
you retry, what you refuse to retry, how long you wait, when you give up, and how you
avoid turning a provider's bad minute into your own bad hour.

The single most important idea in this file: **retries are load amplification.** A
system under stress that retries harder is a system that stays under stress longer.
Every design decision below follows from taking that seriously.

Three problems:

1. [A correct retry decorator](#problem-1-a-correct-retry-decorator)
2. [Circuit breaking](#problem-2-when-retrying-makes-it-worse)
3. [Retry budgets and adaptive behavior](#problem-3-retry-budgets)

Python 3.11+, standard library.

---

## Problem 1: A Correct Retry Decorator

## Scenario

Your team calls an LLM API from a dozen places. Each call site has its own hand-rolled
retry logic, and they're all subtly different: one retries 400s, one has no jitter, one
retries forever, one catches `CancelledError`. Consolidate them.

## Requirements

- **Exponential backoff** with configurable base and cap.
- **Jitter** — and be able to explain which kind and why.
- **Retryable vs. non-retryable** classification, extensible per call site.
- **Maximum attempts** *and* a maximum total elapsed time.
- **`Retry-After` support** — provider guidance overrides your computed delay.
- **Cancellation** must propagate immediately.
- **Observability** — every retry visible, with the reason.
- Works for sync and async callables.

## Starter Code

```python
def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    max_elapsed: float = 60.0,
    retry_on: tuple[type[Exception], ...] = (),
    give_up_on: tuple[type[Exception], ...] = (),
):
    """Decorator adding retry-with-backoff to a callable."""
    raise NotImplementedError
```

## Expected Behavior

```python
@retry(max_attempts=4, base_delay=0.5, max_elapsed=20.0)
async def call_model(prompt: str) -> str:
    ...

# - HTTP 400 raises immediately, no retry, no delay
# - HTTP 429 with Retry-After: 3 waits ~3s (not the computed backoff)
# - HTTP 503 backs off 0.5s, 1s, 2s, each with jitter
# - Total time never exceeds max_elapsed, even with attempts remaining
# - asyncio.CancelledError propagates on the first raise
# - Every retry emits a log line with attempt, delay and cause
```

---

<details>
<summary>💡 Reveal a hint</summary>

**Four things separate a correct implementation from the common one:**

1. **`max_attempts` alone is not a bound.** Three attempts with a 30-second timeout each
   is a 90-second worst case. You need a *deadline* — computed once at the start — and
   every sleep and every attempt must respect it. Callers care about elapsed time, not
   attempt counts.

2. **Jitter is not optional and the kind matters.** With no jitter, N clients that failed
   at the same instant retry at the same instant, forever, in waves. **Full jitter** —
   `uniform(0, exp)` — decorrelates them completely. "Equal jitter" (`exp/2 +
   uniform(0, exp/2)`) retries sooner on average while still decorrelating. Both are
   fine; `exp ± 10%` is not, because the wave survives.

3. **`Retry-After` beats your math.** Your backoff curve is a guess about when the
   service will recover. The header is the service telling you. But: if `Retry-After` is
   300 seconds and your deadline is 20, you must fail fast rather than sleep — sleeping
   past a deadline inside a request handler is how you exhaust a thread pool.

4. **Classification must be extensible.** The decorator can't know what a 429 means in
   your domain. Take a predicate, not just an exception tuple, so a call site can say
   "retry this HTTP error but not that one."

**And the bug that catches everyone:** `except Exception` around an `await` catches
`asyncio.CancelledError` in older Python and, more importantly, catching broad exception
types and retrying means you retry programming errors — `TypeError`, `KeyError` — three
times before surfacing them. Retry an allowlist, not everything-except-a-denylist.
</details>

---

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""Retry with backoff, for sync and async callables.

Design principles:
  - a deadline bounds everything, not an attempt count
  - retry an allowlist; never retry by default
  - provider guidance (Retry-After) overrides computed backoff
  - cancellation is never retried and never swallowed
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

log = logging.getLogger("retry")


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

class RetryDecision(Protocol):
    def __call__(self, exc: BaseException, attempt: int) -> "Verdict": ...


@dataclass(frozen=True)
class Verdict:
    retry: bool
    reason: str
    retry_after: float | None = None    # seconds, from provider guidance


def never(exc: BaseException, attempt: int) -> Verdict:
    return Verdict(False, f"{type(exc).__name__} is not retryable")


def http_policy(
    retryable_status: frozenset[int] = frozenset({408, 409, 425, 429,
                                                  500, 502, 503, 504}),
) -> RetryDecision:
    """Classification for HTTP-shaped errors.

    Note what is NOT retried: 400 (malformed request), 401/403 (auth), 404,
    422 (validation). Retrying these wastes quota and delays the error the
    caller needs to see. 501 is excluded deliberately: it will never work.
    """
    def decide(exc: BaseException, attempt: int) -> Verdict:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status is None:
            # Not an HTTP error. Transport-level failures are retryable;
            # everything else is a bug and should surface immediately.
            if isinstance(exc, (ConnectionError, TimeoutError,
                                asyncio.TimeoutError, OSError)):
                return Verdict(True, f"transport: {type(exc).__name__}")
            return Verdict(False, f"{type(exc).__name__} is not retryable")

        if status in retryable_status:
            hdrs = getattr(exc, "headers", None) or {}
            retry_after = _parse_retry_after(hdrs.get("retry-after")
                                             or hdrs.get("Retry-After"))
            return Verdict(True, f"HTTP {status}", retry_after)
        return Verdict(False, f"HTTP {status} is permanent")

    return decide


def _parse_retry_after(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        # HTTP-date form. Parse it properly in production; returning None
        # falls back to computed backoff, which is safe.
        return None


# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------

def full_jitter(attempt: int, base: float, cap: float) -> float:
    """delay ~ U(0, min(cap, base * 2^attempt)).

    Full jitter maximally decorrelates retries across clients, which is the
    entire point when 500 callers fail simultaneously. It sometimes retries
    very quickly, which is fine — the expected delay still grows exponentially.
    """
    return random.uniform(0.0, min(cap, base * (2 ** attempt)))


def equal_jitter(attempt: int, base: float, cap: float) -> float:
    """Half fixed, half random. Retries sooner on average than full jitter
    while still decorrelating. Prefer when latency matters more than
    smoothing."""
    exp = min(cap, base * (2 ** attempt))
    return exp / 2 + random.uniform(0.0, exp / 2)


def decorrelated_jitter(previous: float, base: float, cap: float) -> float:
    """delay ~ U(base, previous * 3), capped. Grows based on the previous
    delay rather than the attempt number; smooths well under sustained load."""
    return min(cap, random.uniform(base, max(base, previous * 3)))


# --------------------------------------------------------------------------
# Result of a retry sequence
# --------------------------------------------------------------------------

class RetryExhausted(Exception):
    def __init__(self, attempts: int, elapsed: float, last: BaseException) -> None:
        super().__init__(
            f"gave up after {attempts} attempts in {elapsed:.2f}s: {last}"
        )
        self.attempts = attempts
        self.elapsed = elapsed
        self.last = last


class DeadlineExceeded(RetryExhausted):
    """Ran out of time before running out of attempts."""


# --------------------------------------------------------------------------
# The decorator
# --------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    *,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    max_elapsed: float = 60.0,
    policy: RetryDecision | None = None,
    jitter: Callable[[int, float, float], float] = full_jitter,
    on_retry: Callable[[int, float, BaseException, str], None] | None = None,
):
    """Retry a callable with exponential backoff.

    Args:
        max_attempts: total attempts including the first.
        base_delay:   first backoff interval, doubling thereafter.
        max_delay:    cap on any single sleep.
        max_elapsed:  HARD ceiling on total wall time, including sleeps. This,
                      not max_attempts, is what callers actually care about.
        policy:       classifier. Defaults to conservative HTTP handling.
        jitter:       backoff strategy.
        on_retry:     hook(attempt, delay, exc, reason) for metrics.
    """
    policy = policy or http_policy()

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                deadline = time.monotonic() + max_elapsed
                last: BaseException | None = None

                for attempt in range(max_attempts):
                    try:
                        return await fn(*args, **kwargs)

                    except asyncio.CancelledError:
                        # Never retried, never swallowed. The caller is gone;
                        # continuing spends money for nobody.
                        raise

                    except BaseException as exc:            # noqa: BLE001
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                        last = exc
                        verdict = policy(exc, attempt)
                        if not verdict.retry:
                            raise
                        if attempt == max_attempts - 1:
                            break

                        delay = _next_delay(verdict, attempt, base_delay,
                                            max_delay, jitter)
                        remaining = deadline - time.monotonic()
                        if delay >= remaining:
                            # Sleeping would blow the deadline. Fail now with
                            # a distinguishable error rather than sleeping and
                            # then failing anyway.
                            raise DeadlineExceeded(
                                attempt + 1, max_elapsed - remaining, exc
                            ) from exc

                        _emit(attempt, delay, exc, verdict.reason, fn, on_retry)
                        await asyncio.sleep(delay)

                raise RetryExhausted(
                    max_attempts, max_elapsed - (deadline - time.monotonic()), last
                ) from last

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            deadline = time.monotonic() + max_elapsed
            last: BaseException | None = None

            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:                # noqa: BLE001
                    last = exc
                    verdict = policy(exc, attempt)
                    if not verdict.retry:
                        raise
                    if attempt == max_attempts - 1:
                        break

                    delay = _next_delay(verdict, attempt, base_delay,
                                        max_delay, jitter)
                    remaining = deadline - time.monotonic()
                    if delay >= remaining:
                        raise DeadlineExceeded(
                            attempt + 1, max_elapsed - remaining, exc
                        ) from exc

                    _emit(attempt, delay, exc, verdict.reason, fn, on_retry)
                    time.sleep(delay)

            raise RetryExhausted(
                max_attempts, max_elapsed - (deadline - time.monotonic()), last
            ) from last

        return sync_wrapper

    return decorator


def _next_delay(verdict, attempt, base, cap, jitter) -> float:
    """Provider guidance wins; otherwise compute backoff.

    Retry-After is still capped: a provider asking for a 10-minute wait inside a
    request handler is a request to fail, not to sleep.
    """
    if verdict.retry_after is not None:
        return min(verdict.retry_after, cap)
    return jitter(attempt, base, cap)


def _emit(attempt, delay, exc, reason, fn, hook) -> None:
    log.warning(
        "retrying %s attempt=%d delay=%.3f reason=%s error=%s",
        getattr(fn, "__qualname__", "fn"), attempt + 1, delay, reason, exc,
    )
    if hook is not None:
        try:
            hook(attempt, delay, exc, reason)
        except Exception:                                   # noqa: BLE001
            log.exception("on_retry hook failed")           # never break retries
```

**Usage:**

```python
@retry(
    max_attempts=4,
    base_delay=0.5,
    max_delay=20.0,
    max_elapsed=25.0,
    on_retry=lambda a, d, e, r: metrics.increment("llm.retry", tags={"reason": r}),
)
async def complete(prompt: str) -> str:
    resp = await http.post("/v1/complete", json={"prompt": prompt})
    resp.raise_for_status()
    return resp.json()["text"]
```

</details>

---

<details>
<summary>📖 Reveal the explanation</summary>

### The decisions

**The deadline is the real bound.** `max_elapsed` is computed once and checked before
every sleep. Without it, `max_attempts=4` with a 30-second per-attempt timeout is a
2-minute worst case that no caller signed up for. The `delay >= remaining` check
prevents the specific pathology of sleeping 20 seconds and then failing anyway — you
fail at second 5 instead, with `DeadlineExceeded` so the caller can tell "we ran out of
time" from "we ran out of attempts."

**Allowlist classification.** `http_policy` retries a specific set of statuses and a
specific set of transport exceptions. Everything else — including `TypeError` from a bug
in your own code — raises immediately. The alternative, retrying everything except a
denylist, means every programming error takes three attempts and 3.5 seconds to surface,
which is maddening in development and expensive in production.

**Why 501 is excluded but 500 is included.** 500 is "something went wrong," which may be
transient. 501 is "this operation is not implemented," which will never work. Small
distinctions like this are what a shared policy is *for* — no call site should have to
rediscover them.

**Full jitter as the default.** When a provider returns 429 to 500 concurrent clients,
they all fail within milliseconds of each other. With `exp ± 10%` they retry within
milliseconds of each other again, producing a synchronized wave that keeps the provider
saturated. `uniform(0, exp)` spreads them across the whole interval. The cost is that
some retries happen almost immediately; the benefit is that the herd is gone. For
latency-sensitive paths, `equal_jitter` is the reasonable compromise, and offering both
rather than hardcoding one is the right library design.

**`Retry-After` is honored but capped.** The provider knows more than your curve does.
But a `Retry-After: 600` inside a request handler must not become a 10-minute sleep
holding a connection and a worker thread — cap it, and let the deadline check turn it
into a fast failure.

**`CancelledError` is explicitly re-raised before the general handler.** This is the
most common bug in retry code. Catching it and retrying makes tasks uncancellable, which
turns a graceful shutdown into a hang and a canceled request into continued spend.
`KeyboardInterrupt` and `SystemExit` get the same treatment.

**The `on_retry` hook is wrapped in its own try/except.** A metrics call that throws must
not break the retry logic. Small, but it's the kind of thing that causes an outage during
an outage.

**Sync and async share the policy, not the loop.** Duplicating the loop is uglier than
abstracting it but far easier to read and debug, and the two loops genuinely differ
(`await` vs. call, `asyncio.sleep` vs. `time.sleep`). Cleverness here costs more than it
saves.

### Complexity

Worst-case wall time is `min(max_elapsed, Σ delays + Σ attempt durations)`. With
`base=0.5, cap=30, attempts=5`, expected sleep under full jitter is roughly
`0.25 + 0.5 + 1 + 2 = 3.75s`. The deadline dominates in practice, which is the point.

</details>

---

## Production Improvements

- **A circuit breaker** (Problem 2). Retries alone make an outage worse.
- **A retry budget** (Problem 3) — cap retries as a *fraction of total requests* rather
  than per-request, so a broad failure can't multiply your load.
- **Hedged requests** for latency-critical paths: send a second request if the first
  hasn't responded by p95, take whichever returns first. Costs extra requests, cuts tail
  latency dramatically.
- **Idempotency keys** so a retry of a request that actually succeeded upstream doesn't
  double-charge or double-execute.
- **Per-dependency configuration.** A 30-second retry window is right for a batch job and
  absurd for an autocomplete. Defaults should be per-dependency, not global.
- **Metrics**: retries by reason, retry rate as a fraction of requests, `DeadlineExceeded`
  vs. `RetryExhausted` split, and — the one that catches invisible problems —
  *attempts per logical request*, which does not show up in error rate at all.
- **Deadline propagation.** Pass the remaining budget down through service calls so a
  downstream service doesn't start work that its caller has already given up on.

## Edge Cases

| Case | Correct behavior |
|---|---|
| `max_attempts=1` | Calls once, never retries; classification still runs |
| `max_elapsed` shorter than one attempt | First attempt runs to completion; failure raises `DeadlineExceeded` |
| `Retry-After: 3600` | Capped at `max_delay`, then the deadline check fails fast |
| `Retry-After` in HTTP-date format | Falls back to computed backoff rather than crashing |
| Exception has no status attribute | Transport errors retried; everything else raises |
| Function raises `CancelledError` | Propagates immediately, no retry, no logging noise |
| `on_retry` hook raises | Logged, retry continues |
| Clock jumps backwards | `time.monotonic()` is immune — this is why it's used, not `time.time()` |
| Retryable error on the final attempt | `RetryExhausted` with the original as `__cause__` |
| Nested retry decorators | Multiplies attempts (3×3=9). Almost always a bug — check for it in review |

## Follow-up Questions

1. **Why shouldn't every error be retried?** Give three concrete costs.
2. **400 vs. 429 vs. 500** — how do you treat each, and why does 429 differ from 503?
3. **Idempotency.** Your request timed out but the server processed it. What now?
4. **Thundering herd.** 5,000 clients get 429 simultaneously. Walk through what your
   implementation does and why it's better than no jitter.
5. **Retry storms.** How can retries turn a partial outage into a total one?
6. **Nested retries.** Your client retries, the SDK retries, the load balancer retries.
   What's the real attempt count and how do you find out?
7. **Testing.** How do you test backoff timing without making the test suite slow?

## Senior-Level Discussion

**Why not retry everything?** Four costs, and a candidate should name at least three:

1. **Wasted quota.** Retrying a 400 consumes rate-limit budget that a legitimate request
   needed.
2. **Delayed errors.** The caller waits `Σ delays` before learning something that was
   knowable immediately. In a request path, that's user-visible latency for a guaranteed
   failure.
3. **Load amplification.** During a partial outage, retries multiply the load on a
   struggling dependency, extending the outage. This is the one that turns an incident
   into a big incident.
4. **Duplicate side effects.** For non-idempotent operations, a retry after a timeout may
   perform the action twice.

**Idempotency is the question behind retries.** "Is it safe to retry?" is really "is this
operation idempotent, or do I have an idempotency key?" A GET is safe. A completion
request is usually safe (you get a different answer, which is fine). A tool call that
posts a message is not. A request that timed out is the dangerous case: you don't know
whether the server processed it. Without an idempotency key, retrying is a bet.

**Retry storms, concretely.** Service A calls B calls C. C degrades. B retries 3×, so C
sees 3× load. A retries 3×, so C sees 9× load. C, already struggling, now fails
completely. Every layer behaved reasonably; the composition is catastrophic. The
defenses: retry at *one* layer (usually the outermost that has enough context), use retry
budgets, and add circuit breakers so retries stop entirely when a dependency is clearly
down.

**429 is different from 503 and should be treated differently.** A 503 means the service
is struggling — back off and hope. A 429 means *you specifically* exceeded a limit, and
the fix is not just to wait but to slow down your admission rate. If you're getting 429s
regularly, retry logic is treating a symptom; the cure is a client-side rate limiter (see
`rate-limiter.md`).

**Testing timing without slow tests.** Inject the sleep function and the clock. The
decorator should accept them as parameters (or read them from a module-level indirection)
so tests can assert on the *sequence of computed delays* rather than waiting for them.
Testing that jitter produces values in the right range, and that the deadline check
fires, is fast; testing that `time.sleep(30)` actually sleeps is not a useful test.

---

## Problem 2: When Retrying Makes It Worse

## Scenario

Your LLM provider degrades: 95% of requests return 503. Your service has retries with
exponential backoff. Within four minutes:

- Request latency rises to the full retry window on every call.
- Your connection pool is exhausted by requests waiting to retry.
- Health checks fail because workers are all blocked.
- Requests that could have been served from cache also fail, because no worker is free.

Your retry logic converted a dependency outage into a service outage.

## Requirements

Add a circuit breaker. Explain the state machine and every parameter.

<details>
<summary>💡 Reveal a hint</summary>

The insight: **when a dependency is definitively down, the correct number of retries is
zero.** Retrying is a bet that the failure is transient and isolated. When 95% of
requests fail, that bet is clearly lost, and continuing to place it costs you your own
availability.

A circuit breaker is a shared, stateful observation across requests: *this dependency is
down; don't bother.* Three states:

- **CLOSED** — normal. Count failures.
- **OPEN** — fail immediately without calling. This is what protects *you*: no waiting,
  no held connections, no exhausted pools.
- **HALF_OPEN** — after a cooldown, let a small number of probes through. Success closes
  the circuit; failure re-opens it.

The parameters that matter and are usually chosen badly:

- **Failure threshold as a *rate* over a window**, not a raw count. "5 consecutive
  failures" trips on a low-traffic service from noise and takes forever on a high-traffic
  one.
- **A minimum request volume** before the rate is meaningful. 2 failures out of 2
  requests is not 100% failure, it's insufficient data.
- **Cooldown** long enough for recovery, short enough to notice it. Typically tens of
  seconds.
- **Bounded half-open concurrency** — otherwise the whole backlog rushes the recovering
  service and re-opens the circuit immediately.
</details>

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""Circuit breaker with a sliding window and bounded probing."""

from __future__ import annotations

import asyncio
import enum
import time
from collections import deque
from dataclasses import dataclass, field


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    """Raised instead of calling a dependency that is known to be down.

    This is the point: it is fast, it holds no resources, and it lets the
    caller choose a fallback rather than waiting for a timeout."""
    def __init__(self, name: str, retry_at: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry after {retry_at:.1f}s")
        self.retry_at = retry_at


@dataclass
class CircuitBreaker:
    name: str
    failure_rate_threshold: float = 0.5      # trip above 50% failures
    minimum_requests: int = 20               # ...but only with enough signal
    window_seconds: float = 30.0             # sliding observation window
    cooldown_seconds: float = 20.0           # OPEN -> HALF_OPEN delay
    half_open_max_calls: int = 3             # bounded probing
    half_open_successes_to_close: int = 3

    state: State = State.CLOSED
    _events: deque[tuple[float, bool]] = field(default_factory=deque)  # (t, ok)
    _opened_at: float = 0.0
    _half_open_inflight: int = 0
    _half_open_successes: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _failure_rate(self, now: float) -> tuple[float, int]:
        self._prune(now)
        total = len(self._events)
        if total == 0:
            return 0.0, 0
        failures = sum(1 for _, ok in self._events if not ok)
        return failures / total, total

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    async def before_call(self) -> None:
        now = time.monotonic()
        async with self._lock:
            if self.state is State.OPEN:
                elapsed = now - self._opened_at
                if elapsed < self.cooldown_seconds:
                    raise CircuitOpen(self.name, self.cooldown_seconds - elapsed)
                # Cooldown elapsed: probe.
                self.state = State.HALF_OPEN
                self._half_open_inflight = 0
                self._half_open_successes = 0

            if self.state is State.HALF_OPEN:
                if self._half_open_inflight >= self.half_open_max_calls:
                    # Bounded probing. Without this, the entire backlog hits a
                    # recovering service at once and re-opens the circuit.
                    raise CircuitOpen(self.name, self.cooldown_seconds)
                self._half_open_inflight += 1

    async def on_success(self) -> None:
        now = time.monotonic()
        async with self._lock:
            self._events.append((now, True))
            if self.state is State.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_successes_to_close:
                    self.state = State.CLOSED
                    self._events.clear()      # fresh window after recovery

    async def on_failure(self) -> None:
        now = time.monotonic()
        async with self._lock:
            self._events.append((now, False))

            if self.state is State.HALF_OPEN:
                # One failed probe is enough: the dependency is not ready.
                self.state = State.OPEN
                self._opened_at = now
                self._half_open_inflight = 0
                return

            rate, total = self._failure_rate(now)
            if total >= self.minimum_requests and rate >= self.failure_rate_threshold:
                self.state = State.OPEN
                self._opened_at = now


# --------------------------------------------------------------------------
# Composition: breaker OUTSIDE retry
# --------------------------------------------------------------------------

async def call_with_protection(breaker: CircuitBreaker, fn, *args, **kwargs):
    """Order matters.

    The breaker wraps the retry loop, not the other way around. If the breaker
    were inside, each retry would consult it and a single request could trip
    it on its own; worse, retries would continue while the circuit was open.
    Outside, an open circuit means the request fails instantly with no retries
    at all — which is exactly the behavior that protects the caller.
    """
    await breaker.before_call()
    try:
        result = await fn(*args, **kwargs)      # fn already has @retry applied
    except CircuitOpen:
        raise
    except BaseException:
        await breaker.on_failure()
        raise
    else:
        await breaker.on_success()
        return result


# --------------------------------------------------------------------------
# What to do when the circuit is open
# --------------------------------------------------------------------------

async def complete_with_fallback(breaker, primary, secondary, cache, prompt):
    """An open circuit is not an error; it is a signal to take a different
    path. Failing fast is only valuable if there is somewhere to fail to."""
    try:
        return await call_with_protection(breaker, primary, prompt)
    except CircuitOpen:
        cached = await cache.get(prompt)
        if cached is not None:
            return cached
        if secondary is not None:
            return await secondary(prompt)
        raise
```

</details>

## Follow-up Questions

1. Your circuit is open but 5% of requests were succeeding. You're now rejecting all of
   them. Is that right?
2. How do you choose `cooldown_seconds`? What are the costs of too short and too long?
3. Twelve application servers each have their own breaker. Should they share state?
4. The dependency recovers but your circuit stays open because no traffic is probing it.
   What's wrong with the design?
5. How do you test a circuit breaker?
6. Should the circuit be per-provider, per-endpoint, or per-model? Defend it.

## Senior-Level Discussion

**Order of composition is the thing to get right.** Breaker outside retry. If the breaker
is inside the retry loop, a single request's three failures can trip a global circuit,
and retries keep firing against an open circuit. Outside, an open circuit means the
request costs microseconds and holds nothing.

**Failing fast is only useful if there's a fallback.** A breaker that turns a 30-second
timeout into an instant error has improved your resource usage and not your user's
experience. The value is realized when the open circuit routes to a cache, a secondary
provider, or an honest degraded response. Design the fallback at the same time as the
breaker.

**Per-process breakers are usually right, and the reason is worth knowing.** Shared state
across 12 servers gives faster, more accurate tripping — and adds a distributed
dependency to your failure-handling path, which is exactly the code that must work when
things are broken. Local breakers converge quickly enough in practice because all
instances observe the same degradation.

**Granularity matters.** One breaker for an entire provider means a problem with one
model takes down your access to all of them. Per-endpoint or per-model breakers isolate
better, at the cost of slower tripping on each (less traffic per breaker, so
`minimum_requests` takes longer to reach). A common answer: per (provider, model), with a
provider-level breaker as a coarse backstop.

---

## Problem 3: Retry Budgets

## Scenario

Your service handles 2,000 requests/second. A dependency degrades to a 20% error rate —
not enough to trip the circuit breaker at a 50% threshold. With 3 retries per failure,
the dependency now receives:

```
2,000 normal + (2,000 × 0.20 × up to 2 extra attempts) ≈ 2,800 req/s
```

A 40% load increase on a service that is already struggling. It degrades further to 35%
failures, which drives 3,400 req/s, which drives it further. **Retries create a positive
feedback loop that a circuit breaker's threshold never reaches.**

## Requirements

Implement a retry budget: cap retries as a fraction of total requests, globally.

<details>
<summary>💡 Reveal a hint</summary>

The gap in the previous two designs: per-request retry limits bound *one request's*
behavior but say nothing about aggregate load. The circuit breaker handles catastrophic
failure; nothing handles the middle ground of partial degradation, which is where the
feedback loop lives.

A **retry budget** is a global constraint: "retries may be at most 10% of total
requests." Under normal conditions (1% failures) it's never binding. Under partial
degradation it caps the amplification at 1.1× instead of 3×, so the dependency gets a
chance to recover instead of being held down.

Implementation: a token bucket that fills at `rate × request_rate` and is consumed by
each retry. When empty, failures surface without retrying.

This is the mechanism that turns "retry three times" into "retry three times *if the
system can afford it*," and it's the piece most hand-rolled retry code lacks.
</details>

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""Global retry budget: bound retries as a fraction of total traffic."""

from __future__ import annotations

import threading
import time


class RetryBudget:
    """Token bucket where every request contributes budget and every retry
    consumes it.

    ratio=0.1 means: across the whole service, retries may be at most 10% of
    primary requests. Under normal error rates this never binds. Under partial
    degradation it caps amplification at 1.1x instead of 3x, which is the
    difference between a dependency recovering and a dependency being held
    down by its clients.
    """

    def __init__(self, ratio: float = 0.1, min_per_second: float = 5.0,
                 ttl_seconds: float = 10.0) -> None:
        self.ratio = ratio
        self.min_per_second = min_per_second     # floor for low-traffic services
        self.ttl = ttl_seconds
        self._tokens = 0.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _decay(self, now: float) -> None:
        """Tokens expire so that a quiet period doesn't bank retry capacity
        for a later burst."""
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens *= max(0.0, 1.0 - elapsed / self.ttl)
            self._tokens += self.min_per_second * elapsed * self.ratio
            self._last = now

    def deposit(self) -> None:
        """Call once per PRIMARY request (not per attempt)."""
        with self._lock:
            now = time.monotonic()
            self._decay(now)
            self._tokens = min(self._tokens + self.ratio, self.min_per_second * self.ttl)

    def try_withdraw(self) -> bool:
        """Call before each RETRY. False means the budget is exhausted and the
        failure should surface immediately."""
        with self._lock:
            now = time.monotonic()
            self._decay(now)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def available(self) -> float:
        with self._lock:
            self._decay(time.monotonic())
            return self._tokens


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

async def call_with_budget(budget: RetryBudget, breaker, fn, *args, **kwargs):
    """Full protection stack, outermost first:

        retry budget  →  circuit breaker  →  retry  →  the call

    Each layer handles a different failure regime:
      - retry           : isolated, transient failures
      - retry budget    : partial degradation (the feedback loop)
      - circuit breaker : total failure
    """
    budget.deposit()
    await breaker.before_call()

    attempt = 0
    while True:
        try:
            result = await fn(*args, **kwargs)
            await breaker.on_success()
            return result
        except CircuitOpen:
            raise
        except BaseException as exc:                        # noqa: BLE001
            await breaker.on_failure()
            attempt += 1
            if attempt >= 3 or not _retryable(exc):
                raise
            if not budget.try_withdraw():
                # The system as a whole cannot afford this retry.
                raise RetryBudgetExhausted(
                    f"retry budget exhausted; surfacing {type(exc).__name__}"
                ) from exc
            await asyncio.sleep(full_jitter(attempt - 1, 0.5, 20.0))


class RetryBudgetExhausted(Exception):
    """Distinguishable from other failures so it can be alerted on separately:
    a rising rate means a dependency is degrading, before it degrades enough to
    trip a breaker."""
```

**The three layers and what each is for:**

| Regime | Symptom | Mechanism |
|---|---|---|
| Isolated transient failure | 0.5% error rate | Per-request retry with backoff |
| Partial degradation | 20% error rate, sustained | **Retry budget** — caps amplification |
| Total failure | 95% error rate | Circuit breaker — stop calling entirely |

Most systems implement the first and third and skip the middle, which is why partial
degradations are so often the ones that turn into long incidents.
</details>

## Follow-up Questions

1. How do you choose the ratio? What does 10% mean operationally?
2. The budget is per-process. Twelve processes means 12× the retries. Does that matter?
3. A single high-value request must be retried even when the budget is exhausted. How?
4. How does the retry budget interact with the circuit breaker? Can they conflict?
5. What do you alert on: budget exhaustion, or the error rate that caused it?

## Senior-Level Discussion

**The three regimes framing is the takeaway.** Isolated failures, partial degradation,
and total failure need different mechanisms, and a system with only per-request retries
handles exactly one of them. Being able to name the middle regime — and explain why a
circuit breaker's threshold doesn't catch it — is the strongest signal available in this
problem.

**Retry budgets make failures visible earlier.** `RetryBudgetExhausted` is a distinct,
alertable signal that fires during partial degradation, well before the error rate is
high enough to trip a breaker. That's a monitoring benefit as much as a load-shedding
one.

**Per-process vs. global.** A per-process budget with 12 processes allows 12× the
absolute retries, but the *ratio* is preserved per process, so the amplification factor —
which is what causes the feedback loop — is still capped at 1.1×. That's the property
that matters, and it's why per-process is usually adequate. Global budgets need shared
state, which is a dependency in your failure-handling path.

**Priority within a budget.** When the budget is scarce, spend it on what matters:
interactive requests over batch, paying customers over free tier, first retries over
third. A single undifferentiated budget spends itself on whatever fails first, which is
often the batch job. Multiple budgets, or a priority-aware withdrawal, is the refinement.

**The general principle across all three problems:** every retry mechanism is a form of
load shedding in reverse — it *adds* load precisely when the system is least able to
handle it. Good retry design is therefore mostly about the conditions under which you
*stop* retrying: permanent errors, deadlines, open circuits, exhausted budgets, and
cancellation. A candidate who talks more about stopping than about retrying has
understood the problem.
