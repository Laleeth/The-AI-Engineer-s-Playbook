# Coding Round: Async Inference at Scale

The naive version of this problem is a `for` loop. The interview version is everything
that goes wrong when you replace it with `asyncio.gather` on ten thousand items: you
exhaust the provider's rate limit, you hold ten thousand documents in memory, one
failure loses the whole batch, and canceling does nothing because every coroutine is
already scheduled.

Two problems:

1. [Bounded concurrent processing](#problem-1-summarize-10000-documents) — the core
   exercise.
2. [Streaming pipeline with backpressure](#problem-2-an-unbounded-stream) — what changes
   when the input never ends.

Python 3.11+, standard library only (the LLM call is abstracted behind an injectable
async function, which is also how you should design it for testing).

---

## Problem 1: Summarize 10,000 Documents

## Scenario

You have 10,000 documents in object storage that need one-paragraph summaries written
into a database. The current implementation is:

```python
for doc in documents:
    summary = await llm.complete(prompt(doc))   # ~2.5 seconds each
    db.save(doc.id, summary)
```

At 2.5 seconds per document that is **7 hours**. The team needs it in under an hour, it
must be safe to re-run after a failure, and it must not take down the shared LLM quota
that three other services depend on.

## Requirements

Build an async worker system supporting:

- **Concurrency limits** — a hard ceiling on in-flight requests.
- **Retries** — transient failures retried with backoff; permanent failures not.
- **Timeouts** — per-item, so one slow document can't stall the run.
- **Backpressure** — bounded memory regardless of input size.
- **Cancellation** — Ctrl-C stops promptly and cleanly, with no orphaned work.
- **Result collection** — results available as they complete, with failures preserved
  rather than swallowed.
- **Progress and observability** — throughput, failures, and an ETA.
- **Idempotency** — re-running skips work already done.

## Starter Code

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkItem:
    id: str
    payload: Any


@dataclass
class WorkResult:
    id: str
    ok: bool
    value: Any = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0


async def process_all(
    items,                       # iterable or async iterable of WorkItem
    handler,                     # async callable: WorkItem -> Any
    *,
    concurrency: int = 20,
    max_attempts: int = 3,
    timeout_s: float = 60.0,
):
    """Yield WorkResult objects as they complete."""
    raise NotImplementedError
```

## Expected Behavior

```python
async def summarize(item: WorkItem) -> str:
    return await llm.complete(f"Summarize:\n{item.payload}")

count = 0
async for result in process_all(load_documents(), summarize, concurrency=32):
    if result.ok:
        await db.save(result.id, result.value)
    else:
        await deadletter.record(result.id, result.error)
    count += 1

# Properties that must hold:
#   - never more than `concurrency` requests in flight
#   - memory does not grow with the number of items
#   - Ctrl-C stops within a second, no "Task was destroyed" warnings
#   - a single permanently-failing item does not fail the run
#   - re-running skips items already saved
```

---

<details>
<summary>💡 Reveal a hint</summary>

**Why `asyncio.gather` is the wrong primitive here.** `gather` schedules *every*
coroutine immediately. With 10,000 items that means 10,000 coroutine objects, 10,000
copies of the document payload held in memory, and — if your handler doesn't limit
itself — 10,000 concurrent HTTP requests. A semaphore inside the handler fixes the
concurrency but not the memory: the coroutines still all exist, each holding its payload.

**The correct shape is a worker pool over a bounded queue.**

```
producer ──► asyncio.Queue(maxsize=N) ──► worker₁ ─┐
                                          worker₂ ─┼──► results queue ──► caller
                                          ...      │
                                          workerₖ ─┘
```

The bounded queue is the backpressure mechanism: when it's full, the producer *awaits*,
so a lazily-read input source only pulls as fast as workers consume. Memory becomes
`O(concurrency + queue_size)` instead of `O(items)`.

**Three details that are easy to get wrong:**

1. **Results must also be bounded.** If you accumulate results in a list, you're back to
   `O(items)` memory. Yield them through a second bounded queue.
2. **Shutdown needs sentinels.** Workers blocked on `queue.get()` don't know the input
   ended. Push one `None` per worker, or use a task-group with cancellation.
3. **Cancellation must reach the workers.** `asyncio.CancelledError` propagating out of
   your generator has to cancel the producer and every worker, and then *await* them, or
   you get "Task was destroyed but it is pending" warnings and possibly leaked
   connections.
</details>

---

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""Bounded-concurrency async processing with retries, backpressure and
clean cancellation.

The design is a worker pool over a bounded queue. Everything else — retries,
timeouts, idempotency — layers on top of that shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

log = logging.getLogger("async_inference")


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass
class WorkItem:
    id: str
    payload: Any


@dataclass
class WorkResult:
    id: str
    ok: bool
    value: Any = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0


class PermanentFailure(Exception):
    """Raise from a handler when retrying cannot help."""


@dataclass
class Stats:
    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    retried: int = 0
    skipped: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def completed(self) -> int:
        return self.succeeded + self.failed + self.skipped

    @property
    def rate(self) -> float:
        elapsed = time.monotonic() - self.started_at
        return self.completed / elapsed if elapsed > 0 else 0.0

    def eta_seconds(self, total: int | None) -> float | None:
        if not total or self.rate == 0:
            return None
        return max(0.0, (total - self.completed) / self.rate)


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

_SENTINEL = object()


async def process_all(
    items: Iterable[WorkItem] | AsyncIterator[WorkItem],
    handler: Callable[[WorkItem], Awaitable[Any]],
    *,
    concurrency: int = 20,
    max_attempts: int = 3,
    timeout_s: float = 60.0,
    queue_size: int | None = None,
    is_done: Callable[[WorkItem], Awaitable[bool]] | None = None,
    stats: Stats | None = None,
) -> AsyncIterator[WorkResult]:
    """Process items with bounded concurrency, yielding results as they complete.

    Args:
        items:       sync or async iterable. Read lazily — pass a generator for
                     large inputs so the whole input never lives in memory.
        handler:     async function doing the work. Raise PermanentFailure to
                     skip retries.
        concurrency: maximum simultaneous in-flight handler calls.
        max_attempts: attempts per item, including the first.
        timeout_s:   per-attempt timeout.
        queue_size:  input queue depth. Defaults to 2x concurrency, which keeps
                     workers fed without buffering the world.
        is_done:     optional idempotency check; items already done are skipped
                     without invoking the handler.

    Yields:
        WorkResult in completion order (NOT input order).
    """
    stats = stats or Stats()
    queue_size = queue_size or concurrency * 2

    in_q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    out_q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

    async def produce() -> None:
        """Feed the input queue, then push one sentinel per worker."""
        try:
            if hasattr(items, "__aiter__"):
                async for item in items:            # type: ignore[union-attr]
                    await in_q.put(item)            # blocks when full: backpressure
                    stats.submitted += 1
            else:
                for item in items:                  # type: ignore[union-attr]
                    await in_q.put(item)
                    stats.submitted += 1
        finally:
            # Always signal termination, even if the producer raised, so workers
            # don't hang forever on an empty queue.
            for _ in range(concurrency):
                await in_q.put(_SENTINEL)

    async def work(worker_id: int) -> None:
        while True:
            item = await in_q.get()
            if item is _SENTINEL:
                return
            result = await _run_one(
                item, handler,
                max_attempts=max_attempts,
                timeout_s=timeout_s,
                is_done=is_done,
                stats=stats,
            )
            await out_q.put(result)                 # blocks if consumer is slow

    producer = asyncio.create_task(produce(), name="producer")
    workers = [
        asyncio.create_task(work(i), name=f"worker-{i}")
        for i in range(concurrency)
    ]

    async def close_output() -> None:
        """When every worker has exited, close the result stream."""
        await asyncio.gather(*workers, return_exceptions=True)
        await out_q.put(_SENTINEL)

    closer = asyncio.create_task(close_output(), name="closer")

    try:
        while True:
            result = await out_q.get()
            if result is _SENTINEL:
                break
            yield result
    finally:
        # Runs on normal completion, on caller-side break, and on cancellation.
        # Cancel everything, then AWAIT it: not awaiting is what produces
        # "Task was destroyed but it is pending" and leaked connections.
        for task in (producer, closer, *workers):
            task.cancel()
        await asyncio.gather(producer, closer, *workers, return_exceptions=True)
        # Surface any producer error that wasn't a cancellation.
        if producer.done() and not producer.cancelled():
            exc = producer.exception()
            if exc is not None:
                raise exc


async def _run_one(
    item: WorkItem,
    handler: Callable[[WorkItem], Awaitable[Any]],
    *,
    max_attempts: int,
    timeout_s: float,
    is_done: Callable[[WorkItem], Awaitable[bool]] | None,
    stats: Stats,
) -> WorkResult:
    started = time.monotonic()

    if is_done is not None:
        try:
            if await is_done(item):
                stats.skipped += 1
                return WorkResult(item.id, ok=True, value=None, attempts=0,
                                  duration_ms=0.0)
        except Exception as exc:              # noqa: BLE001
            # An idempotency check that fails should not fail the item; do the
            # work again rather than skipping it.
            log.warning("idempotency check failed for %s: %s", item.id, exc)

    last_error = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            value = await asyncio.wait_for(handler(item), timeout=timeout_s)
            stats.succeeded += 1
            return WorkResult(
                item.id, ok=True, value=value, attempts=attempt,
                duration_ms=(time.monotonic() - started) * 1000,
            )

        except asyncio.CancelledError:
            # Do not retry a canceled item and do not swallow the exception:
            # swallowing CancelledError makes the whole pool uncancellable.
            raise

        except PermanentFailure as exc:
            stats.failed += 1
            return WorkResult(
                item.id, ok=False, error=f"permanent: {exc}", attempts=attempt,
                duration_ms=(time.monotonic() - started) * 1000,
            )

        except asyncio.TimeoutError:
            last_error = f"timeout after {timeout_s}s"

        except Exception as exc:              # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_attempts:
            stats.retried += 1
            delay = min(30.0, 0.5 * (2 ** (attempt - 1)))
            await asyncio.sleep(random.uniform(0, delay))   # full jitter

    stats.failed += 1
    return WorkResult(
        item.id, ok=False, error=last_error, attempts=max_attempts,
        duration_ms=(time.monotonic() - started) * 1000,
    )


# --------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------

async def example() -> None:
    async def load_documents() -> AsyncIterator[WorkItem]:
        """Lazy input. Because process_all reads this as fast as the queue
        allows and no faster, the full corpus never lives in memory."""
        for i in range(10_000):
            await asyncio.sleep(0)            # simulate async fetch
            yield WorkItem(id=f"doc-{i}", payload=f"content of document {i}")

    async def already_saved(item: WorkItem) -> bool:
        return False                          # replace with a real DB lookup

    async def summarize(item: WorkItem) -> str:
        await asyncio.sleep(0.05)             # replace with the LLM call
        if "13" in item.id:
            raise PermanentFailure("document is malformed")
        return f"summary of {item.id}"

    stats = Stats()
    failures: list[WorkResult] = []

    async for result in process_all(
        load_documents(), summarize,
        concurrency=32, max_attempts=3, timeout_s=30.0,
        is_done=already_saved, stats=stats,
    ):
        if result.ok:
            pass                              # persist here
        else:
            failures.append(result)

        if stats.completed % 500 == 0:
            eta = stats.eta_seconds(10_000)
            log.info("progress %d/%d rate=%.1f/s eta=%.0fs failures=%d",
                     stats.completed, 10_000, stats.rate, eta or -1, stats.failed)

    log.info("done: %d ok, %d failed, %d skipped, %d retries",
             stats.succeeded, stats.failed, stats.skipped, stats.retried)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example())
```

</details>

---

<details>
<summary>📖 Reveal the explanation</summary>

### The shape, and why

**Worker pool over a bounded queue** rather than `gather` + semaphore. Both bound
concurrency; only the queue bounds *memory*. With `gather`, all 10,000 coroutine
frames exist simultaneously, each holding a reference to its payload. With a queue of
depth 64, at most 64 payloads are in flight. For 10,000 short documents the difference
is annoying; for 10,000 PDFs it is the difference between running and OOM.

**Backpressure is the `await` on a full queue.** This is worth stating precisely: when
`in_q` is full, `await in_q.put(item)` suspends the producer. If the producer is reading
from object storage lazily, it stops fetching. The whole pipeline self-regulates to the
speed of the slowest stage, with no explicit rate control.

**The output queue is bounded too.** A common half-solution bounds the input and then
accumulates results in a list. If each result is a summary that's fine; if each result
is a 2 MB extraction, it isn't. Bounding both means a slow consumer applies backpressure
all the way to the producer.

**Sentinels for shutdown.** Workers blocked on `in_q.get()` have no way to learn that
the input is exhausted. Pushing one sentinel per worker guarantees each worker sees
exactly one and exits. The `finally` in `produce()` matters: if the producer raises
halfway through, the sentinels are still pushed, so workers don't hang.

**Cancellation is handled in `finally`, and it awaits.** Three cases reach that block:
normal completion, the caller breaking out of the `async for`, and cancellation of the
consuming task. In all three, tasks are canceled *and awaited*. Canceling without
awaiting is the classic bug — the event loop reports "Task was destroyed but it is
pending", and in a real system the underlying HTTP connections are not released.

**But `break` alone does not run that `finally` promptly** — and this is the subtlety
worth knowing. When a caller breaks out of `async for` over an async *generator*, Python
does not run the generator's cleanup synchronously; it schedules `aclose()` to run later
(at GC time, or at loop shutdown via `shutdown_asyncgens`). Measured on the code above,
breaking at item 20 leaves the producer task alive for an indeterminate period. The fix is
the caller's, and it should be documented as part of the contract:

```python
import contextlib

async with contextlib.aclosing(process_all(items, handler)) as stream:
    async for result in stream:
        ...
        if enough:
            break            # cleanup now runs deterministically at __aexit__
```

With `aclosing`, the leaked-task count on early break is zero; without it, it is not.
If you cannot rely on callers doing this, don't expose an async generator — return an
object with an explicit `close()`, or take a callback. A candidate who raises this
unprompted has debugged a real async generator leak.

**`CancelledError` is re-raised, never caught by `except Exception`.** In Python 3.8+
`CancelledError` inherits from `BaseException` specifically so that `except Exception`
doesn't swallow it — but `except asyncio.TimeoutError` and friends still need to be
ordered carefully, and an explicit `except asyncio.CancelledError: raise` documents the
intent.

**Retry classification is delegated to the handler** via `PermanentFailure`. The pool
doesn't know what a 400 means; the handler does. This keeps the pool generic and puts
the domain knowledge where it belongs.

**Idempotency is a pluggable check, not an assumption.** `is_done` is called before the
handler. Note the deliberate choice in the error path: if the check itself fails, we do
the work rather than skipping it. Doing work twice is usually cheaper than silently
skipping it.

**Results are yielded in completion order, not input order** — and the docstring says so
loudly, because callers who assume ordering write subtly wrong code. If a caller needs
ordering, they should collect and sort, and pay the memory cost knowingly.

### Complexity

- Time: `O(n / concurrency × per_item_latency)`, plus retry overhead.
- Memory: `O(concurrency + queue_size)` — independent of `n`. This is the property the
  design exists to provide.
- Task count: `concurrency + 2` regardless of input size.

For 10,000 documents at 2.5s each with concurrency 32: `10,000 × 2.5 / 32 ≈ 780s ≈ 13
minutes`, comfortably under the hour, assuming the provider allows 32 concurrent
requests — which is the next question.

</details>

---

## Production Improvements

- **Rate limiting, not just concurrency.** Concurrency limits in-flight requests;
  provider limits are usually requests-per-minute *and* tokens-per-minute. 32 concurrent
  requests at 0.4s each is 80 req/s, which may blow an 8,000/min limit. See
  `rate-limiter.md`; the pool should admit work through a token-aware limiter, not just
  a worker count.
- **Adaptive concurrency.** Start conservative, increase while latency and error rate
  are stable, back off on 429s. A fixed number is either too low (slow) or too high
  (rate-limited) as conditions change.
- **Checkpointing.** Persist completed IDs so a crashed run resumes rather than restarts.
  With `is_done` backed by a durable store, this is nearly free.
- **Dead-letter queue** for permanently failed items with enough context to reprocess.
- **Priority lanes.** If interactive work shares this pool, batch work must yield to it —
  a single weighted queue, not two pools (see
  `../12-senior-scenarios/architecture-tradeoffs.md`).
- **Cost accounting per item**, so a runaway batch is visible in dollars, not just in
  request counts.
- **Graceful shutdown on SIGTERM**: stop accepting new items, let in-flight work finish
  within a grace period, checkpoint, exit.
- **Metrics**: in-flight gauge, throughput, latency histogram, retry counter, failure
  counter by error class.

## Edge Cases

| Case | Correct behavior |
|---|---|
| Input iterator raises halfway | Sentinels still pushed (the `finally`); in-flight work completes; error surfaced to the caller |
| Handler hangs forever | `asyncio.wait_for` cancels it at `timeout_s`; counted as a retryable failure |
| Handler swallows `CancelledError` | Pool becomes uncancellable — document this as a handler contract |
| Consumer stops reading (`break`) | Cleanup runs, but **only promptly if the caller wraps the generator in `contextlib.aclosing`** — a bare `break` defers it to GC (see the explanation) |
| Every item fails permanently | Run completes with all failures recorded; does not raise |
| `concurrency=1` | Degenerates to sequential; still correct |
| Empty input | Yields nothing, terminates cleanly |
| Duplicate item IDs | Allowed by the pool; idempotency is the handler's concern |
| Result larger than available memory | Bounded output queue limits how many are buffered, not how big one is — cap payload size in the handler |
| SIGINT during a run | `KeyboardInterrupt` cancels the consuming task, `finally` runs, tasks awaited |

## Follow-up Questions

1. **What happens when you hit the provider's rate limit?** Your concurrency is 32 and
   you're getting 429s. Walk through the change.
2. **How does memory behave** if each document is 5 MB? What would you change?
3. **Ordering.** A caller needs results in input order. What are the options and what
   does each cost?
4. **Partial failure.** 40 of 10,000 items fail permanently. What do you do, and what do
   you tell the person who asked for the batch?
5. **Idempotency.** The run crashes at item 6,000. What guarantees do you have about
   items 5,990–6,000?
6. **Two runs at once.** Someone starts the job twice. What happens, and how would you
   prevent it?
7. **Testing.** How do you test the cancellation path and the backpressure property?
8. **A caller writes `async for r in process_all(...)` and `break`s early.** What exactly
   happens to the producer and the workers, and when? What would you change about the
   API to make the wrong usage impossible?

## Senior-Level Discussion

**Concurrency is not the same as rate.** This is the point most candidates miss. A
concurrency limit bounds *simultaneous* requests; a rate limit bounds requests *per unit
time*. With fast responses, low concurrency can still exceed a rate limit; with slow
responses, high concurrency may stay under it. If the constraint you're managing is the
provider's quota, concurrency is the wrong knob — and if the constraint is your own
memory or connection pool, it's the right one. Usually you need both.

**Ordering has a cost and it should be paid deliberately.** Completion-order results
allow constant memory. Input-order results require buffering everything that completed
out of order, which in the worst case is the entire batch. Callers who want ordering
should say so and accept the memory. A library that silently preserves order is
silently unbounded.

**Idempotency is a property of the handler, not the pool.** The pool can offer an
`is_done` hook, but true idempotency means the *write* is safe to repeat — an upsert
keyed on `(item_id, prompt_version, model_version)` rather than an append. Including the
prompt and model version in the key is what makes reprocessing after a prompt change
correct rather than duplicative. Candidates who mention this have run a real batch
pipeline.

**Partial failure is a product question.** 40 failures out of 10,000 is 99.6% success,
which is either fine or unacceptable depending on what the data is used for. In a
document-extraction pipeline feeding a legal review, a silently missing document is a
liability. The pool's job is to make failures *visible and reprocessable*, not to decide
whether they matter — but the engineer's job is to make sure someone decides.

**The retry/rate-limit interaction is where batch jobs become incidents.** 10,000 items
failing with 429s and retrying with exponential backoff generates a load pattern that
keeps the provider rate-limited, so more items fail, so more retries. Without jitter,
the retries synchronize into waves. Without a circuit breaker, the run never recovers.
The correct behavior under sustained 429s is to *slow the whole pool down*, not to retry
individual items harder — which is an argument for adaptive concurrency driven by the
error rate.

**Why not use a job queue (Celery, RQ, a cloud queue)?** For a one-off backfill,
in-process is simpler and fine. For recurring work, work that must survive process
restarts, or work that needs distribution across machines, a real queue is the right
answer — and the design above is essentially a single-process version of it. Knowing
when to stop hand-rolling is part of the answer.

---

## Problem 2: An Unbounded Stream

## Scenario

The backfill worked. Now the same processing must run continuously: documents arrive on
a queue at a variable rate averaging 40/second, with bursts to 300/second. Processing
takes 2.5 seconds each. The stream never ends.

## Requirements

- Sustain the average rate; survive the bursts.
- Bounded memory and bounded latency under burst.
- At-least-once processing with idempotent writes.
- Graceful shutdown: finish in-flight work, don't lose queued work.
- Observable lag: how far behind are we?

<details>
<summary>💡 Reveal a hint</summary>

Do the arithmetic first: 40 items/second × 2.5 seconds each = **100 concurrent
requests** just to keep up with the average. Bursts to 300/second cannot be absorbed by
concurrency (that would need 750 concurrent requests); they must be absorbed by the
*queue*, which means accepting latency during bursts rather than scaling instantly.

That's the central design decision and it should be explicit: **during a burst, do you
add latency or drop work?** For document processing, latency. For an interactive
feature, possibly drop or degrade.

The second change from Problem 1: with an unbounded stream there is no "done." Shutdown
becomes a first-class operation, and *lag* — the age of the oldest unprocessed item —
becomes the primary health metric, because throughput alone tells you nothing about
whether you're falling behind.
</details>

<details>
<summary>✅ Reveal the design</summary>

```python
"""Continuous processing with graceful shutdown and lag monitoring.

Differences from the batch version:
  - no terminal condition; shutdown is explicit and signal-driven
  - lag (age of oldest unprocessed item) is the health metric
  - acknowledgement happens after successful processing (at-least-once)
"""

import asyncio
import contextlib
import signal
import time


class StreamProcessor:
    def __init__(self, source, handler, *, concurrency=100, queue_size=500):
        self.source = source                 # async iterator of (item, ack_token)
        self.handler = handler
        self.concurrency = concurrency
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._shutdown = asyncio.Event()
        self._oldest_enqueued_at: float | None = None
        self._tasks: list[asyncio.Task] = []

    @property
    def lag_seconds(self) -> float:
        """Age of the oldest item still waiting. The number to alert on."""
        if self._oldest_enqueued_at is None:
            return 0.0
        return time.monotonic() - self._oldest_enqueued_at

    async def _produce(self) -> None:
        async for item, ack in self.source:
            if self._shutdown.is_set():
                break
            enqueued_at = time.monotonic()
            if self.queue.empty():
                self._oldest_enqueued_at = enqueued_at
            # Blocks when full: this is what pushes backpressure onto the
            # source, which for most queue clients means we stop fetching.
            await self.queue.put((item, ack, enqueued_at))

    async def _work(self) -> None:
        while True:
            try:
                item, ack, enqueued_at = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                if self._shutdown.is_set() and self.queue.empty():
                    return
                continue

            try:
                await self.handler(item)
                await ack()                  # ack ONLY after success:
                                             # at-least-once, never at-most-once
            except Exception:                # noqa: BLE001
                pass                         # leave unacked -> redelivered
            finally:
                self.queue.task_done()
                self._oldest_enqueued_at = (
                    None if self.queue.empty() else self._oldest_enqueued_at
                )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._shutdown.set)

        producer = asyncio.create_task(self._produce(), name="producer")
        self._tasks = [
            asyncio.create_task(self._work(), name=f"worker-{i}")
            for i in range(self.concurrency)
        ]

        await self._shutdown.wait()

        # Graceful shutdown: stop intake, drain what is queued, bounded by a
        # grace period. Anything still unacked will be redelivered.
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
```

**Key differences and why:**

- **Acknowledge after success**, never before. This gives at-least-once delivery: a crash
  mid-processing means redelivery, which is safe *only if* the handler's write is
  idempotent. State that contract explicitly.
- **Lag, not throughput, is the health metric.** Throughput at 40/s looks fine whether
  you're keeping up or 4 hours behind. Lag tells you which. Alert on lag.
- **Bursts are absorbed by the queue**, which converts a throughput problem into a
  latency problem — the right trade for asynchronous work, and a decision to make
  consciously.
- **Shutdown has a grace period.** Drain, but bounded; a worker stuck on a hung request
  must not block deployment forever.
- **Workers poll with a timeout** rather than blocking indefinitely, so they can observe
  the shutdown flag. The alternative is sentinels, as in Problem 1; the timeout approach
  is simpler when shutdown can happen at any moment.
</details>

## Follow-up Questions

1. Lag has been growing for two hours. Walk through the diagnosis and the options.
2. Your handler is idempotent, but the redelivered item now has a *newer* prompt version.
   Is that still idempotent?
3. How would you scale this horizontally across 12 processes? What breaks?
4. A poison item fails every time and is redelivered forever. Design the fix.
5. During deploys you lose 30 seconds of throughput. Is that acceptable? How would you
   reduce it?
6. How do you decide `concurrency=100` empirically rather than by arithmetic?

## Senior-Level Discussion

**Lag is the metric that matters in a streaming system**, and it's the one most teams
don't have. Throughput, error rate, and latency can all look healthy while the backlog
grows without bound. Lag — the age of the oldest unprocessed item — is the only signal
that answers "are we keeping up?" It's also the signal that translates directly into a
business statement: "documents uploaded now will be searchable in 4 hours."

**At-least-once plus idempotent writes is almost always the right contract.**
Exactly-once is expensive and usually illusory across a system boundary. Acknowledge
after the write, make the write an upsert on a deterministic key, and accept occasional
duplicate processing. The cost is a small amount of wasted compute; the alternative —
acknowledging before processing — silently loses data on every crash, and nobody notices
until an audit.

**Horizontal scaling changes the concurrency math.** Twelve processes at concurrency 100
is 1,200 concurrent requests against a shared provider quota. Per-process limits must
become a *distributed* limit, or you'll rate-limit yourself the moment you scale out.
This is the natural bridge to `rate-limiter.md`, and it's the question interviewers use
to see whether a candidate thinks about single-process solutions in a multi-process world.
