# Coding Round: The LLM Client

Interviewers use this problem because it looks like plumbing and turns out to be a
design interview. Everyone can call an HTTP endpoint. Very few people can design a
client that streams, retries correctly, times out at the right granularity, cancels
cleanly, and doesn't turn a provider blip into an outage.

Two problems in this file:

1. [A production LLM client](#problem-1-a-production-llm-client) — the core exercise.
2. [Multi-provider abstraction](#problem-2-supporting-a-second-provider) — the follow-on
   that separates good abstraction from leaky abstraction.

All code is Python 3.11+, standard library plus `httpx`. Nothing here depends on a
specific vendor SDK — that's deliberate, since the point is to understand what an SDK
does for you.

---

## Problem 1: A Production LLM Client

## Scenario

You're building the shared LLM client for a company where six teams will call it. It
sits in the request path of a customer-facing product.

Today it wraps one provider's chat-completions endpoint. It must be good enough that
teams stop writing their own — which means it has to handle the things people get wrong
when they write their own: streaming that breaks halfway, 429s, timeouts that are too
coarse, and requests that keep running after the user has closed the tab.

## Requirements

Implement a client supporting:

- **Streaming** — token-by-token delivery to the caller, with the option to accumulate.
- **Retries** — on transient failures only, with exponential backoff and jitter.
- **Timeouts** — separate connect, time-to-first-token, and total-request budgets.
- **Structured output** — request JSON conforming to a schema, and validate it.
- **Rate limits** — respect `Retry-After`; don't hammer a rate-limited provider.
- **Logging** — one structured record per request with tokens, cost, latency, outcome.
- **Request IDs** — a client-generated ID on every request, propagated and logged.
- **Cancellation** — the caller can abandon a request and resources are released.

Non-goals for the first pass: multi-provider support, caching, batching. (Follow-ups.)

## Starter Code

```python
from dataclasses import dataclass


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    request_id: str
    latency_ms: float
    attempts: int


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        raise NotImplementedError

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_schema: dict | None = None,
    ) -> CompletionResult:
        """Non-streaming completion."""
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """Async iterator yielding text deltas, then a final CompletionResult."""
        raise NotImplementedError
```

## Expected Behavior

```python
client = LLMClient(api_key=KEY, base_url=URL, model="some-model")

# Simple completion
result = await client.complete([Message("user", "Say hello")])
assert result.text
assert result.output_tokens > 0

# Streaming
async for event in client.stream([Message("user", "Count to five")]):
    if isinstance(event, str):
        print(event, end="")          # a token delta
    else:
        final = event                  # CompletionResult

# Structured output
schema = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["pos", "neg"]}},
    "required": ["sentiment"],
}
r = await client.complete([Message("user", "I love this")], response_schema=schema)
json.loads(r.text)["sentiment"]        # must parse and validate

# Cancellation releases the connection
task = asyncio.create_task(client.complete(messages))
await asyncio.sleep(0.1)
task.cancel()                          # no leaked connection, no retry storm

# A 400 is not retried; a 429 waits for Retry-After; a 503 backs off
```

---

<details>
<summary>💡 Reveal a hint (try the problem first)</summary>

Three design decisions carry most of the difficulty:

1. **What is retryable?** Not "any exception." `400` (bad request), `401`, `404`, and
   `422` are permanent — retrying them wastes quota and delays the error the caller
   needs. `429`, `500`, `502`, `503`, `504`, connection errors, and read timeouts are
   transient. This classification belongs in one place.

2. **Where does the timeout live?** A single `timeout=30` is wrong for streaming: a
   long generation legitimately takes 60 seconds while producing tokens the whole time,
   but a stream that produces nothing for 30 seconds is dead. You need at least
   *connect*, *time-to-first-token*, and *total* budgets, and ideally an
   inter-token stall timeout.

3. **Can you retry a stream?** Only before the first token. Once you've yielded text to
   the caller, a retry would duplicate output. Track whether you've emitted anything;
   after that, a failure is a failure. This is the single most-missed requirement in
   this problem.

Two more that separate a good answer from a great one:

- The **total-request budget must bound the retries**, not each attempt. Three attempts
  with a 30s timeout each is a 90-second worst case, which the caller didn't ask for.
- **`Retry-After` overrides your backoff.** If the provider tells you when to come back,
  obey it — your exponential backoff is a guess and their header is information.
</details>

---

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""A production-shaped LLM client.

Design notes are inline. The point of the exercise is the error, timeout and
cancellation handling, not the JSON shapes, which vary by provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

log = logging.getLogger("llm_client")


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass
class Message:
    role: str
    content: str


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    request_id: str
    latency_ms: float
    attempts: int


@dataclass
class Timeouts:
    connect: float = 5.0
    first_token: float = 20.0     # nothing at all from the provider by now => dead
    stall: float = 15.0           # gap between tokens once streaming has started
    total: float = 120.0          # hard ceiling INCLUDING all retries


class LLMError(Exception):
    """Base for client errors."""


class PermanentError(LLMError):
    """Do not retry: the request itself is wrong."""


class TransientError(LLMError):
    """Retryable: the provider or network failed in a way that may not recur."""
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceeded(LLMError):
    """The total time budget was exhausted."""


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------

# Status codes worth retrying. Everything else is the caller's problem.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def classify(status: int, headers: httpx.Headers, body: str) -> LLMError:
    retry_after = None
    if "retry-after" in headers:
        try:
            retry_after = float(headers["retry-after"])
        except ValueError:
            retry_after = None  # HTTP-date form; ignore for brevity

    if status in _RETRYABLE_STATUS:
        return TransientError(f"HTTP {status}: {body[:200]}", retry_after)
    return PermanentError(f"HTTP {status}: {body[:200]}")


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 20.0) -> float:
    """Exponential backoff with full jitter.

    Full jitter (uniform in [0, exp]) rather than exp +/- noise: it decorrelates
    retries across many clients, which is what actually prevents a thundering herd.
    """
    exp = min(cap, base * (2 ** attempt))
    return random.uniform(0, exp)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeouts: Timeouts | None = None,
        max_attempts: int = 3,
        price_per_1k_input: float = 0.0,
        price_per_1k_output: float = 0.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.timeouts = timeouts or Timeouts()
        self.max_attempts = max_attempts
        self.price_in = price_per_1k_input
        self.price_out = price_per_1k_output

        # One shared connection pool. Creating a client per request is the most
        # common performance bug in code like this: it forfeits connection reuse
        # and TLS session resumption, which can double effective latency.
        self._http = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                connect=self.timeouts.connect,
                read=self.timeouts.first_token,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # ---------------- non-streaming ----------------

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_schema: dict | None = None,
        request_id: str | None = None,
    ) -> CompletionResult:
        request_id = request_id or f"req_{uuid.uuid4().hex[:16]}"
        deadline = time.monotonic() + self.timeouts.total
        started = time.monotonic()
        payload = self._payload(messages, max_tokens, temperature,
                                response_schema, stream=False)

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded(f"{request_id}: total budget exhausted")

            try:
                resp = await self._http.post(
                    "/chat/completions",
                    json=payload,
                    headers={"X-Request-Id": request_id},
                    timeout=httpx.Timeout(
                        connect=self.timeouts.connect,
                        read=min(remaining, self.timeouts.total),
                        write=10.0,
                        pool=5.0,
                    ),
                )
                if resp.status_code >= 400:
                    raise classify(resp.status_code, resp.headers, resp.text)

                data = resp.json()
                result = self._parse(data, request_id, started, attempt + 1)
                if response_schema is not None:
                    _validate_json(result.text, response_schema)
                self._log(result, ok=True)
                return result

            except PermanentError as exc:
                self._log_failure(request_id, attempt + 1, exc, retried=False)
                raise

            except (TransientError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt == self.max_attempts - 1:
                    break
                delay = getattr(exc, "retry_after", None) or backoff_delay(attempt)
                # Never sleep past the deadline; fail fast instead.
                if time.monotonic() + delay > deadline:
                    break
                log.warning("retrying %s attempt=%d delay=%.2f err=%s",
                            request_id, attempt + 1, delay, exc)
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                # Propagate immediately. Never retry a cancelled request: the
                # caller has gone away and continuing spends money for nobody.
                self._log_failure(request_id, attempt + 1, "cancelled", retried=False)
                raise

        self._log_failure(request_id, self.max_attempts, last_error, retried=True)
        raise TransientError(f"{request_id}: exhausted retries: {last_error}")

    # ---------------- streaming ----------------

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        request_id: str | None = None,
    ) -> AsyncIterator[str | CompletionResult]:
        """Yield text deltas, then a final CompletionResult.

        Retries are permitted ONLY before the first delta has been yielded.
        Once the caller has seen output, a retry would duplicate it.
        """
        request_id = request_id or f"req_{uuid.uuid4().hex[:16]}"
        deadline = time.monotonic() + self.timeouts.total
        started = time.monotonic()
        payload = self._payload(messages, max_tokens, temperature, None, stream=True)

        emitted = False
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            if emitted:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded(f"{request_id}: total budget exhausted")

            chunks: list[str] = []
            finish_reason = "unknown"
            usage = {"input_tokens": 0, "output_tokens": 0}

            try:
                async with self._http.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                    headers={"X-Request-Id": request_id},
                    timeout=httpx.Timeout(
                        connect=self.timeouts.connect,
                        read=self.timeouts.first_token,
                        write=10.0,
                        pool=5.0,
                    ),
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode(errors="replace")
                        raise classify(resp.status_code, resp.headers, body)

                    async for delta, meta in self._iter_sse(resp, request_id):
                        if meta:
                            finish_reason = meta.get("finish_reason", finish_reason)
                            usage.update(meta.get("usage", {}))
                            continue
                        emitted = True
                        chunks.append(delta)
                        yield delta

                text = "".join(chunks)
                result = CompletionResult(
                    text=text,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"] or _approx_tokens(text),
                    finish_reason=finish_reason,
                    request_id=request_id,
                    latency_ms=(time.monotonic() - started) * 1000,
                    attempts=attempt + 1,
                )
                self._log(result, ok=True)
                yield result
                return

            except PermanentError as exc:
                self._log_failure(request_id, attempt + 1, exc, retried=False)
                raise

            except (TransientError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if emitted:
                    # Partial output already delivered. We cannot retry without
                    # duplicating; surface a partial-failure to the caller.
                    self._log_failure(request_id, attempt + 1, exc, retried=False)
                    raise TransientError(
                        f"{request_id}: stream broke after {len(chunks)} chunks: {exc}"
                    )
                if attempt == self.max_attempts - 1:
                    break
                delay = getattr(exc, "retry_after", None) or backoff_delay(attempt)
                if time.monotonic() + delay > deadline:
                    break
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                self._log_failure(request_id, attempt + 1, "cancelled", retried=False)
                raise

        raise TransientError(f"{request_id}: exhausted retries: {last_error}")

    async def _iter_sse(self, resp: httpx.Response, request_id: str):
        """Parse server-sent events, enforcing an inter-token stall timeout.

        httpx's read timeout applies per read; we additionally guard against a
        stream that stays open but stops producing, which a read timeout alone
        may not catch if the server sends keepalives.
        """
        last_event = time.monotonic()
        async for line in resp.aiter_lines():
            now = time.monotonic()
            if now - last_event > self.timeouts.stall:
                raise TransientError(f"{request_id}: stream stalled")
            last_event = now

            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                # A truncated frame. Skipping is safer than failing the whole
                # stream, but count it — a rising rate means a real problem.
                log.warning("%s: undecodable SSE frame", request_id)
                continue

            choice = (event.get("choices") or [{}])[0]
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                yield delta, None
            meta = {}
            if choice.get("finish_reason"):
                meta["finish_reason"] = choice["finish_reason"]
            if event.get("usage"):
                u = event["usage"]
                meta["usage"] = {
                    "input_tokens": u.get("prompt_tokens", 0),
                    "output_tokens": u.get("completion_tokens", 0),
                }
            if meta:
                yield "", meta

    # ---------------- helpers ----------------

    def _payload(self, messages, max_tokens, temperature, schema, *, stream):
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _parse(self, data, request_id, started, attempts) -> CompletionResult:
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return CompletionResult(
            text=choice["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            request_id=request_id,
            latency_ms=(time.monotonic() - started) * 1000,
            attempts=attempts,
        )

    def _cost(self, r: CompletionResult) -> float:
        return (r.input_tokens / 1000 * self.price_in
                + r.output_tokens / 1000 * self.price_out)

    def _log(self, r: CompletionResult, *, ok: bool) -> None:
        log.info(
            "llm_call",
            extra={"llm": {
                "request_id": r.request_id,
                "model": self.model,
                "ok": ok,
                "attempts": r.attempts,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "finish_reason": r.finish_reason,
                "latency_ms": round(r.latency_ms, 1),
                "cost_usd": round(self._cost(r), 6),
            }},
        )

    def _log_failure(self, request_id, attempts, err, *, retried) -> None:
        log.error(
            "llm_call_failed",
            extra={"llm": {
                "request_id": request_id,
                "model": self.model,
                "ok": False,
                "attempts": attempts,
                "error": str(err),
                "retried": retried,
            }},
        )


def _approx_tokens(text: str) -> int:
    """Rough fallback when the provider omits usage. Never use for billing."""
    return max(1, len(text) // 4)


def _validate_json(text: str, schema: dict) -> None:
    """Validate a structured response.

    In production use a real JSON Schema validator (jsonschema). This shows the
    shape: parse, then validate, and fail with a PermanentError because a schema
    violation will not be fixed by retrying the same request.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PermanentError(f"response was not valid JSON: {exc}") from exc

    for key in schema.get("required", []):
        if key not in obj:
            raise PermanentError(f"response missing required field {key!r}")
```

</details>

---

<details>
<summary>📖 Reveal the explanation</summary>

### Why each decision was made

**One `AsyncClient`, shared.** Creating a new HTTP client per request is the most common
performance defect in hand-rolled LLM clients. It forfeits connection pooling and TLS
session reuse; on a 200ms round trip you can add 100–300ms per call. Owning the pool
also lets you bound concurrency (`max_connections`), which is the first line of defense
against a fan-out that exhausts file descriptors.

**Error classification lives in one function.** `classify()` is the only place that
decides retryable vs. permanent. The alternative — `except Exception: retry` — retries
a malformed request three times, tripling the latency the caller experiences before
getting the error they needed immediately, and burning quota.

**The total budget bounds retries, not each attempt.** `deadline` is computed once and
every attempt's timeout is clamped to what remains. Without this, `max_attempts=3` with
a 30s read timeout is a 90+ second worst case that no caller asked for. The
`time.monotonic() + delay > deadline` check also prevents sleeping past the deadline
just to make a doomed attempt.

**`Retry-After` beats computed backoff.** The provider's header is information; your
exponential curve is a guess. Honouring it also keeps you in good standing with rate
limiters that penalize clients who ignore it.

**Full jitter.** `random.uniform(0, exp)` rather than `exp ± noise`. When 400 clients
get 429s simultaneously, decorrelating their retries is the entire point; partial jitter
leaves a visible retry wave.

**Streaming retries stop at the first token.** The `emitted` flag is the crux. Retrying
after yielding output duplicates it, and the caller — who may have already rendered
those tokens — has no way to undo. After first output, a break is a break, and the
honest thing is to raise with the partial context.

**Separate first-token and stall timeouts.** A long generation is legitimate; a stream
that goes quiet is not. One number cannot express both. The stall check inside
`_iter_sse` catches the case where the connection stays open (keepalives, proxies) but
no content arrives.

**Cancellation propagates immediately and is never retried.** `asyncio.CancelledError`
must not be swallowed by an `except Exception` — a very common bug that makes tasks
uncancellable. The `async with self._http.stream(...)` context manager guarantees the
connection is released on cancellation.

**Schema violations are permanent, not transient.** Retrying the same prompt that
produced invalid JSON usually produces invalid JSON again. If you want a repair loop,
it must change something (add a corrective message), which is a different operation from
a retry.

**One structured log line per request** with tokens, cost, attempts, latency, and
finish reason. This single record is what makes every incident in
`../12-senior-scenarios/production-incidents.md` diagnosable: attempt counts reveal
invisible retry storms, `finish_reason` reveals truncation, and per-request cost reveals
budget anomalies.

### Complexity

Time is dominated by network I/O. Retry logic adds `O(max_attempts)` in the worst case,
bounded by the total deadline. Memory for a streamed response is `O(output_tokens)`
because we accumulate chunks to build the final result — worth noting, since for very
long outputs a caller who only needs the stream shouldn't pay for accumulation. An
option to skip accumulation is a reasonable extension.

</details>

---

## Production Improvements

Things a real client needs that the reference implementation deliberately omits:

- **Circuit breaker.** After N consecutive failures, stop sending and fail fast for a
  cooldown. Without this, a provider outage means every request waits for its full
  timeout and your own thread/connection pools saturate — you take yourself down
  alongside the provider. This is the single most valuable addition.
- **Concurrency limiter / bulkhead.** A semaphore capping in-flight requests, so one
  runaway caller can't consume the whole pool.
- **Token-aware rate limiting** (see `rate-limiter.md`) — request counting is not enough
  when limits are token-based.
- **Prefix-cache-friendly payload construction** — a helper that enforces stable-prefix
  ordering, since a client is the right place to make the fast path the default.
- **Idempotency keys** so a retry that actually succeeded upstream isn't double-charged.
- **Metrics** in addition to logs: histograms for latency and tokens, counters for
  errors by class, gauges for in-flight requests.
- **Tracing** — an OpenTelemetry span per attempt with token and cost attributes.
- **Pluggable serialization** for providers with different wire formats.
- **A test double** — an in-process fake provider that can be told to return 429s, break
  mid-stream, and stall, so callers can test their own error handling.

## Edge Cases

| Case | Correct behavior |
|---|---|
| Provider returns 200 with an empty `choices` array | Treat as permanent; do not retry a well-formed but empty response without investigating |
| Stream ends without `[DONE]` | Accept accumulated text but flag `finish_reason: "unknown"`; count it |
| `finish_reason == "length"` | Not an error, but the caller must know the output is truncated — never silently accept |
| `Retry-After` in HTTP-date format | Parse it, or fall back to computed backoff (do not crash) |
| Provider returns 429 with `Retry-After: 300` | Honour it, or fail fast if it exceeds your budget — don't sleep 5 minutes inside a request |
| Caller cancels mid-stream | `CancelledError` propagates, connection released by the context manager, no retry |
| Two callers pass the same `request_id` | Allowed; logging must not assume uniqueness for correctness |
| Response JSON parses but violates the schema | `PermanentError` — retrying the identical request rarely helps |
| Very long single SSE line | Bounded by the HTTP client's limits; consider an explicit max frame size |
| `max_tokens` larger than the model's remaining context | Provider returns 400; classified permanent, surfaced immediately |

## Follow-up Questions

1. **What happens when the provider returns 429?** Walk through your code path,
   including what `Retry-After: 120` does to a request with a 30-second budget.
2. **The stream breaks halfway through.** What does the caller see, and what are the
   options for a client API that makes this survivable for the *application*?
3. **How would you make this synchronous?** What breaks, and why is async the right
   default here?
4. **Twenty application servers share one rate limit.** Does anything in this client
   help? What has to move elsewhere?
5. **A caller reports that cancelling a request doesn't stop the spend.** Diagnose.
6. **How do you test this?** Specifically: how do you test the stall timeout, the
   mid-stream break, and the cancellation path?
7. **Cost logging is wrong by 8% versus the invoice.** Where would you look?

## Senior-Level Discussion

**On retry semantics and idempotency.** LLM completions are not idempotent in the HTTP
sense — a retry produces a different response. That's usually fine for a chat completion,
but it is *not* fine if the call has side effects (a tool-calling turn that already
executed a tool) or if the caller is charged per call. A senior answer distinguishes
"safe to retry because the operation is pure" from "safe to retry because we have an
idempotency key," and knows which one applies.

**On timeouts as a product decision.** The right total budget depends on what the user
is doing. A 5-second budget for an autocomplete and a 120-second budget for a document
summary are both correct. A client that hardcodes one number forces every caller into
the same product decision. Expose the budget; default it sensibly.

**On the retry/timeout interaction.** These two are frequently designed independently
and then compose into something nobody intended. Three attempts × 30s = 90s. Add a
caller-side 60s timeout and you have a system where the caller gives up during attempt
2, the request continues server-side, and you pay for a completion nobody reads. Budget
first, then divide it among attempts.

**On what belongs in the client vs. the caller.** The client should own transport
concerns: connections, retries, timeouts, error classification, logging. It should *not*
own prompt construction, model selection, or routing — those are application concerns
and putting them in a shared client is how a client becomes a framework nobody can
change. The line is worth defending explicitly.

**On failure modes that don't look like failures.** The most dangerous outcomes here
return HTTP 200: a truncated response (`finish_reason: length`), a refusal, a response
that parses but is semantically wrong. A client that only handles HTTP errors handles
the easy half. Surfacing `finish_reason` in the result type — rather than discarding it
— is a small decision that makes a whole class of production bugs visible.

**On the circuit breaker being the omission that matters.** Of everything left out, the
circuit breaker is the one that turns a provider incident into your incident. When a
provider degrades, retries multiply load, connections pile up, and your service's own
pools exhaust — so requests that could have been served fail too. Candidates who raise
this unprompted have operated a system through a provider outage.

---

## Problem 2: Supporting a Second Provider

## Scenario

Six months later the company adds a second provider — for failover, for cost, and
because one team needs a model the first provider doesn't offer. The two providers
differ in wire format, streaming format, error shapes, structured-output mechanism, and
tool-calling conventions.

Someone proposes a `BaseProvider` class with fifteen abstract methods.

## Requirements

Design the abstraction. Then answer: what should *not* be abstracted?

## Expected Behavior

```python
client = LLMClient(providers=[primary, secondary], model="fast-model")
result = await client.complete(messages)     # transparent failover
result.provider                               # which one served it
```

<details>
<summary>💡 Reveal a hint</summary>

The failure mode of provider abstractions is **the leaky common denominator**: the
interface exposes only what both providers support, so callers lose access to the
features they chose a provider for; or it exposes everything, so it isn't an abstraction.

The useful split:

- **Normalize the transport layer completely** — errors, retries, timeouts, streaming
  events, token accounting. These genuinely are the same concept everywhere.
- **Normalize the common request shape** — messages, max tokens, temperature, stop
  sequences, and a *structured output* request (the mechanism differs; the intent
  doesn't).
- **Do not normalize model-specific capabilities.** Provider-specific parameters should
  pass through explicitly, and it should be visible in the call site that they're
  provider-specific.

And the thing most designs miss: **prompts are not portable.** A prompt tuned for one
model may perform worse on another. An abstraction that makes failover a config toggle
implies prompt portability that doesn't exist. Handle it by attaching a
prompt-per-provider mapping, or by accepting measured quality differences per provider
and gating failover on them.
</details>

<details>
<summary>✅ Reveal the design</summary>

```python
from typing import Protocol, AsyncIterator


class StreamEvent:
    """Normalized streaming event. Providers emit different shapes; callers
    should never see the difference."""
    __slots__ = ("kind", "text", "usage", "finish_reason", "raw")

    def __init__(self, kind, text=None, usage=None, finish_reason=None, raw=None):
        self.kind = kind                # "delta" | "usage" | "done"
        self.text = text
        self.usage = usage
        self.finish_reason = finish_reason
        self.raw = raw                  # escape hatch: the original event


class Provider(Protocol):
    """The narrow surface every provider must implement.

    Deliberately small. Anything provider-specific goes through `extra`, which is
    opaque to the client and visible at the call site.
    """

    name: str

    def build_request(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        response_schema: dict | None,
        stream: bool,
        extra: dict | None = None,
    ) -> tuple[str, dict]:
        """Return (path, json_body)."""
        ...

    def classify_error(self, status: int, headers, body: str) -> LLMError:
        """Map this provider's error shapes onto the common taxonomy."""
        ...

    def parse_response(self, data: dict) -> tuple[str, dict, str]:
        """Return (text, usage, finish_reason)."""
        ...

    def parse_stream_line(self, line: str) -> StreamEvent | None:
        """Parse one line of this provider's streaming format."""
        ...


class MultiProviderClient:
    def __init__(self, providers: list[tuple[Provider, httpx.AsyncClient]], *,
                 timeouts: Timeouts | None = None):
        self._providers = providers
        self._timeouts = timeouts or Timeouts()

    async def complete(self, messages, *, allow_failover: bool = True, **kw):
        """Try providers in order. Failover only on transient errors, and only
        if the caller allows it (some calls must not silently change model)."""
        last: Exception | None = None
        for provider, http in self._providers:
            try:
                return await self._complete_one(provider, http, messages, **kw)
            except PermanentError:
                raise                       # the request is wrong; another
                                            # provider won't fix it
            except (TransientError, BudgetExceeded) as exc:
                last = exc
                if not allow_failover:
                    raise
                log.warning("failing over from %s: %s", provider.name, exc)
        raise TransientError(f"all providers failed: {last}")
```

**What is deliberately *not* in the interface:**

- Tool-calling conventions. They differ enough that a common abstraction either loses
  fidelity or becomes the union of both. Expose them per-provider until you have
  evidence a common shape works.
- Model selection. The caller names a model; the client does not choose one.
- Prompt content. Prompts belong to the application, and per-provider prompt variants
  belong in the application's prompt store, not hidden inside a transport client.
- Cost calculation tables. Prices change; keep them in configuration, not code.

**Failover policy, made explicit:**

- Failover on transient errors only. A 400 fails everywhere.
- `allow_failover=False` for calls where a silent model change is unacceptable — for
  example anything whose output is stored and later compared, or anything with a
  contractual model commitment.
- Every result carries `provider` so downstream logging and evaluation can attribute
  quality correctly. **Without this attribution, a failover event silently contaminates
  your quality metrics.**
- Keep a small percentage of traffic on the secondary continuously. A failover path
  that is never exercised is a failover path that doesn't work.
</details>

## Follow-up Questions

1. Your secondary provider scores 4 points lower on your eval suite. Do you fail over
   automatically? What informs the answer?
2. A caller needs a feature only one provider supports. How does that reach them without
   destroying the abstraction?
3. How do you test failover without an outage?
4. The two providers count tokens differently. What breaks, and how do you handle cost
   reporting?
5. How would you detect that the primary provider silently changed model behavior?

## Senior-Level Discussion

The deep point in this problem is that **a provider abstraction implies a claim about
substitutability that is usually false.** Transport is substitutable; model behavior is
not. A client that makes failover a one-line config change encourages teams to treat
models as interchangeable, and the quality regression that follows is invisible because
nothing errored.

The mature design does two things about this: it makes the serving provider a first-
class field on every result (so evaluation and logging can segment by it), and it makes
failover an explicit, opt-in policy per call site rather than a global default. That
turns an invisible quality risk into a visible engineering decision — which is the
recurring theme of this entire section.
