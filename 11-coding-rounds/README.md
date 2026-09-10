# 11 — Coding Rounds

Scenario-driven coding problems for AI engineering interviews. Every problem starts from
a realistic production requirement, not a puzzle. None of them are LeetCode, and none of
them are solved by calling a framework function — the point is to understand the
machinery that frameworks hide, because that's what you debug at 3am.

## What's here

| File | The problem | Sub-problems |
|---|---|---|
| [llm-api.md](llm-api.md) | A production LLM client: streaming, retries, timeouts, cancellation, structured output | 2 |
| [async-inference.md](async-inference.md) | Processing 10,000 documents with bounded concurrency, backpressure, and idempotency | 2 |
| [rag-from-scratch.md](rag-from-scratch.md) | Chunking, embeddings, index, retrieval, context construction, citations — no vector DB, no framework | 3 |
| [agent-loop.md](agent-loop.md) | A tool-calling agent that terminates correctly and survives hostile tool output | 3 |
| [retry-and-backoff.md](retry-and-backoff.md) | Retry classification, jitter, deadlines, circuit breaking, retry budgets | 3 |
| [rate-limiter.md](rate-limiter.md) | Token-aware limiting, multi-tenant fairness, distributed coordination, adaptive limits | 4 |
| [evaluation-harness.md](evaluation-harness.md) | Running evals, comparing two systems statistically, evaluating RAG and agents | 4 |

## Format

Each problem follows the same structure:

- **Scenario** — the production requirement that motivates the code.
- **Requirements** — what the implementation must handle.
- **Starter Code** — types and signatures to work against.
- **Expected Behavior** — including the properties that must hold, not just example calls.
- **Hint** — behind a spoiler, for when you're stuck but want to keep going yourself.
- **Reference Implementation** — clean, commented Python that explains its decisions.
- **Explanation** — why each decision was made, and what the alternatives cost.
- **Complexity** — time, memory, and (for AI systems) token and dollar cost.
- **Production Improvements** — what a real version needs that the reference omits.
- **Edge Cases** — a table of what should happen and why.
- **Follow-up Questions** — the ones interviewers actually ask.
- **Senior-Level Discussion** — the deeper points that separate levels.

Everything is Python 3.11+, standard library plus `httpx` and `numpy` where genuinely
needed. External calls are injected as functions, so the code is testable without a
provider — which is also how you should design it.

## The code is tested

The reference implementations aren't illustrative pseudocode. All 42 Python blocks in
this section parse, and the core implementations were executed against their stated
claims:

- the async worker pool holds its concurrency limit, reports permanent failures instead
  of raising, and leaks no tasks when the caller uses `aclosing`
- the chunker maintains exact source offsets (`doc.text[start:end] == chunk.text`) with
  100% coverage, and terminates on a sentence longer than the chunk size
- filtered vector search returns `k` results, not `k` minus the filtered ones
- BM25 ranks an exact identifier (`E-4471`) first, which is the whole reason hybrid
  retrieval exists
- the retry decorator does not retry 400s or `TypeError`, honors `Retry-After` over its
  own backoff, and respects the deadline rather than the attempt count
- the rate limiter's reserve-and-refund returns unused output-token capacity
- the agent's controls reduce the runaway-loop incident from 312 side effects to 1
- Wilson intervals stay inside [0, 1] at the extremes, and the paired comparison
  correctly reports that 0.79 vs. 0.83 on 200 cases is *not* significant (p ≈ 0.24,
  needing ~880 cases)

If you change a reference implementation, re-run it. A harness that reports confident
wrong numbers is worse than none.

## How to practice

1. Read the **Scenario**, **Requirements**, and **Starter Code**. Close the file.
2. Implement it. Give yourself 45 minutes — that's the real constraint.
3. Before opening the answer, write down what you think your edge cases are.
4. Compare against the reference. The interesting differences are almost never in the
   happy path.
5. Answer the follow-ups out loud. In a real round, the follow-ups are most of the
   signal.

## Recurring ideas

- **Classify errors before retrying anything.** Permanent vs. transient is the
  distinction that shows up in the client, the worker pool, the agent, and the retry
  decorator.
- **Bound everything.** Context, queues, concurrency, retries, iterations, cost, tool
  output. Every unbounded quantity is an incident waiting for traffic.
- **Cancellation is a first-class path.** `CancelledError` must never be swallowed and
  never retried.
- **Cache and idempotency keys must enumerate every dependency of the result** — query,
  permissions, corpus version, prompt version, model version. Anything omitted is a bug
  you've chosen to ship.
- **Concurrency limits and rate limits are different things,** and you usually need both.
- **Reserve pessimistically, refund quickly** — the pattern that makes token-aware rate
  limiting work at all.
- **Make the unsafe state unrepresentable** rather than documenting that it's unsafe.

## Related

The [senior scenarios](../12-senior-scenarios/) are the design-level counterpart. Several
production incidents there are the direct consequence of getting one of these
implementations wrong — a missing tenant in a cache key, an unbounded agent loop, a retry
storm without jitter, an eval suite that measures the wrong distribution.
