# 07 — Production

Running AI systems that people depend on. What breaks, how you find out, and what you do
about it. Plain language throughout.

The fact underneath every file here:

> **Most AI failures return HTTP 200.** Your error rate stays flat while quality degrades,
> costs triple, or the system quietly starts refusing legitimate requests. Standard
> monitoring is blind to all of it.

That's why this section exists as its own thing. The operational problems of an AI system
aren't the operational problems of a web service with a model bolted on.

## The files

| File | What it covers |
|---|---|
| [observability.md](observability.md) | The signals that aren't blind to silent failure |
| [tracing.md](tracing.md) | Following one request through retrieval, reranking, generation, tools |
| [retries.md](retries.md) | Retry policy — and why retries are load amplification |
| [fallbacks.md](fallbacks.md) | Degrading quality before availability, in measured tiers |
| [rate-limits.md](rate-limits.md) | Living inside a provider quota you don't control |
| [caching.md](caching.md) | The biggest cost lever, and the easiest one to get dangerously wrong |
| [cost-control.md](cost-control.md) | Attribution, budgets, and the conversations that follow |
| [incident-response.md](incident-response.md) | Running an AI incident, from "stop the damage" to the writeup |

## Read in this order

File order works: you need signals before you can trace, traces before you can run an
incident well.

If something is on fire right now, go to
[incident-response.md](incident-response.md) — it's a process, not a lecture.

If the bill is the problem, [cost-control.md](cost-control.md) then
[caching.md](caching.md). Attribution first; you can't fix what you can't attribute.

If you're setting up a system that has nothing yet, start with
[observability.md](observability.md) and instrument three things — attempts per request,
finish-reason distribution, cache hit rate. They're cheap and they catch what nothing else
does.

## Where the boundaries are

These topics appear in three places in this repo and they don't duplicate:

- **07 (here)** — the operational reference: what the policy is and why.
- **[11-coding-rounds/](../11-coding-rounds/)** — the tested implementation. Retry
  decorators, token buckets, circuit breakers, with code that runs.
- **[12-senior-scenarios/](../12-senior-scenarios/)** — the practice. The same problems as
  evolving interview scenarios with constraints that change mid-conversation.

Each file links to its counterparts rather than repeating them.

## The short version

**Attempts per request, finish-reason distribution, cache hit rate.** Three cheap metrics
that need no labels and catch failures your error rate cannot see.

**Track cost per unit of work, not total spend.** Total spend rising while cost per resolved
ticket falls is a success story that looks like a problem.

**Retries are load amplification.** They add work exactly when the system can least handle
it. Retry an allowlist, bound by a deadline rather than an attempt count, use full jitter,
and retry at one layer — check what your provider SDK is already doing.

**Degrade quality before availability.** A worse answer beats no answer. Write the tiers
down and measure each tier's quality *before* you need it.

**Empty retrieval means abstain, not generate anyway.**

**Provider quota is a managed resource with a lead time**, not weather. Alert on headroom at
70% — increases take weeks.

**A cache key must contain everything the answer depends on.** Anything you leave out is a
class of bug you've chosen to ship — and leaving out the permission scope is a data breach.

**Prefix caching first.** Biggest saving, zero correctness risk. Stable content at the top of
the prompt, volatile content at the bottom; one timestamp in the wrong place kills it.

**Alert on change, not level** — and investigate improvements too. A metric that suddenly got
better usually means something stopped happening.

**Half of AI incidents have no deploy.** Put every version identifier — model, prompt, index,
config — on every record, so "what changed?" is a filter instead of a guess.

**Stop the damage before you diagnose.** Agents don't wait for your investigation.

**If customers detect your incidents, the monitoring is the finding.**

## Related

- [06-inference-serving/](../06-inference-serving/) — the serving layer underneath, when
  you run the model yourself
- [10-system-design-patterns/](../10-system-design-patterns/) — the architectural shapes
  around fallbacks, caching and async work
- [08-ai-security/](../08-ai-security/) — the security half of running these systems
- [05-evaluation/](../05-evaluation/) — measuring quality, which is what "degraded" means
- [02-model-selection/cost-quality-latency.md](../02-model-selection/cost-quality-latency.md)
  — the optimization levers behind cost governance
- [00-interview-framework/how-to-answer-debugging.md](../00-interview-framework/how-to-answer-debugging.md)
  — the debugging technique these files assume
