# The AI Engineer's Playbook

Practical engineering knowledge for building and operating production AI systems — LLMs,
RAG, agents, inference, evaluation, system design, and AI infrastructure.

This is **not** a list of AI interview questions. It is preparation material for
experienced engineers interviewing for roles where the job is to reason about, build,
debug, scale, and operate real AI systems:

- Senior AI Engineer · Staff AI Engineer · ML Engineer
- Applied AI Engineer · LLM Engineer · AI Platform Engineer
- AI Architect · Senior Software Engineer working on AI systems

Every question here is scenario-driven, has realistic constraints and numbers, and has
more than one defensible answer. If a question can be answered from memory, it doesn't
belong here.

## Sections

### [12 — Senior Scenarios](12-senior-scenarios/)

Open-ended system design and engineering judgment. 39 scenarios across seven files, many
of which **evolve in rounds**: you design the system, then traffic grows 15×, then costs
triple, then the security team changes the requirements, then accuracy drops 15% and you
have to investigate.

- [architecture-tradeoffs.md](12-senior-scenarios/architecture-tradeoffs.md) — RAG
  architecture, sync vs. async, model routing, caching, platform vs. team-owned
- [production-incidents.md](12-senior-scenarios/production-incidents.md) — latency
  spikes, RAG regressions, token explosions, runaway agents, cross-tenant leakage,
  provider outages
- [cost-vs-quality.md](12-senior-scenarios/cost-vs-quality.md) — unit economics, routing
  and cascades, when to fine-tune, when quality is worth any price
- [build-vs-buy.md](12-senior-scenarios/build-vs-buy.md) — vector search, self-hosted
  inference, evaluation platforms, agent frameworks, observability
- [model-selection.md](12-senior-scenarios/model-selection.md) — extraction at 30M
  documents, multilingual quality, air-gapped constraints, embedding migrations,
  deprecations
- [scaling.md](12-senior-scenarios/scaling.md) — what breaks at 500 req/s, 100× growth,
  embedding backfills, multi-region, rate-limit walls, context explosion
- [staff-level.md](12-senior-scenarios/staff-level.md) — eight teams and eight RAG
  systems, AI strategy, governance after an incident, two-year migrations, influence
  without authority

### [11 — Coding Rounds](11-coding-rounds/)

Implementation problems that start from a production requirement. 21 problems across
seven files, with tested reference implementations in Python.

- [llm-api.md](11-coding-rounds/llm-api.md) — a client with streaming, retries,
  timeouts, cancellation, and multi-provider support
- [async-inference.md](11-coding-rounds/async-inference.md) — bounded concurrency,
  backpressure, idempotency, and continuous stream processing
- [rag-from-scratch.md](11-coding-rounds/rag-from-scratch.md) — chunking through
  citations with no framework; hybrid retrieval; permissions and document updates
- [agent-loop.md](11-coding-rounds/agent-loop.md) — tool calling, termination, loop
  detection, and defending against hostile tool output
- [retry-and-backoff.md](11-coding-rounds/retry-and-backoff.md) — error classification,
  jitter, deadlines, circuit breakers, retry budgets
- [rate-limiter.md](11-coding-rounds/rate-limiter.md) — token-aware limiting, tenant
  fairness, distributed coordination, adaptive limits
- [evaluation-harness.md](11-coding-rounds/evaluation-harness.md) — running evals,
  comparing systems statistically, evaluating RAG and agents

## How to use this

Both sections use progressive disclosure — hints and answers are behind collapsed
sections so you can attempt each problem first. That's the intended workflow:

**For senior scenarios:** read the situation and constraints, close the file, spend
10–15 minutes writing your approach including the numbers you'd ask for, then compare.
The follow-up questions are where the real interview happens — answer them out loud.

**For coding rounds:** implement it in 45 minutes before opening the reference. Write
down your edge cases before you read the answer table. The interesting differences are
never in the happy path.

**With a partner:** the incident scenarios are designed for it. One person plays the
dashboards and reveals evidence only when asked for specific metrics; the other must
state a hypothesis, say what would disconfirm it, and ask for exactly that data.

## The quality bar

Every question was written against one test: *could an experienced AI engineer learn
something from this?*

**Avoided:** trivia, definitions, buzzwords, framework-version questions, anything with
one memorized answer, artificial complexity, unrealistic architectures.

**Preferred:** realistic constraints, genuine ambiguity, trade-offs with conditions,
debugging with partial evidence, actual numbers, production failures, business context,
security, cost, latency, reliability.

The reference implementations are executed, not just written — see the
[coding rounds README](11-coding-rounds/README.md#the-code-is-tested) for what was
verified and how.

## Scope

Content covers current production practice: LLM inference and serving, RAG and hybrid
retrieval, reranking, agents and tool calling, structured outputs, model routing and
cascades, evaluation, inference optimization, GPU serving, vector search, observability,
and AI security. Technology is used where it serves the engineering problem and left out
where it doesn't.

## Contributing

Scenarios should be realistic enough that someone who has run production AI systems reads
them and recognizes the situation. If a proposed question has a single correct answer
that can be memorized, it needs rewriting. New scenarios should follow the existing
format — including the trade-offs, failure modes, follow-ups, and interviewer notes,
which are usually more valuable than the answer itself.

## License

See [LICENSE](LICENSE).
