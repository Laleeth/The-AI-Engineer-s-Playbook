# Roadmap

Ten of twelve sections are written. This file says what's done, what's coming, and why
the numbering has gaps.

---

## Why the numbers skip

Sections are numbered by where they sit in the learning path, not by the order they were
written. `06`, `09` and `10` are planned and unwritten, so their numbers are reserved.

The alternative — renumbering everything as sections land — would break links from every
other file each time. The gaps are deliberate.

---

## Status

| | Section | Files | Status |
|---|---|---|---|
| 00 | [Interview Framework](00-interview-framework/) | 4 | ✅ Done |
| 01 | [LLM Internals](01-llm-internals/) | 7 | ✅ Done |
| 02 | [Model Selection](02-model-selection/) | 5 | ✅ Done |
| 03 | [RAG](03-rag/) | 9 | ✅ Done |
| 04 | [Agents](04-agents/) | 8 | ✅ Done |
| 05 | [Evaluation](05-evaluation/) | 7 | ✅ Done |
| 06 | Inference & Serving | — | 🔜 Planned |
| 07 | [Production](07-production/) | 8 | ✅ Done |
| 08 | [AI Security](08-ai-security/) | 6 | ✅ Done |
| 09 | Data & Pipelines | — | 🔜 Planned |
| 10 | System Design Patterns | — | 🔜 Planned |
| 11 | [Coding Rounds](11-coding-rounds/) | 7 | ✅ Done |
| 12 | [Senior Scenarios](12-senior-scenarios/) | 7 | ✅ Done |

---

## What's planned

### 06 — Inference & Serving

Running models yourself, in production. `01-llm-internals` explains the mechanics; this
section is about operating them.

Likely files: serving stacks and what they do for you · continuous batching · autoscaling
and cold starts · multi-GPU serving and when you need it · load balancing across replicas ·
capacity planning · GPU economics and utilisation.

*Depends on:* [01-llm-internals](01-llm-internals/) for the underlying numbers.

### 09 — Data & Pipelines

Everything upstream of the model. Currently the most under-served topic in this repo, and
in most AI engineering material.

Likely files: document parsing (where a lot of RAG quality is silently lost) · ingestion
pipelines · incremental indexing and freshness · data quality and validation · handling
corpus changes safely · backfills at scale · schema and versioning.

*Depends on:* [03-rag](03-rag/) for what the index is for, and
[08-ai-security/authorization.md](08-ai-security/authorization.md) for indexing ACLs
alongside vectors.

### 10 — System Design Patterns

Reusable shapes that recur across AI products, as reference rather than as scenarios.

Likely files: request/response vs. async job vs. streaming · human-in-the-loop patterns ·
multi-region · batch vs. real-time · the "thin waist" platform pattern. Fallback chains and
caching layers are already covered operationally in
[07-production](07-production/) — this section would cover the architectural shapes around
them.

---

## What isn't planned

Saying this explicitly so nobody waits for it:

**Model training and fine-tuning internals.** Covered where it's a decision
([fine-tuning-vs-rag](02-model-selection/fine-tuning-vs-rag.md)), not as a discipline.
This is a repo about building systems with models, not about making models.

**Framework tutorials.** Framework APIs change faster than anything here would stay
accurate, and the repo's position is that you should understand the machinery frameworks
hide.

**Model comparisons and benchmarks.** Stale within weeks. The repo teaches how to evaluate
a model on *your* data instead.

**Prompt engineering technique catalogues.** Some prompting material will land in `01` or
`10` where it's load-bearing, but a list of techniques isn't planned.

---

## How to help

The most useful contribution is **"this scenario is unrealistic, here's why."** That's the
failure mode this material is most exposed to and the hardest to catch alone.

Also valuable:

- A production incident you've had that isn't represented
- A number in the docs that doesn't match your experience
- A section you'd reorder or cut

If you want to write one of the planned sections, open an issue first so the scope can be
agreed — the sections above are sketches, not commitments.

See the [contributing notes](README.md#contributing) for format and the verification
requirement.

---

## Principles

Constraints the material is written under, listed so contributions can match:

**Scenario-driven.** If a question has one memorised answer, it doesn't belong.

**Numbers over adjectives.** "Expensive" is not information. `$140,000/month at 200M
requests` is.

**Code that runs.** Every Python block is parsed in CI, and the reference implementations
were executed against their claims.

**Claims that are checked.** Numeric figures quoted in prose are re-computed in
[`scripts/verify.py`](scripts/verify.py). Changing a number in the docs fails CI until the
check is updated too.

**Plain language in the reference sections.** Sections 00–08 explain terms on first use.
Sections 11–12 are written at interview register, because that's the register you'll be
answering in.

**Honest about what's unknown.** Where something is genuinely uncertain or
model-dependent, the material says "measure it" rather than inventing a rule.
