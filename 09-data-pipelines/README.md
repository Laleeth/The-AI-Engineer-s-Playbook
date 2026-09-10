# 09 — Data & Pipelines

Everything upstream of the model. [03-rag](../03-rag/) covers what the index is *for*;
this section covers how content gets into it, stays correct, and changes safely. Plain
language throughout.

This is the most under-served topic in AI engineering material, including — until now —
this repo. It is also where a large share of production RAG quality is decided.

The fact underneath every file here:

> **Ingestion fails silently.** A retrieval bug returns something wrong and someone files
> a ticket. A document that failed to index three weeks ago produces "I don't know," and
> nobody files anything.

## The files

| File | What it covers |
|---|---|
| [document-parsing.md](document-parsing.md) | **Where retrieval quality is actually decided.** Read this one. |
| [ingestion-pipelines.md](ingestion-pipelines.md) | Pipeline shape, idempotency, partial failure, and the catalog |
| [incremental-indexing.md](incremental-indexing.md) | Freshness as a number, change detection, and deletes |
| [data-quality.md](data-quality.md) | The gates that keep garbage out, and the checks that find it after |
| [corpus-changes.md](corpus-changes.md) | Treating ingestion as a deploy: evaluate, stage, roll back |
| [backfills.md](backfills.md) | Reprocessing at scale without taking production down |
| [schema-and-versioning.md](schema-and-versioning.md) | What must be versioned, and what can't be added later |

## Read in this order

**Building a pipeline?** [document-parsing.md](document-parsing.md) →
[ingestion-pipelines.md](ingestion-pipelines.md) →
[schema-and-versioning.md](schema-and-versioning.md). The third one contains the decisions
that are painful to retrofit, which is why it comes before you have data.

**Quality is bad and you don't know why?**
[document-parsing.md](document-parsing.md) first — read fifty parsed documents — then
[data-quality.md](data-quality.md).

**Something changed and answers got worse?** [corpus-changes.md](corpus-changes.md).

**Facing a re-embedding or a migration?** [backfills.md](backfills.md), then
[incremental-indexing.md](incremental-indexing.md) for the swap.

## The short version

**Read your parsed output.** Fifty documents, with your eyes, before building anything.
Parsers don't crash — they return something, and something is what gets embedded. Tables
flattened into word salad retrieve confidently and mean nothing.

**The catalog is the pipeline.** A durable record of what was ingested, at which source
version, into which index version. Without it, "is this document indexed?" is
unanswerable and silent gaps are undetectable.

**Make every stage idempotent.** Deterministic chunk ids from document id, chunk index and
content hash; upsert never insert; delete by document before rewriting it.

**Index write throughput degrades as the index grows.** It's the usual cause of a
decelerating backfill, and everyone blames the GPUs. A rate that *falls* means something
is accumulating; a flat rate means a capacity shortfall.

**Freshness is a distribution.** p50, p95, max lag from source change to retrievable. And
compare reindexing capacity against change rate before choosing an architecture — if two
million documents change daily and an overnight window handles 450,000, no amount of
scheduling fixes it.

**Deletes are the failure that ships.** Invisible to most change detection, and a deleted
document keeps answering questions until its chunks leave the index. Mark first, filter at
query time, then sweep and verify.

**Adding good documents can make answers worse.** Near-duplicates split the ranking,
drafts outrank approved versions, and a larger corpus shifts every threshold you tuned. A
corpus change is a deploy.

**Bad content competes and often wins.** Boilerplate matches broadly; duplicates crowd the
top-k. Near-duplicate and boilerplate detection are the two highest-value gates, and both
are cheap.

**Compute wall-clock before starting a backfill.** 900 million chunks in six weeks needs
about 250,000 items/second. Token cost is almost never the constraint — $7,200 re-embeds
the whole thing.

**Bind the embedding model to the index.** Mixed vectors produce normal-looking scores and
meaningless results. Reject the mismatch at query time rather than documenting the rule.

**Some metadata cannot be added later.** Character offsets, stable ids, permission
identifiers, provenance, and effective date separate from ingestion date.

## The arithmetic worth doing out loud

```python
# Can scheduled reindexing keep up at all?
docs_per_hour * window_hours >= documents_changed_per_day

# How long does this backfill actually take?
items / items_per_second / 3600      # hours

# What does re-embedding cost? (usually not the constraint)
chunks * tokens_per_chunk / 1e6 * price_per_million

# Is this chunk a near-duplicate of one already indexed?
index.search(vector, k=1)[0].score >= 0.95
```

## Related

- [03-rag/](../03-rag/) — what the index is for, and how retrieval uses what this section
  produces
- [03-rag/embeddings.md](../03-rag/embeddings.md) — the model-version binding this section
  enforces
- [05-evaluation/regression-testing.md](../05-evaluation/regression-testing.md) — the
  paired comparison a corpus change should be gated on
- [07-production/rate-limits.md](../07-production/rate-limits.md) — lanes, floors, and the
  off-peak arithmetic that makes bulk work a scheduling problem
- [08-ai-security/authorization.md](../08-ai-security/authorization.md) — why permission
  metadata has to be in the index, not applied after
- [12-senior-scenarios/scaling.md](../12-senior-scenarios/scaling.md) — the embedding
  backfill that won't finish, as an interview scenario
