# Backfills at Scale

A backfill is reprocessing content you already have: a new embedding model, a new chunking
strategy, a metadata field added late, a parser fixed. The work is mechanical. The
difficulty is that it is enormous, it competes with production, and it is usually
discovered to be too slow after it has been running for a week.

---

## Compute the wall-clock time before you start

The number that matters is not cost. It's *how long*, and it decides whether the plan is
viable at all.

```python
def backfill_days(items, items_per_second):
    return items / items_per_second / 3600 / 24

def rate_needed(items, days_available):
    return items / (days_available * 24 * 3600)

# 900M chunks — the scenario in 12-senior-scenarios/scaling.md
backfill_days(900_000_000, items_per_second=46)      # ≈ 227 days
backfill_days(900_000_000, items_per_second=162)     # ≈ 64 days
rate_needed(900_000_000, days_available=42)          # ≈ 248 items/sec
```

Six weeks for 900 million chunks means about **250 items per second, sustained, for
forty-two days.** That is the number the whole project turns on, and it reframes the
problem immediately: a pipeline running at 46/sec isn't 250× short, it's about 5× short,
which is an engineering problem rather than an impossibility.

The same arithmetic run the other way is what makes the situation legible. The scenario's
pipeline started at 14 million chunks/day — 162/sec, already within 35% of target — and
fell to 4 million/day. It was never far from viable; it was degrading. Knowing that
changes what you go looking for.

Do this in the first ten minutes. It converts "we should re-embed everything" from an
opinion into a proposal with a delivery date, and it distinguishes the projects that need
a new architecture from the ones that need a bottleneck fixed.

Cost is usually the smaller constraint, which surprises people:

```python
def embedding_cost(chunks, tokens_per_chunk=400, price_per_million=0.02):
    return chunks * tokens_per_chunk / 1e6 * price_per_million

embedding_cost(50_000_000)     # $400
embedding_cost(900_000_000)    # $7,200
```

**$7,200 to re-embed 900 million chunks.** The API spend is a rounding error next to the
engineering time, the index rebuild, and the months of wall-clock. When someone objects to
a backfill on cost, check which cost they mean — it's almost never the tokens.

---

## Find the real bottleneck before optimizing

Backfills are pipelines, and the constraint is rarely where attention goes. The
decelerating backfill in section 12 is the canonical case: 31% GPU utilization, a rate
that fell from 14M/day to 4M/day, and an embedding model that everyone assumed was the
problem.

Deceleration is the diagnostic. A steady-state bottleneck produces a flat rate; a
*degrading* rate means something is accumulating. The usual suspects:

- **Index insert throughput falling as the index grows.** The most common cause.
- **A shared resource being consumed by production traffic** that grew over the same
  period.
- **Memory pressure or fragmentation** in the workers, gradually reducing effective
  concurrency.
- **A retry queue growing**, so an increasing share of work is repeat work.

Instrument per stage — fetch, parse, chunk, embed, write — and compare throughput at each.
One aggregate rate cannot tell you which stage to fix, and the fix for each is different:
more workers for a CPU stage, more quota for an API stage, and for a degrading index
write, a different architecture entirely.

---

## Build offline, swap atomically

The most valuable structural decision: don't write backfilled data into the live index.

Building a fresh index offline and swapping it in avoids the degrading-insert problem
(a bulk build is a different operation from incremental inserts), avoids competing with
production reads, and gives you an instant rollback. It also removes the mixed-vector
hazard entirely — during an in-place re-embedding, the index contains vectors from two
different models and similarity scores across them are meaningless, which is the silent
failure in [embeddings.md](../03-rag/embeddings.md).

```python
def backfill(catalog, source_index_version, target_index_version):
    """Never mutate the serving index. Build beside it."""
    target = create_index(target_index_version)
    for batch in catalog.iter_documents(chunk_size=10_000):
        target.bulk_load(process(batch))          # bulk path, not per-item upsert
    verify(target, expected=catalog.document_count())
    return target                                  # promotion is a separate decision
```

Two details in that sketch carry most of the benefit: **bulk load rather than per-item
upsert**, which is often an order of magnitude faster in a vector store, and **verify
before promote**, because a backfill that silently dropped 3% of documents looks
successful.

---

## Make it resumable, and assume it will be resumed

A backfill running for days will be interrupted — a deploy, a node failure, a quota
exhaustion, someone stopping it because production got slow. If interruption means
starting over, the backfill will never finish, because interruptions arrive faster than
the run completes.

Checkpoint durably and often:

```python
def resumable_backfill(catalog, target, checkpoint_store):
    cursor = checkpoint_store.get() or catalog.start_cursor()
    for batch in catalog.iter_from(cursor, chunk_size=10_000):
        target.bulk_load(process(batch))
        checkpoint_store.set(batch.end_cursor)     # after the write, not before
    return target
```

Checkpoint after the write completes, so a crash re-does one batch rather than skipping
it. Re-doing a batch must be harmless, which is the idempotency requirement from
[ingestion-pipelines.md](ingestion-pipelines.md) — deterministic chunk ids make a repeated
batch a no-op instead of a duplicate.

Order the cursor by something stable. Ordering by `updated_at` while documents are being
updated means you skip and repeat rows unpredictably; order by primary key.

---

## Live traffic keeps changing the corpus underneath you

A backfill over a moving corpus finishes in a state that doesn't match the source: some
documents were processed before an edit, some after, and a few were deleted mid-run.

The pattern that resolves this cleanly:

1. **Record a start timestamp** before the backfill begins.
2. **Run the backfill** over everything as of that timestamp.
3. **Queue live changes** during the run, rather than applying them to the target
   directly.
4. **Drain the queue** into the target when the bulk pass completes.
5. **Verify and promote.**

The queue drain is short if the backfill was fast and long if it wasn't — and if the
change rate exceeds the drain rate, you never converge, which is the same arithmetic as
[incremental-indexing.md](incremental-indexing.md). Compute it before committing.

Deletes during the window need special care: a document deleted mid-run may already have
been written to the target. The delete list must be applied to the target before
promotion, or promoting reintroduces documents that were deleted — including ones deleted
for legal reasons.

---

## Don't take production down with it

The shared-resource problem again, and backfills are its worst case because they are large
and they are not urgent.

- **Separate lane, lower priority, reserved floor for interactive traffic.**
- **Concurrency you can turn down without a deploy.** The single most useful control during
  an incident that a backfill caused.
- **Off-peak scheduling.** The overnight arithmetic in
  [rate-limits.md](../07-production/rate-limits.md) absorbed 450,000 documents in spare
  capacity, without buying anything.
- **A kill switch, tested.** Stopping a backfill should not require a deploy or a
  conversation.

---

## What to monitor

- **Items per second, and its trend.** The trend is the diagnostic; the absolute number is
  just progress.
- **Per-stage throughput**, so the bottleneck is identifiable rather than guessed.
- **Projected completion**, recomputed continuously from the current rate. This is what
  turns "it's running" into a decision.
- **Checkpoint age.** A checkpoint that stops advancing while workers look busy is a stuck
  backfill.
- **Production latency and error rate**, attributed to the backfill's shared resources.
- **Verification counts**: documents expected vs. present in the target, before promotion.

---

## Interview questions

**"You need to re-embed 900 million chunks in six weeks. Where do you start?"**

Arithmetic first: forty-two days means about 250 chunks per second sustained. Compare that
against the current rate — if the pipeline is at 46/sec it's roughly 5× short, which is an
engineering problem, and if it were 250× short I'd be redesigning instead of tuning.
Then per-stage instrumentation to find the actual bottleneck, which with a decelerating
rate and low GPU utilization is usually index write throughput rather than embedding. The
structural fix is building a fresh index offline with bulk load and swapping it in.

**"A backfill has been running for eleven days and is 11% done. What's your first
question?"**

Is the rate flat or falling? Falling means something is accumulating — most often insert
throughput into a growing index — and that changes the fix from "add workers" to "build
offline and swap." Flat means it's a straightforward capacity shortfall and I can compute
how much more I need. I'd also ask what the projected completion has been each day,
because a projection that keeps moving out is a signal nobody acted on.

**"How do you handle documents that change while the backfill runs?"**

Snapshot at a start timestamp, run the bulk pass against that, queue live changes during
the run, drain the queue into the target before promoting. Deletes need applying to the
target separately, or promotion reintroduces deleted documents. And the drain only
converges if change rate is below drain rate — worth computing before starting.

---

## What to remember

- Compute wall-clock duration first. It decides feasibility, and token cost is almost
  never the binding constraint — $7,200 re-embeds 900 million chunks.
- A decelerating rate means something is accumulating. Usually index insert throughput,
  not the GPUs.
- Build offline with bulk load and swap atomically. It avoids degrading inserts, mixed
  vectors, production contention, and gives instant rollback.
- Checkpoint after the write, order by a stable key, and make repeated batches harmless.
- Snapshot, queue live changes, drain before promoting — and apply deletes to the target
  or you will resurrect them.
