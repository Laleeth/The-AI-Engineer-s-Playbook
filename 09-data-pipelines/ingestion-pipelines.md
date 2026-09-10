# Ingestion Pipelines

An ingestion pipeline turns source documents into indexed, searchable chunks. It is
plumbing, and it is the part of a RAG system most likely to be running quietly wrong.

The reason is structural: a query failure is visible and an ingestion failure is not. If
retrieval returns nothing for a document that silently failed to index three weeks ago,
the system answers "I don't know" and nobody files a bug.

---

## The stages, and what each one owes the next

```python
def ingest(document):
    raw = fetch(document.uri)              # source of truth, versioned
    text, meta = parse(raw)                # offsets + permissions preserved
    chunks = chunk(text, meta)             # offsets survive chunking
    vectors = embed([c.text for c in chunks])
    index.upsert(chunks, vectors, meta)    # idempotent, keyed by chunk id
    catalog.record(document, state="indexed", version=document.version)
```

Six stages, and the last one is the stage most pipelines omit. Without a catalog — a
durable record of what has been ingested, at which source version, into which index
version — you cannot answer "is this document in the index?", which means you cannot
detect the failure mode above.

**The catalog is the pipeline.** Everything else is transformation.

---

## Make every stage idempotent

Ingestion will be re-run: a retry, a partial failure, a redeploy, a backfill overlapping
live traffic. If re-running produces duplicates, every one of those events degrades
retrieval quietly — duplicated chunks crowd the top-k with copies of the same passage and
push out the diversity that made retrieval work.

Idempotency comes from deterministic identity, not from careful orchestration:

```python
import hashlib

def chunk_id(document_id, chunk_index, text):
    """Same input, same id, forever — so upsert replaces instead of duplicating."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:{chunk_index}:{digest}"
```

Including the content hash means a chunk whose text changed gets a *new* id, so you can
tell replacement from rewrite. It also means you must delete the old ids when a document
is reprocessed, which is the subject of
[corpus-changes.md](corpus-changes.md) — new chunks appearing without old ones being
removed is the most common way an index accumulates ghosts.

Two rules follow, and both are cheap to adopt and expensive to retrofit:

- **Upsert, never insert.** The operation must be safe to repeat.
- **Delete by document, not by chunk.** Reprocessing a document deletes all its chunks by
  document id, then writes the new set. Trying to diff chunk-by-chunk is where ghosts come
  from.

---

## Partial failure is the normal case

At any real scale, some fraction of every run fails: a document is corrupt, an embedding
call times out, the vector store rejects a batch. The pipeline's job is not to avoid this
— it is to make it visible and resumable.

**Fail per document, not per batch.** One unparseable file must not abort a run of two
million. Collect failures, keep going, report at the end.

**Record the failure with its stage.** "Document 8812 failed" is not actionable; "document
8812 failed at embed, RateLimitError, attempt 3" is. The stage is what tells you whether
this is a data problem or an infrastructure problem, and the distinction determines who
fixes it.

**Distinguish permanent from transient**, exactly as in
[retries.md](../07-production/retries.md). A malformed PDF will fail identically forever
and belongs in quarantine; a 429 belongs in a retry queue. A pipeline that retries
permanent failures spends its whole budget on documents that will never succeed.

```python
def process_batch(documents):
    indexed, transient, permanent = [], [], []
    for doc in documents:
        try:
            ingest(doc)
            indexed.append(doc.id)
        except TransientError as exc:
            transient.append((doc.id, exc.stage, str(exc)))
        except PermanentError as exc:
            permanent.append((doc.id, exc.stage, str(exc)))
    return {"indexed": indexed, "retry": transient, "quarantine": permanent}
```

**A failed document must not look like a succeeded one.** The catalog state should be
`indexed`, `failed`, or `quarantined` — never absent. Absence is indistinguishable from
"never attempted," and that ambiguity is what makes silent gaps possible.

---

## Where the throughput goes

Ingestion is a pipeline of stages with wildly different costs, and the bottleneck is
almost never where people assume.

| Stage | Typical bound | Notes |
|---|---|---|
| Fetch | Network / object storage | Parallelizes trivially; often rate-limited by the source system |
| Parse | CPU | The expensive branches (OCR, layout) can dominate everything else |
| Chunk | CPU, cheap | Rarely the bottleneck |
| Embed | GPU or API quota | Where people look first, and often not the constraint |
| Index write | Vector store insert throughput | **Degrades as the index grows** |

That last row is the one that surprises people, and it's the mechanism behind the
decelerating backfill in
[12-senior-scenarios/scaling.md](../12-senior-scenarios/scaling.md). Insert throughput
into a growing index falls as the structure is maintained; a pipeline that started at 14M
chunks/day and now does 4M has usually hit this, not an embedding limit.

Measure per-stage throughput separately or you will optimize the wrong stage. A GPU at 31%
utilization during an embedding backfill is telling you the GPUs aren't the problem.

---

## Batch, stream, or both

Most systems need both, and the mistake is running them through separate code paths that
drift.

**Batch** — scheduled runs over a set of documents. Simple, efficient, easy to reason
about, and the freshness is bounded by the schedule.

**Stream** — react to change events as they arrive. Fresher, and every hard problem gets
harder: ordering, duplicates, replay, and the fact that a burst of events can arrive
faster than you can embed.

The pattern that holds up is **one transformation path, two triggers**. The same
`ingest(document)` runs whether it was invoked by a nightly sweep or by a change event.
The differences live in scheduling and concurrency control, not in logic. When the batch
path and the stream path are separate implementations, they diverge, and the divergence
shows up as documents that are indexed differently depending on how they arrived.

Reconciliation matters more than either: a periodic sweep that compares the catalog
against the source system and against the index, and repairs the differences. Streams drop
events. Ordering fails. The sweep is what makes the system eventually correct rather than
hopefully correct.

---

## Backpressure, and not taking the system down with you

Ingestion and serving usually share something — an embedding endpoint, a vector store, an
API quota. A backfill that saturates the shared resource turns a data job into a customer
incident.

- **Give ingestion a lower priority lane** and a reserved-floor quota, the pattern from
  [rate-limits.md](../07-production/rate-limits.md).
- **Bound concurrency explicitly**, and make it adjustable at runtime. The setting you can
  change without a deploy is the one that saves the incident.
- **Schedule into off-peak windows.** The arithmetic in that same file is worth
  internalizing: spare overnight quota handled 450,000 documents without buying anything.
  Most "we need more capacity" ingestion problems are distribution problems.
- **Watch the queue, not the worker.** A growing queue with healthy workers means the
  downstream is the constraint.

---

## What to monitor

- **Catalog coverage**: documents in source vs. documents in catalog vs. chunks in index.
  Three numbers that should reconcile, and the alert when they don't.
- **Per-stage throughput and duration**, separately. One aggregate rate hides the
  bottleneck.
- **Failure rate by stage and by type**, with permanent and transient split.
- **Quarantine size and age.** A quarantine nobody drains is a silent data gap.
- **Index write latency over time** — the leading indicator of the degrading-insert
  problem.
- **Time from source change to indexed**, which is the metric your users actually feel.
  See [incremental-indexing.md](incremental-indexing.md).

---

## Interview questions

**"A customer says a document they uploaded three weeks ago isn't searchable. How do you
investigate?"**

Check the catalog first: is the document recorded, at what state and which version? That
single lookup splits the problem — absent means ingestion never saw it, so look at the
source-change path; `failed` or `quarantined` means it was seen and rejected, and the
recorded stage tells you why; `indexed` means the bug is in retrieval or permission
filtering, not ingestion. Without a catalog none of that is answerable, which is the real
finding if it's missing.

**"Your embedding backfill is slowing down over time. GPUs are at 31%."**

Deceleration means something is accumulating, and 31% says it isn't the GPUs. The usual
cause is vector store insert throughput degrading as the index grows. I'd measure
per-stage throughput to confirm, then consider building the index offline and swapping it
in, batching writes differently, or partitioning the index — and I'd check whether the
backfill is competing with production queries on the same store.

**"How do you make ingestion safe to re-run?"**

Deterministic chunk ids from document id, chunk index and content hash; upsert rather than
insert; delete by document id before writing a document's new chunks. Then a catalog that
records what was ingested at which source version, so a re-run is a no-op for unchanged
documents rather than a duplicate write.

---

## What to remember

- The catalog is the pipeline. Without a durable record of what's indexed at which
  version, silent gaps are undetectable.
- Ingestion failures are invisible by construction — a missing document produces "I don't
  know," not an error.
- Make every stage idempotent through deterministic ids, and delete by document before
  rewriting it.
- Fail per document, record the stage, and split permanent from transient.
- Index write throughput degrades as the index grows. It's the usual cause of a
  decelerating backfill, and people blame the GPUs.
- One transformation path, two triggers, plus a reconciliation sweep that makes the system
  eventually correct.
