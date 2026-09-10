# Incremental Indexing and Freshness

Your corpus changes. Documents are edited, added, and deleted, and every one of those
events has to reach the index or the system answers from a world that no longer exists.

"How fresh is your index?" is a question most teams cannot answer with a number. It has
one, it is measurable, and it is usually much worse than anyone assumes.

---

## Freshness is a number, and it's a distribution

Freshness lag is the time from a change at the source to that change being retrievable.
Measure it as a distribution, because the mean hides everything interesting:

```python
def freshness_lag_seconds(indexed_records, now):
    """One value per document: how stale is what we're serving?"""
    return sorted(now - r.source_changed_at for r in indexed_records)
```

The p50 tells you whether the pipeline works. The p99 tells you whether anything is stuck,
and stuck documents are the ones that generate complaints — a single document that failed
to reindex six weeks ago is invisible in a mean and obvious in a p99.

Two things worth pinning down before designing anything:

- **What freshness does the product actually need?** A policy document that changes
  quarterly and a support ticket queue that changes by the minute are different systems.
  Most teams build for minutes and need hours, or build for hours and need seconds for one
  small subset.
- **What's the cost of serving stale?** For a docs assistant, mild embarrassment. For
  pricing, inventory, or anything with legal effect, it's the whole reason the system
  exists. That answer sets the budget.

---

## The arithmetic that decides your architecture

Reindexing capacity has to exceed the change rate. This is the calculation that tells you
whether nightly full rebuilds can work at all:

```python
def can_keep_up(documents_changed_per_day, docs_per_hour, window_hours):
    capacity = docs_per_hour * window_hours
    return {
        "capacity_per_window": capacity,
        "change_rate": documents_changed_per_day,
        "keeps_up": capacity >= documents_changed_per_day,
        "utilization": documents_changed_per_day / capacity,
    }

can_keep_up(documents_changed_per_day=2_000_000, docs_per_hour=45_000, window_hours=10)
# capacity 450,000 — utilization 4.4x. You will never catch up.
```

That 450,000-per-window figure is the overnight capacity computed in
[rate-limits.md](../07-production/rate-limits.md). Against two million changes a day, a
nightly sweep falls further behind every night, and the failure is gradual: freshness
degrades over weeks, nobody sees a broken build, and by the time someone notices, the
index is a month stale.

**Run this calculation before choosing an approach.** If capacity comfortably exceeds
change rate, scheduled reindexing is simpler and you should take it. If it doesn't, no
amount of scheduling tuning fixes it and you need change-driven indexing.

---

## Detecting what changed

Options, in order of preference:

**Change events from the source system.** A webhook or change feed. Fastest and cheapest;
you only touch what moved. But events get dropped, arrive out of order, and don't exist
for every source, so events alone are never sufficient.

**Change tracking columns or logs.** Query for `updated_at > last_run`. Reliable where it
exists, and it misses deletes — a deleted row has no `updated_at` to find.

**Content hashing.** Enumerate sources, hash each, compare against the catalog. Catches
everything including deletes and silent edits. Expensive in enumeration, but the
enumeration is far cheaper than reprocessing:

```python
def find_changes(source_index, catalog):
    """source_index: {doc_id: content_hash} from the source system."""
    known = catalog.hashes()
    added = source_index.keys() - known.keys()
    deleted = known.keys() - source_index.keys()
    modified = {k for k in source_index.keys() & known.keys()
                if source_index[k] != known[k]}
    return {"added": added, "modified": modified, "deleted": deleted}
```

The practical answer is **events for speed, hashing for correctness**: react to change
events for low latency, and run a periodic hash-based reconciliation to catch what the
events missed. Events make the system fast; reconciliation makes it correct. A system with
only events drifts, and nobody notices the drift because there's nothing to compare
against.

---

## Deletes are the failure everyone ships

Additions and edits get tested. Deletes get discovered in production, usually by the
person who deleted the document.

Three reasons deletes are harder:

1. **They're invisible to most change detection.** No `updated_at`, no event in many
   systems, nothing to enumerate. Only a diff against what you have finds them.
2. **A deleted document still answers questions.** Until its chunks leave the index, the
   assistant will happily cite a document that no longer exists — and if it was deleted
   for legal or privacy reasons, that is the incident.
3. **Chunk-level cleanup is easy to get wrong.** Deleting the document row while leaving
   its chunks produces ghosts that retrieve normally and cite a document nobody can open.

The pattern that works is a tombstone plus a filter at query time:

```python
def delete_document(document_id, catalog, index):
    # 1. Mark first — the filter takes effect immediately, before any slow cleanup
    catalog.mark_deleted(document_id)
    # 2. Remove chunks by document id, not by enumerating chunk ids
    index.delete_by_filter({"document_id": document_id})
    # 3. Verify, because a delete that silently no-ops is the whole problem
    assert index.count(filter={"document_id": document_id}) == 0
```

Mark before you sweep. Vector store deletions can be slow or eventually consistent, and
the window between "user deleted it" and "index no longer returns it" is exactly the
window where a privacy deletion has not actually happened. Filtering on the catalog's
deleted flag at query time closes that window immediately, and the physical delete catches
up.

And note where this connects: embeddings are derived personal data, so deletion requests
cover the vector index too — [08-ai-security/pii.md](../08-ai-security/pii.md) makes that
point, and this is the mechanism that satisfies it.

---

## Adding documents can break existing answers

This is the counter-intuitive one, and it's stated in
[03-rag/README.md](../03-rag/README.md) as a rule worth remembering: new content outranks
old correct answers.

Nothing failed. Retrieval is working exactly as designed. A newly added document is a
better vector-space match for a query than the document that used to be returned, and the
answer changes — sometimes for the worse, if the new document is a draft, a duplicate, or
a superseded version.

Which produces the rule: **treat corpus changes as deploys.** A large ingestion is a
change to system behavior, and it deserves the same treatment as a code change — an
evaluation run before and after, on a fixed query set. That is the subject of
[corpus-changes.md](corpus-changes.md).

---

## Reindexing without downtime

Some changes require rebuilding everything: a new embedding model, a new chunking
strategy, a schema change. You cannot mutate the live index in place, because during the
rebuild the index contains vectors from two different models and similarity scores across
them are meaningless — the silent failure described in
[embeddings.md](../03-rag/embeddings.md).

Build alongside, then swap:

1. Build the new index under a new version identifier, from the same catalog.
2. Run your evaluation suite against it and compare with the current index, both on the
   same query set.
3. Shadow it — serve from the old index, query both, log the differences.
4. Swap by pointer, atomically, with the old index kept warm.
5. Keep the old index long enough to roll back without another rebuild.

The cost is running two indexes at once, which is a storage question and usually a small
one: 50 million chunks at 1,536 dimensions is about 307 GB
([03-rag](../03-rag/README.md)), so doubling it is cheap next to a bad swap you can't undo.

---

## What to monitor

- **Freshness lag: p50, p95, p99, and max.** The max is a stuck document with a name.
- **Change detection coverage** — changes found by reconciliation that events missed. If
  this is not near zero, your event path has a hole.
- **Delete completion rate and delete latency**, measured end to end from request to
  no-longer-retrievable.
- **Ghost chunk count**: chunks whose document id is absent or deleted in the catalog.
  Should be zero; alert if not.
- **Reindex queue depth vs. drain rate.** Depth rising steadily is the "never catch up"
  case arriving.
- **Index version served**, on every response, so "which index answered this?" is a filter
  rather than a guess.

---

## Interview questions

**"How fresh is your index?"**

It's a distribution, not a number: p50, p95 and max lag from source change to
retrievable. And the design question underneath it is whether reindexing capacity exceeds
change rate — if two million documents change daily and an overnight window handles
450,000, scheduled reindexing can never work and you need change-driven indexing with
reconciliation.

**"A customer deletes a document for GDPR reasons. Walk me through what happens."**

Mark it deleted in the catalog immediately and filter on that at query time, so it stops
being retrievable within a query rather than within a sweep. Then delete chunks by
document id from the index, and verify the count is zero rather than trusting the call.
The embeddings are derived personal data, so the vector index is in scope, and I'd want
the delete latency measured end to end because the gap between marked and swept is the
window where the deletion hasn't happened.

**"You added 50,000 documents and answers got worse. Nothing else changed."**

Expected, not mysterious. New documents outrank the ones that were previously returned;
if any of them are drafts, duplicates or superseded versions, quality drops with no bug
anywhere. The fix is process: run the eval suite before and after a corpus change, treat
ingestion as a deploy, and check for near-duplicates before indexing.

---

## What to remember

- Freshness is measurable. Report p50, p95 and max lag from source change to retrievable.
- Compare reindexing capacity against change rate before choosing an architecture. It
  decides scheduled versus change-driven, and nothing else does.
- Events for speed, hash-based reconciliation for correctness. Events alone drift
  undetectably.
- Deletes are the failure that ships. Mark first and filter at query time, then sweep and
  verify.
- Adding documents can degrade answers with nothing broken. Corpus changes are deploys.
- Never rebuild in place. Build alongside, evaluate, shadow, swap by pointer.
