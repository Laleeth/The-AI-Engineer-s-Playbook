# Handling Corpus Changes Safely

A corpus change is a deploy. It changes what the system answers, it can break things that
worked yesterday, and it deserves the same rigor as a code change — evaluation before,
rollout in stages, a rollback path, and a version identifier on every response.

Almost nobody treats it that way, which is why "we added some documents" is a common
root cause and an uncommon suspect.

---

## Why adding good content makes answers worse

Nothing has to be broken for this to happen. Retrieval returns the top-k most similar
chunks; adding documents changes which chunks those are.

Concretely, the ways a well-intentioned ingestion degrades quality:

- **Near-duplicates split the ranking.** Three copies of one policy occupy three of your
  five slots, and the distinct document that used to provide the other half of the answer
  is pushed out.
- **Drafts and superseded versions outrank approved ones**, because they often match query
  phrasing more closely than the formal final version does.
- **Broad, shallow documents match everything.** An FAQ or overview page is a decent match
  for a huge range of queries and a good answer to almost none of them.
- **A larger corpus shifts the score distribution.** Thresholds tuned on the old corpus —
  a minimum similarity for "I don't know", a reranker cutoff — no longer mean what they
  meant.

That last one is worth pausing on, because it breaks things far from the ingestion. Any
constant tuned against retrieval scores is corpus-dependent, and a corpus change
invalidates it silently.

---

## Evaluate before, not after

The tooling for this already exists in [05-evaluation](../05-evaluation/); the only new
idea is pointing it at a corpus change instead of a prompt change.

```python
def corpus_change_gate(eval_set, current_index, candidate_index):
    """Same queries, two corpora. Compare paired, not as two averages."""
    before = run_eval(eval_set, index=current_index)
    after = run_eval(eval_set, index=candidate_index)
    return {
        "before": before.score,
        "after": after.score,
        "regressed": [c.id for c in eval_set
                      if before.correct(c) and not after.correct(c)],
        "fixed": [c.id for c in eval_set
                  if not before.correct(c) and after.correct(c)],
    }
```

The `regressed` list is the output that matters. An aggregate that moved from 0.81 to 0.83
can still contain twenty queries that used to work and now don't, and those are the ones
your users will report. Compare case by case — the paired-comparison discipline from
[regression-testing.md](../05-evaluation/regression-testing.md) applies unchanged, as does
the warning that a small aggregate move on a few hundred cases is usually noise.

Segment the results too. A corpus change frequently improves the topic it covers and
damages a neighboring one, and a single number reports that as a mild improvement.

---

## Roll out in stages

Even with an eval gate, the eval set is not the world. Stage the change:

1. **Build a candidate index** rather than writing into the live one. Same catalog, new
   index version.
2. **Gate on evaluation**, with the regression list reviewed by a person when it's
   non-empty.
3. **Shadow.** Serve from the current index, query both, log where the retrieved sets
   differ. This surfaces the queries your eval set doesn't contain, which is most of them.
4. **Canary.** Route a small share of live traffic to the candidate and compare online
   signals — the measurements in
   [online-evaluation.md](../05-evaluation/online-evaluation.md) — because offline and
   online disagree and online is what's true.
5. **Promote** by pointer swap, keeping the previous index warm.

For a small, routine ingestion this is overkill and you should skip to step 1 and 2. For a
new source, a re-chunking, or anything touching a large share of the corpus, every step
earns its cost. The judgment about which is which is the interesting part, and the honest
default is: if you can't name what would break, you haven't thought about it yet.

---

## Version everything, and put it on every response

The single highest-leverage habit here, and it's the same one
[07-production/incident-response.md](../07-production/incident-response.md) prescribes for
incidents generally: *half of AI incidents have no deploy.*

Attach to every response the identifiers of everything that shaped it:

```python
def response_metadata(request, result):
    return {
        "model_version": result.model_version,
        "prompt_version": result.prompt_version,
        "index_version": result.index_version,       # which corpus answered
        "embedding_model_version": result.embedding_version,
        "chunking_version": result.chunking_version,
        "reranker_version": result.reranker_version,
    }
```

With these, "what changed?" is a filter over your logs. Without them it is a conversation
in which several people confidently remember different things, while the actual change was
an ingestion job that ran on Tuesday and belonged to a different team.

The embedding model version is the one that must be bound to the index rather than merely
logged — mixing vectors from two models produces normal-looking scores and meaningless
results, the silent failure in [embeddings.md](../03-rag/embeddings.md). See
[schema-and-versioning.md](schema-and-versioning.md) for how that binding is enforced.

---

## Rollback is a pointer, or it isn't rollback

If reverting a corpus change means re-running ingestion, you don't have rollback — you
have a several-hour recovery with a rebuild in the middle of an incident.

Keep the previous index version intact and warm, and make promotion a pointer swap. Then
rollback is one operation, takes seconds, and can be done by whoever is on call without
understanding the pipeline.

Costs and limits worth stating plainly:

- **Storage.** Two indexes instead of one. 50 million chunks at 1,536 dimensions is about
  307 GB ([03-rag](../03-rag/README.md)); doubling that is inexpensive relative to an
  outage you can't reverse.
- **Deletions don't roll back.** If a document was deleted for privacy or legal reasons,
  rolling back to an index that still contains it re-exposes it. Deletions must be
  reapplied to any index you might promote — which means the delete list is versioned
  separately from the index and applied on promotion.
- **Retention has a limit.** Keeping every index forever is its own compliance problem.
  One or two generations is the usual answer.

---

## Bulk changes need their own lane

A large ingestion competes with production for the embedding endpoint, the vector store,
and the API quota. The controls are in
[rate-limits.md](../07-production/rate-limits.md) and
[ingestion-pipelines.md](ingestion-pipelines.md), and the summary is: separate lane,
reserved floor for interactive traffic, off-peak scheduling, and a concurrency limit you
can turn down without a deploy.

Worth repeating because it reframes most of these conversations: the overnight arithmetic
in that file handled 450,000 documents in spare capacity. Bulk ingestion is usually a
scheduling problem wearing a capacity problem's clothes.

---

## What to monitor

- **Index version served**, on every response, always.
- **Eval score before and after each corpus change**, with the regression list retained —
  not just the aggregate.
- **Retrieval score distribution** across corpus versions. A shift means every tuned
  threshold needs revisiting.
- **Shadow divergence rate**: what fraction of queries retrieve a different set.
- **Time-to-rollback**, tested. An untested rollback is a plan, not a capability.
- **Corpus size and duplicate rate per version**, so a change that doubled the corpus is
  visible as an event.

---

## Interview questions

**"Answer quality dropped last week. No deploy went out."**

Ask what changed that isn't code — and corpus is the first candidate, along with prompt
edits and model-provider changes. If responses carry an index version, this is a filter:
compare quality by index version and find the ingestion. If they don't, that's the finding
and the first fix. Then check the specific mechanism: new near-duplicates, drafts
outranking approved documents, or a shifted score distribution invalidating a threshold.

**"How would you safely add a new document source to a working RAG system?"**

Build a candidate index rather than writing into the live one, run the eval suite paired
against the current index and read the regression list rather than the aggregate, shadow
to catch queries the eval set doesn't cover, canary a small share of traffic, then promote
by pointer with the old index kept warm. And check for near-duplicates against the
existing corpus first — that's the most common way a good source makes answers worse.

**"Your rollback plan is to re-run ingestion. What's wrong with that?"**

It isn't a rollback, it's a rebuild during an incident — hours, under pressure, with the
bad index still serving. Keep the previous index version warm and make promotion a pointer
swap. One caveat I'd raise: deletions must be reapplied on promotion, or rolling back
re-exposes documents that were deleted for legal reasons.

---

## What to remember

- A corpus change is a deploy. Evaluate before, stage the rollout, keep a rollback.
- Good content can degrade answers: duplicates split rankings, drafts outrank approved
  versions, and a bigger corpus shifts every tuned threshold.
- Read the regression list, not the aggregate. Paired comparison, segmented.
- Put model, prompt, index, embedding, chunking and reranker versions on every response.
  "What changed?" should be a filter.
- Rollback is a pointer swap onto a warm previous index — and the delete list must be
  reapplied when you promote.
