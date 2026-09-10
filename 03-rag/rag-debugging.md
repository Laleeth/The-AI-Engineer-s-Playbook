# Debugging RAG

An answer is wrong. Where do you look?

There's one question that splits this problem in half, and asking it first saves days:

> **Was the right information in the prompt?**

If yes, it's a generation problem. If no, it's a retrieval problem. These have completely
different fixes, and teams that skip this question spend two weeks tuning prompts when
the retriever was broken.

---

## The first check

Log what was retrieved on every request. Without this you cannot debug anything.

```python
@dataclass
class RagTrace:
    trace_id: str
    question: str
    rewritten_question: str | None
    retrieved: list[dict]        # id, score, source, first 200 chars
    reranked: list[dict]
    context_sent: str            # exactly what went in the prompt
    answer: str
    citations: list[str]
    # versions — without these you can't attribute anything
    embedding_model: str
    prompt_version: str
    corpus_version: str
    model: str
```

Then debugging a complaint starts here:

```python
def diagnose(trace, correct_answer_doc_id):
    if correct_answer_doc_id not in [r["id"] for r in trace.retrieved]:
        return "RETRIEVAL: the right document was never found"
    if correct_answer_doc_id not in [r["id"] for r in trace.reranked]:
        return "RERANKING: it was found, then dropped"
    if correct_answer_doc_id not in trace.context_sent:
        return "CONTEXT: it survived reranking but didn't fit the budget"
    return "GENERATION: the right document was in the prompt and the answer is still wrong"
```

Four different problems, four different fixes. That function is worth writing on day one.

---

## Retrieval problems

### The right document was never retrieved

Work through these in order.

**1. Is it even in the index?**

Sounds obvious. Check anyway — indexing failures are common and silent.

```python
def check_indexed(doc_id, index):
    chunks = index.chunks_for_doc(doc_id)
    if not chunks:
        return "NOT INDEXED — check the ingestion pipeline"
    return f"{len(chunks)} chunks indexed"
```

**2. Did chunking destroy it?**

Read the actual chunks. Not the document — the chunks.

```python
def inspect_chunks(doc_id, index):
    for c in index.chunks_for_doc(doc_id):
        print(f"--- {c.id} ({count_tokens(c.text)} tokens) ---")
        print(c.text[:300])
```

Look for: sentences cut in half, tables shredded, headers separated from their content,
chunks that are just navigation text.

**3. Does the chunk have enough context to be findable?**

The classic failure: the chunk says "the limit is 30 days" with no indication of what
for. Nobody's query will match it. Fix in
[contextual-retrieval.md](contextual-retrieval.md).

**4. Is it a vocabulary mismatch?**

The user says "money back," the document says "reimbursement." Or the user has an exact
code and dense search returns a near-miss. Fix with
[hybrid search](hybrid-search.md).

Test it directly:

```python
def why_not_retrieved(question, expected_doc_id, index, embedder):
    hits = index.search(embedder(question), k=100)
    ids = [h.id for h in hits]

    if expected_doc_id in ids:
        rank = ids.index(expected_doc_id) + 1
        return f"Found at rank {rank} — retrieve more candidates or add reranking"

    # Is it findable by keyword at all?
    bm25_hits = [i for i, _ in bm25.search(question, k=100)]
    if expected_doc_id in bm25_hits:
        return "Found by keyword but not by vector — add hybrid search"

    return "Not in the top 100 by either method — chunking or context problem"
```

That function gives you the fix directly, and takes ten minutes to write.

### It was retrieved and then dropped

Ranked 30th, you only kept 5.

- Retrieve more candidates before reranking (50 rather than 10).
- Add reranking if you don't have it.
- Check the reranker isn't truncating your chunks — if its input limit is 512 tokens and
  your chunks are 800, it only scores part of each one and doesn't tell you.

### Nothing relevant exists

Sometimes the answer genuinely isn't in your corpus. That's a content problem, not an
engineering one — and the correct behavior is to say "I don't have that information."

Check that path works. Systems that never abstain hallucinate whenever retrieval fails,
and retrieval fails regularly.

---

## Generation problems

The right document was in the prompt and the answer is still wrong.

**1. Was it buried?**

Models attend less well to the middle of long contexts. If the right chunk was 8th of 12
in a long prompt, it may have been effectively invisible.

Test: send only the correct chunk and see if the answer is right. If it is, it's a
context-length or ordering problem, not a comprehension problem.

**2. Is the context too long?**

Cutting from 20 chunks to 5 often *improves* answers. Fewer distractions. Try it —
it's a one-line change and the result is informative either way.

**3. Contradictory context?**

Two retrieved documents say different things — an old policy and a new one. The model
picks one, possibly the wrong one. Fix with recency signals in reranking, or by removing
outdated documents from the index.

**4. Prompt problem?**

Does the prompt actually tell the model to use only the context? Does it tell it to
abstain? Does it ask for citations? These instructions do real work.

**5. Is the model just not good enough?**

Try a bigger model on the failing cases. If that fixes it, you know. But check this
*last* — it's the most expensive fix and usually not the cause.

---

## The symptoms table

Quick lookup by what you're seeing.

| Symptom | Likely cause | Where to look |
|---|---|---|
| Right topic, wrong specifics | Retrieval got a similar-but-wrong document | [hybrid-search](hybrid-search.md) |
| "I don't know" when the answer exists | Chunk lacks context to be found | [contextual-retrieval](contextual-retrieval.md) |
| Exact codes/IDs fail | Dense-only retrieval | [hybrid-search](hybrid-search.md) |
| Follow-up questions fail | No query rewriting | [query-rewriting](query-rewriting.md) |
| Answer good, citations wrong | Chunks too large, or stale IDs | [chunking](chunking.md) |
| Confident and completely wrong | Retrieved wrong docs, model used them faithfully | Retrieval |
| Was working, now isn't | Something changed — see below | Change log |
| Inconsistent across identical questions | Temperature, or ties broken differently | Set temperature 0 |
| Only fails for some users | Permission filter | Retrieval filter |
| Only fails on long documents | Chunking, or context truncation | [chunking](chunking.md) |

---

## When it used to work

Different investigation. Something changed. Find out what.

```python
CHANGE_SOURCES = [
    "embedding_model_version",
    "chunking_config",
    "prompt_version",
    "model_version",          # including silent provider updates
    "corpus_contents",        # documents added or removed
    "reranker_version",
    "retrieval_k",
    "context_budget",
]
```

The two that catch people out:

**Documents were added.** New content can outrank existing correct answers. If someone
imported a large document set, retrieval on old questions can degrade sharply — the new
documents are genuinely more similar to the query than the right answer is.

```python
def check_corpus_change(traces_before, traces_after):
    """Did retrieved documents start coming from somewhere new?"""
    def sources(traces):
        return Counter(r["source"] for t in traces for r in t.retrieved)
    return {"before": sources(traces_before), "after": sources(traces_after)}
```

If 41% of retrieved chunks suddenly come from a source that didn't exist last week,
that's your answer.

**The provider updated the model.** Your config didn't change; their model did. Detect it
by replaying a fixed set of requests periodically and watching for behavior shifts.

---

## The three-question triage

For any complaint, ask these in order. Most cases resolve in the first two.

**1. Was the right document retrieved?**
No → retrieval problem. Stop here and fix retrieval.

**2. Did it make it into the prompt?**
No → reranking or context budget problem.

**3. Was the answer wrong anyway?**
Yes → generation problem. Now look at prompts and models.

The discipline is not skipping ahead. The most common wasted effort in RAG is prompt
engineering on a retrieval problem.

---

## Tools worth building

Small things that pay for themselves quickly.

**A retrieval inspector:**

```python
def inspect(question, index, embedder, k=20):
    """See exactly what comes back and why."""
    hits = index.search(embedder(question), k=k)
    for i, h in enumerate(hits, 1):
        print(f"{i:2d}. [{h.score:.3f}] {h.doc_id} — {h.text[:100]}...")

    scores = [h.score for h in hits]
    print(f"\nscore range: {max(scores):.3f} to {min(scores):.3f}")
    if max(scores) - min(scores) < 0.05:
        print("WARNING: flat scores — nothing is clearly relevant")
```

That flat-score warning is a useful signal on its own. It means the retriever couldn't
distinguish anything, which usually means the answer isn't there.

**A "why didn't you find this" tool:** the `why_not_retrieved` function above.

**Context replay:** re-run a past request with the exact context that was sent, so you
can test prompt changes against a real failure.

```python
async def replay(trace, new_prompt=None, new_model=None):
    """Re-run with the same context, changing one thing."""
    return await generate(
        question=trace.question,
        context=trace.context_sent,        # exactly what it saw before
        prompt=new_prompt or trace.prompt_version,
        model=new_model or trace.model,
    )
```

This isolates generation from retrieval completely. If replay with a better prompt fixes
it, you know it's a prompt problem.

---

## What to monitor

So you find problems before users report them:

```python
MONITOR = [
    "retrieval_score_top1",         # dropping = retrieval degrading
    "retrieval_score_spread",       # flat = nothing clearly relevant
    "retrieved_source_distribution", # shifts when the corpus changes
    "context_tokens_p50_p95",
    "citation_validity_rate",       # free to compute, catches real bugs
    "abstention_rate",              # rising = retrieval failing
    "empty_retrieval_rate",
    "rerank_promotion_rate",        # how often reranking changes the top 5
]
```

Two of these are early warnings most teams don't have:

**Abstention rate.** If "I don't have that information" goes from 4% to 15%, retrieval
broke. Users won't report it — they just leave.

**Retrieved source distribution.** Shifts immediately when documents are added or
removed. This is the alert that would catch the corpus-contamination case on the day it
happens rather than a week later.

---

## Interview questions

**1. "A user says the answer is wrong. Walk me through your investigation."**

The three-question triage: was the right document retrieved, did it reach the prompt, was
the answer wrong anyway. Emphasise not skipping ahead — prompt engineering on a retrieval
problem is the classic wasted week.

**2. "Groundedness is high but answers are wrong. What does that mean?"**

The model faithfully used the wrong documents. It's not hallucinating — it's a retrieval
failure. Two things that sound contradictory can both be true, and knowing that is the
point.

**3. "Someone imported 14,000 documents and quality dropped. Why?"**

New documents outrank the old correct ones. Check the source distribution of retrieved
chunks and recall@5 on your existing question set, before and after. Fix by scoping
retrieval by source, adding hybrid search and reranking, and treating corpus changes as
deploys that run the eval suite.

**4. "It was working last week and now it isn't. Nothing was deployed."**

Enumerate what can change without a deploy: documents added, provider model updated,
embedding model config changed by another team, prompt edited in a UI. Then say you'd
check the retrieval source distribution and replay a fixed request set.

**5. "What would you build to make RAG debuggable?"**

Log the full trace including retrieved chunks and all version identifiers. Build a
retrieval inspector, a "why wasn't this found" tool, and context replay. Monitor
abstention rate and source distribution.

**6. "Users say it's 'making things up.' How do you check?"**

Separate groundedness from correctness. If groundedness is high, it isn't making things
up — it's using wrong documents. If groundedness is low, check whether context is too
long and the right chunk is buried.

---

## What to remember

- One question first: **was the right information in the prompt?**
- Log the full trace — retrieved chunks, context sent, and every version.
- Three-question triage: retrieved → reached the prompt → answer wrong anyway.
- Don't skip ahead. Prompt tuning on a retrieval problem wastes weeks.
- High groundedness plus wrong answers means retrieval, not hallucination.
- Adding documents can break existing answers.
- Flat retrieval scores mean nothing relevant was found.
- Watch abstention rate and retrieved-source distribution — both move early.
- Build the inspector and the replay tool. They pay for themselves in a week.

---

**Back to:** [the RAG index](README.md)
