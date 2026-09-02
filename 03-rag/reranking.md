# Reranking

Retrieve 50 candidates cheaply, score them properly, keep the best 5.

This is usually the single highest-value thing you can add to a working RAG system, and
it has an unusual property: **it makes the system cheaper as well as better.** Most
quality improvements cost money. This one saves it.

---

## Why search alone isn't enough

Your vector search compares two vectors that were computed **separately**. The document
was embedded months ago with no knowledge of the question. The question is embedded now
with no knowledge of the document.

That's called a **bi-encoder** — two separate encodings, compared afterwards.

It's fast, because you can pre-compute every document vector and search millions of them
in milliseconds. It's also crude, because the model never actually reads the question and
the document together.

A **cross-encoder** does read them together:

```
bi-encoder (search):
    embed(question)   →  [0.2, 0.8, ...]
    embed(document)   →  [0.3, 0.7, ...]   (pre-computed)
    score = similarity of the two vectors

cross-encoder (reranking):
    score = model(question + document)     (computed now, together)
```

The cross-encoder can notice things the bi-encoder can't: that the document mentions the
right product but the wrong version, that it answers a similar question about a different
region, that it's about the topic but doesn't actually contain the answer.

**The trade-off is cost.** A bi-encoder searches 50 million documents in milliseconds. A
cross-encoder has to run once *per document*, so it can only score a few dozen.

Which is exactly why you use both: search narrows millions to 50, reranking orders those
50 properly.

---

## The shape

```python
def retrieve(query, k=5, candidates=50):
    # 1. Cheap and broad — get plenty of candidates
    hits = vector_index.search(embed(query), k=candidates)

    # 2. Expensive and accurate — but only on 50 items
    scored = reranker.score(query, [h.text for h in hits])

    # 3. Keep the best few
    ranked = sorted(zip(hits, scored), key=lambda x: -x[1])
    return [hit for hit, _ in ranked[:k]]
```

Three lines of real logic. The gains are large.

---

## Why it saves money

This is the part people find surprising, so it's worth spelling out.

**Without reranking**, you don't trust your ranking, so you send more chunks to be safe:

```
top 20 chunks × 400 tokens = 8,000 tokens of context per request
```

**With reranking**, you trust your top 5:

```
top 5 chunks × 400 tokens = 2,000 tokens of context per request
```

That's a 75% cut in retrieved-context tokens. Since input tokens usually dominate LLM
cost, this is a large saving.

And quality goes **up**, not down, because you removed 15 chunks of distracting,
marginally-relevant text. Models attend worse to long contexts, and irrelevant context
actively hurts — it gives the model wrong things to latch onto.

```python
def compare_cost(chunk_tokens, price_per_million, requests_per_month):
    for k, label in [(20, "no reranking"), (5, "with reranking")]:
        tokens = k * chunk_tokens * requests_per_month
        cost = tokens / 1_000_000 * price_per_million
        print(f"{label:15s} {k:2d} chunks  {tokens:>15,} tokens  ${cost:>10,.0f}")

# 400-token chunks, $0.50/M input, 5M requests/month:
#   no reranking     20 chunks   40,000,000,000 tokens  $   20,000
#   with reranking    5 chunks   10,000,000,000 tokens  $    5,000
```

The reranker itself costs something — but reranking 50 short chunks with a small model is
far cheaper than sending 15 extra chunks to a large model on every request.

---

## Kinds of reranker

### Cross-encoder model (the standard choice)

A small model trained to score query–document relevance. A few hundred million
parameters, runs on CPU at modest volume or cheaply on one GPU.

```python
def rerank(query, documents, model, batch_size=32):
    """Score each document against the query, together."""
    pairs = [(query, doc) for doc in documents]
    scores = []
    for i in range(0, len(pairs), batch_size):
        scores.extend(model.predict(pairs[i:i + batch_size]))
    return scores
```

Best quality-per-cost. This is what "reranking" usually means.

### Hosted reranking API

Several providers offer one. No infrastructure, good quality, per-call cost, and a
network round trip on your critical path.

Fine to start with. Watch the latency — you're adding a hop to every search.

### An LLM as reranker

Ask a general model to score relevance.

```python
PROMPT = """Rate how well this document answers the question, 0-3.
0 = irrelevant, 1 = related, 2 = useful, 3 = directly answers it.

Question: {query}
Document: {doc}

Reply with only the number."""
```

Works, but it's slow and expensive compared to a purpose-built cross-encoder. Reasonable
for prototyping or very low volume. Not a good production default.

### Cheap non-model reranking

Sometimes you don't need a model at all:

```python
def simple_rerank(query, hits):
    """Boost recency, exact matches, and authoritative sources."""
    query_terms = set(tokenize(query))
    for h in hits:
        h.score += 0.1 if query_terms & set(tokenize(h.title)) else 0
        h.score += 0.05 * recency_factor(h.updated_at)
        h.score += 0.1 if h.source == "official_docs" else 0
    return sorted(hits, key=lambda h: -h.score)
```

Free, instant, and it captures signals a cross-encoder doesn't know about — how fresh a
document is, how authoritative its source is. Worth combining with a real reranker
rather than choosing between them.

---

## Choosing the numbers

Two numbers: how many candidates to retrieve, and how many to keep.

### How many candidates?

More candidates means better chances the right document is in there, and more reranking
cost.

```python
async def tune_candidates(cases, options=(10, 20, 50, 100, 200)):
    for n in options:
        r = await measure(cases, candidates=n, final_k=5)
        print(f"candidates={n:3d}  recall@50={r.recall_50:.3f}  "
              f"final_recall@5={r.recall_5:.3f}  rerank_ms={r.latency:.0f}")
```

What you'll typically see: gains flatten out somewhere between 50 and 100. Beyond that
you're paying to rerank documents that were never going to win.

**Check your search recall first.** If recall@50 is 0.95, the reranker has good material
to work with. If recall@50 is 0.60, the right document usually isn't in the candidate
pool at all — and **no reranker can promote something that isn't there.** Fix retrieval
first.

That check is the most useful diagnostic in this file:

```
recall@50 high, recall@5 low   → reranking will help a lot
recall@50 low                  → fix search first; reranking won't save you
```

### How many to keep?

Depends on your token budget and whether answers usually need one source or several.

```python
async def tune_final_k(cases, options=(3, 5, 8, 12)):
    for k in options:
        r = await measure(cases, candidates=50, final_k=k)
        print(f"k={k:2d}  answer_quality={r.quality:.3f}  "
              f"context_tokens={r.avg_tokens:>5.0f}  cost=${r.cost:.4f}")
```

Common finding: quality plateaus around 5–8 chunks. Going from 8 to 20 costs 2.5× the
tokens for no measurable gain, and sometimes a small loss.

---

## Latency

Reranking adds time. Budget for it.

```
vector search:      ~20ms
rerank 50 chunks:   ~80ms      ← added
generation:       ~2000ms
```

Against a 2-second total, 80ms is nothing. Against a 300ms autocomplete requirement, it's
fatal.

Ways to keep it small:

**Batch the scoring.** One forward pass over 50 pairs, not 50 passes.

**Rerank fewer candidates.** 30 instead of 100.

**Skip it when the answer is obvious.** If the top search result is far ahead of the
rest, there's nothing to reorder:

```python
def needs_reranking(hits, margin=0.15):
    """If the top hit dominates, skip the reranker."""
    if len(hits) < 2:
        return False
    return (hits[0].score - hits[1].score) < margin
```

**Run it in parallel with something else** if your pipeline has independent work.

---

## Measuring whether it helped

Compare with and without, on the same queries:

```python
async def evaluate_reranking(cases):
    results = {"without": [], "with": []}
    for case in cases:
        hits = search(case.question, k=50)

        without = hits[:5]                             # trust search order
        with_rr = rerank_and_take(case.question, hits, 5)

        results["without"].append(recall(without, case.relevant_ids))
        results["with"].append(recall(with_rr, case.relevant_ids))

    return {
        "recall_without": mean(results["without"]),
        "recall_with": mean(results["with"]),
        "improvement": mean(results["with"]) - mean(results["without"]),
    }
```

Typical improvement on a system that didn't have it: **10–25 points of recall@5.** It's
one of the biggest single gains available.

Also measure the downstream effects, because they're the point:

- Context tokens per request (should drop a lot)
- Answer quality (should rise)
- Total cost per request (should drop)
- Latency (will rise slightly)

---

## Things that go wrong

**Reranking without enough candidates.** Reranking the top 5 is pointless — you're
reordering five things that were already the top five. Retrieve 30–50 minimum.

**Reranker trained on different content.** A model trained on web search may do poorly
on legal contracts or code. Test on your data.

**Input truncation.** Cross-encoders have input limits, often 512 tokens for query plus
document. If your chunks are 800 tokens, the reranker only sees part of each one — and it
doesn't tell you. Check the limit against your chunk size.

**Losing metadata signals.** A cross-encoder only sees text. It doesn't know a document
is three years out of date or from an unofficial source. Combine model scores with
metadata scores rather than replacing them.

**Becoming a latency bottleneck.** 200 candidates on a CPU can take a second. Measure at
your real candidate count.

**Not re-tuning after upstream changes.** Change chunk size or the embedding model and
your candidate pool changes. Re-check that your candidate count is still right.

---

## Interview questions

**1. "What's a reranker and why use one?"**

Search compares vectors computed separately; a cross-encoder reads the question and
document together and scores them properly. Search is fast and crude, reranking is slow
and accurate — use search to narrow, reranking to order.

**2. "Doesn't that cost more?"**

Usually the opposite, and this is the best answer available here. Reranking lets you send
5 chunks instead of 20, cutting input tokens by 75%. Since input dominates LLM cost, the
saving exceeds the reranker's cost — and quality improves because you removed distracting
context.

**3. "How many candidates should you rerank?"**

Measure it — typically 50 to 100, with diminishing returns after. Then add the important
check: look at recall@50 first. If the right document isn't in the candidate pool,
reranking can't help and you should fix search instead.

**4. "Reranking didn't improve anything. What would you check?"**

Search recall@50 first — if it's low, there's nothing good to promote. Then: are you
retrieving enough candidates? Is the reranker truncating your chunks? Was it trained on
similar content? Are you accidentally reranking only 5 items?

**5. "Where does reranking fit with hybrid search?"**

Hybrid improves recall (getting the right documents into the pool), reranking improves
precision (ordering them). Different jobs, they stack. Hybrid first, then rerank the
combined pool.

**6. "Your reranker adds 400ms and you have a 500ms budget. Options?"**

Fewer candidates, better batching, a smaller reranker, skipping it when the top result
clearly dominates, or running it in parallel with other work. Or accept that this product
can't afford it and improve retrieval instead.

---

## What to remember

- Search compares separately-computed vectors. Reranking reads query and document
  together.
- Retrieve many (50), rerank, keep few (5).
- It makes the system **cheaper and better** — fewer context tokens, less distraction.
- Check recall@50 before adding it. If the right document isn't in the pool, reranking
  can't help.
- Quality usually plateaus around 5–8 chunks in context.
- Watch the reranker's input limit against your chunk size.
- Combine model scores with metadata (freshness, source authority).
- Adds ~50–100ms. Fine for chat, fatal for autocomplete.

---

**Next:** [query-rewriting.md](query-rewriting.md) — fixing the question before you search
with it.
