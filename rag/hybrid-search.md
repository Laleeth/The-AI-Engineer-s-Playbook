# Hybrid Search

Hybrid search means running two kinds of search and combining the results: **meaning-based**
(embeddings) and **word-based** (keyword matching).

You need both because each fails badly where the other is strong.

---

## Where meaning-based search fails

Embeddings are good at "these two things mean roughly the same thing." That's exactly
wrong for exact identifiers.

Consider these:

```
"error code E-4471"
"error code E-4472"
```

To an embedding model these are nearly identical — both are short strings with a code in
the same context. Their vectors will be very close.

**Semantically that's correct. For retrieval it's a disaster.** The user wants E-4471
specifically, and you'll happily return E-4472.

The same problem hits:

- Product codes and SKUs — `WM-2200-B` vs `WM-2200-C`
- Version numbers — "the bug in v2.1.3"
- Person and company names — especially unusual ones
- Legal citations and section numbers
- Rare technical terms the embedding model has barely seen
- Anything where a near-synonym is *wrong* — "negligence" vs "gross negligence"

Keyword search has no such problem. `E-4471` either appears in the document or it
doesn't.

## Where word-based search fails

The opposite case:

```
user asks:  "how do I get my money back"
document:   "Refund Policy: customers may return items within 30 days"
```

Not one word in common except "I" and "my". Keyword search finds nothing. Embeddings
find it immediately.

---

## So: run both

```
query
  ├─→ vector search  → 50 candidates
  └─→ keyword search → 50 candidates
                ↓
            combine
                ↓
          top 10 results
```

The combination step is the interesting part.

---

## BM25: the keyword half

BM25 is the standard keyword ranking function. It scores a document higher when:

- The query terms appear often in it
- Those terms are **rare** across the whole corpus (a rare word matching means more)
- The document isn't padded out with unrelated text

Here's an implementation, because understanding it matters more than importing it:

```python
import math
import re
from collections import Counter, defaultdict

# Keep identifiers intact: e-4471 and v2.1.3 stay as single tokens.
# This is the whole point of adding keyword search — don't tokenize it away.
TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text):
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1          # how much repeated terms help
        self.b = b            # how much to penalise long documents
        self.doc_len = []
        self.postings = defaultdict(dict)    # term -> {doc_index: count}
        self.avgdl = 0.0
        self.n = 0

    def add(self, texts):
        for text in texts:
            tokens = tokenize(text)
            idx = len(self.doc_len)
            self.doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term][idx] = freq
        self.n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

    def _idf(self, term):
        """Rare terms are worth more."""
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return max(1e-6, math.log(1 + (self.n - df + 0.5) / (df + 0.5)))

    def search(self, query, k=50):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            for doc_idx, freq in posting.items():
                norm = 1 - self.b + self.b * (self.doc_len[doc_idx] / (self.avgdl or 1))
                scores[doc_idx] += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * norm
                )
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]
```

The tokenizer is the part to get right. A naive `\w+` split turns `E-4471` into `e` and
`4471`, destroying exactly the signal you added keyword search to capture.

Note also that BM25 only scores documents containing at least one query term — the
inverted index means cost scales with matches, not corpus size. It's fast.

---

## Combining the two lists

You have two ranked lists with incomparable scores. BM25 scores are unbounded and depend
on your corpus. Cosine similarity is between -1 and 1. You can't just add them.

### Option 1: normalise and add (works, but fragile)

```python
def weighted_fusion(dense, sparse, alpha=0.5):
    """Normalise each list to 0-1, then blend."""
    def normalise(results):
        if not results:
            return {}
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {doc: (s - lo) / span for doc, s in results}

    d, s = normalise(dense), normalise(sparse)
    combined = {}
    for doc in set(d) | set(s):
        combined[doc] = alpha * d.get(doc, 0) + (1 - alpha) * s.get(doc, 0)
    return sorted(combined.items(), key=lambda kv: -kv[1])
```

The problem: normalising depends on the min and max *in this result set*. If one search
returns three results and the other returns fifty, the normalisation means different
things. It's unstable, especially on queries where one method finds almost nothing.

### Option 2: Reciprocal Rank Fusion (better default)

Combine by **position**, not score. Position is always comparable.

```python
def rrf(rankings, k=60, weights=None):
    """Reciprocal Rank Fusion.

    Each list contributes 1/(k + rank) for each document. No score
    normalisation needed, which sidesteps the whole comparability problem.

    k=60 is the common default. Smaller k weights the top results more
    heavily; larger k flattens the difference between positions.
    """
    weights = weights or [1.0] * len(rankings)
    fused = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])
```

**Use RRF as your default.** It's simple, robust, needs no tuning, and it can't be
destabilised by one search returning few results.

Weighted score fusion can beat it if you tune the weights on your data — but you have to
actually tune them, and re-tune when anything upstream changes.

---

## Putting it together

```python
def hybrid_search(query, k=10, candidates=50):
    dense_hits = vector_index.search(embed(query), candidates)
    sparse_hits = bm25.search(query, candidates)

    dense_ranking = [d.id for d in dense_hits]
    sparse_ranking = [chunk_ids[i] for i, _ in sparse_hits]

    fused = rrf([dense_ranking, sparse_ranking])
    return [chunks[doc_id] for doc_id, _ in fused[:k]]
```

Two details that matter:

**Retrieve more candidates than you need.** 50 from each, fuse, keep 10. A document
ranked 30th by one method and 5th by the other should surface — and it won't if you only
took the top 10 from each.

**Apply permission filters inside both searches**, not after fusion. Same reasoning as
everywhere else: post-filtering returns too few results and means you loaded data the
user can't see.

---

## When it's worth it

Hybrid search adds a second index to build, maintain, and keep in sync. It's not free.

**Clearly worth it when your content has:**
- Product codes, SKUs, part numbers
- Error codes, log identifiers
- Version numbers
- People's names, company names
- Legal citations, statute references
- Domain jargon the embedding model won't know well

**Probably not worth it when:**
- Content is general prose with no identifiers
- Queries are conversational questions
- You're early and haven't measured a problem yet

**How to tell:** find the queries your current system fails on and look at them. If
you're seeing "user searched for an exact term and got something similar but wrong,"
that's the hybrid search signal.

---

## Measuring the improvement

Measure on the queries hybrid search is meant to fix, not on your whole set. A 2-point
overall improvement can hide a 30-point improvement on identifier queries, which is the
thing you actually built.

```python
async def measure_hybrid(cases, dense_only, hybrid):
    """Segment by query type — the overall number hides the win."""
    by_type = defaultdict(lambda: {"dense": [], "hybrid": []})

    for case in cases:
        qtype = case.segments["query_type"]     # "identifier", "conceptual", "mixed"
        by_type[qtype]["dense"].append(await recall(dense_only, case))
        by_type[qtype]["hybrid"].append(await recall(hybrid, case))

    return {
        qtype: {
            "dense": mean(v["dense"]),
            "hybrid": mean(v["hybrid"]),
            "gain": mean(v["hybrid"]) - mean(v["dense"]),
            "n": len(v["dense"]),
        }
        for qtype, v in by_type.items()
    }
```

Typical result:

```
query_type      n     dense   hybrid   gain
identifier     40     0.42    0.91    +0.49    ← this is why you built it
conceptual    120     0.84    0.83    -0.01    ← no harm
mixed          40     0.71    0.82    +0.11
```

Note the middle row. Hybrid search shouldn't hurt conceptual queries. If it does, your
fusion weights are wrong — you're letting keyword matching dominate where it shouldn't.

---

## Things that go wrong

**The tokenizer destroys identifiers.** `\w+` splits `E-4471` into two useless tokens.
Test your tokenizer on real identifiers from your corpus.

**BM25 statistics go stale.** BM25 needs corpus-wide statistics (how rare is each term).
With incremental indexing these drift. Recompute periodically, or use an implementation
that maintains them.

**Two indexes drift out of sync.** A document is added to one and not the other.
Then fusion silently mixes up documents, or a document is unfindable by one route. Index
both in the same transaction, and add a consistency check.

```python
def check_sync():
    assert vector_index.count == bm25.n, "indexes are out of sync"
```

**Keyword search dominates short queries.** A one-word query gets a strong BM25 signal
and a weak dense signal, so keyword results take over. Watch this in your fusion
weights.

**Stemming decisions.** Should "running" match "run"? Stemming helps prose and hurts
identifiers. If you stem, don't stem tokens that look like codes.

---

## Interview questions

**1. "Why isn't vector search enough?"**

Give the E-4471 vs E-4472 example. Embeddings map them to nearly the same vector, which
is semantically correct and operationally catastrophic. Then list the categories:
identifiers, versions, names, rare terms.

**2. "How do you combine two ranked lists?"**

RRF, and explain why: BM25 and cosine scores aren't comparable, and normalising is
unstable when one list is short. Combining by rank sidesteps it entirely.

**3. "When would you not use hybrid search?"**

General prose with no identifiers, conversational queries, or before you've measured a
problem. It's a second index to maintain — it should earn its place.

**4. "How do you know it helped?"**

Segment by query type. The overall number hides the win. Show that identifier queries
improve a lot and conceptual queries shouldn't get worse.

**5. "Your hybrid search made conceptual queries worse. Why?"**

Fusion weights let keyword matching dominate, probably on short queries where the dense
signal is weak. Retune, or weight by query characteristics.

**6. "How does hybrid search interact with reranking?"**

They stack well. Hybrid gets the right documents into the candidate pool; the reranker
orders them properly. Hybrid improves recall, reranking improves precision — different
jobs.

---

## What to remember

- Embeddings fail on exact identifiers because near-identical strings get near-identical
  vectors.
- Keyword search fails on paraphrase.
- Run both, fuse the results.
- Use RRF by default — no score normalisation, no tuning, robust.
- Your tokenizer must keep identifiers intact. Test it.
- Retrieve 50 from each, fuse, keep 10.
- Filter permissions inside both searches, not after.
- Measure by query type. The overall number hides the point.
- Keep the two indexes in sync, and assert it.

---

**Next:** [reranking.md](reranking.md) — the highest-value addition to most RAG systems.
