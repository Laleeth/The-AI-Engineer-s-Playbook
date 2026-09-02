# Coding Round: RAG From Scratch

The purpose of this round is not to see whether you can call a framework. It is to see
whether you understand what the framework is doing — because when retrieval quality is
bad in production, the fix is almost never "call a different function." It's in the
chunking, the scoring, the fusion, or the context construction, and you can't debug what
you've never built.

Three problems, in increasing order of realism:

1. [Minimal end-to-end RAG](#problem-1-minimal-end-to-end-rag) — the core exercise.
2. [Hybrid retrieval and fusion](#problem-2-hybrid-retrieval) — where most quality lives.
3. [Permissions and updates](#problem-3-permissions-and-document-updates) — where most
   production bugs live.

Python 3.11+, numpy. Embeddings and generation are injected as functions so the code is
testable without a provider.

---

## Problem 1: Minimal End-to-End RAG

## Scenario

Build the pipeline:

```
documents → chunking → embeddings → index → query → retrieval
          → context construction → LLM → answer with citations
```

You may use numpy. You may not use a vector database, a retrieval library, or an
orchestration framework. The corpus is 50,000 chunks — small enough that brute-force
search is legitimate, which is itself worth knowing.

## Requirements

- Chunk documents with configurable size and overlap, **preserving the character offsets
  of each chunk in its source document** (this is what makes citation possible).
- Embed chunks in batches.
- Build a searchable index with cosine similarity.
- Retrieve top-k for a query.
- Construct a context that fits a token budget.
- Generate an answer that cites its sources.
- Return the citations with enough information to locate the original text.

## Starter Code

```python
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    text: str
    metadata: dict


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int          # character offset in the source document
    end: int
    metadata: dict


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    citations: list[Chunk]
    context_tokens: int


class RAG:
    def index(self, documents: list[Document]) -> None: ...
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...
    def answer(self, query: str, k: int = 5) -> Answer: ...
```

## Expected Behavior

```python
rag = RAG(embed_fn=embed, generate_fn=generate)
rag.index(load_documents())

hits = rag.retrieve("what is the refund window?", k=5)
assert all(0.0 <= h.score <= 1.0 for h in hits)
assert hits == sorted(hits, key=lambda h: -h.score)

ans = rag.answer("what is the refund window?")
assert ans.citations                       # never answer without sources
assert ans.context_tokens <= BUDGET
# every citation must be locatable:
doc = corpus[ans.citations[0].doc_id]
assert doc.text[c.start:c.end] == c.text
```

---

<details>
<summary>💡 Reveal a hint</summary>

**Chunking is where quality is won or lost, and it's the part people rush.** Two
properties matter more than the chunk size:

1. **Chunks must not split mid-sentence** if you can avoid it. A chunk that begins
   "...therefore the refund window is" is nearly useless to a model and embeds poorly.
2. **Offsets must be exact.** If `doc.text[chunk.start:chunk.end] != chunk.text`, your
   citations are wrong and you won't discover it until a user does. Assert it in a test.

**Overlap exists to stop answers falling between chunks.** With 512-token chunks and no
overlap, a fact spanning the boundary is in neither chunk in a usable form. 10–20%
overlap is typical. Overlap costs storage and creates near-duplicate retrievals, which
is why deduplication at retrieval time matters.

**Normalize embeddings once, at index time.** If every vector has unit norm, cosine
similarity is a dot product, which is one matrix multiply for the entire corpus:

```python
scores = self.matrix @ query_vec        # (N, D) @ (D,) -> (N,)
```

For 50,000 chunks × 768 dimensions that's ~38M multiply-adds — a few milliseconds in
numpy. **Brute force is genuinely the correct choice at this scale**, and saying so is a
strong signal: a huge number of "we need a vector database" decisions are made for
corpora that fit comfortably in a numpy array.

**Context construction is a budget, not a concatenation.** Take chunks in score order
until the budget is exhausted; skip a chunk that doesn't fit rather than truncating it
mid-sentence.
</details>

---

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""Minimal RAG, built from primitives.

Deliberately simple: no ANN index (brute force is correct at 50k chunks), no
framework. The parts that matter for quality — chunking with offsets, budgeted
context construction, citation validation — are explicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    citations: list[Chunk]
    context_tokens: int
    retrieved: list[RetrievedChunk]


EmbedFn = Callable[[Sequence[str]], np.ndarray]     # (n,) texts -> (n, d) array
GenerateFn = Callable[[str], str]


# --------------------------------------------------------------------------
# Tokenization (approximate)
# --------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Approximate token count.

    In production use the model's real tokenizer: this approximation is fine for
    chunk sizing but wrong enough to matter for a hard context budget, where
    being 20% under is safe and being 5% over is a truncation incident.
    """
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_document(
    doc: Document,
    *,
    target_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Split a document into overlapping, sentence-aligned chunks.

    Invariant maintained: doc.text[chunk.start:chunk.end] == chunk.text.
    Everything downstream (citations, highlighting, provenance) depends on it.
    """
    # Split into sentences while tracking offsets. Regex sentence splitting is
    # crude; production systems use a real segmenter, and for structured formats
    # (Markdown, HTML) you should split on structural boundaries first.
    sentences: list[tuple[int, int]] = []
    pos = 0
    for part in _SENTENCE_END.split(doc.text):
        if not part:
            continue
        start = doc.text.find(part, pos)
        if start == -1:                     # defensive: should not happen
            continue
        end = start + len(part)
        sentences.append((start, end))
        pos = end

    if not sentences:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        start = sentences[i][0]
        end = sentences[i][1]
        tokens = count_tokens(doc.text[start:end])
        j = i + 1

        while j < len(sentences):
            candidate_end = sentences[j][1]
            candidate_tokens = count_tokens(doc.text[start:candidate_end])
            if candidate_tokens > target_tokens:
                break
            end = candidate_end
            tokens = candidate_tokens
            j += 1

        text = doc.text[start:end]
        chunks.append(Chunk(
            id=f"{doc.id}#{len(chunks)}",
            doc_id=doc.id,
            text=text,
            start=start,
            end=end,
            metadata=dict(doc.metadata),
        ))

        if j >= len(sentences):
            break

        # Step back far enough to create the overlap, but always advance at
        # least one sentence or we loop forever on a long sentence.
        back = j - 1
        overlap = 0
        while back > i and overlap < overlap_tokens:
            overlap += count_tokens(doc.text[sentences[back][0]:sentences[back][1]])
            back -= 1
        i = max(i + 1, back + 1)

    return chunks


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

class VectorIndex:
    """Brute-force cosine similarity over normalized vectors.

    At 50k x 768 this is ~150 MB and a few milliseconds per query. An ANN index
    is unnecessary complexity until the corpus is much larger or QPS is high.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None     # (N, D), L2-normalized
        self._deleted: set[int] = set()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunk/vector count mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors.extend(list(vectors / norms))
        self.chunks.extend(chunks)
        self._matrix = None                        # invalidate

    def delete_by_doc(self, doc_id: str) -> int:
        """Tombstone deletion. Compaction is a separate concern."""
        n = 0
        for i, chunk in enumerate(self.chunks):
            if chunk.doc_id == doc_id and i not in self._deleted:
                self._deleted.add(i)
                n += 1
        return n

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.vstack(self._vectors) if self._vectors
                else np.zeros((0, self.dim), dtype=np.float32)
            )
        return self._matrix

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
        *,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> list[RetrievedChunk]:
        matrix = self._ensure_matrix()
        if matrix.shape[0] == 0:
            return []

        q = query_vec / (np.linalg.norm(query_vec) or 1.0)
        scores = matrix @ q                          # cosine, since both unit norm

        # Filtering BEFORE top-k, not after. Post-filtering silently returns
        # fewer than k results and, at scale, returns nothing at all for
        # selective filters — the single most common retrieval bug in
        # multi-tenant systems.
        mask = np.ones(len(scores), dtype=bool)
        if self._deleted:
            mask[list(self._deleted)] = False
        if filter_fn is not None:
            for i, chunk in enumerate(self.chunks):
                if mask[i] and not filter_fn(chunk):
                    mask[i] = False

        scores = np.where(mask, scores, -np.inf)
        take = min(k, int(mask.sum()))
        if take <= 0:
            return []

        idx = np.argpartition(-scores, take - 1)[:take]
        idx = idx[np.argsort(-scores[idx])]
        return [RetrievedChunk(self.chunks[int(i)], float(scores[int(i)])) for i in idx]


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

PROMPT = """Answer the question using ONLY the context below.
Cite the sources you used with their [id]. If the context does not contain the
answer, say "I don't have enough information to answer that." Do not use outside
knowledge.

Context:
{context}

Question: {question}

Answer:"""


class RAG:
    def __init__(
        self,
        embed_fn: EmbedFn,
        generate_fn: GenerateFn,
        *,
        dim: int = 768,
        target_chunk_tokens: int = 400,
        overlap_tokens: int = 60,
        context_budget_tokens: int = 3000,
        batch_size: int = 128,
    ) -> None:
        self.embed = embed_fn
        self.generate = generate_fn
        self.index_ = VectorIndex(dim)
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.context_budget = context_budget_tokens
        self.batch_size = batch_size
        self.documents: dict[str, Document] = {}

    def index(self, documents: list[Document]) -> int:
        all_chunks: list[Chunk] = []
        for doc in documents:
            self.documents[doc.id] = doc
            all_chunks.extend(chunk_document(
                doc,
                target_tokens=self.target_chunk_tokens,
                overlap_tokens=self.overlap_tokens,
            ))

        # Batch embedding: one call per 128 chunks rather than per chunk.
        # This is usually a 10-50x throughput difference.
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            vectors = self.embed([c.text for c in batch])
            self.index_.add(batch, vectors)

        return len(all_chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> list[RetrievedChunk]:
        query_vec = self.embed([query])[0]
        return self.index_.search(query_vec, k, filter_fn=filter_fn)

    def build_context(self, hits: list[RetrievedChunk]) -> tuple[str, list[Chunk], int]:
        """Fill the token budget in score order, skipping chunks that don't fit.

        Two decisions worth noting:
          - dedupe by (doc_id, overlapping span): overlap means adjacent chunks
            can be near-duplicates and waste budget
          - never truncate a chunk mid-way; a partial chunk produces partial
            answers and broken citations
        """
        used = 0
        parts: list[str] = []
        cited: list[Chunk] = []
        seen_spans: list[tuple[str, int, int]] = []

        for hit in hits:
            c = hit.chunk
            if any(c.doc_id == d and not (c.end <= s or c.start >= e)
                   for d, s, e in seen_spans):
                continue                        # overlapping duplicate

            block = f"[{c.id}] (from {c.doc_id})\n{c.text}\n"
            cost = count_tokens(block)
            if used + cost > self.context_budget:
                continue                        # skip, don't truncate
            parts.append(block)
            cited.append(c)
            seen_spans.append((c.doc_id, c.start, c.end))
            used += cost

        return "\n".join(parts), cited, used

    def answer(
        self,
        query: str,
        k: int = 5,
        *,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> Answer:
        hits = self.retrieve(query, k, filter_fn=filter_fn)
        if not hits:
            return Answer(
                text="I don't have enough information to answer that.",
                citations=[], context_tokens=0, retrieved=[],
            )

        context, cited, used = self.build_context(hits)
        text = self.generate(PROMPT.format(context=context, question=query))

        # Only report citations the model actually referenced. A citation list
        # that is just "everything we retrieved" is not a citation list, and
        # users lose trust the first time they click one that isn't relevant.
        referenced = [c for c in cited if f"[{c.id}]" in text]
        return Answer(
            text=text,
            citations=referenced or cited,
            context_tokens=used,
            retrieved=hits,
        )


# --------------------------------------------------------------------------
# Correctness checks worth having as tests
# --------------------------------------------------------------------------

def verify_offsets(doc: Document, chunks: list[Chunk]) -> None:
    for c in chunks:
        assert doc.text[c.start:c.end] == c.text, f"offset mismatch in {c.id}"


def verify_coverage(doc: Document, chunks: list[Chunk]) -> float:
    """Fraction of the document covered by at least one chunk.

    Should be ~1.0. Anything lower means content is unretrievable, which is a
    silent failure: the system will confidently say it has no information about
    text that is sitting in the corpus.
    """
    if not doc.text:
        return 1.0
    covered = np.zeros(len(doc.text), dtype=bool)
    for c in chunks:
        covered[c.start:c.end] = True
    return float(covered.mean())
```

</details>

---

<details>
<summary>📖 Reveal the explanation</summary>

### Decisions that matter

**Sentence-aligned chunking with exact offsets.** The `verify_offsets` assertion is not
decoration — it's the invariant that makes citation, highlighting, and provenance
possible. Systems that chunk by splitting on whitespace and rejoining lose the mapping
back to the source and can never show a user where an answer came from.

**`verify_coverage` catches a silent failure class.** If a chunker drops content — a
common bug with regex-based splitting on unusual inputs — the system will confidently
say it has no information about text that is in the corpus. Nobody reports this bug,
because the failure looks like a correct "I don't know." Measuring coverage at index
time is cheap and catches it.

**Overlap is computed by stepping back over sentences**, with a guaranteed forward
advance (`i = max(i + 1, back + 1)`). Without that guard, a single sentence longer than
the target chunk size loops forever — a real bug in real chunkers.

**Filtering before top-k.** In `VectorIndex.search`, the mask is applied to the scores
*before* the top-k selection. The tempting alternative — take top-k then filter — returns
fewer than k results, and for a selective filter (one tenant out of 500) usually returns
zero. This is the single most common multi-tenant retrieval bug, and the reason it's
worth writing brute-force search yourself once: with a vector database, the equivalent
mistake is applying the filter in application code instead of in the query.

**Normalize at index time.** Storing unit vectors means cosine similarity is a single
matrix-vector product. It also makes scores directly comparable, which matters for
thresholding and fusion.

**Tombstone deletion.** `delete_by_doc` marks rather than removes, because removing from
the middle of a numpy matrix means rebuilding it. Real systems do exactly this, with
periodic compaction — and the tombstones must be excluded from search, which is easy to
forget and produces "deleted documents still appear in results."

**Context as a budget with skip-not-truncate.** Chunks are added in score order until the
budget is exhausted; a chunk that doesn't fit is skipped rather than cut. Truncating
produces sentence fragments that generate partial answers and unverifiable citations.

**Overlap-aware deduplication in context building.** Because chunks overlap by design,
the top-5 can contain three near-identical passages, wasting most of the budget on the
same text. Filtering overlapping spans from the same document typically frees 20–30% of
the budget for genuinely different content.

**Citations filtered to what the model referenced.** Returning every retrieved chunk as a
"citation" trains users to distrust citations, because most of them won't support the
answer. Checking which IDs appear in the output is crude but far better; a production
system would additionally validate that the cited chunk actually supports the claim.

**The prompt's abstention instruction is load-bearing.** "If the context does not contain
the answer, say I don't have enough information" is the difference between a system that
fails visibly and one that hallucinates confidently. It should also be *evaluated* —
abstention is a behavior with its own precision and recall.

### Complexity

- Chunking: `O(n)` in document length.
- Indexing: `O(n_chunks)` embedding calls (batched), `O(n × d)` memory.
- Search: `O(n × d)` per query — 50,000 × 768 ≈ 38M FLOPs, a few milliseconds in numpy.
- Memory: 50,000 × 768 × 4 bytes ≈ **154 MB** for float32. Worth computing out loud;
  this is what tells you whether brute force is viable.

At roughly 1M vectors (~3 GB) brute force becomes uncomfortable and an ANN index earns
its complexity. Below that, it usually doesn't.

</details>

---

## Production Improvements

- **ANN index** (HNSW/IVF) once the corpus outgrows memory-resident brute force. Note
  that this trades exact recall for speed and introduces the filtered-search problem
  described in `../12-senior-scenarios/build-vs-buy.md`.
- **Reranking** with a cross-encoder: retrieve 50, rerank, keep 5. Usually the single
  largest quality improvement available, and it *reduces* prompt cost.
- **Hybrid retrieval** (Problem 2) — dense retrieval alone is weak on exact identifiers,
  product names, and rare terms.
- **Query rewriting** for conversational queries ("what about the second one?" is
  unretrievable without rewriting against history).
- **Structure-aware chunking**: split Markdown by heading, code by function, tables as
  units. Generic character chunking destroys tables in particular.
- **Contextual chunk enrichment**: prepend the document title and section heading to each
  chunk's text before embedding, so a chunk reading "It is 30 days" becomes retrievable.
  Cheap, and often a large recall win.
- **Incremental indexing** with per-document versioning (Problem 3).
- **Embedding model version bound to the index** so a query embedded with model X can
  never search an index built with model Y.
- **Retrieval evaluation**: recall@k and nDCG on a labeled set, run on every change to
  chunking, embedding, or retrieval.

## Edge Cases

| Case | Correct behavior |
|---|---|
| Empty document | Produces zero chunks; must not crash the indexing loop |
| A single sentence longer than `target_tokens` | Emitted as an oversized chunk; the forward-advance guard prevents an infinite loop |
| Document with no sentence terminators | Falls back to one chunk; a production chunker needs a length-based fallback |
| Query embeds to a zero vector | Guarded by the `or 1.0` in normalization |
| `k` larger than the corpus | Returns everything available, not an error |
| All chunks filtered out | Returns empty; `answer` must then abstain, not generate |
| Every retrieved chunk exceeds the budget alone | Context is empty — detect this and either abstain or use a smaller-chunk fallback |
| Duplicate documents indexed twice | Duplicated chunks with different IDs; needs content-hash dedup at ingest |
| Model cites an ID not in the context | Filter it out and count it — hallucinated citations are a measurable failure |
| Unicode / multi-byte text | Offsets are character-based, consistent with Python slicing; be careful if any consumer uses byte offsets |

## Follow-up Questions

1. **How would you improve retrieval quality** without changing the embedding model?
   Name three things and rank them.
2. **How would you add hybrid search?** What's the fusion strategy and why?
3. **How would you evaluate this?** Be specific about the dataset and the metrics.
4. **How do you handle document updates?** A document is edited; what has to happen?
5. **How do you enforce permissions?** The user may only see some documents.
6. **How would you scale to 50M chunks?** What changes first?
7. **A user says the answer is wrong.** How do you determine whether it was a retrieval
   failure or a generation failure?

## Senior-Level Discussion

**Chunking is the highest-leverage and least-glamorous decision in RAG.** Chunk too
small and each chunk lacks context to be interpretable; too large and the embedding
becomes a blurry average of several topics, and you waste prompt budget on irrelevant
text. There's no universally right size — it depends on the document structure and the
query type — which means it's an empirical question you should be measuring, not a
constant you should be copying.

**Retrieval failure and generation failure need to be separable.** When an answer is
wrong, the first question is always: *was the right context in the prompt?* If yes, it's
a generation/prompting problem. If no, it's retrieval. These have completely different
fixes and conflating them wastes weeks. Design the system so this is a one-query
question — log the retrieved chunks with every answer.

**Citations are a correctness feature, not a UI feature.** They let a user verify, which
converts a system that must be right into a system that must be *checkable*. That's a
much lower bar and a much more useful product. But they only work if they're accurate:
a citation that doesn't support the claim is worse than none, because it manufactures
false confidence. Measure citation validity as a separate metric from groundedness.

**Brute force is underrated.** A very large fraction of production RAG systems have
corpora that fit comfortably in memory as a numpy array, and would be faster, simpler,
exactly-accurate, and cheaper without an ANN index. The instinct to reach for a vector
database before doing the arithmetic is one of the more expensive habits in this field.

**The abstention path is a first-class feature.** "I don't have enough information" is
correct behavior and it must be prompted for, tested, and measured. A system that never
abstains is a system that always hallucinates when retrieval fails — and retrieval fails
regularly.

---

## Problem 2: Hybrid Retrieval

## Scenario

Your dense retriever fails on a specific class of query: exact identifiers ("error code
E-4471"), product names, version numbers, and rare technical terms. Users notice,
because those are exactly the queries where they know the right answer exists.

## Requirements

Add lexical retrieval (BM25) and fuse it with dense retrieval. Implement BM25 yourself.

<details>
<summary>💡 Reveal a hint</summary>

**Why dense retrieval fails on identifiers.** Embedding models map text to a semantic
space where "E-4471" and "E-4472" are nearly identical vectors — they're both short
alphanumeric strings in similar contexts. Semantically that's correct; for retrieval
it's catastrophic. Lexical matching has the opposite property: it's exact, and it's blind
to paraphrase.

**Fusion is the interesting part.** Two options:

1. **Score normalization + weighted sum.** Requires the scores to be comparable, which
   they aren't — BM25 is unbounded and corpus-dependent; cosine is in [-1, 1]. You have
   to normalize per query, which is unstable when one retriever returns few results.
2. **Reciprocal Rank Fusion (RRF).** Combine by *rank*, not score:
   `score(d) = Σ 1/(k + rank_i(d))`. Scale-free, needs no normalization, robust,
   and one parameter.

RRF is the right default and knowing why is the signal here: it sidesteps the score
comparability problem entirely.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""BM25 + dense retrieval with reciprocal rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Keep identifier-ish tokens intact: 'e-4471' and 'v2.1.3' survive as
    single tokens, which is the entire point of adding lexical search."""
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25.

    score(q, d) = Σ_t IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.avgdl = 0.0
        self.n = 0

    def add(self, texts: list[str]) -> None:
        for text in texts:
            toks = tokenize(text)
            idx = len(self.doc_tokens)
            self.doc_tokens.append(toks)
            self.doc_len.append(len(toks))
            for term, freq in Counter(toks).items():
                self.postings[term][idx] = freq
        self.n = len(self.doc_tokens)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        # Standard BM25 IDF with the +0.5 smoothing; the max() floor avoids
        # negative weights for terms appearing in >half the corpus.
        return max(1e-6, math.log(1 + (self.n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Score only documents containing at least one query term — the
        inverted index means cost scales with matches, not corpus size."""
        scores: dict[int, float] = defaultdict(float)
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


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked lists by rank rather than score.

    RRF avoids the score-normalization problem entirely: BM25 scores are
    unbounded and corpus-dependent, cosine scores are in [-1, 1], and any
    attempt to make them comparable is fragile. Ranks are always comparable.

    k=60 is the commonly-used constant; larger k flattens the contribution of
    rank position, smaller k weights the top results more heavily.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] += weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class HybridRAG(RAG):
    def __init__(self, *args, dense_weight: float = 1.0,
                 lexical_weight: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bm25 = BM25()
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def index(self, documents: list[Document]) -> int:
        n = super().index(documents)
        # BM25 is built over the same chunk list and in the same order, so
        # indices are shared between the two retrievers. If they ever diverge,
        # fusion silently mixes up documents — worth an assertion.
        self.bm25.add([c.text for c in self.index_.chunks])
        assert len(self.bm25.doc_tokens) == len(self.index_.chunks)
        return n

    def retrieve(self, query: str, k: int = 5, *, candidates: int = 50,
                 filter_fn=None) -> list[RetrievedChunk]:
        dense_hits = super().retrieve(query, candidates, filter_fn=filter_fn)
        dense_ranking = [self.index_.chunks.index(h.chunk) for h in dense_hits]

        lexical = self.bm25.search(query, candidates)
        lexical_ranking = [i for i, _ in lexical]
        if filter_fn is not None:
            lexical_ranking = [
                i for i in lexical_ranking if filter_fn(self.index_.chunks[i])
            ]

        fused = reciprocal_rank_fusion(
            [dense_ranking, lexical_ranking],
            weights=[self.dense_weight, self.lexical_weight],
        )
        return [
            RetrievedChunk(self.index_.chunks[i], score)
            for i, score in fused[:k]
        ]
```

**Note the `.index()` call on a list is O(n)** — fine for illustration, wrong for
production. Store the chunk's position on the chunk itself.
</details>

## Follow-up Questions

1. How do you tune the dense/lexical weights? What data do you need?
2. When would RRF be worse than weighted score fusion?
3. BM25 requires the full corpus statistics. How does that work with incremental
   indexing?
4. Should the query be tokenized the same way for both retrievers? What about stemming?
5. How does hybrid retrieval interact with a reranker?

---

## Problem 3: Permissions and Document Updates

## Scenario

Two requirements that break naive RAG:

1. **Permissions.** Users may only retrieve documents they're authorized to see.
   Authorization is per-document and changes frequently.
2. **Updates.** Documents are edited constantly. An edited document's old chunks must
   stop being retrievable within minutes.

## Requirements

Implement both correctly. State the failure modes of the obvious approaches.

<details>
<summary>💡 Reveal a hint</summary>

**Permissions: filter in the query, never after.** Post-filtering (retrieve top-k, then
drop unauthorized) fails two ways: it returns fewer than k results, and with a selective
filter it returns nothing useful because the top-k was dominated by documents the user
can't see. In an ANN index the failure is worse — the graph traversal spends its budget
on unreachable nodes and recall collapses.

**And permissions must be in the cache key.** A response cache keyed on the query alone
will serve one user's authorized answer to another user. This is the incident in
`../12-senior-scenarios/production-incidents.md#incident-5`. Key on the *resolved
permission set*, not the user ID — keying on user ID gives you a correct but useless
cache.

**Updates: version documents, don't mutate chunks.** The naive approach — find the
document's chunks, delete them, re-chunk, re-embed — has a window during which the
document is partially indexed and retrieval returns half the old version and half the
new. Better: index the new version under a new `doc_version`, then atomically flip which
version is live, then tombstone the old one.

**And the deletion must be visible immediately even if compaction is deferred.** A
tombstone that isn't checked at query time is not a deletion.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""Permission-aware retrieval and versioned document updates."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset[str]

    def cache_scope(self) -> str:
        """Cache key component based on the resolved permission set, NOT the
        user id. Keying on user id would give each user a private cache with a
        near-zero hit rate; keying on the group set lets users with identical
        permissions share entries safely."""
        return hashlib.sha256(
            "|".join(sorted(self.groups)).encode()
        ).hexdigest()[:16]


@dataclass
class DocumentACL:
    doc_id: str
    allowed_groups: frozenset[str]
    classification: str = "internal"

    def visible_to(self, principal: Principal) -> bool:
        return bool(self.allowed_groups & principal.groups)


@dataclass
class VersionedChunk(Chunk):
    doc_version: int = 0
    indexed_at: float = field(default_factory=time.time)
    live: bool = True


class SecureRAG(HybridRAG):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.acls: dict[str, DocumentACL] = {}
        self.live_version: dict[str, int] = {}

    # ---------------- permissions ----------------

    def retrieve_for(
        self, query: str, principal: Principal, k: int = 5
    ) -> list[RetrievedChunk]:
        """Retrieve with the permission filter applied inside the search.

        The filter is evaluated during candidate selection, so k results are
        k results the user may see — not k results minus the ones we had to
        drop afterwards.
        """
        def allowed(chunk: Chunk) -> bool:
            if not getattr(chunk, "live", True):
                return False
            if self.live_version.get(chunk.doc_id) != getattr(chunk, "doc_version", 0):
                return False                      # stale version
            acl = self.acls.get(chunk.doc_id)
            if acl is None:
                return False                      # fail closed on unknown ACL
            return acl.visible_to(principal)

        return self.retrieve(query, k, filter_fn=allowed)

    def answer_for(self, query: str, principal: Principal, k: int = 5) -> Answer:
        hits = self.retrieve_for(query, principal, k)
        if not hits:
            return Answer("I don't have enough information to answer that.",
                          [], 0, [])
        context, cited, used = self.build_context(hits)
        text = self.generate(PROMPT.format(context=context, question=query))
        referenced = [c for c in cited if f"[{c.id}]" in text]
        return Answer(text, referenced or cited, used, hits)

    def cache_key(self, query: str, principal: Principal) -> str:
        """Every dependency of the answer must appear here. A missing dimension
        is a class of bug you have chosen to ship."""
        return "|".join([
            hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16],
            principal.cache_scope(),              # permissions
            self.corpus_version(),                # corpus content
            PROMPT_VERSION,                       # prompt
            MODEL_VERSION,                        # model
            EMBEDDING_MODEL_VERSION,              # retrieval
        ])

    def corpus_version(self) -> str:
        """Cheap corpus fingerprint so a reindex invalidates cached answers."""
        h = hashlib.sha256()
        for doc_id in sorted(self.live_version):
            h.update(f"{doc_id}:{self.live_version[doc_id]}".encode())
        return h.hexdigest()[:16]

    # ---------------- updates ----------------

    def upsert_document(self, doc: Document, acl: DocumentACL) -> int:
        """Index a new version, then flip atomically, then tombstone the old.

        Order matters. Indexing before flipping means there is never a moment
        when the document is partially retrievable; flipping before tombstoning
        means there is never a moment when it is missing entirely.
        """
        new_version = self.live_version.get(doc.id, 0) + 1

        chunks = chunk_document(
            doc,
            target_tokens=self.target_chunk_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        versioned = [
            VersionedChunk(
                id=f"{doc.id}@v{new_version}#{i}",
                doc_id=doc.id, text=c.text, start=c.start, end=c.end,
                metadata=c.metadata, doc_version=new_version,
            )
            for i, c in enumerate(chunks)
        ]

        for i in range(0, len(versioned), self.batch_size):
            batch = versioned[i:i + self.batch_size]
            self.index_.add(batch, self.embed([c.text for c in batch]))
        self.bm25.add([c.text for c in versioned])

        self.acls[doc.id] = acl
        self.documents[doc.id] = doc
        self.live_version[doc.id] = new_version    # <-- the atomic flip

        for chunk in self.index_.chunks:           # tombstone old versions
            if (chunk.doc_id == doc.id
                    and getattr(chunk, "doc_version", 0) < new_version):
                chunk.live = False                 # type: ignore[attr-defined]

        return len(versioned)

    def delete_document(self, doc_id: str) -> None:
        self.live_version.pop(doc_id, None)        # nothing is the live version
        self.acls.pop(doc_id, None)
        for chunk in self.index_.chunks:
            if chunk.doc_id == doc_id:
                chunk.live = False                 # type: ignore[attr-defined]
```

**Tests that must exist for this code:**

```python
def test_no_cross_permission_leakage():
    alice = Principal("alice", frozenset({"engineering"}))
    bob = Principal("bob", frozenset({"sales"}))
    rag.upsert_document(secret_eng_doc, DocumentACL(doc_id, frozenset({"engineering"})))

    assert rag.retrieve_for("secret", alice) != []
    assert rag.retrieve_for("secret", bob) == []
    assert rag.cache_key("secret", alice) != rag.cache_key("secret", bob)


def test_update_is_atomic():
    rag.upsert_document(v1, acl)
    before = rag.retrieve_for("content", principal)
    rag.upsert_document(v2, acl)
    after = rag.retrieve_for("content", principal)
    assert all(c.chunk.doc_version == 2 for c in after)   # no mixed versions


def test_deleted_document_is_unretrievable():
    rag.delete_document(doc_id)
    assert rag.retrieve_for("content", principal) == []
```
</details>

## Follow-up Questions

1. A user's group membership changes. What must be invalidated, and how quickly?
2. Your ACL check is a database call per chunk. How do you make retrieval fast?
3. How do you handle document-level ACLs when a *chunk* contains a mix of sensitive and
   non-sensitive content?
4. Tombstones accumulate. Design compaction. What's the failure mode during compaction?
5. How would you audit that no cross-permission leak has ever occurred?
6. The same document is visible to different groups with different redactions. Now what?

## Senior-Level Discussion

**Fail closed.** `if acl is None: return False` — an unknown ACL means invisible, not
visible. Every default in an authorization path should be the restrictive one, because
the failure mode of the permissive default is a breach and the failure mode of the
restrictive default is a support ticket.

**Cache keys must enumerate every dependency of the answer.** Query, permissions, corpus
version, prompt version, model version, embedding model version. Anything you leave out
is a bug you've chosen to ship, and the permissions one is the bug that ends up in a
regulatory filing. The `cache_key` method above is deliberately verbose; that verbosity
is the feature.

**Version, then flip, then tombstone.** The ordering is the whole design. Any other order
has a window where the document is partially or entirely missing. This pattern —
build-new, atomic-switch, retire-old — is the same one used for embedding model
migrations and index rebuilds, which is why it's worth internalizing.

**Chunk-level sensitivity is the hard version of this problem**, and there's no clean
answer. A document visible to everyone may contain one paragraph that isn't. Options:
classify at chunk level (expensive, and classification is imperfect), redact at ingest
(lossy, and you need per-audience variants), or refuse to index mixed documents (clean
but often impractical). A candidate who identifies that document-level ACLs are an
approximation, and says what breaks, is thinking about the real problem.
