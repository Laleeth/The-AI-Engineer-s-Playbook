# RAG Architecture

RAG means: look things up, put them in the prompt, let the model answer from them.

The idea is simple. Most of the difficulty is in the parts nobody talks about at the
start — chunking, permissions, keeping the index fresh, and knowing when retrieval
failed.

---

## The pipeline

Two halves. One runs when documents change, one runs when a user asks something.

### Indexing (offline, runs when documents change)

```
documents
   ↓ parse         turn PDFs, HTML, Word into text
   ↓ chunk         split into pieces small enough to be useful
   ↓ embed         turn each piece into a vector (a list of numbers)
   ↓ store         save vectors + text + metadata
```

### Querying (online, runs per request)

```
user question
   ↓ rewrite       optional: fix vague or conversational questions
   ↓ embed         turn the question into a vector the same way
   ↓ search        find the closest chunks
   ↓ filter        drop anything this user isn't allowed to see
   ↓ rerank        re-score the candidates properly
   ↓ build context fit the best ones into the token budget
   ↓ generate      model answers, citing sources
```

Two things worth noticing immediately:

**The filter step is not optional.** It's drawn as a step because in a multi-tenant
system, forgetting it is a data breach, not a bug. More on where it belongs below.

**Rerank sits between search and context.** That ordering is what lets you retrieve many
candidates cheaply and send few, which is both cheaper and better.

---

## What each piece does

### Parsing

Getting clean text out of your files. Unglamorous and it decides your ceiling.

PDFs are the hard case: multi-column layouts, tables that lose their structure, headers
and footers repeated on every page, scanned pages that need OCR.

If your parsing mangles tables, no amount of clever retrieval will fix the answers. Look
at your parsed output before you build anything on top of it. Actually read some of it.

### Chunking

Splitting documents into pieces. Too big and each chunk covers several topics, so its
vector is a blurry average and you waste prompt space. Too small and chunks lack the
context to make sense on their own.

Full detail in [chunking.md](chunking.md). The one thing to get right from day one:
**keep the character offsets** so you can point back at the source. Citations depend on
it and retrofitting is painful.

### Embedding

Turning text into a vector — a list of a few hundred to a few thousand numbers — where
similar meanings end up close together.

Two rules that cause most embedding bugs:

- Embed the question the same way you embedded the chunks.
- The index and the query must use the **same model version**. Mixing them produces
  numerically plausible, semantically meaningless results — no error, just nonsense.

See [embeddings.md](embeddings.md).

### Storage and search

Store the vectors and find the closest ones to a query.

At small scale you don't need a vector database. A few tens of thousands of vectors in a
numpy array, searched by brute force, is fast, exact, and simple:

```python
# 50,000 chunks × 768 dims × 4 bytes ≈ 154 MB. Fits in memory easily.
scores = matrix @ query_vector          # one matrix multiply, a few milliseconds
top_k = np.argpartition(-scores, k)[:k]
```

A surprising number of "we need a vector database" decisions are made without doing this
arithmetic. Work out your size first:

```python
def index_size_gb(n_chunks, dimensions, bytes_per_number=4):
    return n_chunks * dimensions * bytes_per_number / 1e9

index_size_gb(50_000, 768)      # 0.15 GB — brute force is fine
index_size_gb(50_000_000, 1536) # 307 GB — you need real infrastructure
```

Above roughly a million vectors, or when you need high query rates, you want an
approximate search index (HNSW, IVF) — which trades exact results for speed.

### Filtering by permission

If different users can see different documents, this belongs **inside the search query**,
not after it.

```python
# WRONG — fetch everything, then remove what they can't see
results = search(query, k=10)
visible = [r for r in results if allowed(r, user)]   # might return 2 results

# RIGHT — the search only ever considers documents they can see
results = search(query, k=10, filter=user_can_see(user))
```

Post-filtering fails two ways: you get fewer than k results, and — worse — in a
multi-tenant index the top 10 are dominated by other tenants' documents, so you filter
almost everything out and return nothing useful.

And it means unauthorised data was loaded into your process. One missed filter, one
exception in the filtering code, and it leaks.

### Reranking

Take 30–50 candidates from search, score them properly with a model that reads the query
and the chunk together, keep the best 5.

This is usually the highest-value single addition to a basic RAG system, and it's
counter-intuitive: it makes the system **cheaper** as well as better, because you send
fewer tokens to the model. See [reranking.md](reranking.md).

### Building the context

Fit the chosen chunks into a token budget. Two rules:

**Skip, don't truncate.** If a chunk doesn't fit, leave it out rather than cutting it
mid-sentence. Half a chunk produces half an answer and a citation you can't verify.

**Remove near-duplicates.** Overlapping chunks mean your top 5 might be three copies of
the same passage, wasting most of your budget.

### Generating

The prompt needs three things:

```
Answer using ONLY the context below.
Cite sources with their [id].
If the context doesn't contain the answer, say you don't have that information.
```

That last line is load-bearing. Without it, the model invents an answer when retrieval
fails — and retrieval will fail regularly. Design the "I don't know" path deliberately
and test it.

---

## Where teams go wrong

**Skipping evaluation.** You cannot improve retrieval you can't measure, and you can't
tell retrieval failures from generation failures by looking at answers. Build the
measurement early — see [../05-evaluation/rag-evaluation.md](../05-evaluation/rag-evaluation.md).

**Assuming a bigger model fixes it.** If the right document wasn't retrieved, a better
model just writes a more fluent wrong answer. Check retrieval first, always.

**Chunking by character count with no structure awareness.** Splitting mid-sentence or
mid-table destroys meaning.

**No plan for updates.** Documents change constantly. If your only path is "rebuild
everything," you'll do it rarely and serve stale answers in between.

**Permissions bolted on later.** It's much harder to add correctly than to build in.

**No version binding between index and query.** The most common silent failure: someone
upgrades the embedding model, the index still holds old vectors, and search returns
plausible nonsense with no error at all.

---

## Handling document updates

The naive approach — delete the old chunks, re-chunk, re-embed — leaves a window where
the document is half-indexed and retrieval returns a mix of old and new.

Better: version it, then switch.

```python
def upsert(doc):
    new_version = current_version(doc.id) + 1

    # 1. Index the new version alongside the old
    chunks = chunk(doc, version=new_version)
    index.add(chunks, embed([c.text for c in chunks]))

    # 2. Switch atomically — one assignment
    set_live_version(doc.id, new_version)

    # 3. Now retire the old version
    mark_dead(doc.id, below=new_version)
```

Build new, switch, retire old. There's never a moment where the document is missing or
mixed. The same pattern works for changing embedding models across a whole corpus.

Queries must then check the live version:

```python
def is_live(chunk):
    return current_version(chunk.doc_id) == chunk.version
```

---

## How complex should yours be?

Start simple. Add pieces when you can point at the problem they solve.

**Level 1 — start here.** Chunk, embed, search, generate. No reranking, no hybrid, no
rewriting. Get this working end to end and measure it. Often good enough.

**Level 2 — when you can name the failure.**
- Answers cite the wrong thing → add reranking
- Exact terms, IDs, product names fail → add [hybrid search](hybrid-search.md)
- Conversational follow-ups fail → add [query rewriting](query-rewriting.md)
- Chunks lack context to be understood alone → add
  [contextual retrieval](contextual-retrieval.md)

**Level 3 — at real scale.** Approximate search index, caching, multi-region, tiered
storage for large and small tenants.

The mistake is building level 3 first. Every piece you add is a thing that can break and
a thing that must be evaluated. Add them when the measurement tells you to.

---

## Interview questions

**1. "Walk me through a RAG system."**

Both halves — indexing and querying — and don't skip parsing, permissions, or updates.
Mentioning those three unprompted is what separates someone who has built one from
someone who has read about one.

**2. "Where would you start if answers are wrong?"**

Ask whether the right documents were retrieved. That single question splits the problem
into retrieval and generation, which have completely different fixes. See
[rag-debugging.md](rag-debugging.md).

**3. "Do you need a vector database?"**

Do the arithmetic. Under about a million vectors, brute force in memory is fast, exact,
and simpler. Above that, or at high query rates, you need a real index.

**4. "How do you handle permissions?"**

Filter inside the query, never after. Explain that post-filtering returns too few
results and means unauthorised data was already loaded.

**5. "A document is edited. What happens?"**

Version, switch, retire. Explain the window problem with delete-then-reindex.

**6. "Your embedding model gets upgraded. What breaks?"**

Everything, silently. Old vectors and new query vectors are in different spaces —
similarity scores look normal and results are meaningless. Bind the model version to the
index and refuse mismatched queries.

---

## What to remember

- Two pipelines: indexing (offline) and querying (online).
- Parsing quality sets your ceiling. Look at the parsed text.
- Keep character offsets from day one — citations need them.
- Filter permissions inside the query, never after.
- Retrieve many, rerank, send few. Cheaper *and* better.
- Skip chunks that don't fit; never truncate them.
- Always include an "I don't know" instruction, and test it.
- Version documents: build new, switch, retire old.
- Start simple. Add pieces when a measurement tells you to.

---

**Next:** [chunking.md](chunking.md) — splitting documents without destroying them.
