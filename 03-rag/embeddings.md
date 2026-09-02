# Embeddings

An embedding turns text into a list of numbers — a vector — where similar meanings end
up close together. That's what lets you search by meaning rather than by exact words.

The model choice feels small. It's one config line. It's also the hardest thing in your
system to change later, so it's worth ten minutes of thought now.

---

## The idea

```
"how do I get my money back"  →  [0.21, -0.05, 0.88, ...]
"refund process"              →  [0.19, -0.03, 0.85, ...]   ← close
"shipping times"              →  [-0.4,  0.72, 0.10, ...]   ← far
```

Similarity is measured with cosine similarity, which is the angle between two vectors:

```python
import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**Normalise once, at index time.** If every stored vector has length 1, cosine
similarity is just a dot product, and searching your whole corpus becomes one matrix
multiply:

```python
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1.0                    # avoid divide-by-zero
matrix = vectors / norms                   # store these

scores = matrix @ query_vector             # entire corpus, one operation
```

---

## The bug that catches everyone

**The index and the query must use the same embedding model.**

Someone upgrades the embedding model in a config file. The index still contains vectors
from the old model. Now you're comparing vectors from two different spaces.

There is no error. Cosine similarity still returns numbers, and they look normal —
0.6, 0.7, plausible values. The results are meaningless.

This is the single most common silent RAG failure. Prevent it structurally:

```python
@dataclass
class Index:
    embedding_model: str        # "provider/model-name"
    embedding_version: str      # exact version
    dimensions: int
    vectors: np.ndarray

    def search(self, query_text, embedder):
        if embedder.model != self.embedding_model:
            raise ValueError(
                f"index was built with {self.embedding_model}, "
                f"query uses {embedder.model} — results would be meaningless"
            )
        ...
```

Make it impossible, not just discouraged. A comment saying "don't change this" is not a
control.

---

## Choosing a model

Public benchmarks tell you how a model does on general web-ish text. Your corpus is
probably not that. Legal documents, medical notes, code, internal jargon — all behave
differently.

**Build a small evaluation set and test on your own data.** 100–200 real questions with
the documents that should come back. A day of work that saves you from a decision you
can't easily reverse.

### What to compare

**Retrieval quality on your data** — recall@5 and recall@50, on your questions. Not a
leaderboard.

**Dimensions.** This decides your storage and memory forever:

```python
def storage_gb(n_chunks, dimensions, bytes_each=4):
    return n_chunks * dimensions * bytes_each / 1e9

storage_gb(10_000_000,  768)   #  30.7 GB
storage_gb(10_000_000, 1536)   #  61.4 GB
storage_gb(10_000_000, 3072)   # 122.9 GB
```

Four times the dimensions is four times the memory and roughly four times the distance
computation per query. Do this arithmetic before you pick.

**Max input length.** If your chunks are 800 tokens and the model truncates at 512,
you're silently losing the end of every chunk. Check this — it fails quietly.

**Cost.** Two costs, and people forget the second:
- One-off: embedding your whole corpus
- Ongoing: embedding every query, forever, plus every new document

**Speed.** Query embedding is on your critical path. A slow embedding model adds latency
to every single search.

**Does it stay available?** You need this exact model for years. A hosted model can be
deprecated. An open-weights model you host yourself can't be taken away — archive the
weights.

### The comparison people skip

Evaluate the embedding model **with your reranker in the pipeline**, not alone.

This changes the answer. If a reranker does the precision work, the embedding model's
job is just to get the right documents into the top 50 — a much easier bar, and much
less sensitive to model quality.

```
                    recall@5 alone    with reranker
small model (768d)       0.71             0.86
large model (3072d)      0.78             0.87
```

Alone, the large model looks clearly better. In the real pipeline the gap is one point —
for four times the storage. That table is the decision, and you only get it by testing
the real configuration.

---

## Embed queries and documents differently

Some embedding models are trained with instruction prefixes, and using them incorrectly
costs real quality.

```python
# Check your model's documentation — this varies
doc_vector = embed("passage: " + chunk_text)
query_vector = embed("query: " + user_question)
```

Two failure modes:

- Model expects prefixes, you don't use them → worse retrieval, no error
- Model doesn't expect them, you add them → worse retrieval, no error

Read the model card. Then verify: embed a question and its known-correct document, check
the similarity is high. If it isn't, something's wrong with how you're calling it.

There's also an asymmetry worth understanding: a short question and a long document
aren't the same shape of text. Some models handle this well; some do better if you embed
a hypothetical *answer* rather than the question (see
[query-rewriting.md](query-rewriting.md)).

---

## Embedding at scale

For a large corpus, throughput matters.

**Batch.** One call per chunk is 10–50× slower than batching.

```python
def embed_corpus(chunks, embedder, batch_size=128):
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors.extend(embedder([c.text for c in batch]))
    return np.array(vectors)
```

**Sort by length within a batch.** Batches are padded to the longest item. Mixing a
50-token chunk with a 2,000-token chunk wastes most of the compute on padding.

```python
def batched_by_length(chunks, batch_size=128):
    ordered = sorted(chunks, key=lambda c: len(c.text))
    for i in range(0, len(ordered), batch_size):
        yield ordered[i:i + batch_size]
```

**Deduplicate first.** Large corpora contain repeated boilerplate — headers, legal
footers, template text. Hash the content and embed unique text only.

```python
def dedupe(chunks):
    seen, unique, mapping = {}, [], {}
    for c in chunks:
        h = hashlib.sha256(c.text.encode()).hexdigest()
        if h in seen:
            mapping[c.id] = seen[h]          # point at the existing vector
        else:
            seen[h] = c.id
            unique.append(c)
    return unique, mapping
```

5–20% savings is typical, and it's free.

**Write vectors to durable storage before building the index.** This one is worth
emphasising:

```
chunks → embed → save vectors to files → build index from files
```

rather than

```
chunks → embed → insert directly into the index
```

Two reasons. Incremental insertion into an ANN index gets slower as the index grows, so
a big backfill decelerates and eventually crawls. And if the index build fails, or you
want to rebuild with different parameters, you don't have to re-embed anything.

Re-embedding 400 million chunks because of an index parameter mistake is the kind of
error you only make once.

**Checkpoint.** A backfill that dies at 80% and restarts from zero costs you days.

---

## Migrating to a new embedding model

You will do this eventually. Better models appear; your corpus grows; a provider
deprecates something.

**Never in place.** Build the new index alongside the old one, compare, then switch.

```
1. Build index B with the new model, while index A serves traffic
2. Run both on live queries, log the differences (dual read)
3. Compare on your evaluation set
4. Switch reads to B behind a flag
5. Watch for a week
6. Retire A
```

**The requirement this implies: you must have room for two indexes at once.** If you
don't, you cannot migrate without downtime, and you'll defer it forever. Treat 2×
storage capacity as a permanent property of your design, not an occasional need.

Also worth doing now rather than later:

**Store the source text with the vector.** Then re-embedding doesn't require re-fetching
and re-parsing every original document.

**Keep chunking separate from embedding.** If changing the model forces you to re-chunk,
your comparison is confounded and your migration is twice the work.

**Write down the migration cost in the original design doc.** "Re-embedding our corpus
would take about two weeks and cost roughly $8,000." Knowing that number is what lets
the company decide rationally later instead of emotionally.

---

## Making it smaller

At scale, storage and memory get expensive. Two options:

**Fewer dimensions.** Some models let you truncate vectors and keep most of the quality.
Test the recall cost on your data — it's usually small but it isn't zero.

**Quantisation.** Store numbers with less precision:

```python
def to_int8(vectors):
    """4x smaller. Measure the recall cost before shipping."""
    scale = np.abs(vectors).max()
    return (vectors / scale * 127).astype(np.int8), scale
```

int8 quantisation cuts memory 4× and usually costs 1–3 points of recall. Product
quantisation cuts far more and costs more.

**Always measure the recall cost on your own hard queries**, not on average. Quantisation
tends to hurt most on the cases that were already marginal — which are the ones you care
about.

---

## Interview questions

**1. "How do you pick an embedding model?"**

Test on your own data with a real evaluation set — not a leaderboard, because benchmarks
measure a different corpus. Compare recall, dimensions (storage), max input length,
cost, and speed. Evaluate *with the reranker in place*, because that changes the answer.

**2. "Someone upgraded the embedding model and search broke. What happened?"**

Query vectors are in the new model's space, the index has old vectors. Similarity still
returns plausible numbers, so there's no error — just meaningless results. Fix by binding
the model version to the index and refusing mismatched queries.

**3. "How would you migrate 50 million vectors with no downtime?"**

Build alongside, dual-read and compare, switch behind a flag, watch, retire. Then say the
real requirement is having capacity for two indexes, and that a design without that room
can't be migrated.

**4. "Your embedding backfill is getting slower every day. Why?"**

Almost certainly you're inserting into the ANN index as you go, and insertion cost grows
with index size. Fix: write vectors to storage, build the index in one pass afterwards.

**5. "Can you reduce storage?"**

Fewer dimensions or quantisation. Both cost recall — measure it on your hard queries,
not on the average.

**6. "Do you embed questions and documents the same way?"**

Depends on the model — some expect instruction prefixes. Getting it wrong degrades
retrieval with no error. Check the model card and verify with a known question/document
pair.

---

## What to remember

- Normalise vectors at index time; then search is one matrix multiply.
- Bind the embedding model version to the index and refuse mismatches. This is the
  most common silent failure in RAG.
- Test on your data, not benchmarks — and test with the reranker in place.
- Dimensions decide storage and query speed forever. Do the arithmetic first.
- Check the model's max input length against your chunk size.
- Batch, sort by length, and deduplicate when embedding at scale.
- Write vectors to storage before building the index. Never re-embed by accident.
- Design for two indexes at once, or you can never migrate.
- Store source text with vectors; keep chunking separate from embedding.

---

**Next:** [hybrid-search.md](hybrid-search.md) — why meaning-based search alone fails on
product codes.
