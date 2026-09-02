# 03 — RAG

Retrieval-augmented generation: look things up, put them in the prompt, let the model
answer from them. Plain language, terms explained on first use.

The running idea: **most RAG problems are retrieval problems, and most teams debug them
as generation problems.** If the right document wasn't found, a better model just writes
a more fluent wrong answer.

## The files

| File | What it covers |
|---|---|
| [architecture.md](architecture.md) | The whole pipeline, and how complex yours should be |
| [chunking.md](chunking.md) | Splitting documents without destroying them |
| [embeddings.md](embeddings.md) | Text into vectors, and the migration you'll regret not planning |
| [hybrid-search.md](hybrid-search.md) | Why meaning-based search fails on product codes |
| [reranking.md](reranking.md) | The highest-value addition to most RAG systems |
| [query-rewriting.md](query-rewriting.md) | Fixing the question before you search with it |
| [contextual-retrieval.md](contextual-retrieval.md) | Giving each chunk enough context to be found |
| [multimodal-rag.md](multimodal-rag.md) | Images, tables, charts, scanned pages |
| [rag-debugging.md](rag-debugging.md) | Finding out why the answer was wrong |

## Read in this order

**Building one?** [architecture.md](architecture.md) → [chunking.md](chunking.md) →
[embeddings.md](embeddings.md). That's a working system.

**Improving one?** [rag-debugging.md](rag-debugging.md) first — find out what's actually
broken, then read the file for that problem.

**Just want the biggest win?** [reranking.md](reranking.md). It's usually the largest
single improvement available, and it saves money.

## What to add, when

Start simple. Add each piece when you can name the problem it solves.

| Symptom | Add this |
|---|---|
| Exact codes, IDs, product names fail | [hybrid search](hybrid-search.md) |
| Right topic, wrong document | [reranking](reranking.md) |
| Follow-up questions fail | [query rewriting](query-rewriting.md) |
| "I don't know" when the answer exists | [contextual retrieval](contextual-retrieval.md) |
| Charts and tables unusable | [multimodal](multimodal-rag.md) — but check parsing first |

Building all of this before measuring anything is the common mistake. Every piece is
something that can break and something that must be evaluated.

## The short version

**Ask one question first: was the right information in the prompt?** It splits every RAG
failure into retrieval problems and generation problems, which have completely different
fixes.

**High groundedness with wrong answers means retrieval failed.** The model isn't
hallucinating — it's faithfully answering from the wrong documents. Both things are true
at once, and knowing that saves weeks.

**Keep exact character offsets when chunking.** `doc.text[start:end] == chunk.text`.
Citations, highlighting, and provenance all depend on it, and you can't add it back
later.

**Bind the embedding model version to the index.** Upgrading the embedding model without
rebuilding gives you query vectors and stored vectors in different spaces. Similarity
scores still look normal. Results are meaningless. No error. This is the most common
silent failure in RAG.

**Filter permissions inside the query, never after.** Post-filtering returns fewer than k
results, and in a multi-tenant index it returns almost nothing useful — plus you loaded
data the user can't see.

**Reranking makes the system cheaper AND better.** Retrieve 50, rerank, send 5. You cut
context tokens by 75% and quality improves, because you removed distracting text. Most
optimisations trade one for the other. This one doesn't.

**Embeddings can't tell E-4471 from E-4472.** Both are short strings with a code in the
same context, so their vectors are nearly identical. Semantically correct, operationally
catastrophic. That's what hybrid search is for.

**Adding documents can break existing answers.** New content outranks the old correct
answers. Treat corpus changes as deploys and run your eval suite on them.

**Always include an "I don't know" instruction, and test it.** Retrieval fails regularly.
Without that instruction the model invents an answer every time it does.

## Before adding a vector database

Do the arithmetic:

```python
def index_size_gb(n_chunks, dimensions, bytes_each=4):
    return n_chunks * dimensions * bytes_each / 1e9

index_size_gb(50_000, 768)       # 0.15 GB — brute force in numpy is fine
index_size_gb(50_000_000, 1536)  # 307 GB  — you need real infrastructure
```

A lot of "we need a vector database" decisions are made without ever computing this. At
50,000 chunks, one matrix multiply searches your entire corpus in a few milliseconds,
exactly, with no extra infrastructure.

## Related

- [05-evaluation/rag-evaluation.md](../05-evaluation/rag-evaluation.md) — measuring
  retrieval and generation separately
- [11-coding-rounds/rag-from-scratch.md](../11-coding-rounds/rag-from-scratch.md) — build
  it from primitives, with tested code
- [12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md)
  — the corpus-contamination incident, as an interview exercise
