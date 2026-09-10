# Contextual Retrieval

A chunk pulled out of a document loses everything around it. Contextual retrieval puts
some of that back before you embed it.

It's one of the highest value-per-effort improvements in RAG, and the cheapest version
takes an hour.

---

## The problem

Here's a chunk from the middle of a policy document:

```
"The limit is 30 days from the date of delivery. Items must be unused
and in original packaging."
```

Thirty days for what? Which items? Which product category? Which country?

The chunk made sense in the document because the heading two pages up said "Returns
Policy — Consumer Electronics — European Union." Once it's chunked, that's gone.

Two consequences:

**It won't be found.** Someone searching "how long to return a laptop in Germany" won't
match this chunk. None of those words are in it.

**It's ambiguous when found.** Even if retrieved, the model doesn't know what it applies
to — and may confidently apply it to the wrong thing.

This is the most common cause of "the answer was right there in our documents and the
system didn't find it."

---

## Fix 1: prepend the headings (do this first)

The cheapest version. Before embedding, add the document title and section headings to
the chunk text.

```python
def contextualise_cheap(chunk, doc):
    """Add structural context. Costs a few tokens, often a big gain."""
    parts = [doc.title]
    if chunk.section:
        parts.append(chunk.section)
    if chunk.subsection:
        parts.append(chunk.subsection)
    header = " > ".join(parts)
    return f"{header}\n\n{chunk.text}"
```

The chunk becomes:

```
Returns Policy > Consumer Electronics > European Union

The limit is 30 days from the date of delivery. Items must be unused
and in original packaging.
```

Now it matches the query. An hour of work, no model calls, no ongoing cost.

**Important detail: embed the enriched text, but store and display the original.**

```python
@dataclass
class Chunk:
    text: str              # original — this is what goes in the prompt
    context: str           # the added header
    embedding_input: str   # context + text — this is what you embedded
```

Otherwise your citations show a chunk with a header you invented, and highlighting the
answer in the source document breaks.

---

## Fix 2: neighbouring text

Include a bit of the surrounding chunks so the piece has local context:

```python
def with_neighbours(chunk, doc, before=100, after=100):
    """Add a window of surrounding characters."""
    start = max(0, chunk.start - before)
    end = min(len(doc.text), chunk.end + after)
    return doc.text[start:end]
```

Cheap. Helps when meaning spills across boundaries. Costs some duplication.

A variant worth knowing: **retrieve small, send big.** Search using small precise chunks,
then expand to neighbors before putting them in the prompt. You get precision in search
and context in generation.

---

## Fix 3: model-generated context

The thorough version. For each chunk, ask a model to write a sentence or two explaining
what this chunk is about, given the whole document.

```python
CONTEXT_PROMPT = """Here is a document:
<document>
{document}
</document>

Here is a chunk from it:
<chunk>
{chunk}
</chunk>

Write 1-2 short sentences placing this chunk in context: what it is about,
what it refers to, and which part of the document it belongs to. This will be
prepended to the chunk to help search find it. Write only those sentences."""


async def contextualise_with_model(chunk, doc, model):
    context = await model(
        CONTEXT_PROMPT.format(document=doc.text[:50_000], chunk=chunk.text),
        max_tokens=100,
        temperature=0.0,
    )
    return f"{context.strip()}\n\n{chunk.text}"
```

Result:

```
This chunk is from the Returns Policy document, in the section covering
consumer electronics sold in the European Union. It specifies the time
limit and condition requirements for returns.

The limit is 30 days from the date of delivery. Items must be unused
and in original packaging.
```

Better than headings alone, especially for documents with weak structure.

### The cost, and how to control it

One model call **per chunk**, at indexing time. For a large corpus that's real money.

```python
def context_generation_cost(n_chunks, doc_tokens, price_per_million):
    """Each call sends the whole document plus the chunk."""
    total_input = n_chunks * doc_tokens
    return total_input / 1_000_000 * price_per_million

# 500,000 chunks from documents averaging 8,000 tokens, at $0.25/M:
# 500,000 × 8,000 = 4,000,000,000 tokens → $1,000
```

Three ways to bring that down:

**Use prompt caching.** The document is identical across all chunks from that document.
Cache it and you pay full price once per document instead of once per chunk. This is the
biggest lever by far — often an 80–90% reduction.

**Use a small model.** This is a simple summarizing task. A small cheap model does it
well.

**Only contextualise where it helps.** Chunks that already contain clear context don't
need it:

```python
def needs_context(chunk, doc):
    """Skip chunks that are already self-explanatory."""
    text = chunk.text.lower()
    # Already names the subject?
    if doc.title.lower().split()[0] in text:
        return False
    # Starts with a heading?
    if chunk.text.lstrip().startswith("#"):
        return False
    # Very short chunks are usually headings or fragments
    if len(chunk.text) < 200:
        return False
    return True
```

---

## Combining with hybrid search

Contextual retrieval and [hybrid search](hybrid-search.md) stack well, and for a specific
reason: the added context contains **words**, and words are what keyword search matches
on.

```python
def index_chunk(chunk, doc):
    enriched = contextualise(chunk, doc)

    vector_index.add(chunk.id, embed(enriched))    # dense: enriched text
    bm25.add(chunk.id, enriched)                    # sparse: also enriched

    store.put(chunk.id, chunk.text)                 # display: ORIGINAL text
```

Adding "European Union" and "Consumer Electronics" to the chunk means a keyword search
for "EU electronics return" now matches, where before there was nothing to match.

The order to add things, if you're doing all of this:

1. Heading prepending (free, immediate)
2. Hybrid search (moderate effort, big gain on identifiers)
3. Reranking (moderate effort, big gain, saves money)
4. Model-generated context (costs money at index time)

---

## Measuring it

Compare retrieval with and without, on the same queries:

```python
async def evaluate_context(cases, plain_index, contextual_index):
    results = defaultdict(lambda: {"plain": [], "contextual": []})

    for case in cases:
        seg = case.segments.get("chunk_type", "all")
        results[seg]["plain"].append(
            recall(plain_index.search(case.question, k=5), case.relevant_ids))
        results[seg]["contextual"].append(
            recall(contextual_index.search(case.question, k=5), case.relevant_ids))

    return {
        seg: {
            "plain": mean(v["plain"]),
            "contextual": mean(v["contextual"]),
            "gain": mean(v["contextual"]) - mean(v["plain"]),
            "n": len(v["plain"]),
        }
        for seg, v in results.items()
    }
```

**Segment by whether the chunk needed context.** The overall number understates the
effect:

```
chunk_type              n     plain   contextual   gain
self_contained         120    0.88      0.89       +0.01
needs_context           80    0.44      0.81       +0.37
```

The second row is what you built. If you only look at the overall average (+0.15), you'd
underrate it — and you wouldn't notice if it started regressing on the segment that
matters.

---

## Things that go wrong

**Embedding the enriched text but displaying it too.** Your citations then show text that
isn't in the source document, and highlighting breaks. Store both; display the original.

**The generated context is wrong.** The model summarizes the chunk incorrectly and now
you've embedded a wrong description. Sample and check some. This is a silent failure —
retrieval just gets worse for reasons nobody can see.

**Context longer than the chunk.** A 50-token chunk with 100 tokens of context is mostly
context, and the vector now represents the context rather than the content. Cap it —
context should be a small fraction of the chunk.

```python
def cap_context(context, chunk_text, max_ratio=0.3):
    limit = int(len(chunk_text) * max_ratio)
    return context[:limit] if len(context) > limit else context
```

**Documents too long for the context prompt.** If a document is 500,000 tokens you can't
send it all with every chunk. Use the surrounding section instead of the whole document,
or a document summary.

**Forgetting to re-contextualise after a document changes.** The document is edited; the
old context is now wrong. Regenerate context when the document changes, not just the
chunks.

**Doing it before the cheap things.** Model-generated context costs money. Heading
prepending, hybrid search, and reranking are cheaper and often give more. Do those first.

---

## Interview questions

**1. "A chunk says 'the limit is 30 days' and nobody can find it. Why, and what do you
do?"**

The chunk lost its context — which policy, which product, which region. Fix by prepending
document and section headings before embedding, or generating context with a model.
Mention that you embed the enriched text but display the original.

**2. "What's the cheapest version?"**

Prepend the document title and section headings. An hour of work, no model calls, and it
handles most of the problem for documents with decent structure.

**3. "Model-generated context costs a call per chunk. How do you make that affordable?"**

Prompt caching on the document — it's identical across all its chunks, so you pay once
per document instead of once per chunk. Plus a small model, plus skipping chunks that are
already self-contained.

**4. "How does this interact with hybrid search?"**

They compound. The added context contains words, and keyword search matches words. Index
the enriched text in both the vector index and the keyword index.

**5. "How do you know it worked?"**

Segment by whether the chunk needed context. The gain is concentrated there and the
overall average understates it.

**6. "What's the risk?"**

Wrong generated context, silently degrading retrieval. And context that overwhelms a
short chunk, so the vector represents the context rather than the content. Cap the ratio
and sample the output.

---

## What to remember

- Chunks lose the context that made them meaningful.
- Prepend document and section headings first. An hour of work, big gain.
- Embed the enriched text; store and display the original.
- Model-generated context is better and costs a call per chunk — use prompt caching on
  the document to make it affordable.
- Skip chunks that are already self-contained.
- Cap context length relative to chunk length.
- This compounds with hybrid search, because context adds matchable words.
- Measure segmented by whether the chunk needed context.
- Do the free things first: headings, hybrid, reranking.

---

**Next:** [multimodal-rag.md](multimodal-rag.md) — when your documents contain images,
tables, and charts.
