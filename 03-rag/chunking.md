# Chunking

Splitting documents into pieces small enough to retrieve. It's the least glamorous part
of RAG and it sets the ceiling on everything downstream.

If your chunks are bad, better embeddings won't save you, reranking won't save you, and
a bigger model definitely won't save you.

---

## Why chunk at all?

Three reasons:

**Precision.** A user asks about refund windows. If your chunk is a whole 40-page
handbook, the model gets 39 pages of noise with the answer buried in it.

**Vector quality.** One vector represents one chunk. If the chunk covers eight topics,
the vector is a blurry average of eight things and matches none of them well.

**Budget.** You can only fit so much in the prompt. Smaller pieces mean you can include
several relevant ones instead of one big mostly-irrelevant one.

---

## The rule that matters most

Whatever else you do:

> **`document.text[chunk.start:chunk.end]` must equal `chunk.text`.**

Keep exact character offsets back into the source document. Write a test for it.

```python
def test_offsets(doc, chunks):
    for c in chunks:
        assert doc.text[c.start:c.end] == c.text, f"offset broken in {c.id}"
```

Why it matters: citations, highlighting the answer in the original document, letting a
user verify, and proving where an answer came from. Systems that chunk by splitting and
rejoining strings lose this mapping, and you can't add it back later without
re-processing everything.

The second test worth having:

```python
def test_coverage(doc, chunks):
    """Every character should be in at least one chunk."""
    covered = bytearray(len(doc.text))
    for c in chunks:
        covered[c.start:c.end] = b"\x01" * (c.end - c.start)
    assert sum(covered) / len(doc.text) > 0.99
```

This catches a silent failure: a chunker that drops content on unusual input. The system
then says "I don't have that information" about text that is sitting in your corpus, and
nobody ever reports it as a bug — it looks like a correct answer.

---

## Ways to chunk, worst to best

### Fixed character count (don't)

```python
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
```

Splits mid-word, mid-sentence, mid-table. A chunk starting "...therefore the refund
window is" embeds badly and reads worse.

Only acceptable as a last-resort fallback.

### Split on sentences

Better. Group whole sentences until you hit a size limit.

```python
import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def chunk_by_sentences(doc, target_tokens=400, overlap_tokens=60):
    """Group whole sentences, keeping exact offsets."""
    sentences = []
    pos = 0
    for part in SENTENCE_END.split(doc.text):
        if not part:
            continue
        start = doc.text.find(part, pos)
        if start == -1:
            continue
        sentences.append((start, start + len(part)))
        pos = start + len(part)

    chunks = []
    i = 0
    while i < len(sentences):
        start = sentences[i][0]
        end = sentences[i][1]
        j = i + 1

        # Add sentences until we hit the size limit
        while j < len(sentences):
            candidate_end = sentences[j][1]
            if count_tokens(doc.text[start:candidate_end]) > target_tokens:
                break
            end = candidate_end
            j += 1

        chunks.append(Chunk(
            id=f"{doc.id}#{len(chunks)}",
            doc_id=doc.id,
            text=doc.text[start:end],       # exact slice — offsets stay true
            start=start, end=end,
        ))

        if j >= len(sentences):
            break

        # Step back to create overlap
        back = j - 1
        overlap = 0
        while back > i and overlap < overlap_tokens:
            overlap += count_tokens(doc.text[sentences[back][0]:sentences[back][1]])
            back -= 1

        # Always move forward at least one sentence, or a very long
        # sentence makes this loop forever
        i = max(i + 1, back + 1)

    return chunks
```

That last line is a real bug guard. Without it, a single sentence longer than
`target_tokens` causes an infinite loop. It's the kind of thing that works on your test
documents and hangs on a customer's.

### Split on structure (best, when you have structure)

If the document has headings, sections, or code blocks, split on those first. Structure
is the author telling you where the topic boundaries are — use it.

```python
def chunk_markdown(doc, max_tokens=500):
    """Split on headings first, then by size within a section."""
    sections = split_on_headings(doc.text)      # returns (start, end, heading)

    chunks = []
    for start, end, heading in sections:
        section_text = doc.text[start:end]

        if count_tokens(section_text) <= max_tokens:
            chunks.append(make_chunk(doc, start, end, heading))
        else:
            # Section too big — split by sentences, but keep the heading
            for s, e in split_sentences_within(doc.text, start, end, max_tokens):
                chunks.append(make_chunk(doc, s, e, heading))

    return chunks
```

Different formats, different boundaries:

| Format | Split on |
|---|---|
| Markdown / HTML | Headings, then paragraphs |
| Code | Functions and classes — never mid-function |
| Legal contracts | Clauses and numbered sections |
| Transcripts | Speaker turns, or topic shifts |
| Slides | One slide per chunk |
| Spreadsheets | Rows or logical groups, keeping headers |

---

## How big should chunks be?

There's no universal answer. It depends on your documents and your questions.

Rough starting points:

| Content | Chunk size | Why |
|---|---|---|
| FAQ, short answers | 100–300 tokens | Each answer is self-contained |
| General documentation | 300–600 tokens | A section of explanation |
| Legal, technical | 500–1,000 tokens | Needs surrounding context |
| Code | One function | Natural boundary |

**Start at 400 tokens with 15% overlap, then measure.** That's a reasonable default, and
measuring beats guessing:

```python
async def compare_chunk_sizes(docs, questions, sizes=(200, 400, 600, 1000)):
    results = {}
    for size in sizes:
        index = build_index(docs, chunk_size=size)
        results[size] = {
            "recall@5": await measure_recall(index, questions, k=5),
            "n_chunks": index.count,
            "avg_context_tokens": await measure_context_size(index, questions),
        }
    return results
```

Watch two things: does recall improve, and what does it cost you in prompt tokens? A
bigger chunk size that improves recall by 2 points while doubling your context cost is
usually a bad trade.

---

## Overlap

Overlapping chunks stop answers falling between the cracks.

Without overlap:

```
chunk 1: "...our refund policy allows returns within"
chunk 2: "30 days of purchase, provided the item is unused..."
```

Neither chunk answers "how long do I have to return something?" properly.

With 15% overlap, chunk 2 starts a bit earlier and contains the whole statement.

**The cost:** more chunks to store and embed, and near-duplicate results in retrieval.
That second one needs handling:

```python
def drop_overlapping(chunks):
    """Remove chunks that mostly repeat one already kept."""
    kept = []
    for c in chunks:
        overlaps = any(
            k.doc_id == c.doc_id and not (c.end <= k.start or c.start >= k.end)
            for k in kept
        )
        if not overlaps:
            kept.append(c)
    return kept
```

Without this, your top 5 results can be three copies of the same passage — you've spent
60% of your prompt budget on one paragraph.

10–20% overlap is the usual range. More than that and you're mostly storing duplicates.

---

## Add context to each chunk

A chunk pulled out of a document loses everything around it. This chunk is nearly
useless on its own:

```
"The limit is 30 days from the date of delivery."
```

Thirty days for what? Which product? Which policy?

Cheap fix — prepend the document and section title before embedding:

```python
def chunk_text_for_embedding(chunk, doc):
    return f"{doc.title} > {chunk.heading}\n\n{chunk.text}"

# becomes:
# "Returns Policy > Consumer Electronics
#
#  The limit is 30 days from the date of delivery."
```

Costs a few tokens, often improves retrieval a lot, and takes an hour to implement. Do
this before anything clever.

There's a more thorough version of this idea — using a model to write a contextual
summary for each chunk — in
[contextual-retrieval.md](contextual-retrieval.md).

---

## Tables, code, and other awkward things

**Tables.** Splitting a table across chunks destroys it — the second half has no headers
and no meaning. Keep tables whole where you can. If a table is too big, repeat the
header row in each piece.

```python
def chunk_table(table_text, max_tokens):
    header, *rows = table_text.split("\n")
    chunks, current = [], [header]
    for row in rows:
        if count_tokens("\n".join(current + [row])) > max_tokens:
            chunks.append("\n".join(current))
            current = [header, row]          # repeat the header
        else:
            current.append(row)
    if len(current) > 1:
        chunks.append("\n".join(current))
    return chunks
```

**Code.** Split on function or class boundaries. A half-function is not retrievable and
not useful. Include the file path and any imports the function depends on.

**Lists.** Keep list items with their introducing sentence. "The following are
prohibited:" followed by nothing is not helpful.

**Headers and footers.** Page furniture repeated on every page pollutes every chunk.
Strip it during parsing.

---

## Retrieve small, send big

A useful pattern when small chunks retrieve well but lack context to answer from.

Retrieve using small precise chunks, then expand to their neighbors before sending to
the model:

```python
def expand(chunk, doc, window=1):
    """Retrieved a small chunk — send it plus its neighbors."""
    siblings = chunks_for(doc.id)
    i = siblings.index(chunk)
    lo = max(0, i - window)
    hi = min(len(siblings), i + window + 1)
    start = siblings[lo].start
    end = siblings[hi - 1].end
    return doc.text[start:end]
```

You get small-chunk precision in the search and big-chunk context in the prompt. It
costs more tokens, so measure whether the quality gain is worth it.

---

## Testing your chunker

Four tests, all cheap, all catching real bugs:

```python
def test_offsets_exact(doc, chunks):
    for c in chunks:
        assert doc.text[c.start:c.end] == c.text

def test_coverage(doc, chunks):
    """No content is silently dropped."""
    covered = sum(c.end - c.start for c in merge_ranges(chunks))
    assert covered / len(doc.text) > 0.99

def test_no_infinite_loop():
    """A sentence longer than the chunk size must not hang."""
    doc = Document("x", "word " * 5000 + ".")
    chunks = chunk_by_sentences(doc, target_tokens=50)
    assert len(chunks) >= 1

def test_size_distribution(chunks):
    """Catch a chunker producing lots of tiny or huge chunks."""
    sizes = [count_tokens(c.text) for c in chunks]
    assert statistics.median(sizes) > 50, "too many tiny chunks"
    assert max(sizes) < 4000, "some chunk is far too large"
```

Run them on your *real* documents, not on invented text. Real PDFs break chunkers in
ways that clean strings don't.

---

## Interview questions

**1. "How do you pick a chunk size?"**

Measure it. Build indexes at several sizes, compare recall@k and context cost. Start at
around 400 tokens with 15% overlap. Say clearly that it depends on document structure
and question type, so there's no universal answer.

**2. "What's the risk of chunks that are too small? Too large?"**

Too small: each chunk lacks the context to be understood or to answer from. Too large:
the vector is a blurry average of several topics, and you waste prompt budget on
irrelevant text.

**3. "How do you handle tables?"**

Keep them whole where possible; repeat headers if you must split. Explain that a table
half has no meaning. Bonus: mention that PDF parsing often destroys tables before
chunking ever sees them.

**4. "Why do offsets matter?"**

Citations, highlighting, verification, provenance. And they're nearly impossible to add
retroactively. Mention the test.

**5. "Your chunker works in testing and hangs on one customer's document. What
happened?"**

Almost certainly a sentence (or a document with no sentence terminators) longer than the
chunk size, with a loop that doesn't guarantee forward progress. Real answer to a real
bug.

**6. "You changed chunk size and quality dropped. Why?"**

Several possibilities: bigger chunks blurred the vectors; smaller chunks lost the
context needed to answer; citation IDs changed and stale references broke; or overlap
now floods the top-k with near-duplicates. The point is to enumerate rather than guess.

---

## What to remember

- Chunking sets the ceiling for everything downstream.
- Keep exact offsets. Test them. Citations depend on it.
- Test coverage too — a chunker that silently drops text produces invisible failures.
- Split on structure where you have it, sentences where you don't, characters never.
- Start at ~400 tokens with 15% overlap, then measure.
- Guarantee forward progress or a long sentence hangs your indexer.
- Prepend document and section titles before embedding. Cheap, effective.
- Keep tables whole; repeat headers when splitting.
- De-duplicate overlapping chunks at retrieval time.

---

**Next:** [embeddings.md](embeddings.md) — turning text into vectors, and the migration
you'll regret not planning.
