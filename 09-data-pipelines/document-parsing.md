# Document Parsing

Parsing is where retrieval quality is decided, and it is the stage nobody looks at.

Teams spend months on rerankers and query rewriting while the index contains tables
flattened into word salad, headers repeated ten thousand times, and half a scanned
archive represented by empty strings. No retrieval technique recovers information that
was destroyed before it was embedded.

[03-rag/architecture.md](../03-rag/architecture.md) calls parsing "unglamorous, and it
decides your ceiling." This file is that ceiling in detail.

---

## Read your parsed output. Actually read it.

The single highest-value hour in a RAG project is spent reading fifty parsed documents
end to end, chosen at random, before building anything on top of them.

People skip it because parsing "worked" — the pipeline ran, the chunk count looks
plausible, nothing errored. That is precisely the failure mode: parsers rarely crash. They
return something, and something is what gets embedded.

```python
def parse_audit_sample(documents, parser, n=50):
    """Not a test. A thing you read with your eyes."""
    import random
    for doc in random.sample(documents, n):
        text = parser(doc)
        yield {
            "source": doc.path,
            "chars": len(text),
            "chars_per_page": len(text) / max(1, doc.page_count),
            "text": text,
        }
```

`chars_per_page` is the cheapest signal you will ever get. A text-heavy page yields
roughly 1,500–3,000 characters. Pages yielding 50 are images your parser silently
skipped; pages yielding 40,000 are usually a parser that repeated the whole document per
page. Both are invisible in aggregate counts and obvious in a sorted list.

---

## Where the information goes

**Tables.** The most common and most damaging failure. A financial table parsed
row-major without structure becomes a sequence of numbers with no column headers
attached, and the embedding of that is meaningless. Worse, it retrieves confidently — the
chunk contains the right words — so the model answers from a table it cannot read.

If your corpus is financial, scientific, or operational, table handling *is* your parsing
strategy, not a detail of it. Options, in ascending cost: extract tables separately and
serialize each row with its headers repeated; convert tables to markdown or HTML and keep
the structure; render the table to an image and use a vision model, which is
[multimodal RAG](../03-rag/multimodal-rag.md).

**Multi-column layouts.** Naive text extraction reads across columns, interleaving two
unrelated sentences into one line. The output looks like text, passes every automated
check, and means nothing. Academic papers, newspapers and many report templates are
affected.

**Headers, footers and boilerplate.** Repeated on every page, they become a meaningful
share of a short chunk's tokens and they are identical across every document — which
makes chunks look similar to each other in embedding space for reasons that have nothing
to do with content. Strip them by detecting lines that repeat across a high fraction of
pages.

**Scanned pages.** These yield empty or near-empty text and need OCR. The failure is
silent: an empty string is a valid parse result. Detect and route them.

**Reading order.** Sidebars, pull quotes, captions and footnotes get spliced into the main
text at whatever position the file happens to store them. A footnote landing mid-sentence
changes what the chunk means.

---

## The arithmetic that justifies spending on this

Parsing is usually a one-time cost per document, and it is small relative to what it
protects:

```python
def parsing_economics(documents, pages_each, cheap_per_page=0.0,
                      good_per_page=0.015, chunks_per_doc=20):
    """Compare a parsing upgrade against the index it feeds."""
    pages = documents * pages_each
    return {
        "cheap_total": pages * cheap_per_page,
        "good_total": pages * good_per_page,
        "chunks_affected": documents * chunks_per_doc,
    }

parsing_economics(documents=200_000, pages_each=12)
# cheap_total 0, good_total $36,000, chunks_affected 4,000,000
```

**$36,000, once**, against four million chunks that every future query depends on and an
index you would otherwise have to rebuild after discovering the problem. Set against
recurring inference spend — the $20,000/month of retrieved context in
[reranking.md](../03-rag/reranking.md) — a one-off parsing spend is rarely the expensive
decision. It only looks expensive because it arrives as a line item while bad parsing
arrives as "the model isn't very good."

That said, don't buy the expensive parser for the whole corpus by default. Route:
cheap extraction for clean digital text, better handling for the document types that fail,
OCR only where it's needed. Which requires knowing which is which — see below.

---

## Classify before you parse

Treat parsing as a routing problem rather than a single function:

```python
def route(document):
    if document.is_scanned:                 # no embedded text layer
        return "ocr"
    if document.has_complex_tables:
        return "table_aware"
    if document.column_count > 1:
        return "layout_aware"
    return "fast_text"
```

Each branch has a different cost and a different quality profile, and you want the
distribution — "12% of this corpus is scanned" — before you commit to an approach. That
number also tells you what your ceiling is if you do nothing.

---

## Quality gates that catch silent failure

Parsing failures don't raise exceptions, so you need explicit checks. These are cheap and
each one catches a real class of problem:

| Check | Catches |
|---|---|
| Characters per page below a floor | Scanned pages, failed extraction, images-only documents |
| Characters per page above a ceiling | Repeated content, parser looping over the document |
| Ratio of alphabetic characters | Binary garbage, encoding failures, OCR noise |
| Detected language matches expectation | Encoding errors, wrong file routed in |
| Table count in output vs. table count detected in source | Tables silently dropped |
| Duplicate line rate across pages | Headers and footers not stripped |
| Chunk offsets satisfy `doc.text[start:end] == chunk.text` | Offset drift, which breaks every citation |

Fail loudly and quarantine. A document that fails these should go to a review queue, not
into the index — a bad document in the index is worse than a missing one, because a
missing document produces "I don't know" and a mangled one produces a confident wrong
answer with a citation.

---

## Keep the offsets, and keep the metadata

Two things must survive parsing or they cannot be added later:

**Character offsets into the source text**, so `doc.text[start:end] == chunk.text` holds
exactly. Citations, highlighting and provenance all depend on it, and retrofitting means
reprocessing the corpus. [chunking.md](../03-rag/chunking.md) makes the same point from
the other side.

**Structural metadata** — page number, section heading, document title, effective date,
and the access-control identifiers this document carries. The last one matters most:
permissions must be indexed alongside the vector or you cannot filter inside the query,
which is the requirement in
[08-ai-security/authorization.md](../08-ai-security/authorization.md). Metadata that
isn't captured at parse time is metadata you don't have.

---

## What to monitor

- **Characters per page**, as a distribution rather than a mean. The tails are the story.
- **Quarantine rate by document type**, trended. A rise means an upstream format changed.
- **OCR share**, and its confidence scores if your OCR reports them.
- **Table extraction rate** — tables found in output versus tables detected in source.
- **Offset assertion failures**, which should be zero and should page someone if not.
- **Parse duration by branch**, because the expensive branch quietly becoming the common
  branch is a cost incident nobody attributes to parsing.

---

## Interview questions

**"Retrieval quality is poor. Rerankers didn't help. Where do you look?"**

At the parsed text, before anything else. Pull fifty documents and read them. If tables
are flattened, columns are interleaved, or a chunk of the corpus parsed to near-empty,
that's the ceiling and no retrieval technique lifts it. The tell that points here is
high groundedness with wrong answers — the model is faithfully reading garbage, which is
the same signal [03-rag/README.md](../03-rag/README.md) describes for retrieval failure.

**"How do you know your parsing is working, at a million documents?"**

Not by reading a million documents. By gating on distributions — characters per page,
alphabetic ratio, duplicate line rate, table counts — and quarantining what fails, then
reading a sample of the quarantine and a sample of what passed. The check that matters
most is the offset assertion, because it's exact rather than statistical.

**"Would you pay $36,000 to reparse a corpus?"**

Probably, and I'd want two numbers first: what share of documents currently fail a quality
gate, and what the retrieval failures cost. It's a one-time spend against an index every
future query reads, set against recurring inference cost — but I'd route by document type
rather than reparse everything, and I'd validate on the failing subset before committing.

---

## What to remember

- Parsing decides the ceiling on retrieval quality, and parsers fail silently rather than
  loudly.
- Read fifty parsed documents with your eyes before building anything. Characters per page
  is the cheapest signal for finding the rest.
- Tables are the most damaging failure: they retrieve well and mean nothing.
- Route by document type instead of buying one parser for everything.
- Capture offsets and permission metadata at parse time. Neither can be added later
  without reprocessing.
