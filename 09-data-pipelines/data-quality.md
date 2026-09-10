# Data Quality and Validation

Retrieval quality has a ceiling set by what's in the index, and most teams cannot describe
what's in their index beyond a chunk count.

This file is about the checks that run before content is indexed, and the ones that run
against the index afterwards. They are cheap, they are mostly arithmetic, and they catch
the class of problem that otherwise surfaces as "the model gives bad answers."

---

## Garbage retrieves confidently

The failure worth understanding first: bad content doesn't sit inertly in the index. It
competes for the top-k slots, and it often wins.

A boilerplate legal footer repeated across ten thousand documents is a strong lexical and
semantic match for a surprising range of queries. A near-duplicate of a good document
splits the ranking between two copies and pushes a genuinely different perspective out of
the top 5. A draft that was never meant to be published outranks the approved version
because it happens to use the query's phrasing.

None of these look like errors. Retrieval works, groundedness is high, and the answer is
wrong — the pattern [03-rag/README.md](../03-rag/README.md) identifies as retrieval
failure rather than hallucination.

---

## Checks before indexing

Run these at ingestion time. Each is cheap, and each maps to a real failure:

| Check | Reject or flag when | Catches |
|---|---|---|
| **Length floor** | Chunk under ~50 tokens of substance | Page numbers, headers, empty fragments |
| **Length ceiling** | Chunk far above target size | Chunking failure, a document with no split points |
| **Alphabetic ratio** | Below ~0.5 | Encoding damage, OCR noise, binary content |
| **Near-duplicate** | Similarity above ~0.95 to an existing chunk | Duplicated documents, versions indexed twice |
| **Boilerplate frequency** | Identical text across a high fraction of documents | Footers, disclaimers, navigation |
| **Language** | Detected language outside the expected set | Wrong document routed in, encoding failure |
| **Required metadata present** | Missing document id, permissions, or timestamp | Documents that can't be filtered or deleted |
| **Offset integrity** | `doc.text[start:end] != chunk.text` | Broken citations |

Two of these deserve more than a table row.

**Near-duplicate detection** is the highest-value check in most corpora, because
duplication is endemic — the same policy attached to forty tickets, a document and its PDF
export, three revisions of one contract. Duplicates waste context budget and crowd the
top-k with the same passage.

```python
def is_near_duplicate(vector, index, threshold=0.95):
    """Cheap: you already computed the embedding to index it."""
    nearest = index.search(vector, k=1)
    return bool(nearest) and nearest[0].score >= threshold
```

Flag rather than reject by default, and keep the newest or the canonical version with a
pointer to the others. Silent rejection of a duplicate that was actually a distinct
document is its own bug.

**Boilerplate detection** works by frequency rather than by rules. Text appearing verbatim
in more than some share of documents is structural, not content:

```python
from collections import Counter

def boilerplate_lines(documents, threshold=0.30):
    counts = Counter()
    for doc in documents:
        for line in set(doc.text.splitlines()):     # count each line once per document
            counts[line.strip()] += 1
    n = len(documents)
    return {line for line, c in counts.items() if line and c / n >= threshold}
```

Deriving the list from your own corpus beats maintaining a rules file, and it adapts when
the source system changes its template.

---

## Checks against the index

Pre-indexing checks look at one chunk at a time. Some problems are only visible in
aggregate:

- **Duplicate rate.** What fraction of chunks have a near-duplicate? Rising means an
  ingestion path is double-writing, which is usually an idempotency bug — see
  [ingestion-pipelines.md](ingestion-pipelines.md).
- **Orphan and ghost chunks.** Chunks whose document id is absent or deleted in the
  catalog. Should be zero.
- **Coverage.** Documents in the source vs. in the catalog vs. represented by chunks in
  the index. Three numbers that must reconcile.
- **Metadata completeness by field.** The permission field being 3% null is a security
  finding, not a data-quality nit.
- **Chunk length distribution.** A shifting distribution means chunking behavior changed,
  which changes retrieval behavior.
- **Retrieval reachability.** Sample documents and check that a query built from their own
  content retrieves them. A document that cannot retrieve itself is in the index and
  invisible, which is the worst state — it looks indexed on every dashboard.

That last check is the one most worth building and the one almost nobody has.

---

## Poisoning, and content you didn't write

Anything ingested from a source users can write to is untrusted input, and the index is a
delivery mechanism for it. A support ticket, a wiki page, a shared drive, a scraped site —
each can carry instructions aimed at the model rather than content aimed at a reader.

This is indirect prompt injection, and the retrieval layer is its distribution channel.
The controls belong in
[08-ai-security/prompt-injection.md](../08-ai-security/prompt-injection.md); what belongs
*here* is the recognition that ingestion is the point where untrusted content enters, and
therefore the point where it can be flagged:

- Flag content containing instruction-shaped text at ingestion, and record which source it
  came from. This is defense in depth, not a guarantee — it belongs alongside real
  controls, never in place of them.
- Keep provenance on every chunk, so an incident can be traced to a source and that
  source's content can be quarantined in bulk.
- Treat "which sources can write into the index, and who can write to them?" as a
  question with a documented answer.

The load-bearing control remains the one in section 08: assume the injection succeeds and
make the outcome harmless.

---

## Sampling beats dashboards

Every check above is statistical, and statistics miss things a person notices in ten
minutes.

Build the habit of reading samples: fifty random chunks weekly, plus fifty from whatever
changed. You are looking for things nobody thought to write a check for — a document
type parsing badly since a source-system upgrade, a template change that turned useful
text into navigation, an entire category of content that is technically fine and useless
in isolation.

When you find one, write the check. That's how the gate list above gets built — every row
in it came from someone reading output, not from a design document.

---

## What to monitor

- **Rejection and flag rate by check**, trended. A spike is an upstream change.
- **Duplicate rate across the index**, which should be stable and low.
- **Metadata completeness by field**, especially permissions.
- **Coverage reconciliation**: source vs. catalog vs. index.
- **Self-retrieval success rate** on a sampled set of documents.
- **Quarantine size and age.** A quarantine that only grows is a data gap with a queue in
  front of it.

---

## Interview questions

**"How do you know what's in your index?"**

Coverage reconciliation between source, catalog and index; distributions for chunk length,
duplicate rate and metadata completeness; and a self-retrieval check on a sample, because
a document that can't retrieve itself is invisible while looking indexed everywhere else.
Then reading fifty random chunks a week, which catches what no check was written for.

**"Answers got worse after ingesting a new source. Retrieval metrics look fine."**

Probably content competition rather than a retrieval bug. The new source may be
near-duplicates that split the ranking, or boilerplate that matches broadly, or drafts
that outrank approved versions. I'd check duplicate rate before and after, look at what's
actually appearing in top-k for the degraded queries, and run the eval suite against the
corpus change — which should have happened before ingesting, treating it as a deploy.

**"A user's wiki page is being ingested. What's the security concern?"**

It's untrusted input entering the index, so the retrieval layer becomes a delivery channel
for indirect prompt injection. I'd flag instruction-shaped content and keep provenance so
a bad source can be quarantined in bulk, but those are defense in depth — the load-bearing
control is that the model has no credential that can do damage if the injection succeeds.

---

## What to remember

- Bad content doesn't sit quietly. It competes for top-k and frequently wins, producing
  confident wrong answers with high groundedness.
- Near-duplicate and boilerplate detection are the two highest-value gates in most
  corpora, and both are cheap.
- Derive boilerplate from corpus frequency rather than maintaining rules.
- A document that cannot retrieve itself is indexed and invisible. Test for it.
- Ingestion is where untrusted content enters. Flag and keep provenance; the real control
  is in section 08.
- Read fifty chunks a week. Every check on your list started as something a person noticed.
