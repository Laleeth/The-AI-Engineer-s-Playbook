# Schema and Versioning

Everything in a retrieval system has a version: the parser, the chunker, the embedding
model, the index, the metadata schema. Most teams version none of them, which makes
"why did this answer change?" unanswerable and makes some upgrades impossible to perform
safely.

This file is about the small number of decisions that are painful to retrofit.

---

## Bind the embedding model to the index

The most important rule in this section, and the source of the most damaging silent
failure in RAG.

If you upgrade the embedding model without rebuilding the index, query vectors and stored
vectors live in different spaces. Similarity scores still come back. They still look
normal — 0.83, 0.79, plausible ordering. The results are meaningless, and nothing errors.

[embeddings.md](../03-rag/embeddings.md) states the rule; here is how to make it
enforceable rather than remembered:

```python
class Index:
    def __init__(self, name, embedding_model, dimensions):
        self.name = name
        self.embedding_model = embedding_model      # immutable for this index
        self.dimensions = dimensions

    def search(self, query_vector, embedding_model, k=10):
        if embedding_model != self.embedding_model:
            raise ValueError(
                f"index {self.name} was built with {self.embedding_model}, "
                f"query used {embedding_model} — results would be meaningless"
            )
        return self._search(query_vector, k)
```

Make the mismatch impossible to express rather than documented. The check costs a string
comparison per query and converts a silent catastrophe into a loud startup failure —
which is the "make the unsafe state unrepresentable" principle from
[11-coding-rounds](../11-coding-rounds/README.md), applied to data.

A dimension check is not sufficient. Two different models with the same output dimension
will pass a shape check and produce nonsense.

---

## What needs a version, and where it lives

| Thing | Where the version lives | Why |
|---|---|---|
| **Embedding model** | Bound to the index, checked at query time | Mixed vectors are silently meaningless |
| **Index** | A pointer the serving layer resolves | Makes promotion and rollback atomic |
| **Chunking strategy** | Recorded per chunk | Changes retrieval behavior; needed to interpret old results |
| **Parser** | Recorded per document in the catalog | Lets you reprocess only what an old parser touched |
| **Metadata schema** | A version on the record | Lets readers handle old and new shapes |
| **Source document** | A content hash or source version in the catalog | Change detection and idempotency both need it |

The pattern throughout: the version travels with the data, not in a config file that
describes what the data is currently assumed to be. Configuration drifts from reality;
data carrying its own version does not.

---

## Schema evolution without a rebuild

Adding a metadata field to a 500-million-chunk index is not a migration you want to run
casually. Design so that most changes don't require one.

**Additive changes are free if readers tolerate absence.** New field, old chunks don't
have it, queries must handle null. That works — as long as filtering on the new field
degrades sensibly rather than silently excluding every older chunk.

This is where a real bug lives. A filter on `region == "EU"` against chunks indexed before
`region` existed excludes them entirely, and the symptom is not an error — it's slightly
worse retrieval for old documents. Decide explicitly:

```python
def build_filter(field, value, schema_version_added):
    """Old chunks lack the field. Say what that means, don't discover it."""
    return {
        "or": [
            {field: value},
            {"schema_version": {"lt": schema_version_added}},   # grandfathered
        ]
    }
```

Grandfathering is right for a field like `topic` and wrong for one like `permissions` —
where an unlabeled chunk must be excluded, not included, because failing closed is the
requirement in
[08-ai-security/authorization.md](../08-ai-security/authorization.md). Make that choice
per field, deliberately, and write down which it is.

**Changing the meaning of an existing field requires a rebuild**, or a new field. Silently
reinterpreting a field means old and new chunks disagree about what it means, and nothing
detects that.

---

## Metadata that must exist from day one

Some fields cannot be added later without reprocessing the corpus, because the information
only exists at ingestion time:

- **Character offsets** into the source (`doc.text[start:end] == chunk.text`). Citations
  and highlighting depend on them.
- **Document id and chunk index**, stable across reprocessing, for deterministic ids and
  delete-by-document.
- **Permission identifiers** — groups, tenants, classification. Filtering must happen
  inside the query, which requires the labels to be in the index.
- **Source URI and source version**, for provenance and change detection.
- **Effective date**, distinct from ingestion date, so recency can be reasoned about.
  Ingestion date is when you saw it; effective date is when it became true.

The last one is quietly important. A policy document ingested today may have been
effective for three years, and a superseded one ingested yesterday is newer by ingestion
date and wrong. Ranking or filtering on the wrong date is a subtle, hard-to-diagnose
quality problem.

---

## Version the query path too

An index version is only half the answer. The same index queried with a different chunk
budget, a different reranker, or a different retrieval strategy gives different results.

Record the whole configuration that produced a response — the metadata block in
[corpus-changes.md](corpus-changes.md) — and version the retrieval configuration as one
unit rather than as loose parameters. "Retrieval config v7" is something you can compare,
roll back, and correlate with a quality change; six independent settings are not.

---

## Deprecating an old index

Old index versions cost storage and carry risk. Retiring them needs a small amount of
process:

- **Keep at least one previous generation warm** so rollback is a pointer swap.
- **Apply deletions to every retained index**, or rolling back reintroduces documents
  deleted for legal reasons. This is the trap in
  [corpus-changes.md](corpus-changes.md) and it is worth stating twice.
- **Set a retention limit.** Indefinite retention of every index generation is its own
  compliance exposure, particularly once you know embeddings are derived personal data
  ([pii.md](../08-ai-security/pii.md)).
- **Alert on queries hitting a deprecated version.** Something still points at it, and you
  want to know before you delete it.

---

## What to monitor

- **Embedding model mismatch attempts.** Should be zero; non-zero means a deploy nearly
  shipped a silent catastrophe.
- **Schema version distribution across chunks**, which tells you how much of the corpus a
  new field actually covers.
- **Chunks missing required metadata**, by field. Permissions being 3% null is a security
  finding.
- **Index versions in existence, and which is served**, with age.
- **Queries by retrieval config version**, so a quality change can be attributed.

---

## Interview questions

**"Someone upgraded the embedding model and search became nonsense. Why, and how do you
prevent it?"**

Query vectors and stored vectors are in different spaces, so similarity scores are
arbitrary — and they still look plausible, so nothing alerts. Prevention is binding the
embedding model identity to the index and rejecting a query that arrives with a different
one, rather than documenting the rule. A dimension check isn't enough: two models can
share a dimension and still be incompatible.

**"You need to add a `region` field to a 500-million-chunk index for data residency. How?"**

Additively, with an explicit decision about what the absence of the field means. For
residency it must fail closed — unlabeled chunks are excluded, not grandfathered — which
means the field isn't usable for enforcement until backfilled. So: add it going forward,
backfill from the catalog where the source system knows the answer, and don't enable the
filter until coverage is complete and verified. The tempting shortcut, grandfathering
unlabeled chunks, is a residency violation.

**"What metadata do you insist on from day one?"**

Character offsets, stable document and chunk ids, permission identifiers, source URI and
version, and effective date separate from ingestion date. All of them are either
impossible or very expensive to add later, and the permission one is the difference
between filtering inside the query and filtering after it.

---

## What to remember

- Bind the embedding model to the index and reject mismatched queries. It's the loudest
  possible fix for the quietest possible failure.
- Versions travel with the data, not in a config file describing what the data is assumed
  to be.
- Additive schema changes are safe only when you decide what absence means — grandfather
  for topics, fail closed for permissions.
- Offsets, stable ids, permissions, provenance and effective date must exist at ingestion.
  None can be retrofitted cheaply.
- Version the retrieval configuration as one unit, so a quality change can be attributed
  to something you can roll back.
