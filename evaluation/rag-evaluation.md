# Evaluating RAG

RAG has two parts that fail separately: finding the right documents, and writing a good
answer from them. If you only measure the final answer, you know something is wrong but
not which half — and the fixes are completely different.

**The single most useful thing in this file:** always ask *was the right information in
the prompt?* before anything else. That one question splits every RAG failure into two
piles with different owners and different solutions.

---

## The four things to measure

| What | Question it answers | Cost to measure |
|---|---|---|
| **Retrieval** | Did we find the right documents? | Cheap (needs labels) |
| **Groundedness** | Is the answer supported by what we found? | Model judge |
| **Citations** | Do the citations point at real, relevant text? | Free (code) |
| **Correctness** | Is the answer actually right? | Model judge or human |

They fail independently, and the combination tells you where to look:

| Retrieval | Groundedness | Correct? | What's wrong |
|---|---|---|---|
| Bad | Good | No | **Retrieval.** The model faithfully used wrong documents. |
| Good | Bad | No | **Generation.** It ignored what you gave it. |
| Good | Good | No | Your documents don't contain the answer — or your label is wrong. |
| Good | Good | Yes | Working. |

That first row is the interesting one, and it's counterintuitive: the model isn't
hallucinating. It's doing exactly what it should with the wrong input. Teams that only
measure "correctness" go and tune prompts for two weeks when the problem was in the
retriever.

---

## 1. Retrieval quality

You need labels: for each question, which documents should come back.

```python
@dataclass
class RagCase:
    id: str
    question: str
    relevant_doc_ids: list[str]     # humans decided these are relevant
    expected_answer: str
    segments: dict[str, str]
```

### Recall — did we find them?

```python
def recall_at_k(retrieved_ids, relevant_ids, k):
    """What fraction of the right documents made it into the top k?"""
    if not relevant_ids:
        return None
    found = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(found) / len(relevant_ids)
```

Measure at several values of k, because they answer different questions:

- **recall@5** — what the model actually sees. This is the number that limits your
  answer quality.
- **recall@50** — what a reranker *could* promote. If recall@50 is high and recall@5 is
  low, a reranker will help a lot. If recall@50 is also low, the documents aren't being
  found at all and a reranker can't save you.

That comparison is genuinely useful and takes one line to produce.

### Ranking — are they near the top?

Position matters, because context is limited and the model attends better to what's
near the edges of the prompt. nDCG rewards putting the best documents first:

```python
def ndcg_at_k(retrieved_ids, relevance_scores, k):
    """relevance_scores: {doc_id: how relevant, e.g. 0-3}"""
    gains = [relevance_scores.get(d, 0) for d in retrieved_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0
```

### The labelling problem, and how to shrink it

Labelling relevant documents for 500 questions is expensive. Two things make it
manageable:

**Pool the candidates.** You don't need to look at every document in your corpus. Run
several retrieval configurations, take the top 20 from each, merge, and only label
those. Anything no configuration retrieves is almost certainly not relevant.

```python
def build_pool(question, retrievers, k=20):
    pool = set()
    for r in retrievers:              # dense, BM25, hybrid, different embeddings
        pool.update(d.id for d in r.search(question, k))
    return pool                       # label this, not the whole corpus
```

**Grade, don't tick.** "Relevant / not relevant" is harder for humans than it sounds.
A 0–3 scale (irrelevant, related, useful, exactly answers it) is easier to agree on and
gives you nDCG for free.

---

## 2. Groundedness

Is every claim in the answer supported by the documents you retrieved?

Note this is **separate from correctness**. An answer can be perfectly grounded and
completely wrong, if the documents were wrong or irrelevant. Keeping them separate is
what makes the diagnosis table at the top work.

```python
GROUNDEDNESS_PROMPT = """Check whether every factual claim in the ANSWER is
supported by the CONTEXT.

Ignore whether the answer is correct. Only check whether the CONTEXT supports it.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only:
{{"total_claims": <int>, "supported_claims": <int>,
  "unsupported": ["<claim>", ...]}}"""


async def groundedness(case, output, judge):
    context = "\n\n".join(d["text"] for d in output.retrieved)
    if not context:
        return None                       # nothing retrieved — not scoreable
    raw = await judge(GROUNDEDNESS_PROMPT.format(context=context,
                                                 answer=output.text),
                      temperature=0.0)
    data = json.loads(raw)
    total = data["total_claims"]
    return data["supported_claims"] / total if total else 1.0
```

The `unsupported` list is worth keeping. Reading the actual unsupported claims tells
you far more than the score — usually you find the model is adding one plausible detail
that wasn't in any document.

---

## 3. Citations

Free to check, and it catches a failure that groundedness misses entirely.

```python
def citation_validity(output):
    """Do the cited IDs correspond to documents we actually retrieved?"""
    cited = re.findall(r"\[([^\]]+)\]", output.text)
    if not cited:
        return None                        # nothing cited — not a score
    available = {d["id"] for d in output.retrieved}
    return sum(1 for c in cited if c in available) / len(cited)
```

Two different failures here:

**Made-up citations.** The model cites `[doc-4471]` which wasn't retrieved and may not
exist. Caught by the code above, free.

**Wrong attribution.** The citation points at a real retrieved document, but that
document doesn't support the specific claim. Needs a judge, and it's the one users
notice — they click a citation, read it, and it doesn't say what the answer claimed.

A system can score 0.94 on groundedness and 0.61 on citation validity at the same time.
Users experience that as untrustworthy even when the answers are right. Measure both.

---

## 4. Abstention

Does it say "I don't know" when it should?

You need test cases where the answer genuinely isn't in your documents. This is the
part most eval sets are missing, because people build datasets by generating questions
*from* their documents — so every question is answerable by construction.

```python
def abstention_correct(case, output):
    should_abstain = case.segments.get("answerable") == "no"
    did_abstain = any(m in output.text.lower() for m in [
        "don't have enough information",
        "cannot answer",
        "no information",
        "unable to determine",
    ])
    return 1.0 if should_abstain == did_abstain else 0.0
```

Report the two error types separately, because they cost very different things:

```python
def abstention_breakdown(results):
    return {
        # Answered when it should have abstained → hallucination
        "false_answer_rate": rate(results, should=False, did=True),
        # Abstained when it could have answered → uselessly cautious
        "false_abstain_rate": rate(results, should=True, did=False),
    }
```

A single score hides the trade-off you're making. A system that always abstains scores
well on hallucination and is worthless.

---

## Turning scores into a diagnosis

This is what people actually want from a RAG evaluation — not "quality is 0.74" but
"work on retrieval."

```python
def diagnose(report):
    r = report.mean("recall@5")
    g = report.mean("groundedness")
    c = report.mean("citation_validity")
    a = report.mean("correctness")

    findings = []

    if r < 0.7:
        findings.append(
            "RETRIEVAL is the bottleneck. The right documents aren't being found. "
            "Look at chunking, hybrid search, and reranking before touching prompts."
        )
    if g < 0.85 and r >= 0.7:
        findings.append(
            "GENERATION is the problem. Documents are found but the answer isn't "
            "grounded in them. Look at the prompt, context ordering, and length."
        )
    if c < 0.9:
        findings.append(
            "CITATIONS are unreliable. Users will lose trust even when answers "
            "are correct."
        )
    if a < 0.7 and g >= 0.85 and r >= 0.7:
        findings.append(
            "Retrieval and grounding look fine but answers are wrong. Either the "
            "documents don't contain the answer, or your labels are wrong. "
            "Read the failures."
        )

    return findings or ["No single stage stands out. Read the worst cases."]
```

---

## Things that break RAG evaluation

### The corpus changes and your labels go stale

You labelled `doc-123` as relevant. Someone rewrote it, or the chunking changed and
`doc-123` is now three chunks with different IDs.

Fix: version the corpus, store its version with the dataset, and re-check labels when
it changes materially. At minimum, warn loudly when the corpus version differs from the
one the labels were made against.

### Several documents could answer the question

Three different pages explain the refund policy. You labelled one. The system retrieves
another and scores as a miss, even though the answer was correct.

Fix: label *all* acceptable documents, or score on whether the answer is right rather
than whether a specific document was found. This is a real source of misleadingly low
retrieval scores.

### Chunk-level vs. document-level

Do you label relevant documents, or relevant chunks? Chunks are more precise but the
labels break every time you change chunking — which you will.

Practical answer: label at the document level, and measure retrieval as "did we get a
chunk from a relevant document." Survives re-chunking.

### Multi-turn questions

"What about the second one?" is meaningless alone. If your dataset stores questions
without their conversation history, you're evaluating a different task from the one
your system does.

Store the full conversation, and tag `turns: multi` so you can see the segment
separately. This is the exact segment that regressed silently in the
[production incident](../12-senior-scenarios/production-incidents.md#incident-6-it-works-in-staging).

---

## What to measure when you change something

Different changes need different checks. A quick guide:

| Change | Check first |
|---|---|
| Chunk size | recall@5, recall@50, citation validity |
| Embedding model | recall@k — and rebuild the index first, or scores are meaningless |
| Adding a reranker | recall@5 before vs. after, and latency |
| Adding hybrid search | recall on identifier/rare-term queries specifically |
| Prompt change | groundedness, correctness, abstention — segmented by turns |
| Adding documents | recall on existing questions (new docs can crowd out old answers) |
| Larger `top_k` | groundedness (more context can mean more to contradict), cost, latency |

That last row on adding documents is worth knowing: importing a new document set can
*reduce* quality on existing questions, because the new documents outrank the old
correct ones. It's a real incident pattern and only a before/after recall comparison
catches it.

---

## Interview questions

**1. "How do you evaluate RAG?"**

Separate the stages. Retrieval, groundedness, citations, correctness. Explain that a
single score can't tell you whether retrieval or generation is at fault, and the fixes
are different.

**2. "Groundedness is 0.94 but answers are wrong. What's happening?"**

The model is faithfully using the wrong documents. It's a retrieval problem, not a
hallucination problem. Say clearly that "not hallucinating" and "wrong" can both be
true.

**3. "How do you get relevance labels affordably?"**

Pool candidates from several retrievers and only label the pool. Use a graded scale
rather than binary. Start with 200 questions rather than 2,000.

**4. "How do you test that the system says 'I don't know'?"**

Include cases where the answer isn't in the corpus. Note that this is missing from most
datasets, because questions generated from documents are always answerable. Report
false-answer and false-abstain rates separately.

**5. "You add 14,000 documents and quality drops. How do you find out why?"**

Compare recall@5 on the existing question set before and after, and check what fraction
of retrieved chunks now come from the new documents. New documents can outrank old
correct ones — the model is then faithfully answering from the wrong source.

---

## What to remember

- Ask "was the right information in the prompt?" first. It splits every failure.
- Retrieval, groundedness, citations, correctness — measure separately.
- Groundedness high + correctness low = retrieval problem, not hallucination.
- recall@5 vs. recall@50 tells you whether a reranker will help.
- Citation validity is free to check and catches what groundedness misses.
- Include unanswerable questions, or you never test abstention.
- Label documents, not chunks — chunking changes.
- Adding documents can make existing answers worse.

---

**Next:** [llm-as-judge.md](llm-as-judge.md) — using a model to grade, without fooling
yourself.
