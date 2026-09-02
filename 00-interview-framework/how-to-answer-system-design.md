# How to Answer System Design Questions

A structure for open-ended questions like *"design a system that…"*. It's not a script —
scripts sound rehearsed — but having a shape stops you rambling and stops you forgetting
the parts that carry the most signal.

---

## The shape

```
1. Clarify        2-4 min    Ask the questions that change the design
2. Frame          1-2 min    State the constraints and what's actually hard
3. Sketch        5-10 min    The simple version, end to end
4. Deepen        10-15 min   Go deep where they push, or where it matters
5. Fail           3-5 min    What breaks, and what you'd measure
6. Sequence       2-3 min    What ships first, what you'd skip
```

Roughly 35 minutes. Adjust to the slot.

**The most commonly skipped steps are 1 and 6, and they carry disproportionate signal.**
Most candidates spend all their time on 3 and 4.

---

## 1. Clarify

Don't ask "any constraints?" — that puts the work on them. Ask the specific questions
whose answers would change what you build.

Four categories, and you want one or two from each:

**Scale**
- How many requests? What's peak versus average?
- How much data, and how fast is it growing?
- How many users, and are they internal or external?

**Data**
- What does it look like? Format, length, structure?
- How often does it change?
- Are there permissions — can everyone see everything?

**Constraints**
- Latency requirement — and is that to first token, or to completion?
- Budget?
- Any residency, compliance, or security rules?

**Success**
- What does a wrong answer cost?
- Is there something today that this replaces? What's its baseline?

Then say why you're asking. This is the part that converts a question into a signal:

> "I'm asking about identifiers because if people search by product code, dense retrieval
> alone will fail — embeddings put E-4471 and E-4472 in almost the same place. That's a
> hybrid-search requirement rather than an optimisation."

Now the interviewer knows you're not asking from a checklist.

**Timebox it.** Two to four minutes. If they say "assume whatever you like," state your
assumptions out loud and move on — don't keep asking.

---

## 2. Frame

Before drawing anything, say what you think the hard part is. This is short and it
orients the whole conversation.

> "Given 20 million documents and 50,000 users, I think there are three real problems
> here: retrieval quality at that corpus size, permissions — because if different people
> see different documents that has to be in the query rather than filtered afterwards —
> and cost, because 15× traffic growth against a per-token bill compounds fast.
>
> The rest is fairly standard. I'd focus on those three unless you'd rather I went
> elsewhere."

Two things this does. It shows you can identify what matters, and it invites the
interviewer to redirect you — which they will, and that's useful.

---

## 3. Sketch

Draw the simple version, end to end. Resist showing off here.

```
documents → parse → chunk → embed → index
                                      ↓
query → embed → search → filter → rerank → context → model → answer
```

Talk through it in one pass, naming each choice briefly:

> "Parse first — worth saying that PDF parsing is where a lot of quality is lost, so I'd
> look at the extracted text before building on it. Chunk on document structure rather
> than character counts. Embed and index.
>
> On the query side: embed, search, then filter by permissions **inside** the query rather
> than after — post-filtering returns fewer than k results and means you loaded data the
> user shouldn't see. Rerank, build the context, generate with citations."

**Common mistakes at this step:**

- Starting with the most complex version. Show the simple one, then add.
- Silently skipping unglamorous parts. Parsing, permissions and updates carry signal
  precisely because most candidates skip them.
- Drawing without narrating why.

---

## 4. Deepen

The interviewer will push somewhere. Follow them — that's what they want to explore. If
they don't push, pick the thing you named as hard in step 2.

Going deep well means **numbers and mechanisms**, not more component names.

**Shallow:**
> "I'd use a vector database for the embeddings."

**Deep:**
> "Let me work out whether we need one. 20 million documents, maybe 50 million chunks, at
> 768 dimensions and 4 bytes a number, that's about 150 GB of raw vectors. That doesn't
> fit comfortably in memory on one machine, so yes, we need a real index — HNSW or
> similar.
>
> But there's a catch with multi-tenancy. If we filter by tenant *after* the ANN search,
> then for a small tenant the top results are dominated by other tenants' documents and we
> filter almost everything out. So the filter needs to be partition pruning — partition
> the index by tenant so the search only ever traverses their data."

That's the same topic answered with arithmetic and a specific failure mode.

**A useful move when you're unsure:** say what you'd measure to decide.

> "I don't know whether reranking is worth the latency here without testing it. What I'd
> check is recall@50 versus recall@5 — if recall@50 is high and recall@5 is low, reranking
> has good material to work with. If recall@50 is also low, the right document isn't in
> the pool and reranking can't help."

Admitting uncertainty *with a plan to resolve it* is stronger than guessing confidently.

---

## 5. Fail

Volunteer this. Don't wait to be asked.

> "Things I'd expect to go wrong:
>
> If someone bulk-imports a new document set, it can outrank the existing correct answers
> and quality drops with no deploy. So I'd treat corpus changes as deploys and run the
> eval suite on them.
>
> If anyone upgrades the embedding model without rebuilding the index, retrieval breaks
> silently — the vectors are in different spaces, similarity still returns normal-looking
> numbers, and there's no error. I'd bind the model version to the index and refuse
> mismatched queries.
>
> And if we add a cache, the key has to include the permission scope, or eventually one
> tenant sees another's answer."

Then measurement:

> "For knowing whether it works: a few hundred real questions with the documents that
> should come back, segmented by query type. I'd measure retrieval and generation
> separately, because a wrong answer with the right documents in the prompt is a completely
> different problem from a wrong answer without them."

**This step is where senior candidates separate themselves**, because it's evidence you've
operated something rather than designed it.

---

## 6. Sequence

Close by saying what you'd build first, and what you'd leave out.

> "Version one: parse, chunk on structure, embed, search, generate with citations, plus
> the eval set. That's a working system and it tells me where the problems are.
>
> I'd skip reranking, hybrid search, and query rewriting initially — not because they're
> not valuable, but because I want to add each one against a measured problem rather than
> a guess. If recall@5 is poor but recall@50 is fine, that's reranking. If exact product
> codes fail, that's hybrid search.
>
> The one thing I'd get right up front is the embedding model choice and the ability to
> hold two indexes at once, because that's the decision that's expensive to reverse."

Three signals in one minute: prioritisation, willingness to defer, and knowing which
decisions are irreversible.

---

## Handling the curveball

Good interviewers change the constraints partway through. That's not you failing — it's
the actual test.

> *"Now the security team says data can't leave your cloud."*

**Weak:** panic, or rebuild everything.

**Strong:** work out what's actually affected.

> "That changes the inference and the embeddings, but not the retrieval logic. The
> question is what fraction of the corpus is affected — if it's all of it, we're
> self-hosting and I'd want to check whether an open model clears the quality bar before
> committing to the infrastructure. If it's 15%, I'd partition: restricted content stays
> in-VPC with a self-hosted model, everything else keeps the current path. That way we buy
> the expensive operational problem only where the requirement is real.
>
> One thing to flag: embeddings are derived data. If the source documents can't leave, the
> vectors can't either. That's the bit people miss."

Structure of a good response to a curveball:
1. Say what it affects and what it doesn't.
2. Ask the question that determines the scale of the change.
3. Give the answer for each case.
4. Name a consequence they might not have considered.

---

## Time management

| Symptom | Fix |
|---|---|
| Still clarifying at 8 minutes | State assumptions, move on |
| No architecture at 15 minutes | Sketch now, deepen later |
| Deep in one component at 25 minutes | Zoom out, cover failure modes |
| Nothing about measurement | Get it in before you finish |
| Finished at 20 minutes | Volunteer failure modes and sequencing |

If you're running out of time, prioritise: **failure modes and measurement over more
architecture.** Two more components is worth less than one good failure mode.

---

## Phrases that carry signal

Not tricks — they're compressed versions of the reasoning.

> "Before I design this, two things would change the answer…"

> "Let me do the arithmetic on that."

> "That's the decision I'd find hardest to reverse, so…"

> "I'd measure X before deciding, because…"

> "I'd skip that in v1 — here's what would make me add it."

> "That changes my answer. If that's the case, then…"

> "The thing that would break first is…"

That last-but-one is worth practising. Updating in response to new information reads as
strength; defending your first idea against evidence reads as the opposite.

---

## A worked example

Compressed version of a full answer, so you can see the shape end to end.

> **Q: "Design a system that answers employee questions over internal documents."**
>
> **Clarify:** "How many documents and how fast do they change? Do all employees see all
> of them, or are there access controls? And roughly what do people ask — is it lookups,
> or reasoning across documents?"
>
> *(Interviewer: 20M documents, changing daily, permissions matter, mostly lookups.)*
>
> **Frame:** "Then the hard parts are permissions, keeping the index fresh against daily
> changes, and retrieval quality at 20 million documents. Mostly-lookups is good news —
> that means a smaller model will probably do, and the quality will live in retrieval
> rather than generation."
>
> **Sketch:** *(pipeline, one pass, naming choices)*
>
> **Deepen:** "On permissions — the filter goes in the search query, not after. On
> freshness — daily changes means I'd version documents: index the new version, switch
> atomically, then retire the old, so there's never a window where a document is half
> indexed."
>
> **Fail:** "Bulk imports can outrank existing answers. Embedding model upgrades break
> retrieval silently. A cache without the permission scope in the key leaks across users."
>
> **Sequence:** "Ship the simple pipeline plus the eval set. Add reranking and hybrid
> search against measured problems. Get the embedding choice and two-index capacity right
> now, because those are expensive to change."

---

## What to remember

- Ask two or three questions that would change the design, and say why you're asking.
- Name what you think the hard part is before drawing anything.
- Sketch simple first, then deepen where they push.
- Go deep with numbers and mechanisms, not more component names.
- Volunteer failure modes and measurement. Don't wait to be asked.
- Close with what you'd build first and what you'd skip.
- When they change the constraints, say what it affects and what it doesn't.
- If short on time, cut architecture, not failure modes.

---

**Next:** [how-to-answer-debugging.md](how-to-answer-debugging.md) — a different structure,
for a different kind of question.
