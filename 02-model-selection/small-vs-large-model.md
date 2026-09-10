# Small Model or Large Model?

Small models are cheaper, faster, and often good enough. Large models are better at hard
reasoning and cost a lot more.

The useful question isn't "which is better" — it's **"what does this specific task
actually need?"** Most teams answer that by reaching for the biggest model available and
never checking.

---

## Rough shape of the difference

Model sizes and prices change constantly, so think in ratios rather than numbers:

| | Small | Mid | Large |
|---|---|---|---|
| Cost per token | 1× | ~4–8× | ~20–50× |
| Speed | Fast | Medium | Slower |
| Simple instructions | Fine | Fine | Fine |
| Multi-step reasoning | Struggles | Usually OK | Good |
| Following complex formats | Sometimes drifts | Usually fine | Reliable |
| Long context | Weaker attention | OK | Best |
| Unusual/rare knowledge | Gaps | Some gaps | Fewer gaps |

The pattern that matters: **the price gap is much bigger than the quality gap on easy
tasks, and much smaller than the quality gap on hard ones.**

---

## Tasks where a small model is usually enough

If your task looks like these, test a small model before assuming you need more:

**Classification.** Sorting things into a fixed set of categories. This is what small
models are best at, especially with a few examples in the prompt.

**Extraction from structured text.** Pulling fields out of a form, an invoice, a
standard document.

**Simple rewriting.** Fix the grammar. Make it shorter. Change the tone.

**Routing decisions.** Which team should handle this? Which tool should we use?

**Yes/no judgements.** Does this text break policy? Is this answer supported by the
context?

**Straightforward question answering with retrieval.** When the answer is sitting in a
retrieved document and the job is to read it out clearly, model size matters much less
than retrieval quality.

That last one surprises people. In a lot of RAG systems, upgrading the model barely
moves the score while improving retrieval moves it a lot — because the bottleneck was
never generation.

---

## Tasks where you probably need a bigger model

**Multi-step reasoning.** Working through a problem where step 3 depends on step 2.

**Ambiguous instructions.** Requests that need interpretation rather than execution.

**Long documents.** Small models attend poorly across very long contexts — they lose
things in the middle.

**Complex structured output.** Deeply nested JSON with conditional fields. Small models
drift.

**Tool selection among many tools.** Picking correctly from 20 tools is a harder
judgement than it looks.

**Anything where being confidently wrong is expensive.** Small models fail *silently* —
they produce fluent, plausible, wrong answers without signaling difficulty. If your
system can't detect that, the cost of the failure is on you.

That last point is the important one and it's about your system, not the model.

---

## How to actually decide

Don't reason about it. Measure it. The whole test takes a day.

### 1. Get real examples

100–200 real requests from your logs. Not invented ones — real ones, with the mess.

### 2. Run both

```python
async def compare_models(cases, small, large, scorer):
    results = []
    for case in cases:
        s = await small(case.input)
        l = await large(case.input)
        results.append({
            "id": case.id,
            "small_score": await scorer(case, s),
            "large_score": await scorer(case, l),
            "small_cost": s.cost, "large_cost": l.cost,
            "small_ms": s.latency_ms, "large_ms": l.latency_ms,
            "segment": case.segment,
        })
    return results
```

### 3. Look at it by segment, not overall

This is the step that changes the answer:

```python
def by_segment(results):
    out = {}
    for seg in {r["segment"] for r in results}:
        rows = [r for r in results if r["segment"] == seg]
        out[seg] = {
            "n": len(rows),
            "small": mean(r["small_score"] for r in rows),
            "large": mean(r["large_score"] for r in rows),
            "gap": mean(r["large_score"] - r["small_score"] for r in rows),
        }
    return out
```

A typical result:

```
segment          n     small   large   gap
simple_lookup   140    0.94    0.95   +0.01     ← small model is fine
standard_query   45    0.88    0.92   +0.04     ← small is probably fine
complex_multi    15    0.61    0.89   +0.28     ← needs the large model
```

**That table is the answer.** It's not "small or large" — it's "small for 185 cases,
large for 15." Which is [routing](model-routing.md).

### 4. Check where it fails, not just how often

Two models with the same score can fail completely differently:

```python
def failure_overlap(results, threshold=0.5):
    small_fails = {r["id"] for r in results if r["small_score"] < threshold}
    large_fails = {r["id"] for r in results if r["large_score"] < threshold}
    return {
        "both": len(small_fails & large_fails),        # genuinely hard
        "only_small": len(small_fails - large_fails),  # the upgrade buys you these
        "only_large": len(large_fails - small_fails),  # surprising — look at these
    }
```

If most failures are in "both," the bigger model isn't buying you much and your problem
is somewhere else — retrieval, prompt, or data.

The "only_large" cases are worth reading. They're rare but informative: sometimes a
larger model over-thinks a simple task, or is more likely to refuse.

---

## Making a small model punch above its weight

Before concluding a small model can't do the job, try these. They're cheap and they
often close most of the gap.

**Give it examples.** Few-shot prompting helps small models much more than large ones.
Three to five good examples in the prompt can move a small model from unusable to fine.

**Narrow the task.** A small model asked to "handle this support ticket" does badly. The
same model asked to "classify this ticket into one of these 8 categories" does well.
Break the job into steps and give it one step.

**Constrain the output.** Enums, strict schemas, structured output modes. Removing the
model's freedom to invent formats removes a whole class of failure.

**Do the retrieval work.** If a task needs knowledge, give it the knowledge rather than
hoping the model has it. Small models have more knowledge gaps; retrieval closes them.

**Fine-tune it.** For narrow, high-volume, stable tasks with lots of examples, a
fine-tuned small model often beats a general large one — see
[fine-tuning-vs-rag.md](fine-tuning-vs-rag.md).

A worked-through version of this ordering: try prompt examples first (hours), then task
decomposition (days), then fine-tuning (weeks). Most teams skip to the last one.

---

## The silent failure problem

The single most important thing to understand about small models.

When a large model doesn't know something, it's more likely to hedge or say so. When a
small model doesn't know, it often produces a fluent, confident, wrong answer.

That's dangerous because it's undetectable from the output alone. You can't look at the
answer and tell.

Two responses:

**Design a detectable failure.** Build an explicit "not sure" branch into the output
schema so the model has somewhere to go other than confabulating:

```python
{
    "answer": "...",
    "status": "answered | insufficient_information | needs_specialist",
}
```

**Verify with code.** For anything checkable — extracted values, citations, numbers,
IDs — verify against the source rather than trusting the output.

If you can do both, small models become much safer to use, because their failures become
visible. If you can't do either, be more careful about where you deploy them.

---

## Latency matters more than people expect

Small models aren't just cheaper, they're faster — often 3–10× on time-to-first-token.

For some products that's the whole argument:

- **Autocomplete** needs to be under ~300ms. A large model can't do that. Nothing else
  matters.
- **Interactive chat** feels much better with fast first tokens, even if total time is
  similar.
- **Batch processing** doesn't care at all.

If latency is your constraint, that may decide the model before quality does. Work out
which constraint is actually binding before optimizing the other one.

---

## Interview questions

**1. "How do you choose between a small and large model?"**

Measure on real traffic, segmented. Show the per-segment table. Then say the answer is
usually "both, routed" rather than one or the other.

**2. "The small model scores 0.88 and the large one 0.92. Which do you ship?"**

Ask what the 4 points are made of — is it spread evenly or concentrated in one segment?
Ask what a failure costs. Ask about latency and volume. If the gap is concentrated in
15% of traffic, route.

**3. "What's the risk of using a small model?"**

Silent confident failures. Then explain how to make failures detectable — abstention
branches in the schema and code-based verification.

**4. "How would you make a small model good enough?"**

Few-shot examples, narrower task, constrained output, better retrieval, then fine-tuning
— in that order, cheapest first. Most teams jump to the last one.

**5. "Your RAG system is inaccurate. Would a bigger model fix it?"**

Probably not. Check whether the right documents were retrieved first. If retrieval is
failing, a bigger model just writes a more fluent wrong answer. This tests whether they
diagnose before upgrading.

---

## What to remember

- Ask what the task needs, not which model is best.
- Small models are fine for classification, extraction, simple rewriting, and routing.
- In RAG, retrieval quality usually matters more than model size.
- Measure on real traffic, segmented. The per-segment table is the decision.
- Small models fail *silently* — make failure detectable with abstention and code checks.
- Try few-shot examples, task narrowing, and output constraints before giving up.
- Latency can decide it before quality does.
- The answer is usually "both, with routing."

---

**Next:** [fine-tuning-vs-rag.md](fine-tuning-vs-rag.md) — teaching a model versus giving
it the information.
