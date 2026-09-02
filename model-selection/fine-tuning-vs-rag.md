# Fine-Tuning or RAG?

Two ways to make a model work with your information.

**RAG** — look things up and put them in the prompt. The model reads them and answers.

**Fine-tuning** — train the model on your examples so the knowledge or behaviour lives
in its weights.

They solve different problems, and the most common mistake is reaching for fine-tuning
when the answer is retrieval.

---

## The one-line rule

> **RAG teaches the model *what*. Fine-tuning teaches the model *how*.**

If you need it to know facts — your policies, your product details, your documents —
that's RAG.

If you need it to behave a certain way — a specific output format, a house tone, a
narrow classification task — that's fine-tuning.

Most "we should fine-tune on our documentation" projects are actually retrieval projects
that will fail as fine-tuning projects, for reasons below.

---

## Why fine-tuning on documents usually fails

This idea is intuitive and wrong, so it's worth being clear about why.

**Facts don't stick reliably.** Training on a document doesn't reliably put its contents
in the model's memory in a retrievable way. You'll get the flavour of your documents —
their style, their vocabulary — without dependable recall of specific facts.

**You can't update it.** Your refund policy changes. With RAG you edit one document.
With fine-tuning you retrain, evaluate, and redeploy. If your information changes
monthly, fine-tuning is a treadmill.

**No citations.** RAG can say "this came from document X, section 3." A fine-tuned model
just asserts. In anything regulated, or anything where users need to verify, that alone
rules it out.

**No permissions.** RAG can filter documents by who's asking. A fine-tuned model has
absorbed everything it was trained on, and there's no way to un-know it for one user.
**If your documents have different audiences, fine-tuning on them is a data leak waiting
to happen.**

That last point ends the discussion in most enterprise settings, and it's the one people
haven't thought about.

---

## When fine-tuning is genuinely right

It's not always wrong — there are cases where it's clearly the best answer.

**Consistent output format.** You need exactly the same structure every time, and
prompting keeps drifting. Fine-tuning nails formats.

**A specific tone or style.** Your brand voice, a particular register, a house style
that's hard to describe but easy to demonstrate with examples.

**A narrow, high-volume classification task.** This is the strongest case. If you're
sorting millions of items into fixed categories and you have lots of labelled examples,
a fine-tuned small model will often beat a large general model at a fraction of the
cost.

**Replacing a huge prompt.** If your prompt is 6,000 tokens of accumulated rules and
you're sending it millions of times, that's a signal. You're paying to re-explain the
task on every single request. Move it into weights.

**A domain the model handles badly.** Unusual notation, specialist jargon, an internal
language the base model has never seen.

### The tell that fine-tuning might be right

Ask: **is your prompt long, rule-heavy, and used at high volume on a narrow task?**

If yes, you're paying an enormous repeated cost to explain something that could be
learned once. That's the clearest fine-tuning signal there is.

---

## Cost comparison

The shapes are different, and this catches people out.

**RAG:**
- Setup: days to weeks
- Per request: you pay for retrieved context tokens on every single call
- Updating: edit a document, re-index it. Minutes.

**Fine-tuning:**
- Setup: weeks — collect data, clean it, train, evaluate
- Per request: cheaper, because your prompt is short
- Updating: retrain. Days.
- **Ongoing: someone has to own retraining forever**

That last line is the one missing from most fine-tuning proposals. A fine-tuned model
isn't a project you finish — it's an artefact you maintain. If nobody owns monthly
retraining, it slowly rots and quality declines without anyone noticing.

A rough break-even: fine-tuning pays off on **narrow, stable, very high-volume** tasks.
If any of those three is missing, RAG or prompting usually wins.

---

## Use both

They're not competitors. The strongest systems often do both:

```python
def answer(question):
    docs = retrieve(question)                  # RAG: current facts
    return finetuned_model(question, docs)     # fine-tuned: format and tone
```

The fine-tuned model has learned how to read your documents and produce your format. The
retrieval gives it today's facts. You get consistent behaviour *and* up-to-date
information.

This combination is underused because people frame it as a choice.

---

## Before you fine-tune, try these

In order, cheapest first. Most teams skip straight to the end.

**1. Better prompting.** Clear instructions, three to five good examples, explicit
constraints. Hours of work.

**2. Structured output.** If the problem is format drift, a strict schema fixes it
without training anything.

**3. Better retrieval.** If the problem is wrong facts, the fix is retrieval, not
training.

**4. Task decomposition.** Split one hard task into two easy ones.

**5. A different model.** Sometimes the base model is just wrong for the task.

**Then** fine-tune, if you still have a gap.

The reason for this order isn't laziness. It's that steps 1–5 take days and can be
reversed in minutes. Fine-tuning takes weeks and creates a permanent maintenance
obligation. **Establish the cheap baseline first**, or you'll credit fine-tuning with
gains that came from cleaning up your prompt.

---

## If you do fine-tune

The practical things that decide whether it works.

### Data quality beats data quantity

500 carefully checked examples beat 50,000 noisy ones. Your model learns your labels,
including your mistakes.

Audit a sample before training. If 5% of your labels are wrong, your ceiling is 95% and
you'll spend weeks chasing failures that aren't failures.

### Split by time and by source, not randomly

This is the mistake that produces a model that scores brilliantly and fails in
production.

```python
# WRONG — near-duplicate examples end up in both train and test
train, test = random_split(examples, 0.8)

# RIGHT — test on data from after the training period
cutoff = "2026-06-01"
train = [e for e in examples if e.date < cutoff]
test  = [e for e in examples if e.date >= cutoff]
```

Why it matters: real datasets contain near-duplicates — the same customer's repeated
tickets, template-generated documents, similar messages from the same source. A random
split puts near-identical examples on both sides, the model memorises them, and your
test score is inflated.

Split by time *and* by customer/source where you can.

### Keep a fallback

Your fine-tuned model will meet inputs unlike anything it was trained on. Keep a general
model available for those:

```python
def answer(request):
    result = finetuned(request)
    if result.confidence < THRESHOLD or result.status == "unknown":
        return general_model(request)      # and log it as a training example
    return result
```

Two benefits: you handle the unusual cases, and every fallback is a labelled hard
example for the next training run. The system improves itself.

### Plan the retraining

Before you start, answer:

- How often do we retrain?
- Who owns it?
- What triggers an off-schedule retrain?
- How do we know quality has drifted?
- What's the rollback if a new version is worse?

If you can't answer these, you're not ready to fine-tune. A model with no owner degrades
quietly.

---

## Interview questions

**1. "Should we fine-tune on our documentation?"**

Almost certainly not — that's a retrieval problem. Give the reasons: facts don't stick
reliably, you can't update it, no citations, and no per-user permissions. Then ask what
they're actually trying to fix; often it's format or tone, which *is* a fine-tuning
case.

**2. "When is fine-tuning the right answer?"**

Narrow task, high volume, stable requirements, lots of good labels. The clearest tell is
a long rule-heavy prompt being sent millions of times — you're paying to re-explain the
task on every request.

**3. "What would you try first?"**

Prompting with examples, structured output, better retrieval, task decomposition, a
different model. Explain that these take days and are reversible, while fine-tuning
takes weeks and creates permanent maintenance.

**4. "How do you split train and test data?"**

By time, and by customer or source. Explain that random splits leak near-duplicates and
inflate scores — this is a strong signal of whether someone has actually trained a model
on real data.

**5. "Your fine-tuned model scores 96% offline and 89% in production. Why?"**

Most likely leakage from a random split, or a distribution shift between training data
and current traffic. Then talk about temporal splits and monitoring for drift.

**6. "What's the hidden cost?"**

You own the model forever. Retraining cadence, an owner, drift monitoring, rollback. A
fine-tuned model with no owner rots.

---

## What to remember

- RAG for *what* (facts). Fine-tuning for *how* (format, tone, narrow tasks).
- Fine-tuning on documents usually fails: facts don't stick, can't update, no citations,
  no permissions.
- The permissions problem alone rules it out for most enterprise document sets.
- The clearest fine-tuning signal: a long rule-heavy prompt at high volume.
- Try prompting, schemas, retrieval, and decomposition first — days versus weeks.
- Use both together: fine-tuned behaviour, retrieved facts.
- Split by time and source, never randomly.
- Keep a general-model fallback, and turn its invocations into training data.
- If nobody owns retraining, don't start.

---

**Next:** [cost-quality-latency.md](cost-quality-latency.md) — the three-way trade-off.
