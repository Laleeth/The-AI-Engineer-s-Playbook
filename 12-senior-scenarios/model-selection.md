# Model Selection

"Which model is best?" is not a question. "Which model, for which slice of my
traffic, under which constraints, verified how?" is.

Every scenario here is designed so that picking a single model is the wrong answer,
and so that the *strategy* — routing, cascading, confidence thresholds, fallbacks,
task decomposition — is what's being evaluated. The recurring senior skill is
**turning a model choice into a measurable engineering decision**: what would you
measure, on what data, to know you were right?

Two things worth carrying into every one of these:

- **Model choice is a decision about a distribution, not about a task.** The right
  model for the median input is often wrong for the tail, and the tail is usually
  where the money and the risk are.
- **Every model choice is temporary.** Prices drop, models are deprecated, better
  ones appear quarterly. The architecture you build to *hold* a model choice matters
  more than the choice.

**Contents**

1. [30 Million Documents, One Extraction Pipeline](#1-30-million-documents-one-extraction-pipeline)
2. [The Multilingual Support Product](#2-the-multilingual-support-product)
3. [Choosing a Model You Can't Change Later](#3-choosing-a-model-you-cant-change-later)
4. [Embedding Model Selection and the Migration You'll Regret](#4-embedding-model-selection-and-the-migration-youll-regret)
5. [The Model You Chose Is Being Deprecated](#5-the-model-you-chose-is-being-deprecated)

---

## 1. 30 Million Documents, One Extraction Pipeline

### Situation

An insurance company is digitizing its claims archive. **30 million documents** must
be processed into structured records: claim number, dates, parties, amounts, policy
references, adjuster notes, and a classification of claim type (of 62 types).

The corpus:

- 41% clean digital PDFs (forms, generated reports)
- 34% scanned documents of varying quality (some from the 1990s)
- 18% contain handwriting (adjuster annotations, signatures, marginal notes)
- 7% are "difficult": faxes of faxes, multi-column layouts, tables spanning pages,
  mixed languages

Output must be **strict JSON** validated against a schema; downstream systems reject
malformed records.

You have:

- a large frontier model (multimodal, strong on hard documents, expensive)
- a smaller open model (weaker, cheap, self-hostable)
- a specialized document-extraction model (fast, cheap, strong on forms, no
  reasoning ability)

### Your Task

Design the model strategy. Then handle the complications the interviewer introduces.

### Constraints

- Total budget for the backfill: **$400k**. That's a ceiling of ~$0.013/document.
- Timeline: 6 months for the backfill, then ~40k documents/day ongoing.
- **Accuracy requirement: 99.2% field-level accuracy on the claim number, dates, and
  amounts.** Errors in these fields cause downstream financial reconciliation
  failures.
- Lower bar (95%) on adjuster notes and classification.
- Regulatory: extraction decisions must be auditable; you must be able to show which
  page and region a value came from.
- 3 engineers.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Step 1 — The document mix is the design.** Four populations with completely
different characteristics are being treated as one problem. Clean digital PDFs are
nearly a solved problem (deterministic parsing + a cheap model, or in many cases no
model at all). Handwriting requires multimodal capability. The "difficult" 7% may
need human review.

The first design move is to **classify documents before extracting them**, and to
route by class. This is cheap (a small classifier on the first page) and it's what
makes the budget work.

**Step 2 — Do the budget math per class before choosing models.**

$400k / 30M = $0.0133/document average. That's tight. But the average is the wrong
frame: you can spend $0.002 on the 41% clean documents and $0.08 on the 7% difficult
ones and still land under budget. Compute the envelope:

```
class          share    count      budget if avg $0.0133
─────────────────────────────────────────────────────────
clean PDF       41%    12.3M      cheap → spend ~$0.002  = $25k
scanned         34%    10.2M      medium → ~$0.010       = $102k
handwriting     18%     5.4M      expensive → ~$0.030    = $162k
difficult        7%     2.1M      most expensive → ~$0.05= $105k
                                                  total  ≈ $394k
```

That fits — barely — and now the model choice per class is a constrained problem
with an actual number attached.

**Step 3 — Separate "extraction" from "verification."** The 99.2% requirement on
three fields is the hard part. No single model call reliably hits 99.2% field-level
accuracy on degraded scans. You need a **verification strategy**, and the good news
is that these three fields are *verifiable*:

- Claim numbers have a checksum or a format pattern and exist in a database — you can
  look them up.
- Dates have valid ranges and internal consistency (loss date ≤ report date ≤ close
  date).
- Amounts appear multiple times in a document and often sum to totals.

This is the crucial insight: **for the high-accuracy fields, don't try to pick a
model good enough; build a verification layer that catches the errors.** Cross-field
validation, database lookup, and multi-pass agreement will get you from a model's
~96% to a system's 99.2%+ far more cheaply and reliably than model selection will.

**Step 4 — Design the human loop.** At 99.2% on 30M documents, 240,000 documents will
still have errors in critical fields. You need a confidence-triggered human review
queue, and you need to size it: if 8% of documents route to review at 2 minutes each,
that's 80,000 hours. Not viable. If 1.5% route at 45 seconds, it's 5,600 hours ≈ 3
FTE-years — plausible. **The confidence threshold is therefore a staffing decision**,
and computing that is the kind of thing that distinguishes a real answer.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Architecture: classify → route → extract → verify → escalate.**

```
document
   │
   ├─► page-1 classifier (cheap: layout + OCR-confidence + a small model)
   │      → {clean_digital, scanned_clean, handwriting, difficult}
   │
   ├─► CLEAN DIGITAL (41%)
   │      deterministic PDF text extraction
   │      + specialized extraction model for field mapping
   │      cost ≈ $0.002/doc
   │
   ├─► SCANNED (34%)
   │      OCR → specialized extraction model
   │      escalate to small open model when OCR confidence is low
   │      cost ≈ $0.008/doc
   │
   ├─► HANDWRITING (18%)
   │      frontier multimodal model, page images
   │      cost ≈ $0.030/doc
   │
   └─► DIFFICULT (7%)
          frontier multimodal, higher effort, page-by-page
          cost ≈ $0.05/doc
   │
   ▼
VERIFICATION LAYER (all classes)
   · JSON schema validation                      (free)
   · claim number: format + checksum + DB lookup (free)
   · date consistency and range checks           (free)
   · amount arithmetic: line items sum to total  (free)
   · provenance check: every value must be
     locatable in source text/region             (free, and satisfies audit)
   │
   ├─ all checks pass + high confidence  ──► accept  (target ≈ 85–90%)
   ├─ checks fail on a low-stakes field  ──► re-extract that field with a
   │                                          stronger model (targeted, cheap)
   └─ checks fail on a critical field
      or confidence below threshold      ──► human review queue
```

**Key decisions and why:**

1. **Classification before extraction** is what makes the budget work. It costs
   ~$0.0005/document and saves an order of magnitude on the 41% clean population.

2. **The specialized extraction model handles the majority.** For structured forms,
   a purpose-built document model outperforms a general LLM on both cost and accuracy,
   and it's the right tool. Reaching for a frontier LLM on a standardized claim form
   is over-engineering.

3. **Verification is free and does most of the accuracy work.** The four checks above
   are deterministic code. They catch the bulk of extraction errors on exactly the
   fields with the 99.2% requirement. This converts an unreachable model-accuracy
   target into an achievable system-accuracy target. *Say this explicitly in an
   interview; it's the core insight.*

4. **Targeted re-extraction beats whole-document re-processing.** If only the amount
   field fails validation, re-extract that field with a stronger model and a cropped
   region — a few hundred tokens, not a full document.

5. **Provenance is built in, not added.** The audit requirement ("which page and
   region did this come from") means every extracted value carries a source
   reference. Design this into the schema from day one; retrofitting it means
   reprocessing 30M documents.

6. **The human queue is sized before you start.** Set the confidence threshold to
   produce a review volume your staffing can absorb, and monitor it. If review volume
   exceeds capacity, the backlog silently becomes "accepted without review," which is
   the worst outcome.

**Sequencing:** process the clean 41% first. It's cheap, fast, low-risk, and it
validates the whole pipeline end to end while producing business value early. Save
the difficult 7% for last, when your verification layer is mature and you have data
on what actually fails.

</details>

### The Interviewer Escalates

**Round 2 — "The smaller model fails on 8% of difficult documents. What now?"**

<details>
<summary>💡 Reveal</summary>

First: **8% of what, failing how?** The response depends entirely on the failure
mode, and asking is the correct first move:

- **Fails loudly** (returns malformed JSON, refuses, returns nulls): this is the good
  case. Detectable, so route those 8% to the stronger model. Cost impact: 8% of that
  class at frontier prices — compute it against the budget line.
- **Fails silently** (returns confident, well-formed, wrong values): this is the
  dangerous case, and it's what "fails on 8%" usually means when discovered via
  sampling rather than errors. Now you can't route on failure because you can't
  detect it.

For silent failure, options in order of preference:

1. **Lean on the verification layer.** If the wrong values fail a checksum, a date
   consistency check, or a database lookup, they become detectable — which is the
   whole argument for building verification independent of model choice.
2. **Two-model agreement** on the risky class: run the cheap model and the
   specialized model, compare structured outputs field by field. Disagreement →
   escalate. This costs ~2× on that class but converts silent failure into detectable
   failure, which is usually worth more than the model upgrade.
3. **Confidence calibration.** If the model exposes token-level log-probabilities or
   you fine-tuned a classifier head, calibrate a confidence score against a labeled
   set and set a threshold by the review-capacity math. Self-reported confidence from
   a chat model is unreliable; measured calibration is not.
4. **Just use the stronger model for that class** and re-do the budget. Sometimes
   correct, and worth stating plainly rather than engineering around.

Then: **is the 8% concentrated?** If those failures are all one document template, one
scanning vintage, or one regional office's forms, the fix is a targeted rule or a
class-specific route, not a global model upgrade. Always look at the failure
*distribution* before treating it as a rate.
</details>

**Round 3 — "Six weeks in, you've processed 9M documents. A downstream audit finds
that the claim-type classification is 91% accurate, not the 95% you promised. The
backfill is running. What do you do?"**

<details>
<summary>💡 Reveal</summary>

Three questions before any action:

1. **Is the 91% uniform or concentrated?** 91% overall might be 98% on the top 10
   claim types and 40% on the rare ones. A 62-class problem always has a long tail,
   and the tail is where errors concentrate. Per-class accuracy is the first thing to
   pull.
2. **What does a misclassification cost?** If it routes a claim to the wrong
   workflow, it's recoverable and annoying. If it affects reserve calculations, it's
   a financial restatement. This determines whether you stop the pipeline.
3. **Is the ground truth right?** Audits find disagreements, and sometimes the audit's
   labels are wrong or the taxonomy is ambiguous (62 classes almost guarantees
   overlapping definitions). Check inter-annotator agreement on the audit sample
   before accepting the number. If human annotators only agree with each other 93% of
   the time, 91% is near the ceiling and the problem is the taxonomy, not the model.

**Then decide about the 9M already processed.** Don't reprocess blindly:
- Reprocessing 9M documents costs real money and time. Instead, re-run *only the
  classification step* — it doesn't require re-extracting anything, so it's cheap.
- Better: keep the extracted data, re-classify with an improved approach, and compare.
  Only records whose class changes need downstream correction.
- Communicate scope precisely to the business: which records, which fields, what the
  correction path is. Understating scope to avoid a difficult conversation is how a
  quality problem becomes a trust problem.

**Whether to pause the backfill:** if extraction accuracy is fine and only
classification is off, don't pause — classification is separable and re-runnable.
Pausing a six-month backfill for a re-runnable step is an over-reaction. Say this
reasoning out loud; the ability to *not* stop everything is as important as the
willingness to.

**The fix itself:** for a 62-class problem at 91%, the highest-value interventions are
usually (a) merge or clarify ambiguous classes with the business, (b) fine-tune on
labeled examples — you now have millions of documents and, crucially, the audit gave
you a labeled set, and (c) hierarchical classification (coarse class first, then
fine-grained within it), which typically outperforms flat 62-way classification
substantially.
</details>

**Round 4 — "Halfway through, a new model is released that's 40% cheaper and
measurably better. Do you switch mid-backfill?"**

<details>
<summary>💡 Reveal</summary>

The tension: better and cheaper is obviously attractive, but **switching mid-backfill
means your 30M records were produced by two different systems**, which matters for
consistency, auditability, and any downstream analysis that assumes uniformity.

Considerations:

- **Is the output consistent?** If the new model classifies borderline cases
  differently, records processed before and after the switch differ systematically.
  For an insurance archive feeding actuarial analysis, a systematic discontinuity at
  document #15,000,000 is a data-quality landmine that will be discovered years later.
- **Can you afford to reprocess?** If the new model is 40% cheaper, reprocessing the
  first half may cost less than you'd think — and gives you a uniform corpus.
  Compute it.
- **What's the evaluation cost?** Switching requires re-validating against your
  labeled set, per document class. That's a week of work minimum, and it must include
  the verification layer's behavior (does the new model fail in different ways that
  your checks don't catch?).
- **Risk of switching mid-flight** vs. **cost of not switching**: if the backfill has
  3 months to run, 40% savings on 15M documents is material.

**A defensible answer:** finish the current class-by-class phase with the current
model, evaluate the new model properly in parallel, and switch at a clean boundary —
e.g. start the handwriting class with the new model — while recording the model
version on every record so the discontinuity is *documented rather than hidden.*
Reprocess earlier records later, if the improvement justifies it, funded by the
savings.

**The general principle, and the thing to say:** *every record carries the version of
every component that produced it.* Model version, prompt version, pipeline version.
This makes model changes tractable instead of contaminating. If that metadata isn't
in the schema, the answer to "should we switch?" is "we can't safely," and that's a
design failure, not a model decision.
</details>

### Trade-offs

**One model for everything.** Simple, uniform outputs, one evaluation, one thing to
upgrade. But it's either too expensive for the easy 41% or too weak for the hard 25%,
and at 30M documents that gap is millions of dollars or hundreds of thousands of
errors. *Choose when* the corpus is homogeneous — it isn't here.

**Route by document class (chosen).** Matches capability to difficulty; makes the
budget work. But requires a classifier (which can be wrong, and misclassification
propagates), multiple models to evaluate and maintain, and inconsistent failure modes
across classes. *Choose when* the input distribution is genuinely heterogeneous and
you can classify cheaply.

**Cascade everything (cheap → verify → escalate).** No classifier needed; routes on
evidence. But pays the cheap call on documents you know will fail (handwriting will
essentially always escalate), and adds latency. *Choose when* difficulty isn't
predictable from the input — here it partly is, so a hybrid (classify first, cascade
within class) is better than either alone.

**Human-in-the-loop for everything above a confidence threshold.** Highest accuracy;
directly tunable. But the threshold is a staffing decision, review capacity is the
binding constraint, and reviewer fatigue degrades quality at high volume.

### What Could Go Wrong?

- **The classifier is the single point of failure.** A misclassified handwriting
  document routed to the cheap path produces silent garbage. Mitigate: bias the
  classifier toward expensive routes on uncertainty (asymmetric cost), and sample-audit
  each route's output.
- **Verification checks pass on plausible-but-wrong values.** A hallucinated claim
  number that happens to satisfy the format and exists in the database (belonging to a
  different claim) passes every check. Add cross-field consistency: does the claim
  number's policy match the policy reference extracted separately?
- **Review queue overflows and becomes a rubber stamp.** Monitor per-reviewer
  throughput and agreement; a sudden throughput increase usually means quality
  collapse, not efficiency.
- **The 62-class taxonomy is ambiguous** and no model can hit 95% because humans
  can't. Measure inter-annotator agreement before setting the target.
- **Provenance retrofitting.** Discovering in month four that the audit requirement
  needs region-level provenance means reprocessing everything. Design it in.
- **Budget consumed by retries.** Failed extractions that retry silently can blow the
  $400k without anyone noticing. Track cost per *successfully accepted* document, not
  per attempt.

### Metrics

Field-level precision and recall **per field and per document class** (never
aggregate — the 99.2% requirement is per-field); verification check pass rates and,
critically, the rate at which verification catches a genuine error vs. produces a
false alarm; human review queue volume, latency, and reviewer agreement; cost per
accepted document by class, tracked against the budget envelope; classification
confusion matrix for the 62 classes; throughput (documents/day) against the 6-month
plan; and provenance completeness (must be 100%).

### Follow-up Questions

1. Your review queue is at 4% of documents, double what you staffed for. Give three
   options and their consequences.
2. How do you build the labeled set to measure 99.2% accuracy? How many documents do
   you need to label to distinguish 99.2% from 98.8%?
3. The ongoing 40k/day stream includes new document templates the backfill never saw.
   How do you detect and handle that?
4. Legal asks whether an extraction was made by a model or a human for a specific
   record from four months ago. Can you answer?
5. Handwriting extraction is at 94% and can't get better with available models. What
   do you propose?
6. How would you decide whether to fine-tune the specialized model on your corpus?

### What Separates Senior From Staff?

**Senior:** designs the routing, builds the verification layer, sizes the human queue,
hits the budget.

**Staff:** recognizes that **the verification layer, not the model choice, is the
product** — and that it's reusable across every document pipeline the company will
build. They insist on version metadata on every record before processing starts,
which is what makes the Round-4 question answerable at all. They question the 62-class
taxonomy with the business rather than accepting an unachievable target, and they size
the human review queue as a staffing commitment negotiated with operations, not as an
engineering parameter. They also sequence the backfill to produce business value and
learning early (clean documents first), so that a six-month project has feedback at
week three.

### Interviewer Notes

The best signal is whether the candidate builds a verification layer rather than
hunting for a model accurate enough. 99.2% field accuracy is not reachable by model
selection on degraded scans, and a candidate who keeps trying to solve it that way
hasn't worked on extraction. Second signal: do they do the per-class budget math? The
envelope calculation is what turns "which model" into an answerable question. Third:
in Round 3, do they check whether the audit's ground truth is trustworthy before
reacting? Fourth: in Round 4, do they raise version metadata? That's the answer that
shows they've lived through a model migration.

---

## 2. The Multilingual Support Product

### Situation

A travel booking platform offers customer support in 14 languages. Volume by language
is extremely skewed:

```
English      48%      German        6%      Portuguese   3%
Spanish      14%      Italian       4%      Dutch        2%
French        9%      Japanese      4%      Polish       2%
                      Korean        3%      Turkish      2%
                      Mandarin      3%      Arabic       1%
                                            Swedish      1%
```

Current approach: one frontier model, prompted in English, handles all languages.
Quality complaints are concentrated in Japanese, Korean, Turkish, Arabic, and Polish
— and in exactly those markets, CSAT is 12–18 points below the English baseline.

The head of international is escalating. Your eval suite is 92% English.

### Your Task

Design a model strategy for multilingual support.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Start with the measurement problem, not the model problem.** A 92% English eval
suite means you have no idea what your quality actually is in the languages that are
failing. Every model decision you make will be unfalsifiable. This is the first
deliverable and it should be stated before anything else.

Building it is non-trivial and is the real work: you need native-speaker-graded
examples per language, sampled from production traffic, covering the actual support
topics. That's a procurement and process problem (annotators, guidelines,
inter-annotator agreement) more than an engineering one.

**Then decompose "quality" per language.** Complaints in Japanese and Arabic may have
completely different causes:

- **Generation quality** — the model writes awkward or unnatural text. Common in
  lower-resource languages.
- **Comprehension** — the model misunderstands the customer's question.
- **Retrieval** — your knowledge base is in English, and cross-lingual retrieval is
  failing. *This is frequently the actual cause and is not a model-selection problem
  at all.*
- **Cultural/register mismatch** — the model is correct but the tone is wrong.
  Japanese and Korean support communication has formality expectations that a model
  prompted in English will systematically miss.
- **Tokenization economics** — many non-Latin scripts consume far more tokens per
  character, so those languages hit context limits earlier, get truncated more, and
  cost more per conversation. A silent, structural disadvantage.

Each cause has a different fix. Jumping to "use a different model for Japanese" without
knowing which of these is happening is guessing.

**The tokenization point is worth checking early** because it's mechanical and often
significant: if a Japanese conversation costs 2.5× the tokens of an equivalent English
one, then your context truncation is firing earlier, your retrieval context is
smaller, and your quality gap may be an artifact of a uniform token budget applied to
non-uniform languages.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Phase 1: build the measurement (weeks 1–6). Non-negotiable first step.**

- Sample 300–500 real conversations per language, stratified by support topic.
- Recruit native-speaker graders (internal support agents in those markets are ideal
  — they know the domain and the register).
- Grade on separate axes: **correctness**, **language quality**, **register/tone**,
  and **resolution**. A single score would hide exactly the distinction you need.
- Measure inter-annotator agreement; if graders don't agree, your rubric is bad.

Simultaneously, instrument the mechanical hypotheses:
- Tokens per conversation by language (check the tokenization inflation).
- Context truncation rate by language.
- Retrieval recall by language (do queries in Turkish find the right English
  documents?).

**Phase 2: fix the causes you find, in the order they matter.**

Typical findings and their fixes, roughly in the order they usually appear:

1. **Cross-lingual retrieval failure** (very common, and rarely the first hypothesis).
   Fix: use a multilingual embedding model; or translate the query to English before
   retrieval; or maintain translated versions of the top knowledge-base articles for
   high-volume languages. This is often the single biggest quality win and it doesn't
   involve changing the generation model at all.
2. **Token budget inequity.** Fix: set context budgets in characters or per-language
   token allowances rather than a global token number, so a Japanese conversation
   isn't structurally starved of context.
3. **Register/tone.** Fix: language-specific prompt sections written by native
   speakers, with concrete examples of appropriate formality. This is cheap and
   effective and does not require a different model.
4. **Generation quality.** Only now does model selection enter. Evaluate candidate
   models *per language* on your new eval set. Expect that different models lead in
   different languages, and that the differences are larger in lower-resource
   languages than in Spanish or French.

**Phase 3: the routing decision.**

Route by language only where the evaluation justifies it:

```
English, Spanish, French, German, Italian, Portuguese, Dutch, Swedish (83%)
    └─► current frontier model (evaluation shows no material gap)

Japanese, Korean, Mandarin (10%)
    └─► whichever model wins your per-language eval; possibly a regional
        provider with stronger performance in these languages

Turkish, Polish, Arabic (5%)
    └─► best-performing model on eval + language-specific prompt +
        higher human-escalation propensity while quality is unproven
```

**Design principles worth stating:**

- **Language is a known, reliable routing key.** Unlike difficulty, language is
  detectable with near-perfect accuracy and cheaply. This is one of the few cases
  where input-based routing is unambiguously correct.
- **Don't route to a model you haven't evaluated in that language.** The temptation is
  to assume a "multilingual model" is uniformly good. Model quality varies enormously
  by language and vendors' claims are not per-language guarantees.
- **Consider translation as an architecture**, and evaluate it honestly: translate the
  customer's message to English, process in English (where your knowledge base, your
  prompts, and your eval suite are strongest), translate the response back. This is
  unfashionable but sometimes measurably better for low-resource languages, and it
  concentrates your quality investment in one language. Its failure mode is losing
  nuance and register in translation — measure it rather than dismissing it.
- **Set per-language SLOs and report them.** A blended CSAT number lets a 1% language
  fail indefinitely.

**Phase 4: sustain it.** Per-language eval runs on every model or prompt change. New
languages don't launch until they have an eval set. The eval sets are refreshed from
production quarterly.

</details>

### Trade-offs

**One model, English prompts, all languages.** Simplest; one prompt to maintain; one
model to evaluate (badly). But quality varies unpredictably by language and you can't
see it. *Choose when* traffic is overwhelmingly one language and others are
negligible — and even then, measure before assuming.

**Per-language model routing.** Best quality per language; matches investment to
where the gaps are. But N models to evaluate, prompt, monitor, and upgrade; N eval
sets to maintain; a much bigger surface when a provider deprecates something.
*Choose when* the quality gaps are measured and material, as here for 5 languages.

**Translate-to-English pivot.** One high-quality pipeline; all knowledge and prompt
engineering concentrated; retrieval works against your English corpus natively.
But translation errors compound, register and idiom are lost, latency doubles, and
it can feel subtly foreign to native speakers. *Choose when* a language's direct
quality is poor and volume doesn't justify dedicated investment.

**Fine-tune per language.** Potentially excellent for high-volume languages with
enough data. But N models to maintain, and the low-volume languages — which are
exactly the ones with problems — have the least data. *Choose when* one or two
languages dominate and you have volume.

### What Could Go Wrong?

- **You "fix" it by switching models and quality doesn't improve**, because the cause
  was retrieval. Six weeks lost. This is why measurement precedes selection.
- **Native-speaker graders disagree** because your rubric doesn't account for regional
  variation (Latin American vs. European Spanish, Modern Standard vs. dialectal
  Arabic). Define the target variety explicitly.
- **Per-language prompts drift** and become 14 divergent forks. Structure them as a
  common base plus a small language-specific overlay, and evaluate the base changes
  across all languages.
- **A model upgrade improves English and regresses Korean.** Without per-language
  eval gating, you ship it. This is the failure mode that per-language SLOs exist to
  prevent.
- **Right-to-left rendering, date/number formats, name ordering.** Not model problems
  at all, but they show up in quality complaints and get misattributed. Check the
  presentation layer.
- **Low-volume languages never get attention** because they're 1% of traffic — while
  being 100% of the experience for those customers, and possibly strategically
  important markets. Set a floor SLO regardless of volume.

### Metrics

Per-language: correctness, language quality, register, and resolution rate (four
separate axes); CSAT; escalation-to-human rate; retrieval recall for that language's
queries; tokens per conversation and context truncation rate; cost per conversation.
Plus: inter-annotator agreement per language (a quality measure of your measurement);
eval coverage per language (share of production topics represented); and the gap
between each language's metrics and the English baseline, tracked over time — that
gap is the actual objective.

### Follow-up Questions

1. Your Japanese eval shows a different model wins by 6 points, but it's from a
   provider you haven't security-reviewed. What's your path?
2. Arabic is 1% of traffic. Justify (or don't) spending a month on it.
3. A model upgrade improves 12 languages and regresses 2. Ship it?
4. How do you evaluate register and formality, which native speakers themselves
   disagree about?
5. The company adds 6 new languages next quarter. What's your launch checklist?
6. Your translation-pivot approach scores better on correctness but worse on CSAT in
   Korean. Which do you believe?

### What Separates Senior From Staff?

**Senior:** builds the per-language eval, diagnoses the causes, designs the routing,
ships measurable improvements.

**Staff:** refuses to make model decisions before measurement exists, and defends that
sequencing to an escalating executive — which requires giving the head of
international a credible timeline and interim mitigations rather than just "we need to
measure first." They recognize that the eval-set construction is a *procurement and
process* problem (annotator recruitment, rubrics, agreement measurement) and staff it
accordingly. They establish per-language SLOs so that low-volume markets can't be
silently abandoned, and they make per-language eval a mandatory gate on every model
change company-wide — turning a one-time fix into a permanent property of the system.

### Interviewer Notes

The candidate should say some version of "I can't answer the model question until I
can measure per-language quality" in the first two minutes. Then watch whether they
enumerate causes beyond the model — retrieval and tokenization are the two that
distinguish real experience, and cross-lingual retrieval failure is the most common
actual root cause in production systems like this. Watch also for whether they treat
translation-pivot as a serious option or dismiss it; dismissing it reflexively
suggests they're following fashion rather than measurement. Finally, the "1% of
traffic" question tests whether they think about markets or only about aggregate
metrics.

---

## 3. Choosing a Model You Can't Change Later

### Situation

A medical-device company is embedding an LLM into a device that runs **on-premise in
hospitals**, air-gapped, updated at most **twice a year** through a validated release
process. The model assists technicians with troubleshooting: it reads device logs and
sensor data and suggests diagnostic steps.

Constraints that make this unlike a web service:

- The model runs on the device's hardware: **24 GB of GPU memory**, shared with other
  device functions.
- **No network.** No API calls, no telemetry back, no dynamic updates.
- Each release requires regulatory validation: 4–6 months and substantial cost.
- Deployed devices are in service **7–10 years**.
- Incorrect diagnostic guidance can lead to device downtime in clinical settings, or
  worse.

### Your Task

Choose a model and design the system around a choice you cannot revise for at least
six months and possibly years.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Every instinct from web-service AI engineering is wrong here.** No routing to a
better model at runtime. No hotfix for a bad prompt. No telemetry to detect
regressions. No A/B test. The system must be *correct by construction and degrade
safely*, because you cannot intervene.

That reframes model selection almost entirely around **predictability over peak
capability**:

- A model that is 88% accurate and fails in consistent, detectable ways is far better
  than one that is 93% accurate and fails unpredictably.
- **Determinism matters.** Greedy decoding, fixed seeds, pinned weights. A device that
  gives different answers to the same input on different days is a validation
  nightmare and a support nightmare.
- **The model must fit with headroom.** 24 GB shared with other functions means
  realistically ~14–18 GB for the model, which after quantization points at something
  in the 7–14B parameter class. Sizing this correctly is table stakes.
- **Refusal and abstention are features.** A model that says "I don't have enough
  information, escalate to support" is behaving correctly. Design and evaluate the
  abstention path as carefully as the answer path — arguably more carefully.

**The second reframe: minimize what the model is responsible for.** Since you can't
update the model, put as much knowledge as possible in *updatable* components:

- A retrieval corpus of service documentation that can be updated more easily than the
  model (if the validation process allows content updates separately from model
  updates — find out, because this distinction may be the most important thing in the
  whole design).
- Deterministic diagnostic logic — decision trees, threshold checks — for anything
  that can be expressed as a rule. The model handles interpretation and
  communication; rules handle diagnosis wherever possible.
- A structured output schema so downstream device software can validate and constrain
  what the model produces.

**Third: what's the failure mode budget?** In a clinical setting, a wrong diagnostic
step that causes a technician to swap the wrong component costs money and downtime. A
wrong step that causes the technician to bypass a safety interlock is catastrophic.
Enumerate the catastrophic paths and make them structurally unreachable — the model
should not be able to suggest actions outside an allowlisted set.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Model selection criteria, in priority order:**

1. **License and provenance.** Must permit commercial on-device redistribution.
   Must have a stable, archivable artifact — you need the exact weights available for
   7–10 years, which means you host them, not a hub. Check the license carefully;
   this eliminates options before capability does.
2. **Fits in budget with headroom.** Target ~10 GB after quantization to leave room
   for KV cache, other device functions, and growth in context length. Quantization
   must be validated — quantized models can degrade unevenly, and the degradation may
   concentrate in exactly the reasoning tasks you care about.
3. **Predictability and instruction-following** over raw capability. Measure variance
   across repeated runs, robustness to input perturbation (log format changes,
   truncated inputs, unusual sensor values), and adherence to output schema.
4. **Abstention behavior.** How does it behave when the answer isn't in the retrieved
   context? A model that confidently confabulates a diagnostic step is disqualifying,
   regardless of benchmark scores.
5. **Fine-tunability.** You will almost certainly want to fine-tune on your service
   documentation and historical support cases; the model must support that and you
   must be able to reproduce the training run years later.

**System design around the model:**

```
device logs + sensor data
        │
        ├─► deterministic rule engine  (updatable content package)
        │      known fault signatures → known procedures
        │      Handles the majority of cases with zero model involvement.
        │
        └─► [unmatched cases only]
                  │
                  ├─ retrieval over service documentation
                  │     (versioned content package, updatable
                  │      separately from the model if regulation allows)
                  │
                  ├─ LLM (pinned weights, greedy decoding, fixed prompt)
                  │     structured output: {diagnosis, confidence,
                  │     steps[], evidence_refs[], escalate: bool}
                  │
                  ├─ output validation:
                  │     · schema valid?
                  │     · every step in the allowlisted procedure set?
                  │     · every claim traceable to a retrieved document?
                  │     · confidence above threshold?
                  │
                  ├─ pass ──► present to technician, with citations
                  └─ fail ──► "escalate to support" with the raw evidence
```

**The critical design choices:**

- **Rules first, model second.** Most diagnostic cases are known fault signatures. A
  rule engine handles them deterministically, is trivially validatable, and can be
  updated as a content package. The model handles the residual — the novel and
  ambiguous cases where it adds real value. This dramatically shrinks the surface
  where the model can be wrong, and shrinks the validation burden.
- **The model never proposes an action outside an allowlist.** Diagnostic steps are
  drawn from a defined procedure library. The model selects and sequences; it does not
  invent. This makes the catastrophic failure modes structurally unreachable rather
  than prompt-discouraged.
- **Citations are mandatory.** Every suggestion references the service document that
  supports it. The technician can verify. This is both a safety feature and a trust
  feature, and it's how you handle the absence of telemetry — the human in the loop
  *is* your feedback mechanism.
- **Escalation is a first-class, well-tested output**, not an error path.
- **Everything is versioned and recorded locally**: model hash, prompt version,
  content package version, rule version. When a hospital reports a problem two years
  later, you must be able to reproduce the exact configuration.

**Validation strategy (this is most of the work):**

- A large, adversarial test suite built with service engineers: known faults, novel
  faults, degraded/corrupted logs, out-of-distribution sensor values, adversarial
  inputs.
- Determinism testing: identical input → identical output, across restarts and
  hardware variations.
- Abstention testing: cases where the correct answer is "escalate."
- Stability testing across the quantized model on the actual device hardware — not on
  a workstation.
- Documented performance characteristics for the regulatory submission.

**And plan the offline feedback loop:** since there's no telemetry, build a
technician-initiated feedback mechanism (a report exportable at service time) so you
accumulate real-world failure data for the *next* release. Without this you are blind
for six months at a time.

</details>

### Trade-offs

**Largest model that fits.** Best capability. But leaves no headroom, risks OOM as
other device functions grow, longer latency on constrained hardware, and more to
validate. *Choose when* the task genuinely requires capability and hardware is
comfortable.

**Smaller model + heavy retrieval + rules.** Predictable, fast, validatable, leaves
headroom, and puts knowledge in updatable content rather than frozen weights.
Weaker at novel reasoning. *Choose when* updates are expensive — as here. **This is
the right default for air-gapped systems.**

**Fine-tuned small model.** Excellent on your domain, small, fast. But you own the
training reproducibility for a decade, and it may be brittle to inputs unlike its
training data — which, over 7–10 years of device evolution, is guaranteed to happen.
*Choose when* the domain is stable and you have strong data.

**No model at all — rules only.** Fully deterministic, trivially validatable, cheapest
to certify. But no help with novel faults, which is the actual value proposition.
Worth stating as the baseline the model must beat: **if the model can't demonstrably
outperform rules on novel cases, don't ship it.** A candidate who names this
alternative is showing good judgment.

### What Could Go Wrong?

- **Quantization degrades reasoning unevenly**, and your workstation testing (full
  precision) doesn't catch it. Always validate on the deployed configuration.
- **The model's behavior drifts with input distribution** as the device is used in new
  contexts, new hospitals, new firmware versions — and you have no telemetry to notice.
  Mitigate with conservative abstention thresholds and the technician feedback channel.
- **Weights become unavailable.** A model is withdrawn, a license changes, a hub goes
  away. Archive everything: weights, tokenizer, config, training code, and the exact
  environment. Treat it like any other long-lived dependency in a regulated product.
- **KV cache OOM under long logs.** A technician pastes a 200k-token log dump. Enforce
  input limits at the boundary with a graceful message, not a crash.
- **The technician over-trusts the output.** Automation bias in a diagnostic context.
  Citations and explicit confidence help; so does designing the UI to present
  suggestions as hypotheses to verify rather than instructions to follow.
- **You cannot fix a bad prompt for six months.** Prompt bugs are as severe as code
  bugs here. Treat prompts as source code with review, testing, and version control.

### Metrics

Because there's no telemetry, most metrics are pre-release: accuracy on the validation
suite by fault class; abstention precision and recall (how often it correctly says "I
don't know"); output-schema conformance rate (must be ~100%); determinism (identical
input → identical output, measured across runs and restarts); latency and memory on
target hardware including worst-case inputs; and the model's performance *relative to
the rules-only baseline* on novel faults.

Post-release, via technician feedback exports: suggestion-acceptance rate, escalation
rate, and reported incorrect suggestions by fault class — the input to the next
release.

### Follow-up Questions

1. Six months after shipping, hospitals report the assistant is unhelpful for a new
   device revision. You can't update for another six months. What can you do?
2. How do you decide the abstention threshold with no production data?
3. A better model is released two months before your validation completes. Do you
   restart?
4. How do you handle a hospital that runs the device in a language you didn't
   validate?
5. What's your plan for the model in year 7, when the ecosystem has moved on entirely?
6. How would you convince regulators that a non-deterministic component is safe?
   (Note: is it non-deterministic, in your design?)

### What Separates Senior From Staff?

**Senior:** picks an appropriate model, designs the rules-plus-retrieval architecture,
builds a thorough validation suite.

**Staff:** identifies the *separability question* — can content and rules be updated
without a full model re-validation? — as the single highest-leverage thing to
establish with the regulatory team, because it determines whether the product can
improve at all between releases. They design the system so the frozen component
(weights) carries as little responsibility as possible and the updatable components
carry as much as possible. They plan the 10-year artifact archival and reproducibility
story, which nobody asks for and everybody needs. And they set up the offline feedback
channel so that six months of blindness produces data rather than nothing.

### Interviewer Notes

This scenario tests whether a candidate can drop web-service assumptions. Watch for
whether they immediately note that routing, hotfixes, A/B tests, and telemetry are all
unavailable — that inventory of missing tools should come fast. The strongest signal
is the rules-first architecture: minimizing the model's responsibility because the
model is the part you can't change. Second-strongest is raising the separability
question about content vs. model updates. Candidates who spend the interview comparing
model benchmarks are answering a question that barely matters here.

---

## 4. Embedding Model Selection and the Migration You'll Regret

### Situation

You're building retrieval for a legal research product. 60M documents (case law,
statutes, filings, secondary sources), ~400M chunks after chunking.

You must choose an embedding model. The decision feels small — it's one config line —
but:

- Re-embedding 400M chunks costs real money and weeks of compute.
- The index structure, storage costs, and query latency all depend on the dimension.
- Retrieval quality is the product; lawyers will not tolerate missing a relevant case.

Candidates:

- **A:** a large hosted embedding API, 3,072 dimensions, strong benchmark scores,
  $0.13/M tokens.
- **B:** a mid-size hosted API, 1,536 dimensions, slightly lower benchmarks,
  $0.02/M tokens.
- **C:** an open-weights model, 768 dimensions, self-hostable, strong on retrieval
  benchmarks, free to run on your own GPUs.
- **D:** a domain-specific legal embedding model, 768 dimensions, trained on case law,
  limited public evaluation.

### Your Task

Choose, and design so the choice can be revisited.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**First: public benchmarks are close to useless for this decision.** General retrieval
benchmarks measure performance on web-ish, general-domain queries. Legal retrieval has
different properties: long documents, dense citation structure, precise terminology
where a near-synonym is *wrong* ("negligence" vs. "gross negligence"), and queries
that are often citations or doctrinal phrases rather than natural questions.

**You must build a domain evaluation set.** For legal research: a few hundred queries
from real researchers, with relevant documents identified by lawyers. This is
expensive and it is the only thing that will actually answer the question. Say this
first.

**Second: compute the storage and cost implications of dimension**, because they're
substantial and often decisive:

```
400M chunks × dimensions × 4 bytes (float32)
   3,072-dim →  4,915 GB  ≈ 4.9 TB
   1,536-dim →  2,458 GB  ≈ 2.5 TB
     768-dim →  1,229 GB  ≈ 1.2 TB
```

That's raw vectors, before index overhead (HNSW adds meaningfully on top). The 3,072-
dim option requires 4× the memory of the 768-dim one. At scale this is the difference
between a manageable cluster and a very expensive one, and it also affects query
latency (distance computation is linear in dimension).

Quantization changes this picture — scalar quantization to int8 cuts it 4×, product
quantization far more — but quantization has a recall cost that must be measured, and
it interacts with dimension.

**Third: cost of the embedding pass itself.** 400M chunks × ~400 tokens = 160B tokens.
- Option A: 160,000 × $0.13 ≈ **$20,800**
- Option B: 160,000 × $0.02 ≈ **$3,200**
- Option C: self-hosted; GPU-hours, probably $2–6k, plus engineering.

None of these is prohibitive for the initial build — which means **the embedding pass
cost is not the deciding factor; storage, latency, and quality are.** Notice this
explicitly, because many candidates anchor on the API price.

**Fourth, and most important: this is a high-lock-in decision, so design for
migration from the start.** The regret comes not from choosing wrong but from choosing
in a way that makes changing impossible.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Process, not a snap answer:**

**Step 1 — Build the legal evaluation set (weeks 1–4).** 300–500 queries from actual
research sessions, with relevance judgments by attorneys, graded (not binary — legal
relevance is a spectrum). Include the hard cases: doctrinal queries, citation lookups,
fact-pattern similarity, and negative queries where the right answer is "nothing
relevant."

**Step 2 — Evaluate all four on a representative subset (weeks 3–6).** Embed 2M chunks
(a stratified sample) with each model and measure recall@10, recall@50, nDCG@10, and —
importantly — performance on the *hard* query subsets separately. Also measure with a
reranker in the pipeline, because that's the real configuration: a weaker embedding
model that retrieves 100 candidates for a strong cross-encoder to rerank may match a
stronger embedding model's top-10 at a quarter of the storage.

**That last point is the key architectural insight.** Bi-encoder retrieval quality and
end-to-end retrieval quality are not the same thing. If the reranker does the precision
work, the embedding model's job is *recall at k=100*, which is a much easier bar and
much less sensitive to model choice. This can flip the decision entirely toward the
cheaper, smaller model.

**Step 3 — Weigh the domain model (D) carefully.** A legal-domain embedding model is
attractive and risky: it may be excellent on case law and poor on the statutes and
secondary sources also in your corpus; "limited public evaluation" means you're the
evaluator; and a small vendor or research artifact may not be maintained for the
lifetime of your product. Evaluate it on the same footing, and weight maintenance risk
explicitly.

**Step 4 — The likely answer and why.** In most real evaluations of this shape,
**option C (768-dim open-weights) with a strong reranker wins on total system value**:
- 1.2 TB instead of 4.9 TB — a 4× infrastructure saving that compounds forever
- lower query latency (shorter distance computations, smaller index)
- self-hosted, so no per-token embedding cost on 400M chunks or on every query
- no vendor dependency for a component that is extremely painful to migrate
- with a reranker, end-to-end quality is typically within noise of the largest option

But **do not assert this without the evaluation.** The answer for a legal corpus might
genuinely be the domain model, or might be that 3,072 dimensions really does capture
distinctions that matter in doctrinal search. The point of the process is that you'll
know.

**Step 5 — Design for migration, which is the actual deliverable:**

1. **Version every vector.** Store `embedding_model_id` and `embedding_version` with
   every chunk. Never allow a query embedded with model X to search an index built
   with model Y — enforce this in code, because the failure is silent and catastrophic
   (see the rapid-fire incident in `production-incidents.md`).
2. **Design for parallel indexes.** The migration pattern is: build the new index
   alongside the old, dual-read and compare on live traffic, cut over, then retire. If
   your storage and infrastructure can't hold two indexes simultaneously, you can never
   migrate without downtime. **Budget for 2× storage capacity as a permanent
   architectural property.**
3. **Keep the chunking pipeline separate from the embedding step.** Re-embedding
   should not require re-chunking, or your migration cost doubles and the comparison
   is confounded.
4. **Store the source text with the vector** so re-embedding doesn't require
   re-fetching and re-parsing 60M documents.
5. **Write down the trigger conditions** for revisiting: a new model beats the current
   one by >5 points nDCG on your eval set; storage cost crosses a threshold; the
   vendor deprecates.

**Estimate the migration cost now**, and put it in the design doc: re-embedding 400M
chunks ≈ 2–4 weeks of compute and $3–20k, plus dual-index storage for the duration,
plus evaluation. Knowing this number is what allows the company to make a rational
decision later instead of an emotional one.

</details>

### Trade-offs

**Large hosted embedding model (3,072-dim).**
*Advantages:* strongest raw retrieval quality; no infrastructure; automatic
improvements.
*Disadvantages:* 4× storage and memory; higher query latency; per-token cost on every
query forever; vendor dependency on the most migration-hostile component in your
stack; dimension is a permanent commitment.

**Mid-size hosted (1,536-dim).** A reasonable middle. Same vendor-dependency concerns,
better economics. Often the pragmatic choice when self-hosting isn't feasible.

**Open-weights self-hosted (768-dim).**
*Advantages:* lowest storage and latency; no per-query cost; full control and
reproducibility; can be fine-tuned on your domain later; weights archivable.
*Disadvantages:* you operate an embedding service (batch throughput for indexing, low
latency for queries); no automatic improvements; quality may lag on hard queries
without a reranker.

**Domain-specific model.**
*Advantages:* potentially much better on in-domain retrieval; small.
*Disadvantages:* unknown behavior outside its training domain; maintenance and
longevity risk; limited community evaluation means you carry all the validation
burden.

### What Could Go Wrong?

- **Query/index model mismatch** after an upgrade — silent, plausible-looking, totally
  broken results. The most common embedding incident there is. Enforce version
  matching structurally.
- **Benchmark-driven choice that doesn't hold in your domain.** Legal terminology
  precision is not what general benchmarks measure.
- **Chunking and embedding coupled**, so a model change forces a full re-chunk and the
  comparison isn't apples to apples.
- **No headroom for a parallel index**, so migration requires downtime and gets
  deferred indefinitely.
- **Quantization applied without measuring recall loss** on hard queries. It's usually
  fine on average and sometimes bad exactly where it matters.
- **The domain model's vendor disappears** and you have 400M vectors from an
  unreproducible artifact.
- **Nobody re-evaluates for two years** because the decision felt settled. Trigger
  conditions with a review date prevent this.

### Metrics

Recall@10/@50/@100 and nDCG@10 on the domain eval set, **broken out by query type**
(citation lookup, doctrinal, fact-pattern, negative); end-to-end quality with the
reranker in place (the number that actually matters); query latency vs. dimension;
storage and memory footprint; embedding throughput for the initial pass and for
incremental updates; index freshness lag; and the recall cost of any quantization.

### Follow-up Questions

1. Your eval shows the 768-dim model at recall@10 of 0.71 and the 3,072-dim at 0.78 —
   but with a reranker they're 0.86 and 0.87. Which do you choose, and what else do
   you need to know?
2. Design the zero-downtime migration for 400M vectors. What's the rollback?
3. How would you fine-tune an embedding model on legal data? What data would you need
   and how would you avoid overfitting to your eval set?
4. Chunking strategy interacts with embedding model choice. How?
5. Two years in, a model appears that's 15 points better. Walk through the decision.
6. How do you handle documents in the corpus that are 400 pages long?

### What Separates Senior From Staff?

**Senior:** builds the domain eval set, evaluates properly with the reranker in the
loop, does the storage math, picks well.

**Staff:** treats the migration path as the primary deliverable. They ensure the
architecture holds two indexes, that vectors are versioned, that chunking and
embedding are decoupled, and that source text is stored with vectors — so that the
inevitable future migration is a project rather than a crisis. They also put the
estimated migration cost in the original design doc, which is what lets the
organization decide rationally in year two. And they push back on benchmark-driven
selection with an argument the business understands: *"public benchmarks measure a
different corpus and different queries than the ones our customers pay for."*

### Interviewer Notes

Two signals dominate. First: does the candidate insist on a domain-specific eval set
rather than citing benchmark leaderboards? Second: do they evaluate the embedding
model *with the reranker in the pipeline*, recognizing that the bi-encoder's job is
recall not precision? That second point is the most common gap even among experienced
candidates and it changes the answer. Also watch for the storage arithmetic — 4.9 TB
vs. 1.2 TB is the kind of number that should be computed in the first five minutes.
Finally, ask how they'd migrate; a candidate whose design can't hold two indexes has
made an irreversible decision without noticing.

---

## 5. The Model You Chose Is Being Deprecated

### Situation

Your provider announces that the model powering your primary product will be
**retired in 90 days**. You have:

- 14 distinct prompts tuned against this model over 20 months, including several with
  carefully-calibrated few-shot examples.
- An eval suite with 18 months of historical scores against this model.
- A cascade whose escalation thresholds were tuned to this model's behavior.
- Three customers with contractual quality commitments.
- A recommended successor model from the provider, plus two competitors' offerings.

### Your Task

Run the migration.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

90 days is enough time if you start immediately and don't waste weeks deciding. The
work splits into four tracks that mostly run in parallel:

1. **Evaluate candidates** (weeks 1–4)
2. **Adapt prompts** (weeks 2–8)
3. **Recalibrate the system around the new model** (weeks 4–10) — the part people
   forget
4. **Roll out with rollback** (weeks 8–12)

**The insight that most candidates miss: it's not just the prompts.** A model swap
invalidates every threshold and heuristic that was tuned against the old model's
behavior:

- cascade escalation thresholds
- confidence cutoffs
- output length assumptions (and therefore `max_tokens`, context budgets, cost models)
- retry logic tuned to specific error patterns
- structured-output parsing that tolerated the old model's quirks
- caching (different model = different cache namespace; hit rate resets)
- latency assumptions in timeouts and SLOs
- the eval suite's baseline — 18 months of history is about to become
  non-comparable

Enumerate this list early. It's the difference between a migration that works and one
that ships and then produces three weeks of mysterious incidents.

**Second insight: don't treat "the provider's recommended successor" as the default.**
A forced migration is a rare opportunity to re-evaluate the whole choice — including
competitors — at a moment when the switching cost is already being paid. Evaluate at
least three candidates. But also don't let the evaluation expand into a six-week
research project; timebox it.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Week 1: inventory and freeze.**

- Enumerate every place the model is referenced: prompts, thresholds, parsers,
  timeouts, cost models, cache keys, eval baselines, documentation, and customer
  commitments.
- Freeze non-essential changes to the AI pipeline for the duration. Migrating a moving
  target is how you end up unable to attribute regressions.
- Notify the three contracted customers proactively with a plan. Doing this in week 1
  rather than week 11 is the difference between a partnership conversation and an
  escalation.

**Weeks 1–4: evaluate candidates.**

- Run the existing eval suite against the successor and two competitors, **with the
  current prompts unchanged**, to establish a baseline. Expect regressions; the
  prompts are tuned to the old model.
- Then run with lightly-adapted prompts to see the achievable ceiling.
- Evaluate on the dimensions that matter, not just accuracy: structured-output
  reliability, instruction-following, refusal behavior, latency, cost per request with
  *your* token profile (output verbosity varies a lot between models and directly hits
  cost and latency), and behavior on your hardest segments.
- Include a "do nothing different" cost baseline so the business can see the trade.

**Weeks 2–8: adapt prompts.**

- Prompts are not portable. Few-shot examples calibrated to one model's tendencies can
  actively hurt on another. Be prepared to rewrite rather than tweak.
- Work prompt by prompt with eval gating. Fourteen prompts is a lot; prioritize by
  traffic share and risk.
- **Keep the old prompts intact and versioned.** You may need to run both.
- Watch for the specific traps: output verbosity changes (breaking length assumptions
  and cost models), formatting differences (breaking parsers), refusal-threshold
  differences (breaking user experience in unexpected segments), and different
  tool-calling conventions.

**Weeks 4–10: recalibrate the surrounding system.**

This is the track that separates a competent migration from a painful one:

- **Re-tune cascade thresholds** against the new model's confidence and error
  distribution. The old thresholds are meaningless.
- **Re-measure token profiles** and update cost models, `max_tokens`, and context
  budgets.
- **Re-validate parsers** against the new model's formatting.
- **Re-namespace caches** by model version (they should already be — if they aren't,
  that's a bug you're about to discover).
- **Re-baseline the eval suite.** Decide explicitly how to handle 18 months of
  history: keep the series but annotate the discontinuity, and re-run a subset of
  historical configurations against the new model to establish a translation factor if
  you need continuity for reporting.

**Weeks 8–12: roll out.**

- Shadow mode first: run both models on live traffic, use the old one's output, log
  and compare. Full-scale comparison at zero customer risk. This is the highest-value
  de-risking step available and it should run for at least a week.
- Then canary: 1% → 5% → 25% → 50% → 100%, with automated rollback on quality, cost,
  or latency regression, and a hold period at each step long enough to see the metrics
  that matter (some quality signals take days).
- Keep the old model available until the deprecation date, and keep the switch a
  config flag.
- For the three contracted customers, consider holding them at a later canary stage
  and communicating the schedule, so their exposure is deliberate.

**Build the muscle, not just the migration.** The deliverable at the end should
include: a documented migration runbook, model version as a config value everywhere
(not a constant in code), a portable prompt structure that separates model-specific
tuning from task definition, and — most importantly — **a standing secondary model
kept warm with continuous traffic**, so the next deprecation is a config change rather
than a project. This will happen again, roughly annually.

</details>

### Trade-offs

**Take the provider's recommended successor.** Fastest, best-supported migration path;
provider tooling and documentation assume it; likely most similar behavior. But it
locks in the same vendor at a moment when switching costs are already sunk, and it
forgoes the evaluation you'll otherwise not do for another two years.

**Switch providers entirely.** Diversifies vendor risk; possibly better or cheaper;
you're paying the migration cost anyway. But you're adding new-provider risks
(different rate limits, different reliability, security review, contracts) to an
already time-boxed migration, and 90 days is not much slack.

**Move to self-hosted open-weights.** Eliminates deprecation risk permanently — nobody
can retire a model you host. But it's a much bigger project than 90 days allows unless
you already have the capability, and it trades a recurring migration cost for a
permanent operational one.

**Split: successor for most traffic, evaluate alternatives for the next cycle.**
Pragmatic. Meets the deadline with the lowest-risk path while starting the diversification
work properly. Often the right answer, and worth proposing explicitly rather than
treating the choice as binary.

### What Could Go Wrong?

- **You migrate the prompts and forget the thresholds.** Cascade escalation rate
  triples or collapses; cost or quality moves and nobody knows why. This is the single
  most common migration failure.
- **Structured-output parsing breaks in a rare case** that wasn't in the eval set.
  Ship with a validation-and-repair path and monitor parse failure rate closely during
  canary.
- **Eval history becomes meaningless** and the team loses its ability to reason about
  trend. Annotate the discontinuity explicitly rather than pretending it didn't happen.
- **The 90 days slips** because the evaluation phase expands. Timebox it hard; a
  good-enough decision in week 4 beats a perfect one in week 9.
- **The contracted customers find out from the provider's blog** rather than from you.
- **Rollback isn't actually possible** because prompts, thresholds, and caches all
  changed together. Keep every change independently revertible and version them
  together as a coherent "configuration bundle."

### Metrics

Per-prompt eval scores on old vs. new (segmented); shadow-mode agreement rate and,
where they disagree, which is better (human-graded sample); cost per request under the
new model with real traffic; latency distribution; structured-output parse failure
rate; cascade escalation rate before and after recalibration; canary-stage quality
deltas; and, for the contracted customers specifically, their own quality metrics
tracked separately.

### Follow-up Questions

1. Two of your 14 prompts can't reach parity on any candidate model. What now?
2. The successor is 30% more expensive per token *and* more verbose. Model the total
   cost impact and your options.
3. Shadow mode shows 94% agreement. Is that good? What do you do with the 6%?
4. A contracted customer refuses to accept a model change without their own
   validation. How do you handle it?
5. How do you make the *next* deprecation cheaper? Be specific about what you'd build.
6. The provider extends the deadline by 6 months two weeks before cutover. Does
   anything change?

### What Separates Senior From Staff?

**Senior:** runs the migration competently — evaluates, adapts prompts, recalibrates,
canaries, ships on time.

**Staff:** treats the migration as a forcing function to fix the underlying fragility.
They ensure that afterwards, model identity is a configuration value everywhere,
prompts separate task definition from model-specific tuning, a secondary model runs
continuously on a slice of traffic, and a written runbook exists. They also handle the
customer commitments as a relationship rather than a risk to be managed quietly —
proactive communication in week 1 converts a liability into evidence of competence.
And they name the recurring cost honestly to leadership: model deprecation is now an
annual event, and the company should budget for it rather than treating each one as a
surprise.

### Interviewer Notes

The discriminating question is what breaks besides the prompts. A candidate who
enumerates thresholds, parsers, cost models, caches, and eval baselines has done this;
a candidate who only talks about prompt engineering has not. Second signal: shadow
mode. It's the standard technique and its absence is telling. Third: do they evaluate
alternatives, or default to the provider's recommendation? Neither answer is wrong but
the reasoning matters. Finally, listen for whether they propose building the capability
to do this cheaply next time — deprecations recur, and a candidate who treats this as a
one-off is missing the pattern.

---

## Model Selection Cheat Sheet

**Questions to ask before naming any model:**

1. What is the *distribution* of inputs? (Not the average — the shape.)
2. What is the cost of being wrong, and is it symmetric?
3. Is the output verifiable by cheaper means (schema, checksum, lookup, arithmetic)?
4. What's the token profile — input-heavy or output-heavy? It determines both cost
   and latency behavior.
5. How will you know if you're wrong? What's the eval set, and does it match
   production?
6. How reversible is this choice? What's the migration cost if you're wrong?

**Patterns that recur:**

| Situation | Usual answer |
|---|---|
| Heterogeneous input difficulty | Classify, then route |
| Difficulty unpredictable from input | Cascade: cheap → verify → escalate |
| High accuracy required on checkable fields | Verification layer, not a bigger model |
| Narrow task, lots of labels, high volume | Fine-tune a small model, keep a fallback |
| Language/locale variation | Route by language; evaluate per language |
| Can't update after deploy | Minimize model responsibility; rules + retrieval |
| Retrieval quality | Evaluate the embedding model *with* the reranker |
| Any model choice | Version everything so it can be changed |

**Things that are almost always wrong:**

- Choosing on public benchmark scores for a domain-specific task.
- Treating "quality" as a scalar across a heterogeneous traffic mix.
- Assuming a bigger model fixes a retrieval problem.
- Making an embedding-model choice without planning the migration.
- Tuning thresholds against one model and keeping them after switching.
- Choosing a model before you can measure whether the choice was right.
