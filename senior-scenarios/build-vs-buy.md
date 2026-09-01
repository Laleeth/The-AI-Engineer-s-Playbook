# Build vs. Buy

The wrong answer is almost always "build," and the second-wrongest is "buy" as a
reflex. The interview signal here is whether a candidate reasons about **total cost
of ownership over a realistic time horizon**, including the costs that never appear
in a proposal: the engineer who maintains it for three years, the on-call rotation,
the migration you'll do in year two, and the opportunity cost of the features you
didn't build.

A framing that works for every scenario in this file:

> **Is this thing a source of competitive advantage, or is it plumbing?**
> Build advantage. Buy plumbing. And be honest about which one you're looking at —
> almost everything feels like advantage while you're building it.

Then apply the second test:

> **What does it cost to reverse this decision in 18 months?**
> Cheap-to-reverse decisions should be made fast and moved on from. Expensive-to-
> reverse decisions deserve the meeting.

**Contents**

1. [Vector Search: Managed, Postgres, or Roll Your Own?](#1-vector-search-managed-postgres-or-roll-your-own)
2. [Self-Hosted Inference vs. API Providers](#2-self-hosted-inference-vs-api-providers)
3. [The Evaluation Platform](#3-the-evaluation-platform)
4. [Agent Orchestration: Framework or Not?](#4-agent-orchestration-framework-or-not)
5. [Observability for AI Systems](#5-observability-for-ai-systems)

---

## 1. Vector Search: Managed, Postgres, or Roll Your Own?

### Situation

You are the first AI engineer at a 60-person B2B analytics company. You're building
semantic search and RAG over customer data: each customer's documents, dashboards,
saved queries, and internal wiki.

Facts:

- **Multi-tenant.** 340 customers today, expected 900 in 18 months.
- Corpus per customer varies wildly: p50 is 4,000 documents, p95 is 180,000, the
  largest is 2.1M. Total today ~14M documents → ~48M chunks.
- Query volume: ~30 QPS average, 120 QPS peak, growing with the customer count.
- **Hard requirement:** strict tenant isolation. A leak is contract-terminating.
- **Hard requirement:** documents update constantly — customers edit dashboards and
  wiki pages all day. Freshness expectation is minutes, not hours.
- Existing stack: Postgres (well-operated, a DBA on staff), Kubernetes, AWS.
- Team: you, plus two backend engineers who have never worked on search.

Three options are being debated in a design doc that has 40 comments:

- **A.** Managed vector database (a hosted vector search service).
- **B.** `pgvector` in the existing Postgres.
- **C.** Build on an open-source ANN library (HNSW/IVF) with your own service.

### Your Task

Choose and defend. Then defend it again when the constraints change.

### Constraints

- Budget: this is a feature, not a product line. Infrastructure spend should stay in
  the low tens of thousands per month.
- Two quarters to GA.
- No dedicated infrastructure team. Your DBA is not signing up for a new datastore.
- The company sells to enterprises with security questionnaires; a new subprocessor
  requires legal review and a customer notification cycle (~6 weeks).

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Step 1 — Establish what's actually hard here.** It isn't ANN search; that's a
solved problem with good libraries. The hard parts are:

- **Multi-tenancy at 900 tenants with a 500× size range.** How do you isolate
  without provisioning 900 indexes, and without letting the 2.1M-document tenant
  degrade the 4,000-document tenant?
- **Incremental updates with minutes-level freshness.** Many ANN index structures
  handle inserts poorly and deletes worse. This is the requirement that eliminates
  the most options and the one candidates most often overlook.
- **Operations.** Index rebuilds, memory pressure, backup/restore, capacity
  planning, and what happens at 3am.

Sort the options against *those* requirements rather than against benchmark QPS
numbers.

**Step 2 — Compute the scale honestly.** 48M chunks × 1,024-dim float32 vectors =
48M × 4KB ≈ **196 GB** of raw vectors. That's the number that decides everything.
It doesn't fit in one commodity machine's RAM comfortably, and HNSW's graph
structure adds overhead on top. Options: smaller embedding dimensions (768 or 512),
quantization (product/scalar quantization cuts this 4–32×), or sharding. Any
candidate who doesn't do this arithmetic is choosing a datastore by vibes.

**Step 3 — Note the tenant size distribution.** p50 4,000 docs, max 2.1M. **A
single global index with tenant filtering behaves badly here**: filtered ANN search
degrades when the filter is highly selective (searching a 4,000-doc tenant inside a
48M-vector index means the ANN traversal discards almost everything it finds, and
recall collapses or latency explodes). **Per-tenant indexes** behave badly too — 900
indexes with wildly varying sizes is an operational and memory nightmare for the
small ones.

The real answer is usually **tiered**: small tenants share a partitioned index (or
even brute-force search, which is genuinely fine at 4,000 vectors — a flat scan of
4,000 × 1,024 floats is a few milliseconds), and large tenants get dedicated
indexes. Recognizing that *brute force is correct for the median tenant* is a strong
signal; a huge fraction of "vector database" problems are solved by noticing the
data is small.

**Step 4 — Now apply build vs. buy.** Is vector search a competitive advantage for
an analytics company? No. It's plumbing. That's a strong prior toward buying, and
the prior should only be overturned by a specific requirement that buying can't
meet.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Recommendation: B (pgvector) with a tiered strategy, and a documented trigger for
moving to A.**

**Why pgvector wins here specifically:**

1. **Transactional consistency with the source of truth.** Documents live in
   Postgres. Embeddings in the same database means a document update and its
   embedding update are one transaction. With an external vector store you own a
   sync pipeline — CDC, a queue, a reconciler, and an inconsistency dashboard —
   forever. Given a minutes-level freshness requirement and constant edits, **this
   sync pipeline is the largest hidden cost of option A**, and it's the argument
   most candidates miss.
2. **Tenant isolation reuses machinery you already have.** Row-level security,
   existing tenant scoping, existing audit logging, existing backup and restore.
   With a new datastore you re-implement and re-verify all of it, and the security
   questionnaire answers change.
3. **No new subprocessor.** Six weeks of legal review and customer notification
   avoided, and no new vendor in the enterprise security review.
4. **Your team can operate it.** You have a DBA. You do not have a search engineer.
   This is the decisive practical constraint and it should be stated plainly.
5. **The median tenant is tiny.** For 4,000–50,000 chunks, an IVFFlat or HNSW index
   in Postgres — or even a sequential scan with a tenant filter — is comfortably
   fast. You're not fighting the tool.

**The tiered design:**

```
tenant size          storage                       search method
──────────────────────────────────────────────────────────────────────────
< 50k chunks         shared partitioned table      HNSW index, tenant-filtered
(≈ 85% of tenants)   partitioned BY tenant hash    within partition

50k – 500k chunks    dedicated partition           HNSW per partition

> 500k chunks        dedicated table + possibly    HNSW, tuned params,
(≈ 5 tenants)        a dedicated read replica      isolated resources
```

Partitioning by tenant is what makes filtered search work: the filter becomes
partition pruning (exact, cheap) rather than post-filtering an ANN traversal
(lossy, slow). **This is the technical crux of multi-tenant vector search and worth
saying out loud.**

**Manage the memory problem:** 196 GB of raw vectors is too much. Reduce
dimensionality (a 768-dim model at equal quality is a 25% cut) and use scalar
quantization for the large tenants. Measure recall impact per tenant tier — don't
assume it's free.

**Write down the trigger for revisiting.** This is the part that makes it a good
decision rather than a lucky one:

> Move to a dedicated vector store when *any* of: (a) p95 search latency exceeds
> 150ms at the 90th-percentile tenant size, (b) index maintenance causes more than
> one incident per quarter, (c) total vector storage exceeds ~500 GB, (d) we need
> capabilities Postgres lacks (native hybrid scoring, sub-second index updates at
> scale, multi-vector search).

Now the decision is reversible-by-design, and the next engineer inherits the
reasoning rather than the conclusion.

**Why not C (build on an ANN library):** you would be building a distributed,
persistent, multi-tenant, incrementally-updatable index service — that is a
multi-year product, and companies exist that do only this. Three engineers with no
search background building it in two quarters produces something that works in the
demo and pages you every week. Reject firmly.

**Why not A (managed) *yet*:** it's a defensible choice and would be right at
larger scale or with a bigger team. What kills it today is the sync pipeline
(against a minutes-freshness requirement), the six-week legal cycle, and adding a
second datastore for a team of three. Say this — the candidate who can articulate
why the rejected option is *reasonable* is showing better judgment than one who
argues it's bad.

</details>

### Trade-offs

**Managed vector DB.**
*Advantages:* purpose-built performance, handles scale and index maintenance,
built-in hybrid search and metadata filtering, someone else is on call for it,
scales past what Postgres will comfortably do.
*Disadvantages:* a second source of truth and the sync pipeline that implies;
per-vector pricing that can grow surprisingly; a new subprocessor with security and
legal overhead; vendor lock-in via a proprietary query interface; your data lives
somewhere your documents don't.
*Choose when:* scale exceeds what your primary database can hold; you need advanced
retrieval features; you have the team to own an additional datastore; freshness
requirements are relaxed enough that async sync is fine.

**pgvector / in-database vectors.**
*Advantages:* one source of truth, transactional updates, existing ops and security
model, no new vendor, no sync pipeline, easy backup/restore, familiar debugging.
*Disadvantages:* index build times get painful at scale; memory pressure competes
with your OLTP workload (consider a read replica); fewer retrieval features; HNSW
index maintenance in Postgres is less mature than in dedicated systems; a very large
tenant can hurt the shared cluster.
*Choose when:* the data already lives in Postgres, scale is in the tens of millions
of vectors, the team is small, and freshness matters.

**Build your own.**
*Advantages:* total control, no vendor, can be optimized for your exact access
pattern, no per-vector pricing.
*Disadvantages:* essentially all of them. Persistence, replication, crash recovery,
incremental updates, deletes, compaction, memory management, sharding, rebalancing —
each is a subsystem, and getting any one wrong is a data-loss or outage class.
*Choose when:* vector search *is* your product, or you have a genuinely unusual
access pattern that no existing system serves and the scale justifies a team.

### What Could Go Wrong?

- **Filtered ANN recall collapse.** Searching a small tenant inside a large shared
  index returns poor results because the ANN graph traversal spends its budget on
  other tenants' vectors. Symptoms: recall is fine in testing (large tenant), bad in
  production (small tenants). Partitioning is the fix; post-filtering is the trap.
- **Index build blocks writes.** A naive `CREATE INDEX` on a large table locks it.
  Use concurrent builds, and plan index maintenance windows for the large tenants.
- **The 2.1M-document tenant degrades everyone.** Noisy-neighbor effects on shared
  Postgres resources. Isolate the largest tenants early rather than after the
  incident.
- **Embedding dimension is chosen once and forever.** Changing it means re-embedding
  everything. Choose deliberately, and store the embedding model version alongside
  every vector so a migration is possible.
- **Nobody plans the reindex.** You will change embedding models. Design for a
  parallel index + dual-read + cutover from day one, or you'll take downtime.
- **Postgres becomes the bottleneck at 900 customers** and the migration you deferred
  is now urgent. This is why the trigger conditions matter — they turn a crisis into
  a planned project.

### Metrics

Search p50/p95/p99 latency **segmented by tenant size tier** (an aggregate number
hides the small-tenant recall problem entirely); recall@k against a labeled set per
tier; index freshness lag (document update → searchable); index build duration and
frequency; memory headroom on the database; storage per tenant and total vector
storage vs. the trigger threshold; cost per tenant per month; and cross-tenant
isolation canary results.

### Follow-up Questions

1. Six months in, one customer's corpus grows to 40M chunks — nearly as large as
   everything else combined. What do you do?
2. You need hybrid search (BM25 + dense). Does that change the decision? How would
   you implement it in each option?
3. Your embedding model needs upgrading. Walk through the migration for 48M vectors
   with zero downtime and no quality gap.
4. A prospective enterprise customer requires that their data be searchable only from
   their own VPC. What changes?
5. The managed vendor cuts prices 70% and adds a Postgres-compatible interface. Do
   you revisit? What's your process for revisiting decisions in general?
6. How would your answer change if the team had ten engineers instead of three?

### What Separates Senior From Staff?

**Senior:** does the arithmetic, picks pgvector for the right reasons, designs the
tiered partitioning, ships it.

**Staff:** writes down the trigger conditions for revisiting so the decision has a
built-in expiry, rather than becoming permanent by inertia. They recognize that the
*embedding dimension and model choice* is the truly irreversible decision in this
design — far more so than the datastore — and put the migration path in place before
it's needed. They also handle the 40-comment design doc as an organizational
problem: converge the discussion by naming the decision criteria explicitly
("freshness requirement, team capability, subprocessor cost") so the team is arguing
about the right things, and they document why the rejected options were reasonable
so nobody relitigates it in Q3.

### Interviewer Notes

The 196 GB calculation is the fork in the road: candidates who do it are engineering,
candidates who don't are choosing tools by reputation. The second signal is whether
they identify the sync pipeline as the real cost of an external vector store —
it's the single most under-appreciated cost in this decision and it's directly
implied by the freshness requirement. The third is the small-tenant filtered-ANN
problem; almost nobody raises it unprompted, so ask about the p50 tenant explicitly
and see if they reason to it. A candidate who says "brute force is fine for 4,000
vectors" has actually thought about the numbers.

---

## 2. Self-Hosted Inference vs. API Providers

### Situation

Your company runs a content-moderation pipeline for a social platform. Every post,
comment, and image caption is classified for policy violations before publication.

- **Volume: 240M items/day** (≈ 2,800/second average, 7,000/second peak).
- Average item: 180 input tokens, ~30 output tokens (a structured classification).
- Currently: a mid-tier API provider. **$1.9M/month.**
- Latency requirement: p99 under 400ms — moderation is in the publish path.
- Accuracy requirement: false-negative rate on severe categories under 0.05%.
- Team: 6 engineers, 2 with GPU/serving experience. An existing platform team
  operates Kubernetes.

Leadership asks: should we self-host?

### Your Task

Give a recommendation with the numbers to support it, including what it costs to be
wrong.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

This is the scenario where self-hosting is most likely to be correct, because every
factor points the same way: enormous steady volume, short prompts, short outputs, a
narrow task, and a hard latency requirement. Do the math rather than asserting it.

**Throughput requirement.** 2,800 items/sec average, 7,000 peak. Each item is
180 in / 30 out. So:
- Input: 2,800 × 180 ≈ **504k tokens/sec** average prefill.
- Output: 2,800 × 30 ≈ **84k tokens/sec** decode.

**What that means for hardware.** A small model (say 3–8B parameters) on a modern
data-center GPU with continuous batching and paged attention can produce on the
order of a few thousand output tokens/sec and handle substantially more prefill
throughput, depending heavily on model size, quantization, batch composition and
sequence length. Take an order-of-magnitude estimate: if one GPU sustains ~2,000
output tokens/sec for your model and sequence profile, you need ~42 GPUs for
average decode load, plus headroom for prefill and peak — call it **60–90 GPUs**
with redundancy and multi-region.

**Cost comparison.**
- At roughly $2/GPU-hour for reserved capacity: 75 GPUs × $2 × 730h ≈
  **$110k/month** in compute.
- Add: engineering (2 FTE ongoing for serving infra ≈ $60k/month fully loaded),
  monitoring, spare capacity, model development, and a fallback API path for
  overflow. Call it **$220–280k/month** all-in.
- vs. **$1.9M/month** on the API.

That's roughly an **85% reduction**, i.e. ~$19M/year. At that magnitude, the answer
is obvious and the interesting questions are all about *risk and sequencing*, not
about whether.

**Why this workload is unusually favorable:**
- Short prompts (180 tokens) mean prefill is cheap and batching is efficient.
- Short outputs (30 tokens) mean decode is cheap — this is the dominant cost driver
  in most self-hosting math and here it's tiny.
- Narrow classification task → a small fine-tuned model likely matches or beats a
  general mid-tier model.
- Steady, predictable, high volume → high GPU utilization, which is the single
  factor that makes or breaks self-hosting economics.
- Latency requirement of 400ms p99 is *easier* self-hosted (no network hop to a
  provider, no shared-tenant queueing, no provider rate limits).

Compare with a workload that's 10k input tokens, 2k output tokens, spiky, and
low-volume: the same analysis would say "absolutely not."
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Recommendation: yes, self-host — but as a staged migration with the API retained
as permanent overflow and fallback, not as a cutover.**

**Stage 1 (weeks 1–6): prove the model, not the infrastructure.**

Before buying a single GPU, answer: can a small model hit the accuracy bar? Fine-tune
a candidate on your moderation labels (you have enormous quantities — every
moderation decision, every appeal, every human review). Evaluate against the current
system on a held-out set with per-category breakdowns, focusing on the severe
categories where the 0.05% false-negative bar lives.

**If the model doesn't clear the bar, the infrastructure question is moot.** Ordering
matters: many teams build the serving stack first and then discover the quality gap.

**Stage 2 (weeks 4–12): build the serving path in parallel, take 5% of traffic.**

- Use a mature inference server (continuous batching, paged attention, tensor
  parallelism if needed). Do not write your own serving loop.
- Shadow mode first: run self-hosted inference on 100% of traffic *without* using
  its output, and compare decisions against the API. This gives you a full-scale
  accuracy and latency comparison at zero risk. It costs GPU time and is worth every
  dollar.
- Then serve 5% live behind a feature flag with instant rollback.

**Stage 3 (weeks 12–24): ramp to 90%, keep 10% on the API permanently.**

The retained API path is not waste, it's insurance, and it does three jobs:
- **Overflow capacity** for traffic spikes beyond provisioned GPUs (a viral event
  can 3× your volume in minutes; GPUs don't autoscale in seconds).
- **A live fallback** that is continuously exercised, so it works when you need it.
- **A continuous quality reference** — an ongoing comparison between your model and a
  frontier baseline, which is how you detect model rot.

**Stage 4: operationalize.**

- Capacity planning against traffic seasonality; peak is 2.5× average, so provision
  for peak plus headroom and use the API for the rest.
- Model lifecycle: retraining cadence, eval gates, canary deploys, version rollback.
  Policy changes (a new category, a changed threshold) must have a defined lead time.
- On-call: this is now a tier-1 service. Who carries the pager? Get that answered
  before Stage 3, not after.

**Name the risks explicitly in the proposal:**

- **Utilization risk.** The economics assume high utilization. If traffic is more
  variable than modeled, or if you overprovision from caution, cost per item can
  double. Model the *actual* diurnal curve, not the average.
- **Hidden costs.** GPU capacity is often reserved (annual commitment) to get the
  price used in the model. That's a multi-million-dollar commitment that reduces
  flexibility. Spot/preemptible capacity is cheaper but needs a resilience story.
- **The 2 GPU-experienced engineers.** If both leave, you have a $19M/year system
  nobody can operate. That's a real organizational risk and it belongs in the doc.
- **Model quality is now your problem.** No more "the provider improved the model and
  we got better for free." You own every accuracy improvement forever.

**What would change the recommendation:**
- If volume were 10× lower ($190k/month API spend), the savings wouldn't justify the
  team and the answer would be no.
- If prompts were 10k tokens instead of 180, the GPU count and the analysis change
  dramatically.
- If the accuracy bar couldn't be met by a small model, no.
- If traffic were spiky rather than steady, utilization would kill the economics.

</details>

### Trade-offs

**API provider.**
*Advantages:* zero infrastructure, elastic capacity, automatic model improvements,
no capacity planning, no on-call for inference, fast to change models.
*Disadvantages:* cost scales linearly forever with no leverage; rate limits; provider
outages are your outages; model deprecations force migrations on their schedule;
silent model updates can change your behavior; data leaves your environment.
*Choose when:* volume is low or unpredictable, the team is small, the task benefits
from frontier capability, or you're still finding product-market fit.

**Self-hosted.**
*Advantages:* dramatically lower marginal cost at scale; predictable performance and
no rate limits; lower latency (no external hop); full control of model version, so
no surprise behavior changes; data never leaves; the ability to fine-tune for your
exact task.
*Disadvantages:* capital and commitment; GPU capacity planning; cold starts and
scaling lag; you own model quality, upgrades, and drift; on-call; specialized
expertise that is scarce and expensive to replace.
*Choose when:* volume is high and steady, the task is narrow, utilization will be
high, and latency or data residency requirements push you there.

**Hybrid (recommended here).**
*Advantages:* captures the economics while retaining elasticity and a real fallback;
the fallback is continuously validated; a graceful path if the self-hosted model
degrades.
*Disadvantages:* two systems to maintain and evaluate; two quality baselines; routing
complexity; the temptation to let the API path rot because it's "only 10%."

### What Could Go Wrong?

- **A viral event triples traffic in 90 seconds.** GPUs don't scale in 90 seconds.
  Without the API overflow path, you either queue (blocking publishes) or fail open
  (publishing unmoderated content). Fail-open in moderation is a headline.
- **The fallback path rots.** Prompts drift, the provider deprecates the model, and
  the fallback is broken when you need it. Continuous traffic is the only remedy.
- **KV-cache memory pressure** under long-tail long inputs causes OOM and cascading
  failures. Enforce input length limits at admission, not in the model server.
- **Model rot.** Policy evolves, adversaries adapt, and your model's accuracy on new
  evasion patterns silently degrades. The API comparison path is your smoke detector.
- **Reserved capacity commitment** signed at peak volume, then volume drops. You're
  paying for idle GPUs. Structure commitments conservatively and use on-demand or
  the API for the top of the curve.
- **Both GPU engineers leave.** Bus factor of 2 on a $19M/year system. Document,
  cross-train, and use managed inference platforms where they reduce the specialized
  surface.

### Metrics

GPU utilization (the number the entire business case rests on) and its distribution
over the day; throughput in items/sec per GPU; p50/p95/p99 latency end-to-end;
queue depth and admission rejections; false-negative rate per severity category
(the safety metric); agreement rate between self-hosted and API paths on the shadow
sample; cost per million items, all-in including engineering; capacity headroom vs.
peak; time-to-deploy a model update; and incidents per quarter attributable to
inference infrastructure.

### Follow-up Questions

1. Your utilization is 45%, not the 80% you modeled. Cost per item is now within 30%
   of the API. What do you do?
2. A new policy category is added and must be live in 48 hours. Walk through it for
   both architectures.
3. How do you handle a 5× traffic spike with 90 seconds of warning?
4. The provider releases a model that's 3× cheaper and better than your fine-tune.
   What's your process for deciding whether to unwind?
5. How would you structure the GPU capacity commitment to avoid being locked into a
   volume forecast?
6. What's your plan if both GPU-experienced engineers resign next month?

### What Separates Senior From Staff?

**Senior:** does the throughput and cost math, designs the staged migration with
shadow mode, builds the serving stack, ships it.

**Staff:** treats the recommendation as a portfolio decision. They insist the model
question is answered before the infrastructure question, structure the capacity
commitment to preserve optionality, and make the retained API path a permanent
architectural principle rather than a transitional artifact. They surface the bus-
factor risk and the "we now own model quality forever" commitment to leadership as
part of the proposal, because a $19M/year saving that creates an unstaffed permanent
obligation is not the win it looks like. They also define, up front, what evidence
would make them unwind the decision — most organizations can only make this kind of
decision in one direction, and naming the reverse path is what makes it a real
choice.

### Interviewer Notes

This scenario is a good test of quantitative reasoning under uncertainty. The
candidate can't know exact GPU throughput numbers, and shouldn't pretend to — what
matters is that they *set up the calculation correctly*: throughput requirement,
tokens per item split into prefill and decode, GPU count, utilization, all-in cost
including people. Watch for whether they identify that short outputs make this
workload unusually favorable — it's the key technical insight and it distinguishes
people who understand inference economics from people who've read a pricing page.
Then watch for whether they propose shadow mode; running the new system on full
traffic without using its output is the standard de-risking technique and its absence
suggests limited migration experience. Finally, probe the "what would change your
mind" question — a candidate who can't answer it made a decision, not an analysis.

---

## 3. The Evaluation Platform

### Situation

Your company has 11 AI features across 5 product teams. Evaluation is a mess:

- Two teams have real eval suites, in different formats, run manually before releases.
- Two teams have "some test cases in a notebook."
- One team ships on manual spot-checks.
- Nobody can compare quality across features, and leadership has no view of whether
  AI quality is improving.
- A recent incident (a regression that reached customers) has created appetite for
  investment.

You've been asked to fix evaluation company-wide. Options on the table:

- **Build** an internal eval platform.
- **Buy** one of several commercial LLM-evaluation products.
- **Adopt** an open-source framework and standardize on it.

You have two engineers for two quarters.

### Your Task

Recommend an approach, and say what "done" looks like.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**The most important realization: the hard part of evaluation is not the tooling.**

Teams without evals do not lack a test runner. They lack:

- labeled datasets that represent their production traffic,
- agreed definitions of what "correct" means for their feature,
- the discipline to run evaluation before every release,
- someone whose job it is to maintain the dataset as the product changes.

A platform solves none of those. Buying one solves none of them either. Any answer
that centers on tooling is solving the visible problem instead of the real one.

**Second realization: the thing that must be standardized is the *interface*, not
the implementation.** What the company needs is:

- every AI feature reports a score, versioned, to one place,
- score definitions are documented and stable enough to trend,
- releases are gated on it.

How each team computes their score can differ, because a RAG assistant, a
classification pipeline, and an agent are genuinely different evaluation problems and
forcing them into one harness produces a harness that fits none of them.

**Third: 2 engineers for 2 quarters cannot build an eval platform.** They can build a
thin reporting layer, a good reference harness, and — most valuably — spend most of
their time *embedded with the teams that have no evals*, building the first dataset
with them. That's the highest-leverage use of the headcount, and proposing it is a
strong signal.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Recommendation: adopt open-source for the harness, build a thin results registry,
and spend most of the headcount on datasets and adoption. Don't buy, yet.**

**What to build (small):**

1. **A results registry.** A service (or honestly, a table plus a dashboard) where
   every eval run posts:
   ```
   feature_id, eval_suite_version, dataset_hash, git_sha, model_id,
   prompt_version, judge_version, metric_name, score, n, ci_low, ci_high,
   run_timestamp, environment
   ```
   This is a week of work and it is the single highest-value artifact. It gives
   leadership a view, gives teams a history, and — critically — makes scores
   *comparable over time* by pinning dataset and judge versions. Most eval failures
   in practice are "the score moved and we don't know if the system or the
   measurement changed"; this schema prevents that.

2. **A CI gate**, as a shared GitHub Action / pipeline step: run your suite, post to
   the registry, fail the build on regression beyond a threshold the team sets.

3. **A reference harness** built on a well-maintained open-source evaluation library,
   with the company's conventions baked in (dataset format, judge prompts,
   statistical comparison, report format). Make it genuinely good so teams adopt it
   because it's easier than not.

**What to buy: nothing yet, and here's the trigger.** Commercial eval platforms are
good at tracing, dataset management UIs, and human-annotation workflows. You will
want those — but buying before teams have datasets means paying for a tool to manage
data that doesn't exist. Revisit when: three or more teams have mature suites, human
annotation volume exceeds what a spreadsheet handles (say 500+ items/week), or you
need non-engineers (PMs, domain experts) in the loop. Write the trigger down.

**Where the two engineers actually go (the real recommendation):**

- **~30% on the registry, CI gate, and reference harness.**
- **~70% embedded with the three teams that have no evals**, one at a time, for
  4–6 weeks each. Their job is not to build tooling for those teams; it's to sit with
  them and produce: a dataset sampled from *their* production traffic, stratified
  properly; an agreed scoring definition; the first suite; and a team member who
  owns it.

This is unglamorous and it's the work. A platform team that builds tools and waits
for adoption will have 11 features and 2 eval suites in a year.

**What "done" looks like — define it up front:**

- 11 of 11 features have a suite in the registry with a score from the last 30 days.
- Every AI feature's release pipeline blocks on eval.
- Each suite has a named owner and a documented refresh cadence.
- A dashboard shows quality trend per feature, and leadership looks at it monthly.
- The registry has enough history that anyone can answer "did quality regress when we
  changed X?"

Note that none of these are "the platform has feature Y." Define success in terms of
organizational behavior, not software.

**On evaluation methodology, publish standards, not code:**

- Datasets sampled from production, stratified, refreshed quarterly, with a frozen
  slice retained for historical comparison.
- Report scores with confidence intervals; a 2-point move on 200 examples is noise
  and teams must be taught this.
- Judge models pinned and versioned; when the judge changes, re-score history or
  clearly break the series.
- Segment scores; never gate on a single blended number.
- Every production incident contributes a case to the suite.

</details>

### Trade-offs

**Build a platform.**
*Advantages:* fits your workflows exactly; no vendor; can integrate deeply with
internal systems (feature flags, deploy pipeline, data warehouse).
*Disadvantages:* eval platforms have a large surface (datasets, runs, comparisons,
annotation UIs, tracing, statistics) and two engineers will build 15% of it; you own
it forever; every hour on the platform is an hour not spent on the actual bottleneck
(datasets).
*Choose when:* you have unusual requirements, a real platform team, and existing
mature eval practice to build on.

**Buy a commercial platform.**
*Advantages:* immediately good tracing, annotation workflows, and dataset management;
non-engineers can participate; someone else maintains it; often includes production
monitoring integration.
*Disadvantages:* your evaluation data — arguably your most valuable AI asset — lives
with a vendor; cost scales with usage; lock-in through dataset and annotation
formats; it will not create the discipline you're missing.
*Choose when:* teams already have datasets and the bottleneck is workflow, annotation
volume, or non-engineer participation.

**Adopt open-source + thin internal layer (recommended).**
*Advantages:* fastest to something real; no vendor; small surface to maintain; teams
keep flexibility; the registry is the only thing you own and it's simple.
*Disadvantages:* open-source eval libraries vary in maturity and can be abandoned;
you're responsible for the glue; less polish means lower voluntary adoption; no
annotation UI, so human review stays awkward.
*Choose when:* early in the journey, small team, adoption is the bottleneck — i.e.
here.

### What Could Go Wrong?

- **You build tooling and nobody uses it.** The default outcome. Prevented only by
  embedding with teams and making adoption nearly free.
- **Teams game the metric.** Once eval scores are visible to leadership, scores go
  up and quality doesn't. Mitigate by keeping a held-out set the teams don't tune
  against, and by tracking online quality alongside offline.
- **Benchmark overfitting.** Six months of optimizing against a fixed 400-example set
  produces a system excellent at those 400 examples. Rotate datasets, keep a frozen
  slice for comparability *and* a rolling slice for honesty.
- **The judge model becomes the standard.** If everyone uses an LLM judge with the
  same biases, the company optimizes toward that judge's preferences (verbosity,
  formatting, hedging) rather than user value. Periodically validate the judge
  against human grading — and measure the *agreement rate* as a metric in its own
  right.
- **Registry becomes a compliance exercise.** Teams post a score to satisfy the
  dashboard and nobody acts on it. The CI gate is what makes it real; without the
  gate it's decoration.
- **Statistical illiteracy.** Teams celebrate noise and panic at noise. Bake
  confidence intervals into the reporting format so it's impossible to report a bare
  number.

### Metrics

*Adoption:* features with a live suite (target 11/11); features with a blocking CI
gate; days since last eval run per feature; suites with a named owner.
*Quality of the evaluation itself:* dataset size and its representativeness vs.
production traffic; judge/human agreement rate; offline/online metric agreement;
share of production incidents that had a corresponding eval case *before* the
incident (the real measure of whether your suites are any good).
*Outcomes:* regressions caught pre-release vs. post-release; time from a quality
regression to detection.

### Follow-up Questions

1. A team says their feature "can't be evaluated automatically." How do you respond?
2. Your judge model is upgraded by the provider and every historical score shifts.
   What do you do — and what should you have done?
3. Leadership wants a single "company AI quality score." Do you give them one?
4. How do you evaluate an agent that takes 15 steps and calls 6 tools, where only the
   end state is checkable?
5. Two teams' scores aren't comparable but leadership compares them anyway. How do
   you handle that?
6. When *would* you buy a commercial platform? Be specific about the trigger.

### What Separates Senior From Staff?

**Senior:** picks the right build/buy split, ships the registry and CI gate, builds a
good reference harness.

**Staff:** identifies that the bottleneck is organizational, not technical, and
allocates 70% of scarce headcount to embedded dataset work rather than to building
software — which is a harder decision to defend and the right one. They define
success as behavior change with measurable criteria and a date. They anticipate
metric-gaming and design the held-out set before scores become visible to leadership.
And they treat evaluation methodology (sampling, statistics, judge validation) as a
standard the company publishes and teaches, because a platform with bad methodology
industrializes bad decisions.

### Interviewer Notes

The strongest signal is whether the candidate says, in some form, "the tool is not
the problem." Candidates who dive into platform architecture are answering a question
nobody asked. The second signal is headcount allocation: proposing that most of the
team embeds with product teams rather than builds is a mature, slightly
counter-intuitive call that senior candidates rarely make. Probe the statistics — ask
what they'd do about a 2-point score change on 200 examples. And ask when they'd buy;
a candidate with no trigger condition is making a permanent decision by accident.

---

## 4. Agent Orchestration: Framework or Not?

### Situation

Your team is building an agent that handles supply-chain exception workflows: a
shipment is delayed, and the agent gathers information from four internal systems,
determines the impact, drafts customer communications, and proposes remediation
options for a human to approve.

The agent needs: tool calling across 9 internal APIs, multi-step planning,
conversation state across a workflow that can span days, human approval checkpoints,
retries, and full auditability (this touches customer commitments).

The team is split:

- Half want to use a popular agent framework: "we'll ship in weeks not months."
- Half want to build the loop directly: "frameworks hide the parts we need to
  control."

You have four engineers and a hard six-month deadline for a customer commitment.

### Your Task

Decide, and structure the work so the decision is survivable if you're wrong.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Decompose "the framework" into what it actually provides**, because the answer
differs per piece:

| Concern | Framework value | Notes |
|---|---|---|
| The agent loop itself | **Low** | ~200 lines. Not the hard part. |
| Tool/function schema plumbing | Medium | Useful but thin |
| Provider abstraction | Medium | Real value if multi-provider |
| Prompt/message management | Low–medium | Easy to outgrow |
| State persistence across days | **High** | Genuinely hard; frameworks vary wildly |
| Human-in-the-loop checkpoints | **High** | Hard; few frameworks do it well |
| Observability / tracing | **High** | Expensive to build well |
| Retries, timeouts, rate limits | Medium | Well-understood, easy to build |
| Evaluation | Low | You'll build your own anyway |

Notice that the loop — the thing people argue about — is the least valuable part.
The valuable parts are **durable state, human checkpoints, and tracing**, and those
are the parts most agent frameworks are weakest at, because they're the parts that
are actually workflow-engine problems.

**Which reframes the question entirely:** this is not "agent framework or not." It's
"this is a **long-running, human-in-the-loop, auditable business workflow** that
happens to call an LLM." Multi-day duration, approval gates, and auditability are
durable-workflow requirements. A workflow engine (Temporal, Step Functions, or your
company's existing orchestration) plus a hand-written agent loop is very likely the
right shape.

Candidates who reach that reframe have understood the problem. Candidates arguing
framework-vs-no-framework are answering the question as posed.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Recommendation: durable workflow engine for orchestration + a small, owned agent
loop + bought/adopted observability. No general-purpose agent framework.**

**Reasoning by component:**

1. **Orchestration → buy/adopt a durable workflow engine.** The requirements —
   multi-day execution, resumability, human approval steps, retries with state,
   auditability, exactly-once side effects — are precisely what durable workflow
   engines exist for and are genuinely hard to build. This is plumbing, it's mature,
   and the company may already run one. **This is the single most important decision
   in the design** and it's the one the team's debate wasn't about.

2. **The agent loop → build.** It is small (a few hundred lines: message history,
   tool dispatch, termination conditions, error handling) and it is where all your
   product-specific control lives — how you handle a malformed tool call, how you
   detect non-progress, how you decide to ask the human, how you constrain tool
   selection. Frameworks abstract exactly these decisions, and you need them
   explicit and auditable. Also: debugging a production agent means reading the exact
   message sequence sent to the model, and every layer of abstraction between you and
   that sequence costs you hours during an incident.

3. **Tool definitions → build thin, own the schemas.** Nine internal APIs need
   typed, validated, permission-scoped wrappers with idempotency keys. That's your
   integration code; a framework doesn't help and its abstractions get in the way of
   the permission model.

4. **Observability → adopt.** Tracing an agent run (nested tool calls, token counts,
   costs, decisions) is expensive to build well. Use an existing LLM observability
   tool or your company's tracing stack with a good span model. Do not build this.

5. **Provider abstraction → build thin, or use a small library.** A minimal
   interface over one or two providers. Don't take a heavy dependency for this.

**Structure the decision to be survivable:**

- Keep the agent loop behind an interface with a small surface. If you're wrong and
  a framework would have been better, swapping is bounded.
- Do not let framework types leak into your domain model. The most common way this
  decision becomes irreversible is that framework objects end up in your database and
  your API contracts.
- Timebox a spike: two engineers, one week, build the same narrow workflow both ways.
  Compare on the specific criteria you care about — can you inject a human approval
  mid-run? can you resume after a 14-hour pause? can you see the exact prompt? — not
  on lines of code. **Then decide with evidence.** A week is cheap against a
  six-month deadline and it ends the argument permanently.

**Address the "ship in weeks" argument honestly.** It's often true for the first
demo and false for production. Frameworks accelerate the first 60% and can decelerate
the last 40%, especially around state, errors, and observability. But that's an
empirical claim about *this* framework and *this* workload, which is exactly what the
spike measures. Don't win the argument by assertion.

</details>

### Trade-offs

**Use an agent framework.**
*Advantages:* fast start; conventions for tools, memory and planning; community
patterns; less boilerplate; good for exploration and prototypes; abstracts provider
differences.
*Disadvantages:* abstractions hide the message sequence, which is what you debug;
upgrade churn in a fast-moving ecosystem; opinionated state models that may not match
a multi-day workflow; hard to audit; performance and cost overhead from extra calls
you didn't ask for; your team's understanding of their own system degrades.
*Choose when:* prototyping, short-lived agents, standard patterns, small team
optimizing for speed over control, or the agent is not business-critical.

**Build the loop.**
*Advantages:* complete control over context, termination, errors, and permissions;
trivially auditable; no upgrade churn; the team understands the system deeply, which
matters enormously during incidents; easy to optimize cost and latency.
*Disadvantages:* you write the boilerplate; you reinvent patterns others have
refined; onboarding is slower without community knowledge; risk of building a worse
framework accidentally.
*Choose when:* the agent is business-critical, auditability matters, requirements are
unusual, or the team will operate it for years — i.e. here.

**Workflow engine + owned loop (recommended).**
*Advantages:* buys the genuinely hard part (durable state, resumption, human steps)
and owns the part that carries product logic; excellent auditability; the workflow
engine is stable, boring technology.
*Disadvantages:* a heavier operational dependency; workflow engines have their own
learning curve and constraints (determinism requirements can be surprising); two
concepts for engineers to hold.

### What Could Go Wrong?

- **You accidentally build a framework.** Six months in, the "small loop" is 6,000
  lines with a plugin system. Guard by keeping the loop boring and pushing complexity
  into tools and workflows.
- **Framework lock-in via persistence.** Framework-serialized state in the database
  makes the decision irreversible. Persist your own domain state.
- **Determinism constraints bite.** Durable workflow engines typically require
  deterministic replay; an LLM call is non-deterministic and must be an activity, not
  workflow logic. Getting this wrong causes bizarre replay bugs. Know the rule before
  you start.
- **Human approval steps time out.** A workflow waiting days for approval must handle
  the approver leaving the company, going on vacation, or the shipment resolving
  itself. Design escalation and expiry.
- **Auditability is claimed but not implemented.** "Full auditability" means being
  able to reconstruct, for any run, exactly what the model saw and what it did.
  That requires logging the complete message sequence, which requires deciding what
  to do about PII in it. Decide early.
- **The spike is run badly** — one engineer who loves frameworks vs. one who hates
  them, each building the version they prefer. Define the evaluation criteria before
  the spike starts.

### Metrics

Workflow completion rate and time-to-completion (including human wait time,
separately); agent steps per workflow (p50/p95) and cost per workflow; tool call
success rate by tool; human approval rate, rejection rate, and rejection *reasons*
(the highest-value quality signal in a human-in-the-loop agent); resumption success
rate after interruption; audit completeness (can you reconstruct 100% of runs?);
and time-to-debug a failed workflow — a direct measure of whether your architecture
choice was right.

### Follow-up Questions

1. Your spike shows the framework version working in 3 days and the custom version
   in 9. Does that settle it?
2. How do you test an agent that takes 12 steps across 4 systems over 2 days?
3. A tool returns malformed data and the agent produces a wrong customer email that
   gets sent. Walk through prevention.
4. The framework you rejected releases a version that solves your state problem.
   Would you revisit? How?
5. Six months in, the agent works but nobody except you can debug it. What went
   wrong, and how do you fix it?
6. How does your answer change if this agent were customer-facing and autonomous?

### What Separates Senior From Staff?

**Senior:** decomposes the framework question by concern, chooses well per component,
runs a spike to settle the disagreement with evidence.

**Staff:** reframes the problem — this is a durable workflow with an LLM in it, not
an agent project — which changes the architecture and the risk profile entirely.
They also handle the split team as a first-class problem: an unresolved architectural
disagreement among four engineers on a six-month deadline is a delivery risk
regardless of which side is right. They design the spike with pre-agreed criteria so
it produces a decision rather than more debate, and they make the decision reversible
by constraining where framework or engine types are allowed to appear. They also
recognize that "auditability" is a compliance requirement that needs a definition and
an owner, not a bullet point.

### Interviewer Notes

The reframe (durable workflow, not agent framework) is the thing to listen for. It
comes from noticing "spans days" and "human approval" in the requirements, which are
stated plainly and which most candidates skim past. Second signal: do they decompose
the framework into components with different answers, or treat it as a single
yes/no? Third: do they propose a timeboxed spike with pre-agreed criteria? That's a
process answer to a technical disagreement and it's a senior behavior. Candidates who
argue strongly for either side without proposing a way to test the claim are showing
you how they'd handle team disagreement in general.

---

## 5. Observability for AI Systems

### Situation

You run AI infrastructure at a 200-engineer company with 6 production LLM features.
Your existing observability stack (metrics, logs, distributed tracing, an APM vendor)
is mature and well-adopted for conventional services.

For AI features it is inadequate in specific ways:

- Traces show "POST /v1/messages — 3.2s" and nothing about prompts, tokens, or
  retrieval.
- Cost is invisible per request; you get a monthly provider invoice.
- Nobody can answer "what exactly did the model see?" for a given user complaint.
- Quality is not measured in production at all.
- Debugging a bad answer takes an engineer 45+ minutes of manual reconstruction.

Options: extend the existing stack, or buy a specialized LLM observability product.

### Your Task

Decide, and define what AI observability should actually capture.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**First, define the requirements independent of the tool.** What does AI
observability need that conventional observability doesn't?

1. **The full prompt and completion**, not just the request metadata. This is the
   single biggest gap and it is fundamentally a *data volume and privacy* problem,
   not a tooling problem — a 6,000-token prompt per request at high volume is a lot
   of storage, and it contains user data.
2. **Token accounting** — input, output, cached — per request, aggregatable by any
   dimension (feature, tenant, model, prompt version).
3. **Cost per request**, derived from tokens and pricing, attributable to teams and
   customers.
4. **Retrieval visibility** — what was searched, what came back, with scores, and
   what made it into the context.
5. **Agent step traces** — nested tool calls with arguments, results, and the
   decisions between them.
6. **Version dimensions** on every span: prompt version, model version, embedding
   model version, retrieval config, judge version. Without these, no incident is
   attributable.
7. **Quality signals** — user feedback, escalation, edit distance, validation
   failures — joined to the trace that produced them.

Now look at that list: items 2, 3, 6 are *just structured attributes on spans* —
your existing tracing does this natively. Item 5 is *nested spans* — also native.
Items 1 and 4 are large payloads with privacy implications, which is where existing
stacks get expensive and awkward. Item 7 is a data-joining problem.

So the honest split is: **most of AI observability is conventional observability with
the right attributes, and the genuinely new part is payload storage and quality
measurement.** That reframe usually decides the build/buy question.

**Second: the privacy question is the real design constraint.** Storing full prompts
means storing user data in your observability system, which likely has different
retention, access control, and residency properties than your production database.
Get this decided before choosing a tool, because it may eliminate options.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Recommendation: extend the existing stack for structure and metrics; add a
purpose-built payload store you control; buy only if you need annotation and
evaluation workflows for non-engineers.**

**Concretely:**

**1. Standardize the span model (build, one week).** Define a company-wide
convention — ideally on OpenTelemetry semantic conventions for GenAI, so you're not
inventing a schema that diverges from the ecosystem — for:

```
span: llm.call
  attributes:
    gen_ai.system, gen_ai.request.model, gen_ai.response.model
    gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
    gen_ai.usage.cached_tokens
    app.prompt_version, app.feature_id, app.tenant_id
    app.route, app.cost_usd, app.finish_reason
    app.payload_ref            ← pointer, not the payload

span: retrieval
  attributes:
    app.index_id, app.embedding_model_version, app.top_k,
    app.candidates_returned, app.scores_summary, app.payload_ref

span: tool.call
  attributes:
    app.tool_name, app.tool_version, app.idempotency_key,
    app.duration_ms, app.error_class
```

This gets you items 2, 3, 5, 6 immediately in tooling your company already knows,
with existing dashboards, alerting, and on-call familiarity. **This is the highest
value per unit of effort in the whole project.**

**2. Payload store (build, small).** Prompts and retrieved documents go to
object storage keyed by trace ID, with:
- a short TTL (e.g. 14–30 days) set by legal,
- access control and an access audit log,
- PII scrubbing on write for known-sensitive fields,
- sampling for high-volume features (100% of errors and low-quality outcomes, 1–5%
  of successes).

Keeping payloads in your own storage rather than a vendor's resolves the privacy
question cleanly and is genuinely cheap to build. The observability UI links out to
it by `payload_ref`.

**3. Quality signal pipeline (build).** Every user-visible feedback event (thumbs,
edits, escalation, retry, abandonment) is emitted with the originating trace ID. Now
"show me the traces for the 200 worst-rated responses this week" is a query. This
is what takes debugging from 45 minutes to 5, and it's the deliverable people will
actually notice.

**4. Buy: consider a specialized platform for the annotation and evaluation
workflow**, not for tracing. If PMs and domain experts need to review outputs, label
them, and build datasets, that UI is expensive to build and commercial products are
good at it. Trigger: when human review volume exceeds what a spreadsheet handles.

**5. Dashboards and alerts that matter for AI systems** (and that most teams lack):
- cost per request, per feature, with anomaly detection
- token distributions (input p50/p95/p99, output p50/p95/p99)
- `finish_reason: length` rate
- prefix cache hit rate
- retry/attempt count per logical request
- retrieval score distribution and source composition
- refusal rate, validation failure rate, schema parse failure rate
- escalation/fallback rate

Each of these corresponds to a real incident class documented in
`production-incidents.md`. Build the dashboard from the incident list, not from the
tool's default template.

**Why not buy the whole thing:** you'd be running two observability systems, and
during an incident engineers would have to correlate across them manually — the
exact problem you're solving. AI features are not separate from the services around
them; a latency problem might be in retrieval, in the model call, or in the Postgres
query feeding the prompt, and you want one trace across all three.

</details>

### Trade-offs

**Extend the existing stack.**
*Advantages:* one system, one trace across AI and non-AI components; existing
adoption, alerting, on-call familiarity; no new vendor or security review; cost is
incremental.
*Disadvantages:* payload storage is expensive and awkward in most APM products; no
built-in annotation or eval workflows; you build the AI-specific dashboards yourself;
some vendors price by attribute cardinality, which token-level attributes explode.

**Buy a specialized LLM observability platform.**
*Advantages:* purpose-built for prompts, traces, evaluations, datasets, and
annotation; fast to value; good UIs for non-engineers; often includes eval tooling
and prompt management.
*Disadvantages:* a second observability system and the correlation problem that
creates; your prompts (containing user data) at a vendor; cost scales with volume;
lock-in via datasets and annotations; the AI-specific view can encourage treating AI
features as separate from the systems they live in.

**Hybrid (recommended).**
*Advantages:* correlation preserved in the primary stack; privacy controlled;
specialized tooling adopted only where it's genuinely better.
*Disadvantages:* integration work; two places to look for different questions, which
must be documented clearly or it becomes the worst of both.

### What Could Go Wrong?

- **Logging prompts creates a compliance problem.** Full prompts contain user data,
  possibly regulated data. If your observability system has broader access than your
  production database, you've created a data-access backdoor. Get legal involved
  before, not after.
- **Cardinality explosion.** Adding `tenant_id` and `prompt_version` as attributes on
  every span can multiply your metrics cost by orders of magnitude in some APM
  pricing models. Know your vendor's model; use exemplars and log-based metrics where
  appropriate.
- **Sampling hides the rare failure.** 1% sampling means the 0.3% failure mode is
  nearly invisible. Sample by outcome: 100% of errors and negative feedback, low rate
  for successes.
- **Payload retention set by engineering rather than legal**, then a subject-access
  request or an audit arrives. Decide retention with the people accountable for it.
- **Nobody uses it.** Observability that isn't wired into on-call runbooks and
  incident practice becomes shelfware. Ship the dashboards *with* the runbook
  updates.
- **Cost of observability exceeds the cost of the feature.** Genuinely possible with
  full-payload logging at high volume. Budget it explicitly.

### Metrics

*Of the system itself:* trace completeness (share of AI requests with a full trace);
payload availability rate; observability cost as a percentage of AI infrastructure
cost; attribute coverage (share of spans carrying prompt/model version).
*Of its value:* median time-to-diagnose an AI incident (before/after); share of user
complaints that can be traced to a specific request within 5 minutes; number of
incidents where the root cause was identified from the trace vs. by reconstruction.

### Follow-up Questions

1. Legal says prompts may not be stored at all for one product line. How do you debug
   that product?
2. Your payload store is 40 TB after two months. What do you do?
3. Design the sampling policy. What do you sample at 100%, and why?
4. How do you correlate a user complaint received via support three days later with
   the trace that produced it?
5. What would you instrument differently for an agent than for a single-shot RAG
   call?
6. Your APM vendor's bill triples after you add token attributes. Now what?

### What Separates Senior From Staff?

**Senior:** defines the span model, builds the payload store, wires up quality
signals, ships useful dashboards.

**Staff:** recognizes that most of "AI observability" is conventional observability
with the right attributes, which reframes an expensive buy decision into a cheap
standardization project. They drive the privacy decision to legal *before* choosing a
tool, because it constrains the option space and is the thing that will otherwise
force a rebuild in year two. They adopt OpenTelemetry GenAI conventions rather than
inventing a schema, so the company isn't locked into homegrown semantics. And they
build the dashboard set from the organization's actual incident history, making
observability a direct response to how the company has failed rather than a generic
template.

### Interviewer Notes

The insight to listen for: "token counts, cost, and versions are just span
attributes — we already have tracing." Candidates who immediately propose buying a
specialized tool haven't asked what's actually missing. The second signal is privacy:
storing prompts is a data-governance decision, and a candidate who doesn't raise it
is going to create a compliance problem. Third: sampling by outcome rather than
uniformly — it's the correct answer and it requires having thought about rare failure
modes. Finally, ask what dashboards they'd build; a candidate who names cache hit
rate, `finish_reason: length`, and attempt count is drawing on real incident
experience.

---

## The Build vs. Buy Checklist

Run any decision through these. If you can't answer five of them, you don't have
enough information to decide.

**Strategic**
- Is this a source of competitive advantage, or plumbing?
- What do we lose if a competitor has the same thing?

**Cost**
- Fully-loaded cost of building: engineering months × cost, plus ongoing maintenance
  (typically 15–30% of the build cost per year, forever).
- Vendor cost at today's scale, and at 10× scale. Check the pricing model's shape,
  not just the current number.
- Opportunity cost: what doesn't get built?

**Capability**
- Does the team have the expertise? If two people leave, can it still be operated?
- Who is on call for it in 18 months?

**Risk**
- What breaks if the vendor raises prices 3×, gets acquired, or shuts down?
- What breaks if our build has a bug at 3am and its author has left?
- Security review, subprocessor approval, data residency — what's the calendar cost?

**Reversibility**
- What does it cost to switch in 18 months? What creates lock-in (data formats,
  APIs, persisted state, team knowledge)?
- Can we structure the decision to keep it cheap to reverse?

**Trigger**
- Under what conditions should this decision be revisited? Write it down with the
  decision, or it becomes permanent by default.
