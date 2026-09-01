# Architecture Trade-offs

Scenarios in this file are about *choosing*. Every one of them has at least two
defensible answers. The interview signal is not which box you draw — it is
whether you can state the conditions under which your choice stops being correct.

A useful internal test while reading: **what number would have to change for me to
pick the other option?** If you can't answer that, you haven't understood the
trade-off yet.

**Contents**

1. [The Enterprise Assistant That Outgrew Its Prototype](#1-the-enterprise-assistant-that-outgrew-its-prototype)
2. [Synchronous or Asynchronous: The Contract Review Product](#2-synchronous-or-asynchronous-the-contract-review-product)
3. [Model Routing: Where Does the Router Live?](#3-model-routing-where-does-the-router-live)
4. [Caching an LLM Application Without Lying to Users](#4-caching-an-llm-application-without-lying-to-users)
5. [Centralized AI Platform vs. Team-Owned Stacks](#5-centralized-ai-platform-vs-team-owned-stacks)

---

## 1. The Enterprise Assistant That Outgrew Its Prototype

### Situation

You joined a 900-person financial services company as a Senior AI Engineer. Two
years ago a small innovation team built an internal assistant over the company's
document corpus. The prototype was a success and was quietly promoted to
production without ever being redesigned.

Today:

- **20 million documents** in the corpus: policies, contracts, meeting notes,
  Confluence pages, PDFs of regulatory filings, ~40 TB raw.
- **50,000 employees** have access; ~12,000 weekly actives.
- Traffic has grown **15×** in six months and is still growing ~8% month over month.
- Peak load is **~180 queries/second**, heavily concentrated 09:00–11:00 local
  time in three regions.
- **p95 end-to-end latency is 5.1 seconds.** The product team says users abandon
  above ~3 seconds.
- **Inference spend has grown 4×** to roughly **$310k/month**, against an approved
  annual AI budget of $1.2M.
- Architecture: a managed vector database (single region, us-east), a hosted
  embedding API, and a frontier LLM API. Retrieval is dense-only, top-k = 20,
  every chunk stuffed into the prompt.
- The security team has issued a finding: **documents classified Internal-Restricted
  must not leave the company's cloud tenancy.** About 15% of the corpus is
  classified that way, but it is the 15% people ask about most.

### Your Task

Redesign the system. You have four engineers, two quarters, and you are expected
to keep the existing system running the entire time.

### Constraints

- Latency: p95 ≤ 2.5s to first token, ≤ 6s to completion for long answers.
- Budget: get to ≤ $120k/month within two quarters.
- Security: Internal-Restricted content and its embeddings must stay in the
  company VPC. Audit trail required for every retrieval of a restricted document.
- Compliance: answers touching regulatory filings must cite sources; retention of
  prompts/completions capped at 30 days.
- Reliability: the assistant is now in the critical path for the support desk.
  99.5% availability, no single-provider dependency.
- Engineering capacity: 4 engineers. No dedicated SRE. No GPU expertise on the team today.
- You may not ask users to change how they use the product.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path (try it yourself first)</summary>

**Step 1 — Refuse to redesign anything until you have a latency and cost budget
broken down by component.** "5 seconds p95" is not actionable. You need the
decomposition:

```
retrieval (embed query + ANN search + fetch)   ?
reranking                                       ?
prompt construction                             ?
LLM time-to-first-token                         ?
LLM generation                                  ?
```

In systems like this the answer is almost always that generation dominates, and
generation time is driven by **output tokens**, while cost is driven by **input
tokens**. Top-k = 20 with no reranking means you are paying for ~20 chunks of
context on every request — that is the single biggest cost lever and it is also
inflating time-to-first-token via prefill.

**Step 2 — Segment the traffic before optimizing it.** Almost no enterprise
assistant has uniform queries. Expect something like: 60% are short factual
lookups ("what's the travel policy for Frankfurt"), 25% are summarization of a
known document, 15% are genuinely multi-hop reasoning. These have wildly
different cost and latency profiles and should not share one pipeline.

**Step 3 — Separate the security problem from the performance problem.** They
feel entangled ("we must self-host, which is slow and hard") but they are not.
The security requirement is about *where restricted data and its derivatives
live*, not about which model answers a question about non-restricted data. This
suggests a **partitioned architecture**, not a wholesale migration.

**Step 4 — Sequence by risk-adjusted value.** Cheap reversible wins first
(reranking + smaller k, caching, routing), expensive irreversible commitments
last (self-hosted inference, self-managed vector infrastructure). Two quarters is
not enough time to do the irreversible thing badly *and* recover.

</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Target architecture, in the order I would ship it.**

**Phase 1 (weeks 1–4): measure and cut waste. No architecture change.**

- Instrument per-stage latency and per-request token accounting with a trace ID
  that follows the request through retrieval and generation. Emit
  `input_tokens`, `output_tokens`, `retrieved_chunks`, `cache_hit`, `route`,
  `doc_classifications_touched` on every request.
- Replace `top_k=20 → prompt` with **retrieve 50, rerank with a cross-encoder,
  keep top 5–8**. A small reranker (a few hundred million parameters, CPU-servable
  at this QPS or cheap on one GPU) typically recovers or improves answer quality
  while cutting input tokens by 60–70%. This is usually the single highest-ROI
  change in an over-grown RAG system: it *reduces* cost, *reduces* prefill
  latency, and *increases* quality at the same time.
- Add an **exact-match + semantic cache** on (normalized query, permission scope,
  corpus version). In an enterprise assistant, duplicate questions are extremely
  common — onboarding questions, policy questions, "how do I file X". A 25–35%
  hit rate is realistic. Cache the *final answer with its citations*, keyed
  including the permission scope so you can never serve a cached answer across a
  permission boundary.

Expected outcome of Phase 1 alone: input tokens down ~65%, cost down 40–50%,
p95 down 1–1.5s. No new infrastructure to operate.

**Phase 2 (weeks 5–10): routing and streaming.**

- **Stream tokens to the client.** Perceived latency, not total latency, is what
  drives abandonment. Time-to-first-token becomes the metric that matters for the
  3-second bar; total completion time can be 6s if the user is reading.
- **Route by query class.** A small classifier (or the cheap model itself, with a
  strict structured output) assigns each query to: `lookup` (small model, 5
  chunks), `summarize` (mid model, whole document, cached aggressively), or
  `reason` (frontier model, full pipeline, allowed to be slower). Route on
  features you actually have: query length, presence of a document reference,
  retrieval score distribution, conversation depth.
- Set an explicit **fallback chain**: primary provider → secondary provider →
  degraded "here are the top documents, we couldn't generate an answer" response.
  Never let a provider outage return a 500 to the support desk.

**Phase 3 (weeks 11–20): the security partition.**

Split the corpus by classification, not by convenience:

```
                     ┌──────────────── query + user identity ────────────────┐
                     │                                                        │
             permission resolution (groups → doc ACLs)                        │
                     │                                                        │
        ┌────────────┴─────────────┐                                          │
        │                          │                                          │
  General partition          Restricted partition                             │
  (85% of corpus)            (15%, in-VPC)                                    │
  managed vector DB          self-hosted ANN index in VPC                      │
  hosted embeddings          embedding model served in VPC                     │
        │                          │                                          │
        └────────────┬─────────────┘                                          │
                     │                                                        │
             merge + rerank (in VPC)                                          │
                     │                                                        │
        does the context contain restricted content?                          │
             ┌───────┴────────┐                                               │
            yes              no                                               │
             │                │                                               │
   in-VPC self-hosted    external LLM API ◄───────────────────────────────────┘
   open-weights model    (frontier, streaming)
```

Key decisions and why:

- **Only the restricted 15% gets self-hosted.** Self-hosting inference for the
  whole product with four engineers and no GPU experience would consume both
  quarters and probably fail the latency bar. Partitioning means you buy the hard
  operational problem only where the requirement is real.
- **The routing decision is made after retrieval, not before.** You cannot know
  whether a question touches restricted material until you have retrieved. This
  costs you an extra hop but is the only correct ordering; deciding up front
  either leaks or over-restricts.
- **Embeddings are treated as derived data subject to the same classification.**
  This is the point security teams care about most and engineers most often miss.
  An embedding of a restricted document is a lossy but real representation of it;
  it cannot sit in a multi-tenant managed service if the source cannot.
- For the in-VPC model, pick something in the 8–30B class served on a small
  fixed GPU fleet with a mature serving stack (continuous batching, paged
  attention). At 15% of 180 QPS ≈ 27 QPS peak, this is a small fleet — this is
  the argument that makes the plan feasible for four engineers.

**Phase 4 (ongoing): evaluation as a gate.**

None of the above ships without a regression suite: a few hundred queries with
graded reference answers, split by query class and by classification, run on every
prompt/model/retrieval change, blocking deploys on retrieval recall@k and answer
faithfulness. Without this, Phase 1's k-reduction is indistinguishable from a
quality regression and you will get rolled back by anecdote.

</details>

### Trade-offs

**Approach A — Full migration to self-hosted inference in the VPC.**

*Advantages:* one architecture, no data-residency ambiguity, marginal cost per
token drops sharply at high utilization, no provider rate limits, full control
over model versions (no silent behavior changes under you).

*Disadvantages:* you own GPU capacity planning, autoscaling with multi-minute cold
starts, KV-cache memory pressure, model upgrades, and 24/7 on-call — with four
engineers and no SRE. Quality of open-weights models on the 15% "hard reasoning"
traffic is likely below frontier, and you'd feel it exactly where users are least
tolerant. Utilization at 180 QPS peak / ~30 QPS trough means you either overprovision
(cost) or accept queueing at peak (latency).

*Choose when:* volume is high and steady, the security requirement covers most of
the corpus, and you have or can hire real inference-serving expertise.

**Approach B — Stay fully managed; solve security with contractual controls
(zero-retention agreements, private networking, BYOK).**

*Advantages:* fastest path, no new operational surface, keeps frontier quality
everywhere, lets four engineers spend both quarters on retrieval and evaluation —
which is where most of the quality is anyway.

*Disadvantages:* it is a legal answer to a technical requirement, and it fails the
moment the requirement is "must not leave our tenancy" rather than "must not be
trained on." Leaves you with concentrated vendor risk and no leverage on price.

*Choose when:* the security finding is negotiable, the corpus is not
highly regulated, and time-to-value dominates.

**Approach C — Partitioned (the answer above).**

*Advantages:* buys the expensive operational problem only where it's required;
each partition can evolve independently; gives you a working self-hosted
capability at low risk that you can expand later if economics shift.

*Disadvantages:* **two of everything.** Two indexes, two embedding models (or a
careful commitment to keep them identical), two inference paths, two eval suites,
and a merge/rerank step that must reconcile scores across them. This is real,
permanent complexity, and it is the strongest argument against this design.

*Choose when:* the restricted fraction is a minority of traffic and the team is
small — i.e., this scenario.

### What Could Go Wrong?

- **Score incomparability across partitions.** Two indexes with different embedding
  models produce scores you cannot merge by sorting. Mitigation: use the same
  embedding model in both partitions (this constrains your VPC model choice), and
  make the cross-encoder reranker the only thing that produces the final ordering.
- **Permission-scoped cache poisoning.** A cache key that omits the permission
  scope will eventually serve a restricted answer to someone without access. This
  is the single highest-severity bug available in this design. Key on the
  *resolved ACL set*, not the user ID (or you lose all cache value), and add a
  test that asserts a cache miss across scope boundaries.
- **The reranker becomes the latency bottleneck.** Cross-encoding 50 candidates is
  50 forward passes. Budget for it, batch it, and cap candidate count adaptively.
- **Classification drift.** Documents get reclassified. If reclassification doesn't
  trigger re-indexing into the correct partition, you have a slow leak. Treat
  classification as an event stream, not a batch job.
- **Quality regression attributed to the wrong change.** Phases 1–3 ship
  overlapping changes. Without per-change eval runs and the ability to attribute,
  you will spend a month bisecting.
- **Cost moves from inference to GPUs and nobody notices.** Self-hosting doesn't
  eliminate spend; it converts variable spend into fixed spend. If utilization is
  low, the $/query can be *worse*. Track $/query per partition, not total spend.

### Metrics

*Technical*

- Time-to-first-token p50/p95/p99; total completion time p95 — segmented by route.
- Retrieval recall@k and nDCG against a labeled set; reranker lift over raw ANN.
- Input tokens and output tokens per request (p50/p95) — the true cost drivers.
- Cache hit rate, and *cache correctness* (sampled audit).
- Route distribution and per-route error rate; fallback activation rate.
- GPU utilization, queue depth, and time-in-queue for the VPC partition.
- Groundedness/faithfulness score and citation-validity rate.

*Business*

- Cost per query, and cost per *resolved* query (not the same thing).
- Weekly active users and query abandonment rate (correlate with TTFT buckets).
- Support-desk deflection rate — the actual reason this system is funded.
- Answer thumbs-down rate segmented by route; escalations to human support.
- Audit completeness for restricted-document retrievals (must be 100%).

### Follow-up Questions

1. Six weeks in, retrieval quality is up but users complain the assistant "got
   dumber." Your eval suite shows no regression. What is your investigation plan,
   and what does this tell you about your eval suite?
2. Legal now says the retention cap is 7 days, and that prompts containing
   restricted content may not be logged at all. Your entire debugging workflow
   depends on logged prompts. Redesign your observability.
3. The company acquires a subsidiary in a jurisdiction with data-residency laws.
   You now need a third partition in a different region. What breaks in your
   design, and what would you have done differently in Phase 3 knowing this?
4. Your frontier provider deprecates the model you route `reason` traffic to, with
   90 days notice. Walk me through the migration, including how you decide whether
   the replacement is acceptable.
5. Traffic grows another 5× but budget is frozen. Where does the next 50% of cost
   come out of, and what quality are you willing to trade for it?
6. A director asks why you didn't just fine-tune a small model on the corpus and
   skip retrieval entirely. Give the honest answer, including the conditions under
   which they'd be right.

### What Separates Senior From Staff?

A **senior** engineer designs the partitioned architecture correctly, sequences
the phases sensibly, and lands the latency and cost targets. They own the system.

A **staff** engineer additionally:

- Recognizes that "the same embedding model must serve both partitions" is a
  *long-lived organizational constraint*, writes it down as an architectural
  standard, and makes sure the next three teams don't violate it.
- Reframes the budget conversation: instead of "get to $120k," they establish
  cost-per-resolved-query as the unit metric and show the CFO that the system is
  net-positive at $310k if deflection is real — then negotiates the target based
  on evidence rather than accepting an arbitrary number.
- Notices that the security finding will recur for every AI product in the company
  and builds the classification-aware retrieval layer as a shared component with a
  clear contract, rather than as part of this app.
- Plans for the successor: documents what would have to be true (utilization,
  open-model quality, team size) to collapse the partitions back into one
  architecture, so the next team doesn't have to rediscover the reasoning.

### Interviewer Notes

*What you're actually evaluating:*

- **Does the candidate measure before designing?** Anyone who starts drawing boxes
  before asking for the latency decomposition is showing you how they'd behave on
  day one of a real incident.
- **Do they separate coupled requirements?** The strongest signal in this scenario
  is realizing that "security" and "latency/cost" are independent problems that a
  partition can decouple. Weak candidates propose self-hosting everything.
- **Do they know that reranking reduces cost?** Many candidates treat quality and
  cost as always-opposed. Retrieve-more-then-rerank-to-fewer is a case where they
  aren't, and knowing that is a strong signal of hands-on RAG experience.
- **Do they respect the team size?** A four-engineer plan that requires an SRE org
  is a wrong answer regardless of its technical elegance.
- **Cache key discipline.** If they propose caching and don't mention permissions
  unprompted, ask directly. Their reaction tells you whether they've operated a
  multi-tenant system.

*Red flags:* naming specific vendors as the answer; proposing a rewrite in a
framework; treating embeddings as non-sensitive; no rollback story; no evaluation
gate.

*Green flags:* asks for the query distribution; asks what "5 seconds" is composed
of; asks whether users are internal (changes the abuse and latency calculus);
proposes streaming early; explicitly names what they would *not* do this year.

---

## 2. Synchronous or Asynchronous: The Contract Review Product

### Situation

You are the founding AI engineer at a 30-person legal-tech company. The product
reviews commercial contracts and produces a risk report: every clause extracted,
classified, compared against the customer's playbook, and annotated with
suggested redlines.

The current implementation is a single synchronous HTTP endpoint. The user uploads
a PDF and waits. It works, because the original design assumption was
20-page NDAs.

Reality now:

- Median contract: 34 pages. p95: 180 pages. p99: **900 pages** (master service
  agreements with exhibits).
- Processing is per-clause: a 34-page contract produces ~120 clauses, each needing
  1–3 model calls (extract, classify, compare-to-playbook).
- p50 request time: 40 seconds. p95: 6 minutes. p99: times out at the load
  balancer's 15-minute limit and the user sees nothing.
- Your largest customer just signed and will upload **~4,000 contracts in a single
  batch** during their migration, then ~200/day steady-state.
- Support tickets are dominated by "it's stuck" and "I refreshed and lost my place."

### Your Task

Redesign the processing architecture. The CEO wants the batch migration to work in
three weeks and does not want the UX to feel like a step backwards for the common
case of a 12-page NDA.

### Constraints

- The 12-page NDA must still feel interactive: first results visible in < 10s.
- The 900-page MSA must complete reliably, even if it takes 40 minutes.
- Provider rate limit: 8,000 requests/minute and 4M input tokens/minute, shared
  across all customers.
- Reprocessing must be cheap: customers update their playbook and expect the
  report to refresh.
- Small team: three backend engineers, one of whom is you.
- Costs are billed per contract to the customer; unit economics must stay positive.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

The instinct is "make it async." That is directionally right and insufficiently
specific. The interesting questions are:

1. **What is the unit of work?** The contract, or the clause? This is the central
   design decision. Clause-level units give you parallelism, granular retries, and
   partial results — at the cost of a coordination layer that must know when a
   contract is "done."
2. **Async doesn't mean the user waits differently — it means the user waits
   *visibly*.** A job that returns a job ID and a polling endpoint is strictly
   worse UX than the current synchronous call unless you also stream partial
   results. The right frame: *the report is built incrementally and rendered as it
   fills in.*
3. **Two workloads are hiding here.** Interactive single-contract review and bulk
   backfill of 4,000 contracts have opposite requirements: latency vs. throughput.
   Running them through one queue means the migration starves the interactive
   users. This is a priority/isolation problem, not a scaling problem.
4. **Idempotency is a first-class requirement**, not a nice-to-have. With
   clause-level work items, retries, and playbook-triggered reprocessing, you will
   execute the same logical work many times. If a work item isn't keyed by
   `(contract_version, clause_id, playbook_version, prompt_version)`, you will
   both waste money and produce inconsistent reports.

</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Design: clause-level work items on a priority queue, with incremental
materialization of the report.**

```
upload ──► ingest (parse + segment into clauses) ──► emit N work items
                                                          │
                                    ┌─────────────────────┴──────────────────┐
                                    │  priority queue                        │
                                    │  interactive lane   |  bulk lane       │
                                    └─────────────────────┬──────────────────┘
                                                          │
                                        worker pool (bounded concurrency,
                                        token-aware rate limiting)
                                                          │
                                        write clause result (idempotent upsert)
                                                          │
                                        ─► push update to client (SSE/WebSocket)
                                                          │
                                        contract completion detector ─► final report
```

**Decisions:**

- **Unit of work = clause.** A 900-page MSA becomes ~3,000 independent items
  rather than one 40-minute request. Retries are per-clause, so one malformed
  clause can't fail a contract. Progress is meaningful ("1,840 of 3,000 clauses
  reviewed") instead of a spinner.
- **Ingest stays synchronous and fast.** Parsing and clause segmentation is CPU
  work, typically 1–4 seconds. Return the clause skeleton immediately — the user
  sees the structure of their contract in under 10 seconds, and cells fill in.
  This is how you keep the NDA case feeling interactive without keeping the
  request open.
- **Two lanes, one worker pool.** Interactive work is dequeued preferentially; bulk
  work fills idle capacity. A strict weighted policy (e.g. 4:1) prevents bulk
  starvation entirely while guaranteeing interactive latency. Separate pools would
  waste capacity; separate *lanes* on a shared pool doesn't.
- **Rate limiting is token-aware, not request-aware.** The provider limit that will
  actually bite is 4M input tokens/minute, and clause sizes vary by 50×. A
  request-counting limiter will either underuse the quota or blow through it.
  Estimate tokens per item at enqueue and admit work against a token budget.
- **Idempotency key** = hash of `(contract_version, clause_id, playbook_version,
  prompt_version, model_version)`. Results are upserted on this key. Playbook
  updates change the key for comparison steps but *not* for extraction, so a
  playbook change reprocesses ~30% of the work instead of 100%. This is the
  design decision that makes "refresh the report" economically viable.
- **Backpressure is explicit.** Queue depth above a threshold rejects new *bulk*
  submissions with a documented retry-after; it never rejects interactive ones.
  The migration customer gets a throttled ingest rate, communicated up front, not
  a degraded product for everyone.

**Why not just a bigger timeout and more replicas?** Because it doesn't solve
partial failure (one bad clause kills a 900-page job), doesn't give progress,
doesn't isolate the migration from interactive traffic, and doesn't make
reprocessing cheap. It only buys headroom, and the p99 will eat it.

</details>

### Trade-offs

**Approach A — Synchronous with streaming, no queue.**
*Advantages:* far simpler; no queue to operate; no completion-detection logic; easy
to reason about and debug; results stream as they're produced.
*Disadvantages:* the connection is the state. A dropped connection loses work.
No priority control. Bulk ingest is impossible without a parallel mechanism.
Retries restart everything.
*Choose when:* p99 work is bounded (say under 60s) and there is no bulk workload.

**Approach B — Contract-level async jobs.**
*Advantages:* simple job model; one row per contract; trivially resumable at the
contract level; much less coordination code than clause-level.
*Disadvantages:* no partial results, so the NDA UX regresses; retry granularity is
the whole contract, so a transient failure at clause 2,900 costs you the previous
2,899; a playbook change reprocesses everything.
*Choose when:* contracts are small and uniform, or the product is genuinely
batch-oriented ("we'll email you when it's ready").

**Approach C — Clause-level work items (chosen).**
*Advantages:* fine-grained retries, real progress, cheap reprocessing, natural
parallelism, priority control, and the 900-page case stops being special.
*Disadvantages:* you now own a distributed system. Completion detection, partial
failure semantics ("2,997 of 3,000 succeeded — is the report valid?"), ordering
for cross-clause context, and observability across thousands of small units. This
is meaningfully more code and more ways to be subtly wrong.
*Choose when:* work decomposes naturally and item counts vary by orders of
magnitude — as here.

### What Could Go Wrong?

- **Cross-clause context loss.** Clause 14 may only be interpretable given the
  definitions in clause 2. Naive independent processing produces confidently wrong
  classifications. Mitigation: a two-pass design — pass 1 extracts definitions and
  document-level metadata, pass 2 processes clauses with that context injected.
  This introduces a barrier in your DAG; plan for it rather than discovering it.
- **"Done" is ambiguous.** With 3,000 items, some will be permanently failing.
  Define completion as *all items terminal* (succeeded or permanently failed), and
  surface partial reports with an explicit coverage figure. A report that silently
  omits 3 clauses is a legal-liability bug, not a UX bug.
- **Retry storms during a provider degradation.** 3,000 items × exponential backoff
  with no jitter is a self-inflicted DDoS on your own quota. Jitter, cap
  concurrency globally, and add a circuit breaker that drains rather than retries.
- **Poison items.** A clause containing 40k tokens of table gibberish will fail,
  retry, and consume budget forever. Cap attempts, dead-letter, alert.
- **Bulk migration wins the queue anyway.** If the weighted policy is implemented
  as "check bulk when interactive is empty," a burst of interactive work will
  starve bulk indefinitely and the migration never finishes. Both directions of
  starvation are real; use explicit weights.
- **Cost per contract becomes invisible.** With clause-level items, nobody notices
  that the 900-page MSA costs $240 to process until the invoice arrives. Attribute
  cost to contracts at write time.

### Metrics

*Technical:* time-to-first-clause-result (the real interactive metric); contract
completion time p50/p95/p99 by page-count bucket; per-item success rate and retry
distribution; queue depth and time-in-queue per lane; token-budget utilization vs.
provider limit; dead-letter rate; duplicate-work rate (idempotency effectiveness).

*Business:* cost per contract by size bucket vs. price charged (unit economics per
customer); migration throughput (contracts/hour) and projected completion date;
support ticket volume for "stuck" reports; report coverage percentage — the share
of clauses successfully reviewed, which is the quality floor the product sells.

### Follow-up Questions

1. The migration customer uploads all 4,000 contracts in the first ten minutes.
   Your queue is 12M clauses deep and interactive users are fine, but the customer
   asks when it will finish. How do you answer, and what would you have built to
   answer it?
2. Legal discovers that clause classifications changed between two runs of the
   *same* contract with the same playbook. Why might that happen, and how do you
   make the system reproducible?
3. Your provider raises your rate limit 5×. Does anything in this design change?
   What if they *lowered* it 5×?
4. A customer wants to review a contract collaboratively with three lawyers
   editing simultaneously while it's still processing. What changes?
5. Now the report must be produced within 90 seconds for a "quick scan" tier at
   1/10th the price. Design that path — does it share this architecture?

### What Separates Senior From Staff?

A senior engineer builds this correctly and ships it in three weeks.

A staff engineer asks the question that isn't in the ticket: **is per-clause
model invocation the right shape at all?** Three calls per clause × 120 clauses is
360 calls for a 34-page NDA. A batched design — 20 clauses per call with
structured output — could cut cost 5× and latency 3× at some accuracy cost. The
staff-level move is to *run that experiment* before committing three weeks of
engineering to scaling an expensive primitive. They also notice that the pricing
model (per contract) and the cost model (per clause) are misaligned, and raise it
with the business before the 900-page contracts destroy the margin.

### Interviewer Notes

Evaluate whether the candidate treats "make it async" as an answer or as the
beginning of one. The strongest candidates immediately ask about the *distribution*
of contract sizes, not the average, and separate the interactive and bulk
workloads without being prompted. Probe idempotency — it separates people who have
run these systems from people who have designed them on whiteboards. If they never
mention that partial results have legal meaning in a contract-review product, they
are optimizing a system without understanding the product.

---

## 3. Model Routing: Where Does the Router Live?

### Situation

Your company runs a customer-facing support assistant. 1.2M conversations/month.
You've established, with evaluation data, that:

- 71% of turns are answerable by a small, cheap model at 96% of the frontier
  model's quality.
- 22% need a mid-tier model.
- 7% genuinely require the frontier model; on these, the small model is 61% as good
  and its failures are *confident* — it doesn't know it's wrong.

Leadership wants routing. Three teams have opinions:

- The application team wants routing logic in the app: "we know our traffic."
- The platform team wants a shared routing gateway: "eight teams will need this."
- A vendor sells a managed routing product that claims 40% cost reduction.

### Your Task

Decide where routing lives and how routing decisions are made. Defend it to all
three teams.

### Constraints

- Eight product teams will eventually need routing; today only two have the volume
  to care.
- The gateway would be operated by a 3-person platform team already at capacity.
- Any routing mistake on the 7% hard traffic is a visible customer-facing failure.
- Latency budget for the routing decision itself: < 50ms p95.
- The company has no shared eval infrastructure yet.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

Separate three things that get conflated:

1. **The routing policy** (which model for which query) — inherently
   application-specific. Support traffic and code-generation traffic have nothing
   in common.
2. **The routing mechanism** (call model A vs. B, handle fallback, retry, record
   the decision) — entirely generic.
3. **The evidence** (evaluation data proving the policy is safe) — a capability,
   not a component.

Once separated, the answer usually stops being "app vs. platform" and becomes
"mechanism in the platform, policy in the app, evidence as a shared capability."

Then ask the harder question: **what signal does the router route on?** Options,
in increasing order of cost and quality:

- Static rules (query length, feature flags, customer tier) — free, brittle.
- A learned classifier over query features — cheap, needs labeled data, drifts.
- The cheap model itself with a confidence/abstention signal — no extra latency
  hop, but LLM self-reported confidence is famously unreliable.
- **Cascade**: run cheap, evaluate the result, escalate if it fails a check. Costs
  the cheap call twice-over on escalation but routes on *the actual output*,
  which is far more reliable than routing on the input.

For a case where the failure mode is *confident wrongness on 7% of traffic*, input-based
routing is dangerous and cascading is attractive — but only if you have a cheap,
reliable escalation check.

</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Where it lives:** the *mechanism* goes in a thin shared gateway library (not a
network hop) — provider abstraction, fallback, retries, decision logging, cost
attribution. The *policy* stays in the application, expressed as configuration the
gateway executes. The vendor product is declined for now, for a specific reason:
it optimizes a metric (cost) against a quality proxy it defines, and you cannot
audit its decisions against your 7% failure class.

Making it a **library rather than a service** is the decision that respects the
3-person platform team's capacity and the 50ms budget. A network hop for a routing
decision spends most of the budget on the hop itself. Revisit as a service only
when you need centralized, real-time policy changes across teams.

**How decisions are made — a hybrid cascade:**

```
query
  │
  ├─ hard gate: rules that must never be routed down
  │  (payment disputes, account security, legal escalation, VIP tier)
  │      └──► frontier model
  │
  ├─ cheap model + structured output including an explicit
  │  "insufficient_information" / "requires_specialist" branch
  │      │
  │      ├─ escalation check (cheap, deterministic where possible):
  │      │    · did it use the abstention branch?
  │      │    · did retrieval scores fall below threshold?
  │      │    · did it fail schema validation or citation validation?
  │      │    · is the conversation already 3+ turns (a proxy for "not going well")?
  │      │
  │      ├─ pass ──► return
  │      └─ fail ──► mid or frontier model, decision logged
```

Key points:

- **The abstention branch is designed into the prompt and schema**, so "I don't
  know" is a first-class, cheap-to-detect output rather than something inferred
  from log-probs. This is what makes cascading work: you converted "is the model
  confident?" from an unanswerable question into a structural one.
- **Hard gates come first** because some traffic must never be routed by
  statistics. Cost optimization on account-security questions is a business risk,
  not an engineering trade-off.
- **Every decision is logged with its features and outcome**, which within a month
  gives you the labeled dataset you need to consider a learned router — and, more
  importantly, gives you the ability to answer "why did this conversation get the
  cheap model?" during an incident.
- **Escalation rate is a monitored SLO.** If it drifts from an expected ~25% to
  60%, something upstream broke (retrieval, a prompt change, traffic mix) and you
  learn it from the router before you learn it from customers.

Expected economics: ~71% of traffic served cheap, ~25% escalating (paying both
calls), net cost reduction in the 45–60% range with quality held at or above the
current single-model baseline — *provided* the escalation check is good. That
proviso is the entire risk, which is why it ships behind an eval gate and a
shadow-mode period where routing decisions are made and logged but not enforced.

</details>

### Trade-offs

**Routing on input (classifier):** one model call total, lowest latency and cost.
But it predicts difficulty from the question, and difficulty often depends on
whether retrieval found the answer. Drifts as traffic changes. *Choose when* the
query classes are genuinely distinguishable up front (e.g. language, or "summarize
this document" vs. "reason about these documents") and the cost of a wrong route
is low.

**Routing on output (cascade):** routes on evidence, degrades gracefully, and the
escalation signal doubles as a quality monitor. But escalated requests pay twice
and take longer — the p95 latency of the 25% is now cheap-call + frontier-call,
which may bust your latency SLO exactly for the hardest queries.
*Choose when* failures are expensive and confidence is not observable from input.

**Managed routing product:** fastest to value, no code. But the vendor's notion of
"equivalent quality" is not yours, decisions are hard to audit, and you've added a
dependency in the critical path of every request. *Choose when* you have no eval
capability and low stakes — which is a reason to be worried, not reassured.

### What Could Go Wrong?

- **Cascade latency blowup.** The 7% hardest queries — from your most frustrated
  users — become the slowest. Mitigate by starting the frontier call speculatively
  in parallel for high-risk segments, accepting the double cost on a small slice.
- **Escalation check trained on yesterday's failures.** Prompt changes silently
  invalidate the check. Version the check with the prompt and re-validate together.
- **Gaming by the cheap model.** If the model learns (via prompt tuning against
  your metric) that abstaining is rewarded, escalation rate creeps and savings
  evaporate. Monitor the ratio, and evaluate abstention precision, not just recall.
- **Policy sprawl.** Eight teams, eight policies, no shared vocabulary. Within a
  year nobody can answer "what does this company spend per conversation and why."
  This is the argument the platform team is actually making, and it's a good one —
  address it with shared *decision logging schema*, not shared policy.
- **The 96%-as-good measurement was made on a stale eval set.** Traffic mix shifts;
  the small model's relative quality is not a constant. Re-measure quarterly.

### Metrics

Route distribution vs. expected; escalation rate and escalation precision
(what fraction of escalations actually needed it — measured by sampling and grading
both answers); cost per conversation and per resolved conversation; p95 latency
*per route* and blended; quality by route against a graded set; hard-gate hit rate;
customer satisfaction and resolution rate segmented by route — the number that
tells you whether the cheap model is actually 96% as good in production, not in eval.

### Follow-up Questions

1. Your escalation check uses retrieval score thresholds. The embedding model is
   upgraded and all scores shift. What happens, and how do you prevent it?
2. Finance asks you to hit a hard $/conversation ceiling. Would you route
   dynamically against a budget? What are the failure modes of a budget-aware
   router?
3. A second team adopts the gateway and immediately wants a policy your config
   language can't express. Do you extend the language or let them fork?
4. The frontier model's price drops 70% overnight. What do you do — and how quickly
   can you do it?
5. How would you A/B test a routing policy change safely, given that the effect on
   customer satisfaction takes weeks to materialize?

### What Separates Senior From Staff?

The senior answer is a good router. The staff answer notices that **three teams
disagreeing is the actual problem**: they'd get the two high-volume teams to agree
on a decision-logging schema and a shared eval harness *first*, ship the library
into those two, and let adoption by the other six be pulled rather than mandated.
They'd also frame the vendor decision as reversible-vs-irreversible and put a
re-evaluation date on it rather than winning the argument permanently. And they'd
say out loud that with no shared eval infrastructure, *any* routing decision the
company makes is unfalsifiable — and fix that first.

### Interviewer Notes

The best signal here is whether the candidate distinguishes policy from mechanism
without prompting, and whether they notice that the 7% failure mode ("confidently
wrong") rules out naive input-based routing. Candidates who immediately reach for
"train a classifier" without asking what labels they'd have are pattern-matching.
Push on the latency consequence of cascading — many candidates propose it and never
compute that they've made their hardest queries their slowest. Also worth probing:
do they treat the vendor option seriously, or dismiss it reflexively? Reflexive
build bias is a real weakness at senior level.

---

## 4. Caching an LLM Application Without Lying to Users

### Situation

A consumer-facing nutrition app uses an LLM to answer questions about food, recipes,
and the user's logged meals. 400k daily active users, ~2.6M LLM calls/day, and the
per-request cost is now the single largest line item in COGS.

Analysis of a week of traffic:

- 31% of queries are **exactly identical** strings to another query that week
  ("is oat milk healthy").
- Another ~24% are near-duplicates by embedding similarity (> 0.94 cosine).
- 18% reference the user's own logged data ("how much protein did I have today").
- The rest are long-tail.

The obvious move is caching. The last engineer who tried it was rolled back after
users received answers referencing *other people's* meal logs.

### Your Task

Design the caching layer. Explain exactly what is safe to cache and what is not.

### Constraints

- Answers must remain personalized where the user expects personalization.
- Nutrition claims are regulated in some markets; a stale answer that contradicts
  updated guidance is a compliance issue.
- Target: 40% cost reduction from caching.
- p95 latency currently 2.4s; caching should improve it, not complicate it.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

The previous failure wasn't a caching bug — it was a **key-design** bug. The
discipline: *the cache key must contain everything the answer depends on.* So the
real work is enumerating the dependencies of an answer:

- the user's question
- the retrieved context (which may be personal)
- the user's profile (allergies, goals, dietary restrictions)
- the prompt version
- the model version and decoding parameters
- the knowledge base version
- locale/market (for regulated claims)

Anything in that list that you leave out of the key is a class of bug you have
chosen to ship.

Then notice: caching is not one thing. There are at least four distinct caches with
different keys, hit rates, TTLs, and risk profiles:

1. **Exact response cache** — for impersonal queries.
2. **Semantic cache** — near-duplicate matching; higher hit rate, real risk of
   serving a subtly-wrong answer to a subtly-different question.
3. **Retrieval cache** — cache the retrieved documents, not the answer. Lower
   savings (you still pay generation) but far safer and personalization-compatible.
4. **Prefix / KV cache** — cache the shared prompt prefix at the provider or serving
   layer. Zero semantic risk, works on 100% of traffic, and is usually the most
   underrated option in the room.

Ranking these by (savings × safety) rather than savings alone is the senior move.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Step 0: classify every query as personal or impersonal, and make this a
first-class field, not an inference.** The classification happens before
retrieval. If the query or its resolved context touches user-specific data, it is
personal and is *never* eligible for the response caches. Enforce this as a
type-level or middleware invariant, not a convention — the previous incident
proves conventions fail here.

**Layer 1 — Prompt-prefix caching (all traffic).** The system prompt, tool schemas,
and nutrition guideline snippets are identical across requests and often account
for 60–80% of input tokens. Provider-side prefix caching (or a serving stack with
prefix reuse if self-hosted) cuts input cost and prefill latency with **zero
semantic risk**. Ship this first. It is the only change on this list that cannot
produce a wrong answer.

**Layer 2 — Exact response cache (impersonal only).**

```
key = H(normalized_query, locale, user_dietary_flags, prompt_v,
        model_v, decoding_params, kb_version)
```

Normalization: lowercase, collapse whitespace, strip trailing punctuation. Do *not*
strip negation or quantities. TTL 24h, plus explicit invalidation on `kb_version`
or `prompt_v` change. Expected hit rate on impersonal traffic: high, because the
31% exact-duplicate figure is measured on exactly this population.

Note `user_dietary_flags` in the key: "is oat milk healthy" has a different correct
answer for a user with a nut allergy. Including a small, low-cardinality set of
flags preserves the personalization users care about while keeping the key space
small enough to hit. This is the design compromise that makes caching viable — and
it's a decision to make explicitly, with product input, about *which* dimensions of
personalization are cache-visible.

**Layer 3 — Semantic cache (impersonal only, gated).** Embed the normalized query,
ANN-search the cache, accept a hit only if similarity ≥ a threshold *calibrated on
labeled data*, not chosen by intuition. The calibration procedure matters more than
the threshold: sample a few thousand near-duplicate pairs, have them graded for
"would the same answer be correct for both," and pick the threshold at your
tolerated false-hit rate (for a health product: very low, so a conservative
threshold and a lower hit rate). Log every semantic hit with both queries so the
threshold can be audited and re-tuned.

Guard against the classic failure: high embedding similarity between *opposite*
questions ("is X safe during pregnancy" vs. "is X unsafe during pregnancy"). Add a
cheap negation/quantity check that vetoes hits when the two queries differ on
negation tokens, numbers, or units.

**Layer 4 — Retrieval cache (all traffic, including personal).** Cache the
document-retrieval step keyed on `(normalized_query, kb_version)` — the retrieved
*nutrition facts* are not personal even when the question is. This gives latency and
cost savings on the 18% personal traffic where response caching is forbidden.

**Never cached:** anything whose context includes another user's data — enforced by
the personal/impersonal gate; anything where the answer includes a
time-relative claim ("today", "this week") unless the time bucket is in the key.

**Expected result:** prefix caching alone typically yields 25–40% input-token
savings across all traffic. Exact + semantic on the impersonal 82% adds substantial
call elimination. The 40% total cost reduction target is reachable primarily
through layers 1 and 2, which are the two lowest-risk layers — a good sign that the
plan isn't betting on the dangerous part.

**Compliance:** `kb_version` in the key means a guideline update invalidates every
affected cached answer atomically. Cap TTL at 24h regardless, so the blast radius
of a missed invalidation is bounded.

</details>

### Trade-offs

**Aggressive semantic caching (low threshold):** big savings, big hit rate, and a
steady trickle of subtly wrong answers that are extremely hard to detect because
they look plausible. In a health context this is a product risk, not a tech-debt
item.

**Conservative caching (exact only):** provably safe, lower savings, and leaves the
24% near-duplicate traffic on the table.

**No response caching, only prefix + retrieval caching:** zero semantic risk,
moderate savings, and it scales with traffic without accumulating a correctness
liability. For regulated or high-stakes domains this is often the right permanent
answer, not a stepping stone.

The general rule to articulate: **cache the deterministic parts of the pipeline
aggressively and the generative parts conservatively.**

### What Could Go Wrong?

- **Cache serves across a permission or personalization boundary** — the incident
  that already happened. Prevented only by making the personal/impersonal gate
  structural.
- **Stale answers after a guideline change.** Missed invalidation → contradicting
  your own updated content. Bounded TTL is the safety net.
- **Cache stampede.** A popular query expires and 4,000 concurrent requests all
  miss and call the provider. Use request coalescing / single-flight.
- **Hit-rate collapse from a harmless-looking change.** Adding a timestamp or a
  session ID into the prompt (or the key) drops hit rate from 35% to 2% overnight
  and cost triples. This is a real and common outage. Alert on hit-rate drop as a
  first-class signal.
- **Semantic cache drift after an embedding model upgrade.** All stored vectors are
  now in a different space. Cache must be versioned by embedding model and flushed
  on change, or you'll serve nonsense matches.
- **Cached errors.** Never cache a failed or truncated generation. Cache only
  responses that passed validation.

### Metrics

Hit rate per layer; **false-hit rate** for the semantic layer (sampled and human-
graded — the only metric that keeps you honest); cost per DAU; p95 latency for hits
vs. misses; invalidation lag after a `kb_version` bump; stampede events; share of
traffic classified personal (drift here means the classifier broke); and
user-reported answer complaints segmented by cache-hit status — the ground truth on
whether caching is degrading quality.

### Follow-up Questions

1. Your semantic cache false-hit rate is 0.4%. Product says that's fine. What
   additional information do you need before agreeing, and what's your position if
   0.4% of 2.6M/day means 10,000 wrong health answers?
2. A regulator asks you to reproduce the exact answer given to a specific user on a
   specific date. Can you? What would you need to have built?
3. Traffic doubles but hit rate falls from 34% to 19%. Give three plausible causes
   and how you'd distinguish them.
4. How would caching change if 90% of users were on a paid tier with strict
   personalization expectations?
5. Would you cache tool-call results in an agentic version of this product? Which
   ones, and how do you decide?

### What Separates Senior From Staff?

Senior: builds the four layers correctly, calibrates the threshold with data, ships
the safety gate.

Staff: recognizes that "which personalization dimensions are cache-visible" is a
**product decision with cost consequences**, and brings product and legal into it
explicitly rather than choosing unilaterally in a config file. They also establish
false-hit rate as an SLO with an owner, because a cache's savings are visible on a
dashboard while its errors are invisible by construction — and a metric only one
side of which is measured will always drift in the wrong direction. And they'd note
that if 55% of traffic is duplicate questions, the *product* answer might be a
curated content library, not a cache.

### Interviewer Notes

This scenario is a strong discriminator because caching sounds boring and is
actually where multi-tenant AI systems break. Listen for: does the candidate
enumerate the dependencies of an answer before designing a key? Do they separate
prefix caching (safe) from semantic caching (risky) or lump them together? Do they
propose a *calibration procedure* for the similarity threshold or just pick 0.95?
Do they mention hit-rate collapse as an incident class? A candidate who volunteers
the negation-similarity failure mode has almost certainly shipped a semantic cache.

---

## 5. Centralized AI Platform vs. Team-Owned Stacks

### Situation

A 400-engineer product company. Over 18 months, six product teams independently
shipped LLM features. There is no platform team. The current state:

- Four different LLM providers, eleven API keys, no consolidated billing view.
  Finance can't attribute $480k/month of AI spend to products.
- Three vector databases. Two teams built their own retrieval on Postgres+pgvector,
  one uses a managed service, one built an in-memory index that reloads on deploy.
- Five different prompt-management approaches, including one team keeping prompts
  in a Google Doc that a PM edits.
- No shared evaluation. Two teams have no evaluation at all and ship on vibes.
- One security review has been done, for one product.
- Two teams have independently built (buggy) retry-and-rate-limit logic; one caused
  a provider-side quota exhaustion that took down another team's feature.

Leadership asks you to "create an AI platform." Two teams are enthusiastic; three
are indifferent; one — the highest-revenue product — is actively hostile and says a
platform will slow them down.

### Your Task

Propose what to centralize, what to leave alone, and how you'd sequence it. You
will be given three engineers.

### Constraints

- No mandate. You cannot force adoption; you report to Engineering, not to the
  product orgs.
- The hostile team ships weekly and has the CEO's attention.
- Any migration must be incremental — no product can pause feature work.
- The quota-exhaustion incident is your political capital. It expires.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

Sort the problems by a two-axis test: **how much does inconsistency here hurt the
company**, and **how much does standardization here constrain a product team?**

- High harm, low constraint → centralize first. (Cost attribution, rate-limit
  coordination, security review, provider credentials.)
- High harm, high constraint → standardize the *interface*, not the
  implementation. (Evaluation: mandate that every AI feature has a suite and
  reports a score; don't mandate the harness.)
- Low harm, high constraint → leave alone. (Prompt authoring workflow, chunking
  strategy, choice of reranker.)
- Low harm, low constraint → ignore. Don't spend capital here.

The failure mode of platform teams is inverting this: they build the fun thing (a
framework, an orchestration DSL) which is high-constraint, and skip the boring
thing (credential and quota management) which is where the actual harm is.

Second insight: **with no mandate, adoption must be earned by solving a problem the
teams already have.** The quota incident is that problem, today. The hostile team is
hostile to *process*, not to *not being paged at 3am*. Build the thing they'd
choose voluntarily.

</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Centralize now (weeks 1–8): the gateway, scoped narrowly.**

A single egress path for all provider calls. Not a framework — a thin proxy/SDK
that provides:

1. **Credential management.** One set of provider keys, held by the platform. Teams
   get a platform token. This alone fixes the eleven-keys problem and makes
   provider migration possible later.
2. **Quota coordination.** Per-team token and request budgets enforced centrally,
   so one team cannot exhaust another's capacity. This is the direct fix for the
   incident, and it's the pitch to the hostile team: *"this exists so the other
   teams can't take you down."*
3. **Cost attribution.** Every call tagged with team, product, feature, and
   environment. Finance gets its answer; teams get their own dashboard. Costs
   become a thing people can manage instead of a mystery.
4. **Correct retry, backoff, and fallback**, implemented once. Two teams have
   already written this badly; nobody wants to own it.
5. **Structured request/response logging** with a common schema, PII-scrubbed, so
   there is one place to answer "what did we send the model."

That is the entire v1. Note what is *not* in it: no prompt management, no
orchestration, no chunking opinions, no framework. Adoption cost for a team should
be changing a base URL and a key.

**Standardize the interface (months 3–6): evaluation.**

Do not build "the eval platform." Instead:

- Publish a **contract**: every production AI feature has a versioned eval dataset,
  a scoring function, and reports `score@version` to a central registry on every
  release. How they compute it is theirs.
- Provide a *reference* harness that's genuinely good, so most teams adopt it
  because it's easier than writing one — pull, not push.
- Make the registry visible: a dashboard showing every AI feature, its current
  score, and when it was last evaluated. Teams with no evaluation show as blank.
  Visibility does the work a mandate would.

**Leave alone, deliberately and loudly:** prompt authoring, chunking strategy,
retrieval implementation, model choice, framework choice. Say this out loud in the
kickoff. A platform that announces its boundaries is far less threatening than one
that doesn't, and the hostile team's objection is mostly about unbounded scope.

**Vector databases: converge by attrition, not migration.** Three vector stores is
genuinely wasteful, but forcing a migration is expensive, risky, and buys you a
year of resentment. Instead: publish a recommended default with real operational
support attached, and offer to *run* it. Teams migrate when their current choice
hurts — the in-memory index that reloads on deploy will hurt soon. Spend the
platform's scarce credibility on the gateway, not on this.

**Security: one review, one checklist, applied at the gateway.** Since all traffic
now flows through one path, prompt-injection logging, output filtering, and data
classification checks can be implemented once. This is the argument that makes
the gateway strategically important rather than merely convenient.

**Sequencing rationale:** the gateway is high-value, low-constraint, and its
adoption is nearly free — so it converts one incident's worth of political capital
into permanent structural leverage. Everything harder is attempted *after* you have
that leverage and a track record.

</details>

### Trade-offs

**Strong central platform (mandated stack, shared services, teams build on top).**
*Advantages:* consistency, cost control, one security posture, one place to upgrade;
juniors on product teams don't have to become AI infrastructure engineers.
*Disadvantages:* the platform becomes a bottleneck; product velocity depends on a
3-person team's roadmap; local optimization becomes impossible; the platform's
abstractions ossify around the first two use cases and fit nobody later.
*Choose when:* the domain is stable, use cases are similar, and there's a real
mandate with headcount behind it.

**Fully federated (teams own everything, platform publishes guidance).**
*Advantages:* maximum velocity, teams optimize for their own workload, no
bottleneck, no coordination overhead.
*Disadvantages:* exactly the current state — duplicated work, no cost visibility,
inconsistent security, cross-team blast radius, and a compounding integration debt.
*Choose when:* few teams, low spend, exploratory phase.

**Thin waist (chosen):** centralize the narrow layer everything must pass through;
federate everything above it.
*Advantages:* one integration point gives leverage on cost, security, quota and
observability without constraining design; adoption is cheap; scope is defensible.
*Disadvantages:* a shared component in the critical path of six products — its
availability is now everyone's availability, and a bad deploy is a company-wide
incident. Requires real operational maturity from a 3-person team.

### What Could Go Wrong?

- **The gateway becomes a single point of failure.** Six products down because the
  platform team shipped a bad config. Mitigation: SDK-with-direct-fallback rather
  than a mandatory proxy hop; the platform's own SLO published and monitored;
  progressive rollout.
- **Scope creep kills it.** Someone asks for prompt templating, then orchestration,
  then an agent framework. Eighteen months later there's a half-built internal
  LangChain that nobody wants. Write down what the platform does *not* do, and
  defend it.
- **The hostile team never adopts** and, being the revenue leader, legitimizes
  non-adoption. Mitigation: solve their problem specifically and first; get their
  name on the design doc; measure and publicize their benefit.
- **Cost attribution reveals something political.** The moment finance can see
  spend by product, someone's project looks bad. Expect this and decide in advance
  whether you're the messenger or the analyst.
- **Standardizing evaluation surfaces that two products are bad.** True, useful,
  and dangerous to the platform's popularity. Introduce the dashboard with scores
  private-to-team first, then aggregate, then public.

### Metrics

*Platform health:* gateway availability and added latency (p99 must be small —
this is your adoption tax); percentage of company AI traffic flowing through the
gateway (the real adoption metric); number of provider credentials outstanding
(should trend to one set).

*Company outcomes:* AI spend attributable to a product (target 100%); cross-team
quota incidents (target zero); number of production AI features with a live eval
score; time-to-first-LLM-call for a new team (a proxy for platform value); security
reviews completed per AI feature.

*Anti-metrics to watch:* if product teams' feature velocity drops after adopting
the platform, the platform is failing regardless of how good its dashboards are.

### Follow-up Questions

1. Six months in, adoption is at 60% and the hostile team still hasn't moved. What
   do you do — and what would you *not* do?
2. A team wants a provider you haven't integrated. They can ship it themselves in a
   day; through you it's three weeks. How do you handle this without either
   blocking them or losing the single-egress property?
3. The gateway has a 12-minute outage during a product launch. Write the follow-up
   plan, including the architectural change you'd make.
4. Leadership now wants to consolidate to one vector database and gives you the
   mandate you didn't have. Do you take it?
5. Two years out, what does this platform look like if it succeeded? What if it
   failed? Describe both concretely.

### What Separates Senior From Staff?

This scenario is mostly a staff-level scenario in disguise, and the difference
shows in what the candidate optimizes for. A senior engineer designs a good
gateway. A staff engineer designs a good gateway *and* a strategy for adoption
without authority: they identify the specific incident that creates the opening,
scope the first deliverable to be nearly free to adopt, name the hostile team's
real objection (unbounded scope, not the technology), and write down the platform's
boundaries as a commitment. They think in terms of **leverage per unit of political
capital**, sequence irreversible decisions late, and define what failure looks like
in advance. They also accept that convergence on vector databases may simply never
happen, and are at peace with that, because it wasn't where the harm was.

### Interviewer Notes

Ask this of anyone interviewing at staff level, and of senior candidates who claim
platform experience. The tell is scope: candidates who propose building a framework
in the first six months have not run a platform team with three people. Listen for
whether they distinguish "centralize the interface" from "centralize the
implementation" — it's the single most useful distinction in platform work. Also
listen for whether they treat the hostile team as an obstacle or as a customer;
the latter is the staff-level posture. A candidate who asks "what happens to my
headcount if adoption stalls?" is thinking about the right incentives.

---

## How to Practice With These

1. Read only **Situation**, **Your Task**, and **Constraints**. Close the file.
2. Spend 10 minutes writing your approach — including the numbers you'd ask for.
3. Open the reasoning path. Note where you jumped to a solution before measuring.
4. Read the strong answer and the trade-offs. Ask yourself the test at the top of
   this file: *what number would have to change for the other option to win?*
5. Answer the follow-ups out loud, to a wall or a friend. The follow-ups are where
   the interview actually happens.
