# Scaling

Scaling interviews for AI systems reward **quantitative reasoning under uncertainty**.
You will not know the exact throughput of a GPU you've never benchmarked, and you
aren't expected to. What you're expected to do is set up the calculation correctly,
carry the units, identify which term dominates, and say what you'd measure to replace
your estimate with a fact.

Three things that make LLM scaling different from ordinary service scaling, and that
candidates consistently miss:

1. **Requests are not units of work.** A request with 200 input tokens and one with
   20,000 input tokens differ by 100× in cost and in the load they place on a server.
   Any capacity model built on requests-per-second is wrong from the first line.
2. **Prefill and decode are different workloads sharing one machine.** Prefill is
   compute-bound and parallel; decode is memory-bandwidth-bound and sequential. They
   compete, and the ratio between them determines nearly everything about your
   serving behavior.
3. **Memory, not compute, is usually the binding constraint** — specifically KV cache
   memory, which grows with concurrent sequences × their lengths. When you run out,
   throughput doesn't degrade gracefully; it falls off a cliff as the scheduler starts
   preempting.

**Contents**

1. [500 Requests per Second: What Breaks First?](#1-500-requests-per-second-what-breaks-first)
2. [100× in Eight Months](#2-100-in-eight-months)
3. [The Embedding Backfill That Won't Finish](#3-the-embedding-backfill-that-wont-finish)
4. [Multi-Region Inference](#4-multi-region-inference)
5. [The Provider Rate Limit Wall](#5-the-provider-rate-limit-wall)
6. [Context Length Explosion](#6-context-length-explosion)

---

## 1. 500 Requests per Second: What Breaks First?

### Situation

You are designing a self-hosted inference tier for an internal AI assistant.

```
throughput requirement:   500 requests/second (sustained)
average input:            2,000 tokens
average output:             500 tokens
p95 latency requirement:  2 seconds (end to end)
model:                    a ~13B parameter model, served on modern data-center GPUs
```

### Your Task

Identify the bottleneck. Show your reasoning. Then say what you'd measure to confirm.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Step 1 — Convert requests into tokens, split by phase.**

```
prefill: 500 req/s × 2,000 tokens = 1,000,000 input tokens/second
decode:  500 req/s ×   500 tokens =   250,000 output tokens/second
```

Already the shape of the problem is visible: this workload is **4:1 input-heavy**.
Prefill will dominate compute; decode will dominate the latency the user perceives.

**Step 2 — Check the latency budget against decode, first.**

Decode is sequential: 500 output tokens must be generated one after another for a
given request. If the model produces, say, 40 tokens/second per sequence under load
(a plausible figure for a 13B model at high batch occupancy — the per-sequence rate
falls as batch size grows), then:

```
500 tokens ÷ 40 tokens/sec = 12.5 seconds
```

**The latency requirement is violated by more than 6×, before considering queueing.**

This is the answer. The bottleneck is not throughput — it's that **the p95 latency
requirement is incompatible with generating 500 output tokens**, unless per-sequence
decode speed is much higher than typical, which means much smaller batches, which
destroys throughput.

Stating this clearly is the point of the exercise: *the requirements as given are
mutually infeasible, and the first job is to say so and find out which one is real.*

**Step 3 — Now check throughput feasibility separately.**

Take rough order-of-magnitude figures for a 13B model on a modern data-center GPU with
continuous batching and paged attention:
- prefill: on the order of 10,000–30,000 tokens/second
- decode: on the order of 1,500–3,000 tokens/second aggregate across the batch

Then:
```
prefill:  1,000,000 ÷ ~20,000  ≈ 50 GPUs
decode:     250,000 ÷ ~2,000   ≈ 125 GPUs
```
Decode dominates the GPU count even though prefill dominates the token count, because
decode is memory-bandwidth-bound and far less efficient per token. **This inversion is
the single most useful intuition in inference capacity planning.**

So: on the order of 100–150 GPUs, with the exact number depending on batch composition,
quantization, and how much you're willing to sacrifice per-request latency for
aggregate throughput.

**Step 4 — Check KV cache memory, which is often the real wall.**

KV cache per token, roughly: `2 × num_layers × num_kv_heads × head_dim × bytes`.
For a 13B model with grouped-query attention and fp16, this lands somewhere in the
region of 100–200 KB per token of context (varies enormously by architecture — GQA
and MQA cut this dramatically).

At 2,500 tokens of context per sequence and, say, 150 KB/token:
```
per sequence: 2,500 × 150 KB ≈ 375 MB
```
On an 80 GB GPU with ~26 GB taken by fp16 weights, ~50 GB available for KV cache:
```
50 GB ÷ 375 MB ≈ 133 concurrent sequences per GPU
```
That's the concurrency ceiling per GPU, and it's what actually determines whether you
can batch enough to hit the decode throughput above. If your arithmetic says you need
more concurrency than memory allows, **memory is the bottleneck, not compute** — and
the fixes are different (quantization, shorter contexts, paged attention, offloading)
than the fixes for a compute bottleneck (more GPUs, better kernels).

**Step 5 — Say what you'd measure.** All of the above is estimation. What you actually
do is benchmark with *your* traffic profile: tokens/second at various batch sizes and
sequence lengths, time-to-first-token vs. concurrency, KV cache occupancy, and the
throughput/latency curve. Estimates set expectations; benchmarks set capacity.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**"The requirements are infeasible as stated, and here's which one has to give."**

Generating 500 output tokens in under 2 seconds requires ~250 tokens/second
*per sequence*, which is far above what a 13B model achieves at any batch size that
makes the economics work. So one of these must change:

**Option A — Change the latency requirement to time-to-first-token.** If the product
streams, the user sees output in a few hundred milliseconds and reads while the rest
generates. "p95 latency 2 seconds" almost always means "the user waits 2 seconds
staring at nothing," and streaming solves that without any capacity change. **This is
the first question to ask the product owner**, and in most cases it dissolves the
problem entirely.

**Option B — Reduce output length.** 500 tokens is ~375 words. Is that the product
requirement, or an artifact of an unconstrained prompt? Structured, concise outputs
of 150 tokens cut decode time by 70% and decode cost proportionally.

**Option C — Accept lower throughput per GPU** by running smaller batches for lower
per-sequence latency, and buy more GPUs. Compute the cost; usually this is a bad
trade compared to A or B.

**Option D — Use a smaller or more heavily-quantized model** for the latency-sensitive
portion, routing only the traffic that needs the bigger model.

**Assuming streaming is acceptable (Option A), the design:**

```
capacity plan
  · decode-bound: ~125–150 GPUs for 250k output tokens/sec
  · prefill headroom: sufficient, but schedule prefill and decode carefully —
    chunked prefill so a long prompt doesn't stall the decode batch
  · KV cache: ~133 concurrent sequences/GPU at 2.5k context; verify
  · provision to peak, not average; measure the diurnal curve

serving stack
  · continuous batching (do not use static batching)
  · paged attention for KV memory efficiency
  · prefix caching — with a 2,000-token input, much of it is likely shared
    system prompt; this is a large win on the prefill side
  · admission control: reject or queue at a bounded depth rather than
    accepting unboundedly and blowing latency for everyone
  · separate queues (or separate pools) for long-context requests so a
    20k-token prompt doesn't stall a batch of short ones
```

**Key architectural decisions to name:**

- **Prefix caching is the highest-value optimization here** given 2,000 input tokens,
  much of which is probably a shared prompt. It cuts prefill load substantially and
  improves time-to-first-token — the metric that now matters most.
- **Chunked prefill** prevents head-of-line blocking: without it, a single long prompt
  monopolizes the GPU and every concurrent decode stalls, spiking p99.
- **Admission control is not optional.** Under overload, an unbounded queue converts a
  throughput problem into a total outage, because every request times out after
  consuming resources. Bound the queue and fail fast.
- **Segregate by sequence length.** Mixed batches of very short and very long sequences
  waste memory and hurt both. Bucketing by length is a standard and effective move.

**What to measure to replace the estimates:**
tokens/second at various batch sizes; time-to-first-token vs. concurrency; KV cache
utilization and preemption rate; queue depth and time-in-queue; GPU utilization split
between prefill and decode; and the actual input/output token distributions in
production (the averages given here almost certainly hide a long tail that dominates
the p99).
</details>

### Trade-offs

**Throughput vs. latency, the central tension.** Larger batches → higher GPU
utilization and lower cost per token, but higher per-request latency (more sequences
sharing the same memory bandwidth). Smaller batches → better latency, worse economics.
There is no configuration that optimizes both; you pick a point on the curve and you
should be able to draw the curve.

**Bigger model vs. more replicas of a smaller one.** A larger model may need tensor
parallelism across GPUs, adding communication overhead and reducing efficiency; a
smaller model replicated N times is simpler, scales linearly, and fails more
gracefully. Prefer replication of a model that fits on one GPU whenever quality
allows.

**Provision to peak vs. autoscale.** GPUs take minutes to come up (image pull, weight
loading, warmup). Autoscaling on a 5-minute lag doesn't help a 90-second spike.
Provision to peak and use an API provider for overflow, or accept queueing.

### What Could Go Wrong?

- **Averages hide the tail.** "Average input 2,000 tokens" may mean p50 of 800 and p99
  of 30,000. Long sequences consume KV memory quadratically in aggregate effect and
  will cause preemption and latency spikes. Always plan against the distribution.
- **KV cache exhaustion → preemption thrashing.** When memory fills, the scheduler
  evicts and recomputes sequences, throughput collapses non-linearly, and the system
  enters a death spiral. Cap concurrency and maximum context length at admission.
- **Head-of-line blocking from long prompts** without chunked prefill: p99 latency
  becomes hostage to the longest prompt in flight.
- **Cold starts.** A GPU node loading 26 GB of weights takes minutes. Any scaling
  strategy that assumes fast scale-up is wrong.
- **Prefix cache thrash.** If prompts differ in their first tokens (a timestamp, a user
  ID at the top), prefix caching does nothing. Order prompts stable-first.
- **Overload behavior is untested.** Systems behave qualitatively differently at 110%
  of capacity than at 90%. Load-test past the knee and know what happens.

### Metrics

Tokens/second (prefill and decode, separately) per GPU and aggregate; time-to-first-token
p50/p95/p99; inter-token latency; end-to-end completion time; batch size distribution;
KV cache utilization and preemption/recompute rate; queue depth, time-in-queue, and
admission rejections; GPU utilization (and, more useful, memory-bandwidth utilization);
input and output token distributions in production; cost per million tokens by phase.

### Follow-up Questions

1. Traffic doubles overnight. Walk through what degrades first, second, and third.
2. Half your requests now have 32,000-token inputs. What changes?
3. Your p99 is 8× your p50. Give three plausible causes and how you'd distinguish them.
4. How would you decide between tensor parallelism across 2 GPUs and running 2
   independent replicas?
5. Your GPU utilization reads 95% but throughput is low. What's happening?
6. Design the admission control policy. What do you reject and what do you queue?

### What Separates Senior From Staff?

**Senior:** does the token arithmetic, identifies the decode bottleneck, sizes the
fleet, configures the serving stack correctly.

**Staff:** notices immediately that the stated requirements are infeasible and goes to
the product owner to find out which requirement is real — rather than engineering
heroically against a specification that was never validated. They reframe "p95 latency"
as "time-to-first-token" and, in doing so, remove the constraint that would otherwise
have cost 100 extra GPUs. They also insist on the token *distribution* before sizing
anything, because a capacity plan built on averages is a plan to be paged.

### Interviewer Notes

This is a calculation question and the calculation is the answer. Give the candidate a
whiteboard and see whether they carry units, split prefill from decode, and notice the
decode/latency contradiction. The single best signal is asking "does this stream?" —
it's the question that resolves the problem and very few candidates ask it. The second
is recognizing that decode dominates GPU count despite prefill dominating token count.
Don't penalize candidates for not knowing exact GPU throughput numbers; do penalize
them for not asking for the token distribution.

---

## 2. 100× in Eight Months

### Situation

Your AI product is a coding assistant embedded in an IDE plugin. It has been growing
steadily and a large enterprise deal just closed.

```
today                        in 8 months (projected)
─────────────────────────────────────────────────────
4,000 daily active users     400,000
~180k requests/day           ~18M requests/day
~2 req/sec average           ~210 req/sec average
~9 req/sec peak              ~950 req/sec peak
API provider, single region  ?
$21k/month                   ?
p95 latency 1.4s             must stay under 2s
```

The current architecture: a stateless service calling an LLM API, a Postgres database
for user state, a managed vector database for codebase indexing (per-user indexes),
and Redis for caching.

### Your Task

Produce a scaling plan. Identify what breaks at each order of magnitude.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**100× growth doesn't break everything at once — it breaks things at different
thresholds, in a predictable order.** The useful exercise is to walk the growth curve
and identify what fails at 3×, 10×, 30×, 100×. That structure is the answer.

**At 3× (~12,000 DAU):** probably nothing. Your provider's rate limits might start to
bind at peak. Costs triple to ~$65k/month, which is when someone starts asking
questions.

**At 10× (~40,000 DAU):**
- Provider rate limits become a hard wall at peak.
- Cost is ~$210k/month — now a budget line item with a stakeholder.
- The vector database's per-user index model starts to strain: 40,000 indexes is a lot
  of small indexes, and most managed vector databases price and perform poorly in that
  shape.
- Postgres connection counts from a scaled-out stateless tier become a problem.

**At 30× (~120,000 DAU):**
- Per-user vector indexes are now clearly the wrong architecture. 120,000 indexes,
  most of them tiny and idle.
- Cost ~$630k/month; the economics of self-hosting inference start to make sense.
- Single-region latency hurts users in distant geographies.
- Redis working set may exceed a single instance.

**At 100× (~400,000 DAU):**
- Everything above, plus organizational load: on-call, incident volume, and the fact
  that a single team cannot operate this.

**The key insight is that the *vector database architecture* is the thing that breaks
structurally**, while everything else breaks quantitatively. Rate limits are negotiable,
costs are optimizable, Postgres can be scaled — but "one index per user" is a design
that has a ceiling built into it, and changing it later means re-indexing every user's
codebase. That's the change to make early.

**Second insight: the enterprise deal changes the shape of traffic, not just the
volume.** 400k users concentrated in a few large enterprises means: heavily correlated
usage (everyone's workday starts at the same time), much larger codebases,
per-tenant data isolation requirements, and possibly VPC/residency requirements. The
peak-to-average ratio (950:210 ≈ 4.5:1) is already high and enterprise concentration
will make it worse.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Structure the plan as: fix the structural thing early, optimize the quantitative
things continuously, and prepare the organizational thing before you need it.**

**Now (months 1–2): instrumentation and the structural fix.**

1. **Per-request cost and latency attribution** by user, tenant, and request type. You
   cannot manage any of what follows without it, and adding it later is harder.
2. **Redesign the vector index architecture** before it's urgent. Move from
   per-user indexes to a tiered model:
   - Shared partitioned index for small codebases (the majority), partitioned by
     tenant so filtering is partition pruning rather than post-filtering.
   - Dedicated indexes only for large codebases.
   - Index only what's needed: most of a repository is rarely queried. Index on demand,
     evict cold indexes, and rebuild lazily — the working set is far smaller than the
     total.
   This is the change with the longest lead time and the biggest migration cost, so it
   goes first.
3. **Establish load-testing at 10× current traffic** as a routine practice, not a
   one-off. You'll do this repeatedly.

**Months 2–4: cost and rate-limit headroom.**

4. **Negotiate rate limits and committed-use pricing** with the provider now, on the
   strength of the enterprise deal. Provider quota increases take weeks and are much
   easier to obtain before you need them.
5. **Prefix caching and prompt discipline.** A coding assistant sends large amounts of
   repeated context (system prompt, file context, project conventions). This is the
   single largest cost and latency lever.
6. **Response caching** where safe: identical completions for identical contexts are
   common in coding assistance (boilerplate, common patterns). Key carefully — code
   context is per-tenant and often confidential.
7. **Route by task type.** Autocomplete, explain, refactor, and generate-tests have
   wildly different requirements. Autocomplete needs sub-300ms and can use a small
   model; refactoring can be slower and needs a strong one. **A coding assistant is
   one of the clearest routing cases there is because the user's action tells you the
   task.**
8. **Add a second provider** and keep 2–5% of traffic on it continuously.

**Months 4–7: capacity architecture.**

9. **Evaluate self-hosting for the autocomplete path.** Autocomplete is high-volume,
   short-input, short-output, latency-critical, and served well by a small model — the
   ideal self-hosting candidate. If it's 70% of requests, moving it self-hosted
   transforms the cost curve. Follow the shadow-mode migration pattern.
10. **Multi-region** for latency, once the enterprise footprint is known. Start with
    read-path components (vector search, caches) close to users; inference can follow.
11. **Autoscaling and admission control** on the stateless tier; connection pooling
    (PgBouncer or equivalent) in front of Postgres before connection counts bite.

**Months 6–8: organizational.**

12. **On-call, runbooks, SLOs.** At 400k users this is a tier-1 system. Define the SLOs
    now and instrument against them.
13. **Per-tenant isolation and quotas** so one enterprise's batch usage can't degrade
    another's interactive experience.
14. **Capacity review cadence** — a monthly forecast against actual, so the next
    surprise is a small one.

**The sequencing principle to state explicitly:** *do the irreversible and
long-lead-time things first (index architecture, provider negotiation, self-hosting
evaluation), and the reversible optimizations continuously.* Most scaling failures are
failures of lead time, not of engineering.

</details>

### Trade-offs

**Scale the current architecture vs. re-architect.** Scaling what you have is faster
and less risky, but per-user vector indexes have a hard ceiling. Re-architecting mid-
growth is painful; re-architecting after you hit the ceiling is worse. The judgment
call is *when*, and the answer is "before the ceiling, when you still have slack."

**Self-host vs. stay on the API.** At $2M+/month projected, self-hosting the
high-volume path is compelling. But it adds an operational burden precisely when the
team is stretched by growth. Splitting — self-host the uniform high-volume path, API
for everything else — captures most of the value at a fraction of the risk.

**Provision for peak vs. degrade at peak.** A 4.5:1 peak ratio means provisioning for
peak wastes most of your capacity most of the time. Alternatives: API overflow for the
top of the curve, or graceful degradation (smaller model at peak). Both are better than
paying for 4.5× idle capacity.

### What Could Go Wrong?

- **The growth projection is wrong.** It usually is, in one direction or the other.
  Build for 10× with a clear path to 100×, rather than building for 100× and
  discovering you needed 5×. Keep decisions reversible where you can.
- **The enterprise deal changes requirements, not just volume** — VPC deployment, data
  residency, SSO, audit logging, a dedicated tier. These can dominate the roadmap and
  they're not in the traffic numbers.
- **Correlated peaks.** Enterprise users all start at 09:00 in their timezone. The peak
  is sharper and less predictable than a consumer curve.
- **Per-tenant noisy neighbors.** One customer indexing a monorepo saturates your
  embedding pipeline. Quota per tenant from the start.
- **The team doesn't scale.** 100× traffic with the same three engineers and no on-call
  rotation ends in burnout and a bad incident. This is a real scaling constraint and it
  belongs in the plan.
- **Cost growth outruns revenue growth.** At 100× users, is cost per user flat, rising,
  or falling? If enterprise users are heavier than consumer users, unit economics can
  degrade while revenue grows. Model it.

### Metrics

Requests and tokens per second (average and peak, and the peak/average ratio); cost per
DAU and per tenant; p95/p99 latency by task type; provider rate-limit headroom
(percentage of quota used at peak — alert well before 100%); vector index count, size
distribution, and query latency by index size; Postgres connection count and query
latency; cache hit rates; error and retry rates; and a monthly capacity forecast vs.
actuals.

### Follow-up Questions

1. Growth stalls at 40,000 DAU instead of 400,000. What did you over-build, and what
   was still worth doing?
2. The enterprise customer requires deployment in their VPC. How does that change
   everything above?
3. Your provider caps you at 60% of the rate limit you need for peak. What's your plan
   for next Tuesday?
4. Design the per-tenant quota system. What happens when a tenant exceeds it?
5. At what point would you hire, and for what roles?
6. How do you load-test a system whose bottleneck is a third-party API?

### What Separates Senior From Staff?

**Senior:** produces the phased plan, identifies the breaking points, executes the
architecture changes competently.

**Staff:** distinguishes structural limits from quantitative ones and sequences by
lead time rather than by urgency — doing the vector index redesign while it's cheap
rather than when it's on fire. They engage the provider on quota and pricing eight
months early, because that's a relationship with a calendar. They surface that the
enterprise deal brings requirements, not just load, and get those into the roadmap
before they're emergencies. And they name the organizational scaling problem — team
size, on-call, SLOs — as part of the technical plan, because a system nobody can
operate is not scaled.

### Interviewer Notes

The best answers walk the growth curve and identify what breaks at each order of
magnitude, rather than designing the end state directly. That structure demonstrates
they've done this. Watch for whether they identify the per-user index model as the
structural problem — it's the only thing in the architecture with a built-in ceiling.
Watch also for whether they mention provider rate limits and negotiate early; it's a
weeks-long process that catches teams off guard constantly. A candidate who raises team
scaling and on-call is showing operational maturity.

---

## 3. The Embedding Backfill That Won't Finish

### Situation

You need to embed **900 million text chunks** for a new retrieval system. You started
the backfill 11 days ago. Progress:

```
day 1–3:    ~14M chunks/day
day 4–7:    ~9M chunks/day
day 8–11:   ~4M chunks/day
completed:  ~97M chunks (11%)
```

At the current rate, completion is in roughly 200 days. You have six weeks.

Additional facts:

- The embedding model is self-hosted on 8 GPUs.
- Source documents are read from object storage; chunking happens in the same worker.
- Vectors are written to a vector database that is also serving production queries for
  an existing (smaller) index.
- GPU utilization is reported at 31%.

### Your Task

Diagnose why it's slowing down and get it done in six weeks.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**31% GPU utilization with a decelerating rate is the whole story: the GPUs are not
the bottleneck, and something is getting progressively worse.**

Deceleration over time is the key signal. A steady-state bottleneck gives a flat rate.
A *degrading* rate means something is accumulating. Candidates:

1. **The vector database is degrading as the index grows.** Insert throughput into an
   ANN index typically falls as the index grows (graph construction gets more expensive
   in HNSW; IVF needs retraining). This is the most likely cause and it fits the curve
   precisely.
2. **Contention with production queries.** The same database is serving live traffic;
   as the index grows, memory pressure increases, and both insert and query performance
   degrade — possibly harming production too, which nobody has checked.
3. **Object storage read throughput** hitting a rate limit or a hot-partition problem
   (many object stores throttle by key prefix).
4. **Worker-side memory leak or growing batch state** slowing the pipeline over time.
5. **Chunk size distribution changing** — if the corpus is processed in an order that
   correlates with document size, later chunks may be longer. Check.

**The diagnostic move: instrument the pipeline stage by stage.** Time spent in read,
chunk, embed, and write, per batch, over time. The stage whose time is growing is the
answer. Without this you're guessing.

**Second observation: the target rate.** 900M chunks in 6 weeks = 900M / 42 days ≈
**21.4M chunks/day ≈ 250 chunks/second.** You've never exceeded 14M/day. So even
fixing the deceleration isn't enough — you need roughly 1.5× your best-ever rate.
That means both a fix *and* a capacity increase.

**Third: question the requirement.** Do all 900M chunks need to be embedded before
launch? Common alternatives:
- Embed the most-accessed subset first and backfill the tail lazily.
- Embed on first access for cold documents.
- Reduce chunk count (are you chunking too finely? 900M chunks from how many
  documents?).

The best scaling answer is frequently "do less work," and it should be on the table
before you buy more GPUs.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Step 1 — Instrument, then fix the deceleration (days 1–3).**

Add per-stage timing. Expect to find that the vector-database write stage has grown
from a small fraction of batch time to the dominant one.

If that's confirmed, the fix is to **decouple embedding from indexing**:

```
before:  read → chunk → embed → insert into live ANN index
after:   read → chunk → embed → write vectors to object storage / parquet
                                        │
                                        └─► bulk index build (offline)
```

Reasons this is the right architecture, not just a workaround:
- **Bulk index construction is dramatically faster than incremental insertion** for
  most ANN index types. Building an index from a complete dataset in one pass avoids
  the incremental graph-maintenance cost entirely.
- It removes contention with production queries.
- The embedded vectors become a durable artifact: if the index build fails, or you
  want to rebuild with different parameters, you don't re-embed anything. **This alone
  justifies the change** — re-embedding 900M chunks because of an index parameter
  mistake is a catastrophe you've now made impossible.

**Step 2 — Fix GPU utilization (days 2–5).**

31% utilization means the GPUs are starved. Likely causes and fixes:
- **Synchronous pipeline:** the worker reads, chunks, embeds, and writes in sequence,
  so the GPU idles during I/O. Fix: decouple stages with queues, and prefetch. A
  producer/consumer pipeline with a deep enough buffer keeps GPUs saturated.
- **Small batches:** embedding models are throughput-oriented; batch sizes should be
  large (hundreds of sequences), sorted by length to minimize padding waste.
- **Padding waste:** if chunks vary in length and you batch naively, you may be
  computing on 50% padding. Sort by length within a batch window.
- **Tokenization on the critical path:** CPU-bound tokenization can starve the GPU.
  Parallelize it across CPU workers.

Getting from 31% to 85% utilization is a ~2.7× throughput gain with zero additional
hardware. **Do this before adding GPUs**; adding hardware to an underutilized pipeline
multiplies the waste.

**Step 3 — Scale out (days 3–7).**

With the pipeline fixed, compute what you need:
- Measure actual chunks/second/GPU at high utilization.
- Target 250 chunks/second sustained, with margin for failures — plan for ~350.
- Scale GPU count accordingly. If 8 GPUs at 85% give you 120 chunks/sec, you need
  ~24 GPUs. Rent them for six weeks; this is a temporary workload and should not
  drive permanent capacity.

**Step 4 — Reduce the work (parallel track).**

- **Deduplicate.** Content-hash chunks. Large corpora frequently have 5–20% exact
  duplicates (boilerplate, headers, repeated legal text, template content). Free
  savings.
- **Question the chunk count.** 900M chunks — from how many documents? If it's 900M
  chunks from 30M documents, that's 30 chunks/document, which may be over-chunking.
  Fewer, larger chunks with a reranker often retrieves better *and* costs less.
- **Prioritize.** Embed the corpus in order of expected query frequency so the system
  is useful at 40% completion rather than at 100%.

**Step 5 — Make it restartable and observable.**

- Checkpoint progress at the chunk-range level so a failure resumes rather than
  restarts.
- Idempotent writes keyed on `(chunk_id, embedding_model_version)`.
- A dashboard showing rate, ETA, per-stage timing, and error rate — so the next
  deceleration is caught on day 1, not day 11.

**Expected outcome:** decoupled writes remove the deceleration; pipeline fixes give
~2.7×; scaling GPUs gives the rest; dedup and prioritization provide margin. Six weeks
is achievable, with the offline index build as a separate, well-understood step at the
end.

</details>

### Trade-offs

**Incremental indexing during embedding vs. bulk build after.** Incremental means the
index is queryable as it fills (useful for progressive launch) but is far slower and
contends with production. Bulk build is much faster and produces a better-quality
index, but nothing is queryable until the end. *Choose bulk* unless progressive
availability is a genuine product requirement — and if it is, do both: bulk-build in
chunks and merge.

**More GPUs vs. better pipeline.** Pipeline fixes are free and multiply the value of
every GPU. Always fix utilization first. But there's a point where engineering time
costs more than rented GPUs; for a six-week deadline, spend a few days on pipeline and
then buy hardware.

**Full backfill vs. lazy/prioritized embedding.** Full backfill is simple and
complete. Lazy embedding launches faster and may never need to embed the cold tail —
but adds a cold-start path in production and makes coverage unpredictable, which is
bad for a retrieval product where "we don't have that document" is invisible.

### What Could Go Wrong?

- **The production index degrades** while you hammer the same database. Check
  production query latency now; you may already have an incident nobody attributed.
- **A failure at 80% with no checkpointing** costs you three weeks. Checkpoint first,
  before scaling anything.
- **Embedding model version drift.** If someone updates the model mid-backfill, you
  have a corpus with mixed embeddings — silently broken retrieval. Pin the version and
  record it per vector.
- **Object storage throttling** from 24 GPUs' worth of parallel reads hitting the same
  key prefix. Distribute reads across prefixes.
- **The bulk index build itself takes a week** and nobody planned for it. Benchmark it
  on a sample and put it in the schedule.
- **Cost surprise.** 24 GPUs for six weeks is real money; get it approved rather than
  discovered.

### Metrics

Chunks/second overall and per stage (read, chunk, tokenize, embed, write); GPU
utilization and memory-bandwidth utilization; batch size and padding ratio; queue
depths between stages; error and retry rates; checkpoint lag; projected completion date
updated continuously; production query latency on the shared database (to catch
collateral damage); and duplicate rate.

### Follow-up Questions

1. You fix the pipeline and reach 22M chunks/day, then it drops to 16M after four days.
   What's your first hypothesis?
2. The vector database vendor says bulk import is available but requires a specific
   file format and a full index rebuild. Does that change your plan?
3. Halfway through, the team wants to switch embedding models. What do you say?
4. How would you validate that the 900M embeddings are correct — not just present?
5. Design the incremental-update path for after the backfill. New documents arrive at
   200k/day.
6. What would you have designed differently on day one?

### What Separates Senior From Staff?

**Senior:** diagnoses the deceleration, decouples the write path, fixes GPU
utilization, scales out, and lands the deadline.

**Staff:** recognizes that writing embeddings to durable storage as an intermediate
artifact is not a workaround but the correct architecture — it makes index rebuilds,
parameter changes, and model migrations tractable forever, and it is the difference
between "we can re-index" and "we would have to re-embed." They question whether all
900M chunks are needed and whether the chunking strategy is right, because doing less
work beats doing work faster. And they build the observability that makes the *next*
large batch job diagnosable on day one, since this company will run many of them.

### Interviewer Notes

The deceleration is the clue and candidates should latch onto it: a degrading rate
implies an accumulating cost, which points at index insertion. The 31% GPU utilization
is the second clue and should prompt "the GPUs aren't the bottleneck." Watch for
whether the candidate computes the required rate (21.4M/day) and notices it exceeds
their best-ever throughput — many propose fixes without checking whether the fixes are
sufficient. The strongest candidates propose writing vectors to durable storage as an
intermediate artifact and explain why that's architecturally right rather than
expedient.

---

## 4. Multi-Region Inference

### Situation

Your AI product has grown into three major regions: North America (58% of traffic),
Europe (31%), and Asia-Pacific (11%). Everything runs in a single US region.

Symptoms:

- European users see p95 latency of 3.1s vs. 1.6s for US users.
- APAC users see 4.4s.
- A European enterprise customer has raised GDPR concerns about data leaving the EU.
- An APAC customer is asking for a data-residency commitment.

Architecture: stateless API tier, Postgres (single primary, US), managed vector
database (US), LLM API provider (US endpoints), Redis cache (US).

### Your Task

Design the multi-region architecture. Be specific about what is replicated, what is
partitioned, and what stays single-region.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**First, decompose the latency.** 3.1s for Europe vs. 1.6s for US is a 1.5s gap.
Transatlantic round-trip is ~80–120ms. So network latency alone does not explain a
1.5s gap unless there are *many* round trips.

Count them: client → API tier (1 RT), API tier → Postgres (1+), API tier → vector DB
(1+), API tier → LLM provider (1, but this one is long-lived for streaming), plus
retrieval possibly involving multiple sequential calls. Six sequential cross-Atlantic
round trips at 100ms each is 600ms; add TLS handshakes on cold connections and you're
closer to the observed gap.

**This reframes the problem: the fix is not "put everything in Europe" — it's "stop
making six sequential transatlantic calls."** Moving the API tier and the read path
close to users, while keeping some components central, may capture most of the benefit
at a fraction of the complexity.

**Second, separate latency from residency.** They're different requirements with
different solutions:
- *Latency* is solved by putting the read path near users.
- *Residency* is solved by ensuring specific data never leaves a jurisdiction, which
  is a much stronger and more expensive requirement — it constrains where data is
  stored, processed, backed up, and logged.

A candidate who conflates them will over-build for latency or under-build for
compliance.

**Third, classify the data.** Which data is:
- **Global and read-mostly** (product config, shared knowledge base) → replicate
  everywhere.
- **User-owned and region-affine** (a European customer's documents and embeddings) →
  partition by region, never replicate.
- **Global and write-heavy** (billing, cross-region analytics) → keep central, accept
  the latency, make it asynchronous.

This classification *is* the architecture.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Architecture: region-affine data partitioning with a global control plane.**

```
                    ┌──────── global (US) ────────┐
                    │  control plane:              │
                    │   · account/billing          │
                    │   · product config           │
                    │   · aggregate analytics      │
                    │   (async replication out;    │
                    │    never on the request path)│
                    └──────────────────────────────┘
                             │ (async, non-blocking)
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ── US region ──      ── EU region ──      ── APAC region ──
   API tier             API tier             API tier
   Postgres (primary    Postgres (primary    Postgres (primary
     for US tenants)      for EU tenants)      for APAC tenants)
   vector DB (US data)  vector DB (EU data)  vector DB (APAC data)
   Redis                Redis                Redis
   LLM: US endpoints    LLM: EU endpoints    LLM: APAC endpoints
```

**Key decisions:**

1. **Tenants are pinned to a home region.** Their documents, embeddings, conversation
   history, and caches live only there. This solves residency *and* latency in one
   move, because a request never leaves the region for anything on the critical path.
2. **Routing is by tenant, not by user location.** A European employee of a US company
   is served from US — because that's where their data is. Geo-routing to the nearest
   region would be wrong and would break residency. Route on the tenant's home region,
   resolved at the edge from the auth token.
3. **The control plane is global but off the request path.** Account state is
   replicated asynchronously into each region and read locally. If replication lags,
   requests still serve. Never make an authorization check a synchronous cross-region
   call.
4. **LLM provider endpoints must be regional.** This is the part that requires vendor
   negotiation: confirm that EU endpoints process and do not store data outside the EU,
   and get it contractually. A regional API endpoint that routes to US compute solves
   latency but not residency — verify which you're getting.
5. **Caches are per-region and never replicated.** A cache is a data store; replicating
   it across a residency boundary defeats the purpose.
6. **Cross-region operations are explicitly enumerated and made asynchronous**: usage
   reporting, aggregate analytics, model evaluation datasets (which must be
   de-identified or region-scoped), and admin tooling.

**Migration sequencing:**

- Phase 1: deploy the API tier and Redis in EU; keep data in US. Captures a portion of
  the latency win by reducing round trips (connection pooling, batching) but not
  residency. Fast and low-risk.
- Phase 2: EU Postgres and vector database; migrate EU tenants with a per-tenant
  cutover (read-only window, copy, verify, switch). Per-tenant migration means the
  blast radius is one customer at a time.
- Phase 3: EU LLM endpoints, with a per-region eval run to confirm behavioral
  equivalence (regional endpoints can serve different model versions — check).
- Phase 4: APAC, same pattern, informed by what went wrong in EU.

**What deliberately stays single-region:** batch analytics, the data warehouse (with
appropriate de-identification), model training/fine-tuning pipelines, and internal
tooling. Replicating these triples cost for little benefit, and they're not on the
request path.

</details>

### Trade-offs

**Full active-active replication everywhere.** Lowest latency for everyone, resilient
to regional failure. But it's the most expensive, requires conflict resolution for
writes, and is **incompatible with data residency** — replicating EU data to the US is
exactly what the customer is objecting to. Often the wrong answer for AI products
specifically.

**Region-partitioned (chosen).** Satisfies residency, good latency, cost scales with
regions not with replication factor. But no cross-region failover for a tenant's data
(a regional outage takes that region's tenants down), cross-region features are hard,
and tenant migration between regions is a real operation you'll need to build.

**Single region + edge caching/CDN.** Cheapest, simplest. Helps static content and
maybe cached responses. Does nothing for residency or for the dynamic path, which is
all of it. *Choose when* latency complaints are mild and residency isn't required —
not here.

**Regional read replicas, central writes.** Middle ground: read latency improves,
writes stay slow, residency is not satisfied (data still lives centrally). Useful as an
intermediate step, not an end state.

### What Could Go Wrong?

- **A cross-region call sneaks onto the critical path.** An auth check, a feature-flag
  lookup, a rate-limit counter. One synchronous transatlantic call undoes the entire
  project. Enforce with an architectural test: fail the build if a service in region X
  can reach a datastore in region Y synchronously.
- **Residency is violated by observability.** Logs, traces, and prompt payloads flow to
  a central observability system in the US. This is the most commonly missed residency
  leak and it's exactly the data the customer cares about. Regional log storage, or
  strict scrubbing.
- **Model behavior differs between regional endpoints** (different versions, different
  safety configurations). Run per-region evals; don't assume equivalence.
- **Tenant migration between regions** is requested (a customer relocates) and nobody
  built it. Design the tenant-move operation early.
- **Regional outage has no failover** for that region's tenants. Decide explicitly
  whether that's acceptable, and if not, design a residency-compatible DR strategy
  (e.g. a second EU region rather than failover to US).
- **Cost triples** because everything is now deployed three times, including
  under-utilized components. Deploy only what must be regional.

### Metrics

p95/p99 latency by region and by tenant; count of cross-region calls on the request
path (target: zero); residency compliance audit — for a sample of requests, prove no
data left the region, including logs; replication lag for the control plane; per-region
cost and utilization; per-region model eval scores (to catch behavioral drift between
endpoints); and tenant-to-region mapping correctness.

### Follow-up Questions

1. A customer with employees in all three regions wants low latency for all of them
   while keeping data in the EU. What do you tell them?
2. The EU region goes down. What's the customer experience, and what's your DR plan
   given residency constraints?
3. Your LLM provider doesn't offer an APAC endpoint. Now what?
4. How do you run a company-wide evaluation when the datasets can't leave their
   regions?
5. Design the tenant-migration operation between regions. What's the downtime?
6. How do you prevent a well-meaning engineer from adding a cross-region call next
   quarter?

### What Separates Senior From Staff?

**Senior:** designs the partitioned architecture, sequences the migration, handles the
data classification correctly.

**Staff:** separates latency from residency at the outset and solves them with
different mechanisms rather than over-building one to satisfy the other. They identify
observability as the residency leak nobody looks at. They build the *architectural
enforcement* — a test or a network policy that makes cross-region calls impossible
rather than discouraged — because a design that depends on everyone remembering will
be violated within two quarters. And they treat the LLM provider's regional guarantees
as a contractual question to be verified in writing, not an API endpoint to be
configured.

### Interviewer Notes

The decomposition of the 1.5s latency gap is the first test: candidates who assume
"network is slow" without counting round trips are not debugging. The second test is
whether they separate latency from residency — conflating them is the most common
error and leads to either over-engineering or non-compliance. The third is routing:
by tenant, not by user geography, and the reasoning behind that is the whole design.
Ask about logs; if they haven't thought about where trace payloads go, they'd have
shipped a compliance violation.

---

## 5. The Provider Rate Limit Wall

### Situation

Your document-processing product runs on an LLM API. Your limits:

```
8,000 requests/minute
4,000,000 input tokens/minute
800,000 output tokens/minute
```

You are hitting 429s during business hours. Analysis:

```
peak sustained:   6,200 requests/minute
                  4,300,000 input tokens/minute   ← over the limit
                    520,000 output tokens/minute
```

The provider says a limit increase will take 3–4 weeks and is not guaranteed. A
customer's 400,000-document backfill starts in nine days.

### Your Task

Get through the next nine days and design so this doesn't recur.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**The binding constraint is input tokens, not requests.** You're at 78% of the request
limit and 108% of the input-token limit. Any mitigation that reduces request count
without reducing input tokens (batching more work per request, for instance) may make
things *worse*, since larger requests carry more tokens.

This is the first thing to say, and it eliminates a category of wrong answers.

**Reduce input tokens — the direct attack:**
- What's in the 4.3M tokens/minute? Decompose per request: system prompt, retrieved
  context, document content. If the system prompt is 1,200 tokens repeated across
  6,200 requests/minute, that's 7.4M... no, 6,200 × 1,200 = 7.4M/minute, which
  exceeds your entire budget. So either the request rate is lower than the peak
  figure or the prompt is smaller — but the point stands: **shared prompt content is
  probably a large fraction of your token consumption.**
- Prefix caching: check whether cached input tokens count against the limit. With many
  providers, cached tokens are billed differently but may still count toward rate
  limits — **verify, because it changes the strategy entirely.**
- Reranking to reduce retrieved context: the standard lever.
- Truncation and context scoping.

**Smooth the demand — the scheduling attack:**
- Peak is what breaks you. If some of the work is asynchronous (and in a
  document-processing product, most of it is), shift it off-peak. You have a
  4M-tokens/minute budget for 1,440 minutes/day — that's 5.76B tokens/day. Your
  problem is distribution, not volume.
- A token-aware scheduler that admits work against a rolling budget is the durable fix.

**Add capacity — the procurement attack:**
- A second provider, even at worse quality or price, for overflow. Nine days is enough
  to integrate one if the gateway abstraction exists.
- Multiple accounts/projects with the provider, if permitted (check the terms — some
  providers prohibit this and it can get you suspended).
- Escalate the limit increase commercially: a 400k-document customer backfill is a
  revenue argument, and providers move faster for revenue than for tickets.

**Then the backfill specifically:** 400,000 documents in a customer backfill is
*asynchronous by nature*. It should never compete with interactive traffic. This is a
prioritization problem more than a capacity problem.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Days 1–2: stop the bleeding and measure precisely.**

1. **Implement a client-side token-aware rate limiter.** Right now you're discovering
   the limit by getting 429s, which wastes requests and triggers retries that make it
   worse. Track a rolling window of input tokens sent and admit requests against
   `limit × 0.9`. Estimate tokens before sending. This converts hard failures into
   controlled queueing.
2. **Separate lanes: interactive vs. batch.** Interactive requests get priority
   admission; batch work fills the remainder. One weighted queue, not two pools.
3. **Instrument token consumption by workload**, so you know where the 4.3M is going.

**Days 2–5: reduce input tokens.**

4. **Prefix caching** — and confirm with the provider whether cached tokens count
   toward the rate limit. If they don't, this is an enormous immediate win. If they do,
   it still saves cost but not quota.
5. **Rerank and trim retrieved context.** Retrieve 40, rerank, send 6. Typically a
   50–60% input-token reduction with no quality loss.
6. **Audit the prompt** for accumulated dead weight. Long-lived prompts collect rules;
   ablate and measure.
7. **Right-size per task.** Does the classification step need the full document, or the
   first page?

Realistic combined effect: 4.3M → ~2M input tokens/minute at the same request rate.
That puts you at 50% of the limit with headroom for the backfill.

**Days 3–7: capacity and scheduling.**

8. **Integrate a second provider** behind the gateway for overflow. Route the least
   quality-sensitive workload there (e.g. the classification step, not the final
   extraction), evaluate it, and keep a small share of traffic on it permanently so it
   stays warm.
9. **Build the backfill scheduler.** 400,000 documents against a nightly window: if
   off-peak hours (say 20:00–06:00) give you 10 hours at 3M tokens/minute of spare
   capacity, that's 1.8B tokens/night. At ~4,000 input tokens/document that's 450,000
   documents/night — the backfill can complete in a single overnight window without
   ever touching business hours. **Do this arithmetic and show the customer the
   schedule.**
10. **Negotiate the limit increase commercially**, with the backfill as the business
    justification, in parallel.

**Day 8–9: rehearse.** Run 10% of the backfill through the scheduler during a business
day and confirm interactive latency and 429 rate are unaffected. Do not discover the
interaction on the customer's start date.

**The durable design:**

- **All provider calls go through a gateway** that enforces token-aware limits,
  priority lanes, and per-workload quotas.
- **Rate-limit headroom is a monitored SLI** with alerting at 70% and 85% — you should
  never learn about a limit by hitting it.
- **Every provider has a configured secondary** with continuous traffic.
- **Capacity forecasting** as a monthly exercise: projected token consumption vs.
  contracted limits, with lead time for increases.

</details>

### Trade-offs

**Client-side rate limiting vs. reacting to 429s.** Proactive limiting wastes no
requests and gives you control over which work gets shed; it requires accurate token
estimation and adds complexity. Reactive handling is simpler but you burn quota on
rejected requests and your retry logic can amplify the problem.

**Second provider vs. optimizing the first.** A second provider adds real capacity and
resilience but doubles the evaluation, prompt-maintenance, and monitoring surface.
Optimization is cheaper and improves cost too, but has a ceiling.

**Batch scheduling vs. more capacity.** Scheduling is free and uses capacity you're
already paying for; it requires the work to be deferrable and adds a scheduler to
operate. More capacity is simpler and costs money. For asynchronous work, scheduling
wins almost every time.

### What Could Go Wrong?

- **Retry storms.** 429s trigger retries which consume quota which cause more 429s. If
  your backoff lacks jitter, all clients retry in sync. This is self-inflicted and
  common.
- **Token estimation is wrong** and your client-side limiter admits too much (429s
  continue) or too little (wasted capacity). Reconcile estimates against the provider's
  reported usage and correct the estimator.
- **The backfill starves interactive traffic anyway** because the lane weighting is
  implemented as "batch runs when interactive queue is empty," which never happens.
  Use explicit weights.
- **The second provider's quality is unevaluated** and you route customer-visible work
  to it during a capacity crunch. Evaluate before you need it.
- **Multiple accounts violate the provider's terms** and your access is suspended
  mid-backfill. Read the contract.
- **The limit increase arrives** and the team removes the rate limiter because "we
  don't need it now." Keep it; the next wall is closer than it looks.

### Metrics

Rate-limit headroom as a percentage of each limit (requests, input tokens, output
tokens) — three separate SLIs, because any one can bind; 429 rate and retry count;
queue depth and time-in-queue per lane; token estimation error (estimated vs. actual);
backfill progress and projected completion; interactive p95 latency during backfill
windows (the number that proves isolation works); and secondary-provider traffic share.

### Follow-up Questions

1. The provider denies the limit increase. What do you tell the customer?
2. Your token estimator is 15% low. What breaks, and how do you fix it?
3. The customer wants the backfill done in 24 hours, not seven nights. Options?
4. How do you rate-limit across 20 application servers without a central bottleneck?
5. The provider changes their limits with 48 hours' notice. How resilient are you?
6. Design the priority scheme. What are the classes and what happens under pressure?

### What Separates Senior From Staff?

**Senior:** identifies input tokens as the binding constraint, implements token-aware
limiting and lanes, reduces context, schedules the backfill, lands it.

**Staff:** treats provider quota as **a managed resource with a supply chain** —
forecast, lead time, commercial relationship, secondary supply — rather than as an
environmental constraint to be worked around. They get the limit-increase conversation
started through the commercial relationship on day one, because that's a 3–4 week lead
time that only a person can shorten. They make rate-limit headroom a standing SLI so
this never again arrives as a surprise. And they push the gateway pattern as company
infrastructure, because every team will hit this wall.

### Interviewer Notes

The first test: does the candidate notice that input tokens, not request count, are the
binding constraint? It's stated in the numbers and half of candidates miss it, then
propose batching, which makes it worse. Second: do they do the off-peak capacity
arithmetic and discover the backfill fits comfortably in unused nightly capacity? That
calculation turns a crisis into a scheduling exercise. Third: do they mention retry
amplification? It's the mechanism by which rate-limit problems become outages.

---

## 6. Context Length Explosion

### Situation

A customer-support assistant has been in production for a year. Conversations are
multi-turn; the assistant retrieves relevant knowledge-base articles and prior tickets.

Over 12 months:

```
                          launch      now
average conversation      3.2 turns   7.8 turns
average context sent      2,100 tok   14,600 tok
p95 context sent          4,400 tok   47,000 tok
p99 context sent          6,100 tok   118,000 tok
cost per conversation     $0.019      $0.240
p95 latency               1.3s        4.9s
```

Nobody made a decision that caused this. It grew.

### Your Task

Diagnose the growth and design context management.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Context growth is almost always the sum of several independent decisions, none of
which looked significant.** Enumerate the likely contributors:

1. **Conversation history accumulates.** 7.8 turns × (user message + assistant
   response + any tool results) grows linearly, and if every turn's *retrieved context*
   is also retained, it grows quadratically.
2. **Retrieved context per turn grew.** Someone raised `top_k` from 5 to 10 to fix a
   recall complaint. Nobody lowered it back.
3. **The system prompt accumulated rules.** Every incident, every edge case, every
   product launch added a paragraph. A year of this is thousands of tokens.
4. **Tool schemas grew** as tools were added.
5. **Retrieved documents got longer** — the knowledge base grew, articles got more
   detailed, or the chunking changed.
6. **Nothing was ever removed**, because removing context feels risky and there was no
   measurement to justify it.

**The p99 is the alarming number.** 118,000 tokens means some conversations are sending
an enormous amount of context — likely long conversations where history was never
truncated. These are the requests that dominate cost (a small fraction of conversations
can be a large fraction of spend) and blow the latency tail.

**The key diagnostic:** decompose the context. What fraction of those 14,600 tokens is
system prompt, tool schemas, conversation history, and retrieved context? You cannot
manage what you haven't measured, and the answer usually surprises people.

**The key architectural insight:** context is a *budget* that must be allocated, not a
list that grows. The design question is "given 8,000 tokens, what is the best possible
context for this turn?" — a selection problem, not an accumulation problem.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Design: a context budget with explicit allocation and eviction.**

```
CONTEXT BUDGET: 8,000 tokens (a decision, enforced)

  system prompt + policies        1,200   fixed, stable prefix (cached)
  tool schemas                      400   fixed, stable prefix (cached)
  conversation summary              600   rolling, regenerated every N turns
  recent turns (verbatim)         2,400   last 3 turns, truncated if needed
  retrieved knowledge             2,800   top 5 after reranking
  user's current message            600   truncated with notice if longer
  ─────────────────────────────────────
  total                           8,000   hard cap, enforced before send
```

**The mechanisms:**

1. **A hard cap, enforced in code.** Every request is assembled by a context builder
   that cannot exceed the budget. If a component wants more, another must give it up,
   and the trade is explicit. This single change prevents recurrence permanently.

2. **Rolling conversation summarization.** After turn 3, older turns are compressed
   into a running summary (regenerated every 3 turns, cached). This converts linear
   growth into constant space. The summary must preserve: the user's actual problem,
   identifiers (order numbers, account IDs), decisions made, and things already tried —
   which is what a support agent would keep in their head. Test the summarizer against
   these specifically.

3. **Retrieve per turn, don't accumulate.** Retrieved context from turn 2 is not
   retained into turn 5. Retrieve fresh each turn based on the current question plus
   the summary. This is the change that eliminates quadratic growth, and it's the one
   most systems get wrong.

4. **Rerank aggressively.** Retrieve 40 candidates, rerank, keep 5. Typically better
   quality than keeping 15 unranked, at a third of the tokens.

5. **Stable prefix ordering.** System prompt and tool schemas first (cached), volatile
   content last. Preserves prefix caching, which after this redesign covers 20% of the
   context on every request.

6. **Handle the p99 explicitly.** A conversation that has gone 25 turns is a signal in
   itself — it's not going well. Options: escalate to a human, or start a fresh context
   with a summary and an explicit "let me make sure I have this right" recap. Both are
   better product behavior *and* cheaper than sending 118,000 tokens.

**Expected outcome:** average context 14,600 → ~7,000 (with prefix caching making ~1,600
of that nearly free); p99 bounded at 8,000 instead of 118,000; cost per conversation
$0.240 → ~$0.05; p95 latency substantially improved because prefill dominates
time-to-first-token.

**Critically: this must be evaluated, not assumed.** Cutting context can hurt quality.
Ship behind an eval gate with segmented scores (by turn count especially — the
regression risk is concentrated in long conversations), and canary it.

**Prevent recurrence:**

- The context budget is a code-enforced constant with an owner.
- Context composition (tokens by component) is a monitored metric with an alert on
  drift.
- Any change that increases a component's allocation requires taking it from another —
  making the trade-off explicit and visible in code review.
- A quarterly prompt audit: ablate accumulated rules and remove those that don't
  measurably help.

</details>

### Trade-offs

**Full history vs. summarized history.** Full history preserves everything and never
loses a detail; it also grows without bound and eventually degrades quality (models
attend poorly to very long contexts, and relevant details get buried). Summarization
bounds cost and often *improves* quality by removing noise, but risks losing a detail
that mattered — and summarization errors compound over turns.

**Larger budget vs. smaller.** More context can improve quality up to a point, past
which it adds cost and latency for no benefit and can actively hurt (the "lost in the
middle" effect). The right budget is an empirical question; measure quality against
budget size on your own eval set rather than assuming more is better.

**Retrieve per turn vs. accumulate.** Per-turn retrieval is fresh and bounded but may
lose context that was relevant three turns ago and isn't retrieved now. Accumulation
keeps everything and grows quadratically. A middle path — retrieve per turn but pin
a small set of "sticky" documents the conversation is clearly about — often works best.

### What Could Go Wrong?

- **Summarization drops an identifier** (an order number, a case ID) and the assistant
  asks for it again. Users find this maddening. Extract structured entities separately
  from the prose summary and always retain them.
- **Quality regresses on long conversations**, which is exactly where users are most
  frustrated. Segment your eval by turn count or you'll ship this blind.
- **Summarization cost is not free.** Regenerating a summary every 3 turns is an extra
  model call. Cache it and amortize; measure the net.
- **The budget is bypassed.** A new feature adds context "just this once" outside the
  builder. Enforce structurally — one code path, no exceptions.
- **Prefix caching breaks** because someone puts a timestamp in the system prompt.
- **Truncation is silent.** A user's 4,000-token pasted log gets cut at 600 and the
  assistant answers about the visible part. Tell the user, and design the truncation to
  keep the most informative portion.

### Metrics

Context tokens by component (the decomposition is the key metric); total context
p50/p95/p99; cost per conversation and per turn; quality **segmented by turn count**;
summarization quality (does the summary retain identifiers, decisions, and attempted
solutions? — measurable with a targeted test set); prefix cache hit rate; truncation
rate and truncation-triggered complaint rate; conversation length distribution and
escalation rate for long conversations.

### Follow-up Questions

1. Your context budget cuts quality 3 points on conversations longer than 10 turns.
   What do you do?
2. A customer pastes a 60,000-token log file. Design the handling.
3. Would a model with a 1M-token context window change your design? Should it?
4. How do you evaluate a summarizer whose failures are "lost a detail that mattered
   later"?
5. Context grew 7× in a year with no decisions. What process change prevents the next
   one?
6. How would you design this differently for an agent that takes 30 tool-calling steps?

### What Separates Senior From Staff?

**Senior:** decomposes the context, designs the budget and summarization, ships it with
segmented eval gates.

**Staff:** recognizes that the growth happened because **no one owned the context and
there was no mechanism that made growth visible** — and fixes the mechanism, not just
the number. The code-enforced budget where adding tokens to one component requires
taking them from another turns an invisible drift into an explicit trade-off at review
time. They also notice that a 25-turn support conversation is a product signal, not
just a cost problem, and connect the engineering fix to a better user outcome
(escalation). And they resist the "just use a bigger context window" answer, because
larger windows make the problem invisible rather than solving it.

### Interviewer Notes

The p99 of 118,000 tokens is the detail to probe: ask what those conversations look
like and whether the candidate connects long conversations to failing conversations.
The core insight to listen for is treating context as a budget to allocate rather than
a list to append to — that reframe is the answer. Watch for whether they propose
per-turn retrieval instead of accumulation (the fix for quadratic growth) and whether
they insist on segmenting eval by turn count. A candidate who suggests a
million-token-context model as the solution should be asked what that does to cost and
latency; the answer reveals whether they've thought about prefill.

---

## Scaling Cheat Sheet

**Always convert to the right unit.**
Requests/second is not a workload. Convert to prefill tokens/second and decode
tokens/second, separately. Then check memory.

**The three constraints, in the order they usually bind:**

1. **KV cache memory** → limits concurrency → limits batch size → limits throughput.
2. **Memory bandwidth** (decode) → limits tokens/second per sequence → limits latency.
3. **Compute** (prefill) → limits how fast you can ingest prompts.

**Quick estimates worth memorizing the *shape* of:**

```
KV cache per token   ≈ 2 × layers × kv_heads × head_dim × bytes_per_element
vector storage       = n_vectors × dimensions × 4 bytes  (fp32)
prefill time         ≈ input_tokens / prefill_throughput      (parallel)
decode time          ≈ output_tokens / per_sequence_decode_rate (sequential)
```

**Latency levers, in order of effect:**

1. Stream (changes the metric that matters to TTFT)
2. Reduce output tokens
3. Prefix caching (cuts prefill)
4. Reduce input tokens (rerank, trim, scope)
5. Smaller model / quantization
6. More capacity (last, because it's the expensive one)

**Throughput levers:**

1. Continuous batching (not static)
2. Fix pipeline utilization before adding hardware
3. Chunked prefill (prevents head-of-line blocking)
4. Length-bucketed batching (cuts padding waste)
5. Quantization (more concurrency per GPU)
6. Horizontal replication

**Things that break at scale in AI systems specifically:**

- Averages, everywhere. Plan against distributions.
- Per-entity resources (one index per user, one namespace per tenant).
- Anything unbounded: context, queues, retries, conversation length.
- Rate limits you didn't monitor headroom on.
- Batch and interactive work sharing a queue without weights.
- Retries without jitter (retry storms) and without circuit breakers.
- Cold starts, wherever autoscaling is assumed to be fast.

**The question to ask before any capacity plan:** *what is the distribution of input
and output tokens?* Everything follows from that.
