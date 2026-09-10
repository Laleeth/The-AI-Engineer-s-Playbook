# Batch vs. Real-Time

A large share of AI work does not need to happen when the user asks. Deciding what runs
ahead of time and what runs on demand is one of the cheapest architectural levers
available, and it is routinely left on the table because "real-time" sounds like the
better answer.

It usually costs five to ten times more and delivers nothing the user notices.

---

## The cost difference, and where it comes from

Batch work is cheaper for reasons that have nothing to do with the model:

- **Batch size is unconstrained.** No latency target means you fill the batch, which is
  where throughput comes from — see
  [continuous-batching.md](../06-inference-serving/continuous-batching.md).
- **No headroom.** Interactive serving runs at 70–80% utilization to protect tail latency;
  batch runs at whatever the hardware sustains.
- **Interruptible capacity works.** Spot and preemptible instances are 60–75% cheaper and
  are fine for work that can be restarted.
- **Scheduling into troughs.** Batch fills the idle hours you're already paying for, which
  is the diurnal gap in [gpu-economics.md](../06-inference-serving/gpu-economics.md).

```python
def batch_vs_realtime_cost(items, tokens_each,
                           realtime_utilization=0.45, batch_utilization=0.90,
                           realtime_rate=2.00, batch_rate=0.80,
                           tokens_per_gpu_second=2_000):
    gpu_seconds = items * tokens_each / tokens_per_gpu_second
    return {
        "realtime": gpu_seconds / 3600 * realtime_rate / realtime_utilization,
        "batch": gpu_seconds / 3600 * batch_rate / batch_utilization,
    }

batch_vs_realtime_cost(items=10_000_000, tokens_each=300)
# realtime ≈ $1,852   batch ≈ $370
```

**Five times cheaper**, on identical work and the same model. The difference is
utilization and the price of interruptible capacity, and neither is visible in a
per-request cost comparison.

---

## What can move to batch

Ask of every model call: *does the answer depend on something that only exists at request
time?* If not, it is a candidate for precomputation.

Things that usually can move:

- **Embeddings for a corpus.** Already batch in any sane design.
- **Summaries, tags, categories and extracted fields** on content that exists before a
  user asks about it.
- **Contextual chunk annotations** — the technique in
  [contextual-retrieval.md](../03-rag/contextual-retrieval.md), whose $1,000 for 500,000
  chunks is a one-time batch cost precisely because it doesn't depend on the query.
- **Recommendation and ranking features** for a known user set.
- **Evaluation runs**, which have no user waiting at all and are frequently run
  interactively out of habit.

Things that cannot:

- Anything conditioned on the user's actual question.
- Anything that must reflect state as of now — inventory, prices, account status.
- Anything with a combinatorial input space, where precomputing means computing an
  infinity of answers nobody will request.

That last one is the real boundary, and it's where "just cache everything" fails.

---

## The middle: precompute the expensive part

Most valuable work isn't cleanly one or the other. The pattern is to split the request
into what is knowable in advance and what isn't, and precompute the first half.

RAG is the canonical example and it's worth noticing how much of it is already batch:
parsing, chunking, embedding and indexing all happen ahead of time. Only the query
embedding, the search, the rerank and the generation are real-time. Nobody calls RAG a
batch system, but the majority of its compute is.

Extending the same idea:

- **Precompute per-document analysis** — summaries, key entities, structured fields — so
  the real-time path retrieves rather than derives.
- **Precompute per-user context** on a schedule, so the request doesn't assemble it.
- **Cache the answers to questions people actually repeat.** Which is
  [caching.md](../07-production/caching.md), and where its warning applies: a cache key
  must contain everything the answer depends on, and omitting the permission scope is a
  data breach.

The design question that produces these: *what does this request compute that it could
have looked up?*

---

## Freshness is the constraint, and it's a product question

Precomputing means serving something derived from an earlier state. How much earlier is
acceptable is not an engineering decision, and engineers routinely make it by accident.

```python
def staleness_budget(recompute_interval_hours, change_rate_per_day):
    """Expected share of items serving a stale derived value."""
    return min(1.0, change_rate_per_day * recompute_interval_hours / 24)

staleness_budget(recompute_interval_hours=24, change_rate_per_day=0.02)   # 2% stale
staleness_budget(recompute_interval_hours=24, change_rate_per_day=0.50)   # 50% stale
```

Same daily schedule, wildly different outcomes, entirely determined by how fast the
underlying data changes. Two percent stale is usually fine; fifty percent means you have
built a system that is wrong half the time and calls itself a cache.

The measurement discipline from
[incremental-indexing.md](../09-data-pipelines/incremental-indexing.md) applies here
too: staleness is a distribution and the tail is what generates complaints.

Where staleness is unacceptable for a subset, the answer is usually hybrid rather than
abandoning precomputation — precompute for the slow-changing majority, compute on demand
for the fast-changing minority, and know which is which.

---

## Making the same code do both

The failure to avoid is two implementations of the same logic, because they drift and the
drift shows up as "the batch results don't match what the API returns."

```python
def derive(item):
    """One function. The only difference is who calls it and how many at a time."""
    return pipeline.run(item)

def realtime(item):
    return derive(item)

def nightly(items):
    for batch in chunked(items, 1_000):
        store.write([derive(i) for i in batch])
```

Differences that are legitimately real — concurrency, checkpointing, which capacity pool
it runs on, error handling — belong around `derive`, not inside it. This is the same
"one transformation path, two triggers" rule as
[ingestion-pipelines.md](../09-data-pipelines/ingestion-pipelines.md), and it is broken
for the same reason each time: the batch version gets written later, by someone in a
hurry, in a different repository.

---

## When batch is the wrong call

Reasons to keep something real-time even though it could be precomputed:

- **The input space is too large.** Precomputing answers to arbitrary questions is not
  precomputation, it's enumeration.
- **The precomputed set is mostly unused.** Deriving summaries for ten million documents
  when a thousand get read is worse than deriving on demand and caching. Compute the hit
  rate before committing.
- **Staleness is unacceptable and the data moves.** See above.
- **The operational cost exceeds the saving.** A scheduled job is a thing that fails at
  3am, needs monitoring, and drifts from the real-time path. For small volumes, don't.

That last one is a real judgment: batch trades unit cost for operational surface area.
At ten thousand items a day the saving is a few dollars and the cost is a pipeline
somebody maintains.

---

## What to monitor

- **Share of model calls served from precomputed results**, which is the lever's actual
  usage.
- **Precomputed hit rate** — how much of what you derived gets read. Low means you're
  paying to compute things nobody wants.
- **Staleness distribution** of served derived values, p50 through max.
- **Batch job completion time and its trend**, since a job that's grown past its window
  starts colliding with peak traffic.
- **Cost per item, batch vs. real-time**, so the lever's value stays visible when someone
  proposes removing it.
- **Divergence between batch and real-time outputs** on a sampled set — the early warning
  that the two paths have drifted.

---

## Interview questions

**"Your inference bill is dominated by summarizing documents when users open them. What
would you change?"**

Ask what fraction of documents get opened, and how often the same document is opened more
than once. If documents are read repeatedly, summarize once at ingestion — it's the same
work moved to batch, roughly five times cheaper per item on interruptible capacity at full
utilization, and it removes the latency from the user path entirely. If most documents are
never opened, precomputing everything is worse and I'd cache on first access instead. The
hit rate decides it.

**"When is real-time worth the premium?"**

When the answer depends on the request — the user's actual question, current state, or an
input space too large to enumerate. Also when the precomputed set would mostly go unread,
or when staleness is unacceptable and the data changes quickly. Otherwise real-time is
usually paying a five-times premium for latency nobody perceives.

**"You precompute summaries nightly. A customer says theirs is wrong."**

Staleness, most likely: they edited the document after the last run. The question I'd want
answered is what share of items are stale at serving time, which is change rate times
recompute interval — if the corpus turns over quickly a nightly job can be wrong for a
large fraction of the day. Fixes in order of cost: recompute on change for edited
documents, shorten the interval, or fall back to on-demand for recently-modified items.

---

## What to remember

- Batch is typically five times cheaper for identical work, from utilization and
  interruptible capacity rather than from anything about the model.
- The test is whether the answer depends on something that only exists at request time.
- Most systems are hybrids already. RAG precomputes nearly everything expensive and nobody
  calls it batch.
- Staleness = change rate × interval. Compute it; it's a product decision, not an
  engineering one.
- One derive function, two callers. Separate implementations drift and the drift is
  invisible until a customer finds it.
- Batch trades unit cost for a pipeline someone maintains. Below a certain volume that's a
  bad trade.
