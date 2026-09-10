# Multi-Region

Multi-region is three different projects that share a name, and the first job in any
discussion about it is finding out which one is being asked for:

- **Latency** — serve users from somewhere near them.
- **Availability** — survive losing a region.
- **Data residency** — some data legally may not leave a jurisdiction.

They have different designs, different costs, and different urgency. Residency is usually
a hard requirement with a deadline; the other two are trade-offs. Building all three at
once, because they were named together in one meeting, is how this becomes a year-long
program.

---

## Latency: mostly not your problem

For AI systems the network is rarely the bottleneck, and this is the correction worth
making early.

```python
def latency_budget(network_ms, retrieval_ms, generation_ms):
    total = network_ms + retrieval_ms + generation_ms
    return {"total_ms": total, "network_share": network_ms / total}

latency_budget(network_ms=150, retrieval_ms=200, generation_ms=3_000)
# total 3,350 ms — network is 4.5% of it
```

Cross-continent round trips cost roughly 150 milliseconds against a response that takes
three seconds to generate. Moving inference closer saves about 4% of perceived latency and
costs you a distributed system.

Two caveats that do justify regional deployment for latency:

- **Time to first token** is what a streaming user feels, and 150 ms against a 600 ms TTFT
  is 25%, not 4%. If TTFT is your metric, the network share is much larger.
- **Multi-turn or agentic flows** pay the round trip repeatedly. Ten sequential calls turn
  150 ms into 1.5 seconds of pure network.

So: compute the network share of the metric you actually care about before agreeing that
latency requires multi-region. Frequently the honest answer is that a CDN for static
assets and a single well-placed inference region is enough.

---

## Availability: the region isn't usually what failed

Multi-region for availability is expensive, and it defends against a failure mode that is
rarer than the ones that actually take AI systems down.

The realistic failure list, roughly in order of frequency:

1. **The model provider degrades or rate-limits you.** Not regional. The defense is a
   fallback chain — [fallbacks.md](../07-production/fallbacks.md).
2. **Your own deploy.** Not regional, and multi-region makes it worse by giving you more
   places for a bad version to be.
3. **A dependency fails** — vector store, database, queue. Sometimes regional.
4. **A zone fails.** Handled by multi-zone, which is far cheaper than multi-region and
   which you should have anyway.
5. **A whole region fails.** Real, and rare.

Multi-region only helps with the last two, and the fourth is handled by multi-zone. If
availability is the goal, do provider fallbacks and multi-zone first — they cover more of
the failure distribution for a fraction of the cost.

When you do build it, the expensive question is state:

- **Stateless inference** replicates easily. This part is not hard.
- **The vector index** must exist in every serving region. That's a full copy of the index
  per region, plus a replication path, plus the risk of versions diverging — and a query
  served from a region running an older index version is a correctness problem that looks
  like flakiness.
- **Conversation and session state** needs to be reachable, or sessions must be pinned to
  a region and fail over badly.
- **Caches are per-region**, so failover into a cold region arrives with a cold cache and
  the cost and latency profile that implies. Size for that, or the failover itself becomes
  the incident.

---

## Data residency: the one that's usually mandatory

This is the requirement that actually forces multi-region, and it's the least negotiable.
It also reaches further into the system than people expect.

The scope question — *what counts as the data?* — has a broader answer than the obvious
one:

- **The documents.** Obvious.
- **The embeddings.** Derived personal data, in scope. This is the point
  [08-ai-security/pii.md](../08-ai-security/pii.md) makes, and it means the vector index
  is a residency-controlled asset, not infrastructure.
- **The prompts**, which contain retrieved content and user input.
- **Logs and traces**, which contain the prompts. This is the one that gets missed most
  often — a centralized observability stack quietly exports everything to one region.
- **Evaluation datasets**, which are usually sampled from production traffic.
- **Any third-party model provider**, and where their inference physically runs.

The architectural consequence: a residency boundary is a full stack per region — index,
serving, cache, logs — not a routing rule in front of a shared backend.

Enforcement must fail closed, which is the same rule as
[authorization.md](../08-ai-security/authorization.md):

```python
def route_to_region(request, allowed_regions):
    region = request.data_residency_region
    if region is None:
        raise ResidencyError("no region determined — refuse rather than guess")
    if region not in allowed_regions:
        raise ResidencyError(f"no compliant capacity in {region}")
    return region
```

An unlabeled request must be refused, not sent to a default. A default region is how data
crosses a boundary silently, and silent is the only way this failure ever happens.

Two consequences worth stating in an interview because they surprise people:

- **You cannot pool capacity across residency boundaries.** Each region needs its own
  peak headroom, so utilization falls and cost per request rises — the arithmetic in
  [capacity-planning.md](../06-inference-serving/capacity-planning.md) applied once per
  region.
- **Residency and a global model provider can conflict outright.** If the provider has no
  compliant region, the requirement forces self-hosting, which changes the build-vs-buy
  calculation entirely. That's often the real finding.

---

## Routing, and what a region means for a single user

Decide deliberately, because the default is usually wrong for at least one of the three
goals:

- **Nearest region** — right for latency, wrong for residency.
- **Home region** — right for residency, and it means a European user in New York gets
  European latency. Correct, and worth telling the product team out loud.
- **Any healthy region** — right for availability, and it must be constrained by residency
  where residency applies.

When these conflict, residency wins, because it's the only one with a legal consequence.
Write that precedence down; it's the kind of rule that gets quietly inverted during an
incident when someone fails traffic over to keep the service up.

---

## What to monitor

- **Cross-region request rate**, which should be zero where residency applies. This is the
  alert that catches a boundary violation.
- **Requests refused for missing residency labels.** Non-zero means an upstream system
  isn't labeling, and someone will be tempted to add a default.
- **Index version per region**, which must match. Divergence produces
  region-dependent answers that read as flakiness.
- **Per-region utilization and headroom**, separately. Pooled numbers hide that one region
  is at capacity.
- **Failover time, tested**, and cache hit rate immediately after failover.
- **Replication lag** for the index and any shared state.

---

## Interview questions

**"Users in Europe say the product is slow. Should we deploy inference in Europe?"**

Probably not for that reason alone. A cross-continent round trip is around 150 ms against
multi-second generation — about 4% of total latency. I'd decompose their latency first,
and check time to first token specifically, since 150 ms against a 600 ms TTFT is a
quarter of it and does matter for streaming. If TTFT is the complaint, a regional
deployment helps; if total generation time is, the fix is in the model or the pipeline.

**"What does data residency actually require?"**

More than routing. Documents, embeddings, prompts, logs, traces and eval datasets all
contain the data, so a residency boundary is a full stack per region rather than a rule in
front of a shared backend. Logs are the one usually missed — a centralized observability
stack exports everything to one place. And enforcement has to fail closed: an unlabeled
request gets refused, never defaulted, because a default is how the boundary gets crossed
silently.

**"We want to survive a region outage. Where do you start?"**

By checking whether region outages are the failure mode actually hurting us. In most AI
systems the top causes are the model provider degrading and our own deploys, and
multi-region helps with neither — it makes the second worse. I'd do provider fallbacks and
multi-zone first, which cover more of the failure distribution far more cheaply, then
scope multi-region around the stateful parts: index replication, session state, and the
cold-cache cost of failing over.

---

## What to remember

- Three different projects share this name. Ask which one, and check whether residency —
  the only one with a legal deadline — is really the driver.
- Network is roughly 4% of a multi-second response, so latency rarely justifies it. Time
  to first token and multi-turn flows are the exceptions.
- For availability, provider fallbacks and multi-zone cover more failures for far less
  money.
- Residency covers embeddings, prompts, logs and eval data — not just documents. It means
  a full stack per region.
- Fail closed on unlabeled requests. A default region is how data crosses a boundary
  silently.
- You can't pool capacity across residency boundaries, so utilization drops and unit cost
  rises.
