# Load Balancing Across Replicas

Round-robin is the default in every load balancer you'll inherit, and it is close to the
worst choice for inference. It assumes requests are interchangeable and short. Inference
requests are neither: one takes 200 milliseconds, the next takes 40 seconds, and the
replica that would serve the second one fastest is the one that already has its prompt in
cache.

Getting this layer right is often worth more than adding replicas, which makes it a good
interview topic — it rewards understanding the workload rather than the tool.

---

## Why the usual algorithms fail here

| Algorithm | Assumption | Why inference breaks it |
|---|---|---|
| **Round-robin** | Requests cost about the same | Output lengths vary by 20× or more, so equal request counts mean wildly unequal load |
| **Least connections** | A connection is a unit of work | Closer, but a streaming request holds a connection for its whole generation regardless of remaining work |
| **Least response time** | Past latency predicts future load | Past latency mostly reflects the *output length* of finished requests, which says nothing about what's still running |
| **Random** | Load averages out | It does, slowly. With heavy-tailed request costs, the variance that matters is exactly what doesn't average out |

The common failure they share: none of them can see the thing that actually determines a
replica's spare capacity, which is how much KV cache is free and how many sequences are
already decoding.

---

## What to balance on instead

**Least outstanding tokens** is the best simple default. Track, per replica, an estimate of
the work still in flight rather than the number of requests:

```python
def outstanding_tokens(replica):
    """Work still to do, not requests still open."""
    return sum(
        max(0, r.max_tokens - r.tokens_generated)
        for r in replica.in_flight
    )

def choose_replica(replicas):
    healthy = [r for r in replicas if r.healthy and r.accepting]
    return min(healthy, key=outstanding_tokens)
```

This is an estimate — `max_tokens` is an upper bound and most requests finish well before
it — but it is a much better one than request count, and it degrades gracefully. Requests
that are nearly done stop counting against their replica, which is exactly the behavior
round-robin lacks.

**Queue depth reported by the replica** is better still when your serving layer exposes it,
because it reflects the replica's own admission decisions and its real KV cache pressure.
Balancing on a number the replica computes beats balancing on a number the proxy guesses.

---

## Cache-aware routing

This is the inference-specific one, and the one that surprises people.

If requests share a prompt prefix — a system prompt, a document being asked about across
several turns, a tenant's few-shot examples — then the replica that served the last such
request already holds those KV blocks. Routing the next one elsewhere throws that away and
pays full prefill again.

```python
def prefix_key(request):
    """Hash the stable head of the prompt: system prompt, tools, tenant preamble."""
    return hash(request.system_prompt + request.tool_definitions)

def route_with_affinity(request, replicas, load_threshold=0.8):
    preferred = consistent_hash(prefix_key(request), replicas)
    # Affinity is a preference, never a guarantee — shed it under load
    if preferred.load < load_threshold:
        return preferred
    return min(replicas, key=outstanding_tokens)
```

The saving is the whole prefill of the shared prefix. With a 2,000-token system prompt and
a 200-token user turn, a prefix cache hit removes about 90% of prefill work for that
request — which at scale is a large fraction of your prefill fleet.

Two rules keep this from backfiring:

- **Affinity must be soft.** A popular prefix — one tenant, one hot document — will
  overload its assigned replica if affinity is absolute. Always fall back to load-based
  routing above a threshold.
- **The prefix must be byte-identical.** A timestamp or a per-request ID injected into the
  system prompt makes every prefix unique and the hit rate goes to zero silently. This is
  the same failure described in
  [positional-encoding.md](../01-llm-internals/positional-encoding.md) and in
  [caching.md](../07-production/caching.md), and it is worth checking before you blame the
  router.

---

## Long requests need their own lane

A single 100k-token request can consume the KV cache budget of forty ordinary ones (see
[capacity-planning.md](capacity-planning.md)). Route a few onto the same replica and its
batch size collapses; everyone assigned there sees latency several times worse, and no
per-request metric explains why.

Separate the traffic:

- **A lane for long-context requests**, with its own replicas, its own concurrency limit,
  and possibly a different serving configuration tuned for it.
- **A lane for interactive short requests**, protected from the above.
- **A lane for batch work**, which should be preemptible and should never compete with
  interactive traffic. If batch and interactive share a pool, batch will win at the wrong
  moment.

This is the same lane-and-floor pattern as
[rate-limits.md](../07-production/rate-limits.md), applied to your own fleet rather than
to a provider's quota.

---

## Health checking that means something

A replica that responds to `GET /health` with 200 may still be useless — out of KV cache,
queue thirty deep, and about to time out everything you send it. Shallow health checks are
why load balancers keep routing into a replica that everyone can see is dead.

Health for an inference replica should be graded, not binary:

```python
def replica_state(replica):
    if not replica.responding:
        return "down"            # remove from the pool
    if replica.queue_depth > replica.queue_limit:
        return "saturated"       # stop sending new work; keep existing streams
    if replica.kv_utilization > 0.95:
        return "degraded"        # send only if nothing better is available
    return "healthy"
```

The "saturated" state is the one that matters and the one most systems lack. It lets a
replica stop accepting work without being marked down and losing its in-flight streams —
draining rather than failing.

Two more things worth doing: **check the health of the model, not just the process** — a
replica that has loaded the wrong model version answers `/health` perfectly — and **fail
open on the health check itself**. If the health endpoint breaks and you mark every replica
down, you've caused an outage with a monitoring bug.

---

## Retries, and how they take you down

Retrying a failed inference request is more dangerous than retrying a web request, because
the work is expensive and the failure is often *caused* by load. A retry storm against a
saturated fleet adds load exactly when the fleet can least absorb it, and the first
symptom is that a partial degradation becomes a total one.

The rules from [retries.md](../07-production/retries.md) apply with more force here:

- Retry on a **different replica**, never the same one.
- **Budget retries** as a share of total traffic — a few percent, enforced globally — so a
  broad failure can't multiply load.
- **Never retry a timeout** on a request that may still be generating. You'll pay for the
  same expensive work twice and possibly return it twice.
- Prefer **hedging on the tail** — issue a second request only after p95 has elapsed, and
  cancel the loser — over blanket retries. And make sure cancellation actually frees the
  GPU, which requires the serving layer to honor it.

---

## What to monitor

- **Load distribution across replicas** — p50 and p95 of per-replica in-flight work. If
  they diverge, the algorithm is wrong for the traffic.
- **Prefix cache hit rate per replica**, if you route with affinity. This is how you find
  out affinity stopped working.
- **Time-in-queue by replica.** One hot replica is a routing problem; all of them is a
  capacity problem. Distinguishing these quickly is most of the value of the dashboard.
- **Requests routed to degraded/saturated replicas**, which should be near zero.
- **Retry rate as a share of traffic**, with an alert well below your budget.

---

## Interview questions

**"p95 latency is bad but average GPU utilization across the fleet is 55%. What's wrong?"**

Load is unevenly distributed. Look at per-replica in-flight work rather than the fleet
average: with round-robin and variable output lengths, some replicas will be saturated
while others idle, and the fleet average hides it completely. Switch to least-outstanding-
tokens or replica-reported queue depth, and check whether long-context requests are
concentrated somewhere.

**"How would you exploit a shared system prompt across replicas?"**

Cache-aware routing: hash the stable prefix and prefer the replica that already holds those
KV blocks, falling back to load-based routing when that replica is busy. With a 2,000-token
system prompt and short user turns it removes most of the prefill for a hit. Then verify
the prefix is byte-identical — an injected timestamp silently zeroes the hit rate, and that
failure looks like the router not working.

**"A replica is unhealthy but the load balancer keeps sending it traffic."**

The health check is shallow — it proves the process is up, not that the replica can serve.
Add a saturated state driven by queue depth and KV utilization so replicas can shed new
work without dropping in-flight streams, and check model version in the health response.
Then confirm the health check fails open, so a broken check can't take the whole fleet out.

---

## What to remember

- Round-robin assumes equal-cost requests. Inference requests differ by more than an order
  of magnitude, so it distributes badly by construction.
- Balance on outstanding work, or better, on queue depth the replica reports itself.
- Cache-aware routing turns a shared prefix into a large prefill saving — as a soft
  preference, always with a load-based fallback.
- Long-context, interactive and batch traffic need separate lanes. One long request can
  consume the memory of forty short ones.
- Health must be graded. "Saturated" is the state that lets a replica drain instead of
  fail, and most systems don't have it.
