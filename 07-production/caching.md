# Caching

The biggest cost lever available in most AI systems, and the easiest one to get dangerously
wrong.

Two facts that shape everything in this file:

> **Caching is not one thing.** There are four distinct layers with completely different
> risk profiles, and the safest one usually has the biggest payoff.
>
> **A cache key must contain everything the answer depends on.** Anything you leave out is
> a class of bug you have chosen to ship — and in a multi-tenant system, one of those
> classes is a data breach.

---

## The four layers, ranked by safety

| Layer | What it caches | Savings | Can it return a wrong answer? |
|---|---|---|---|
| **1. Prefix / KV** | Prompt prefill work | 20–50% of input cost | **No** |
| **2. Retrieval** | Search results | Latency + some cost | **No** (if keyed on corpus version) |
| **3. Exact response** | Whole answers, exact match | 10–40% of requests | Only via a bad key |
| **4. Semantic response** | Answers to *similar* questions | Higher hit rate | **Yes** |

**Do them in this order.** Layer 1 is free of correctness risk and often the largest single
saving. Layer 4 is where teams go first because it sounds clever, and it's the one that
serves confidently wrong answers.

The general principle:

> **Cache the deterministic parts of the pipeline aggressively. Cache the generative parts
> conservatively.**

---

## Layer 1: prefix caching

If part of your prompt is identical on every request — system prompt, tool definitions,
policy text — the model's work on that prefix can be reused.

This is usually 60–80% of your input tokens, and input tokens are usually most of your
bill. **It is the highest value-per-risk change available**, and it costs nothing in
quality because it changes no output.

Two rules, and violating either kills it entirely:

**Stable content first.**

```
[system prompt]        ← never changes      ┐
[tool definitions]     ← rarely changes     │  cacheable prefix
[policy text]          ← rarely changes     ┘
[retrieved documents]  ← varies per request
[conversation history] ← varies
[user message]         ← varies
```

**Nothing volatile above the line.** A timestamp, a request ID, a user's name, an A/B
bucket — one of those at the top and you have no cache at all.

```python
def build_prompt(system, tools, policy, docs, history, user_message):
    """Order is load-bearing. Everything above the marker must be byte-identical
    across requests or the prefix cache is useless."""
    return "".join([
        system, tools, policy,          # ---- cacheable ----
        format_docs(docs),
        format_history(history),
        user_message,
    ])
```

### The incident this causes

The most common cache failure in production, and it has no deploy to point at:

> Someone edits one word in the system prompt through a management UI. Every request's
> prefix hash changes. Hit rate drops from 87% to 4%. Latency and cost triple. Nobody
> looks at deploys, because there wasn't one.

Two defences:

**Monitor hit rate as a first-class SLI**, and alert on a drop. It moves immediately and
points at the cause; a latency alert only tells you something is wrong.

**Treat prompt edits as deploys** — review, canary, rollback. A UI that lets someone change
production behaviour with no gate is the actual root cause; the cache is just the
mechanism.

Also worth surfacing the consequence at the moment of the decision:

> "This edit adds ~460 input tokens (+$1,900/month at current volume) and will invalidate
> the prefix cache."

Almost nobody builds that, and it's remarkably effective.

---

## Layer 2: retrieval caching

Cache the *documents* you retrieved, not the answer.

```python
def retrieval_key(query, corpus_version, embedding_model_version, k):
    return hash_key(normalize(query), corpus_version,
                    embedding_model_version, k)
```

Lower savings than response caching — you still pay for generation — but two properties
make it valuable:

**It's safe for personalised requests.** "How much protein did I eat today?" is personal;
the *nutrition facts* retrieved to answer it are not. So you can cache the expensive
retrieval step even where you must not cache the answer.

**It's deterministic.** Given the same query and corpus version, retrieval returns the same
documents. There's no correctness gamble.

---

## Layer 3: exact response caching

Cache whole answers, keyed on an exact normalised match.

This is where key design starts to matter, so start by enumerating what an answer actually
depends on:

```python
def response_key(query, principal, ctx):
    return hash_key(
        normalize(query),                 # the question
        principal.permission_scope(),     # WHO CAN SEE WHAT
        ctx.user_visible_flags,           # personalisation that changes the answer
        ctx.corpus_version,               # the documents
        ctx.prompt_version,               # the instructions
        ctx.model_version,                # the model
        ctx.decoding_params,              # temperature etc.
        ctx.locale,                       # regulated claims differ by market
    )
```

Miss any one of those and you have a specific, nameable bug:

| Omitted | Bug you shipped |
|---|---|
| Permission scope | **One customer sees another's data** |
| Corpus version | Stale answers after a document update |
| Prompt version | Old behaviour persists after a fix |
| Model version | Mixed behaviour across a migration |
| Locale | Wrong regulated claims in a market |

### Permissions: key on the scope, not the user

This is the detail that decides whether the cache is both safe and useful.

```python
def permission_scope(principal):
    """Key on the RESOLVED permission set, not the user id.

    Keying on user_id is safe but gives every user a private cache with a
    near-zero hit rate. Keying on the group set lets users with identical
    permissions share entries — safely."""
    return hashlib.sha256(
        "|".join(sorted(principal.groups)).encode()
    ).hexdigest()[:16]
```

And write the test that proves it:

```python
def test_no_cross_permission_reuse():
    alice = Principal("alice", frozenset({"engineering"}))
    bob   = Principal("bob",   frozenset({"sales"}))
    assert response_key(q, alice, ctx) != response_key(q, bob, ctx)
```

Run it on every PR. This is the incident in
[../12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md)
— a semantic cache keyed without a tenant served cross-tenant answers for three weeks.

### Personalisation that's cache-visible

You often want *some* personalisation in the key. A nut-allergy flag changes the correct
answer to "is oat milk healthy?"

Include a small set of low-cardinality flags. Which dimensions are cache-visible is a
**product decision with cost consequences**, not something to decide silently in a config
file — bring product into it.

---

## Layer 4: semantic caching

Return a cached answer for a *similar* question, matched by embedding similarity.

Higher hit rate. Real risk of serving a subtly wrong answer, because "similar" is a
judgement your embedding model makes and it doesn't understand your domain.

**If you use it, do these three things:**

**1. Calibrate the threshold on labelled data.** Not by intuition. Sample a few thousand
near-duplicate pairs, have humans grade "would the same answer be correct for both," and
pick the threshold at your tolerated false-hit rate. For anything health, legal, or
financial, that tolerance should be very low.

**2. Veto on negation and quantities.** The classic failure: "is X safe during pregnancy"
and "is X *unsafe* during pregnancy" are near-identical in embedding space.

```python
NEGATION = {"not", "no", "never", "unsafe", "without", "avoid", "don't", "cannot"}

def veto_semantic_hit(query_a, query_b):
    """Reject a similarity match if the two queries differ on negation or numbers."""
    a, b = set(tokenize(query_a)), set(tokenize(query_b))
    if (a & NEGATION) != (b & NEGATION):
        return True
    if numbers(query_a) != numbers(query_b):
        return True
    return False
```

**3. Measure false-hit rate, not just hit rate.** A cache's savings appear on a dashboard;
its errors are invisible by construction. A metric measured on only one side will drift in
the wrong direction. Sample semantic hits, log both queries, and have them graded.

**And version by embedding model.** Upgrade the embedding model and every stored vector is
in a different space. Similarity still returns plausible numbers. Results become nonsense
with no error.

---

## What never to cache

```python
def is_cacheable(request, response, ctx):
    if response.failed or response.truncated:
        return False                      # never cache a broken answer
    if request.touches_personal_data:
        return False                      # unless the key includes it
    if response.contains_time_relative_claim:
        return False                      # "today", "this week", "currently"
    if ctx.compliance_sensitive and not ctx.short_ttl:
        return False
    return True
```

**Caching a failed or truncated response** is a mistake that keeps paying: you now serve
the broken answer for the whole TTL.

**Time-relative answers** need the time bucket in the key, or they're wrong tomorrow.

---

## Invalidation

Two mechanisms, and you want both:

**Version-based (primary).** Put `corpus_version`, `prompt_version`, `model_version` in the
key. Changing any of them makes the old entries unreachable — atomic, no deletion pass
needed.

**TTL (safety net).** Cap it regardless, so a missed invalidation has a bounded blast
radius. 24 hours is a common default; shorter where content is regulated.

```python
def cache_ttl(request, ctx):
    if ctx.regulated_market:
        return 3600            # 1 hour — guidance changes matter
    if request.is_time_sensitive:
        return 300
    return 86400               # 24 hours default
```

---

## Cache stampede

A popular entry expires and 4,000 concurrent requests all miss and hit the provider at
once.

```python
async def get_or_compute(key, compute):
    if (hit := await cache.get(key)) is not None:
        return hit

    # Single-flight: one caller computes, the rest wait for that result
    async with in_flight_lock(key):
        if (hit := await cache.get(key)) is not None:   # check again
            return hit
        value = await compute()
        await cache.set(key, value)
        return value
```

Without this, your busiest cache entries become your worst traffic spikes.

---

## What to monitor

```python
CACHE_METRICS = [
    "prefix_cache_hit_rate",       # a drop = cost/latency incident, no deploy
    "response_cache_hit_rate",
    "semantic_false_hit_rate",     # sampled + graded — the honesty metric
    "key_cardinality",             # sudden growth = something volatile got in the key
    "stampede_events",
    "invalidation_lag",            # corpus change -> stale entries gone
    "cross_scope_canary",          # must stay green
    "latency_hit_vs_miss",
]
```

Two that catch things nothing else does:

**Key cardinality.** A sudden jump means a per-request value leaked into the key —
a timestamp, a session ID — and your hit rate is about to collapse.

**Cross-scope canary.** A continuous check that a known-unique document for tenant A never
appears for tenant B. It should be green forever, and it's the only thing that proves the
permission dimension is working.

---

## Interview questions

**1. "How would you cache an LLM application?"**

Four layers, and start with the safe one: prefix caching (no correctness risk, biggest
input-token saving), then retrieval caching, then exact response caching with a careful
key, then semantic caching only with a calibrated threshold. Say the principle: cache
deterministic parts aggressively, generative parts conservatively.

**2. "What goes in the cache key?"**

Everything the answer depends on: query, permission scope, corpus version, prompt version,
model version, decoding params, locale. Then name the bug each omission causes — the
permission one is a data breach.

**3. "Why key on permission scope rather than user ID?"**

User ID is safe but gives everyone a private cache with a near-zero hit rate. The resolved
group set lets users with identical permissions share entries safely.

**4. "Your cache hit rate dropped from 34% to 3% overnight with no cache deploy."**

Something volatile entered the key — a timestamp, session ID, or a field a frontend deploy
started sending. Check key cardinality. Also check whether a prompt edit invalidated the
prefix.

**5. "What's the risk of semantic caching?"**

Serving a subtly wrong answer that looks plausible, and it's undetectable from the output.
Calibrate the threshold on graded data, veto on negation and numbers, and measure
false-hit rate — because savings are visible and errors aren't.

**6. "A regulator asks you to reproduce the exact answer given to a user last Tuesday."**

Only possible if you logged the response with its cache status and all version identifiers,
with retention covering that period. Worth saying that caching makes reproducibility harder
and needs to be designed for.

---

## What to remember

- Four layers. Do prefix caching first — biggest saving, zero correctness risk.
- Stable content first in prompts, volatile last. One timestamp at the top kills it.
- Prefix hit rate is a first-class SLI. A drop is a cost incident with no deploy.
- The key must contain everything the answer depends on. Each omission is a named bug.
- Key on resolved permission scope, not user ID. Test it on every PR.
- Never cache failed or truncated responses.
- Semantic caching needs a calibrated threshold, a negation veto, and a measured false-hit
  rate.
- Version-based invalidation as the mechanism; TTL as the safety net.
- Single-flight to prevent stampedes.
- Watch key cardinality — it predicts a hit-rate collapse.

---

**Next:** [cost-control.md](cost-control.md) — governing the spend.
