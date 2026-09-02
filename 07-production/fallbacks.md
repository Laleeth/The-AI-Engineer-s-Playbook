# Fallbacks

What your system does when the normal path doesn't work.

The rule underneath all of it:

> **Degrade quality before you degrade availability.**

A worse answer beats no answer for almost every product. A system with no fallback has
decided that the only acceptable outcome is a perfect one, which means its actual outcome
is an error page.

---

## Define the tiers before you need them

Not "we'll figure it out." A written ladder, with each rung's quality measured *in advance*
so you know what you're choosing.

```
Tier 0  Normal            primary model, full retrieval, full context
Tier 1  Cheaper model     secondary or smaller model, same pipeline
Tier 2  Reduced pipeline  skip reranking, fewer chunks, shorter context
Tier 3  Retrieval only    "here are the 5 most relevant documents"
Tier 4  Cached            a previous answer to a similar question
Tier 5  Honest error      "we can't answer right now, here's who to contact"
```

```python
@dataclass
class Tier:
    name: str
    handler: Callable
    quality_score: float      # measured on your eval set, not guessed
    cost_per_request: float
    p95_latency_ms: int
```

**Measure each tier's quality before the incident.** During an outage is the wrong time to
discover that your fallback model scores 0.61 on the traffic you just routed to it.

That's the most common gap: teams build a fallback, never evaluate it, and then degrade to
something unmeasured at exactly the moment everyone is watching.

---

## Which tier for which failure

Different failures want different fallbacks. Mapping them explicitly avoids the "always
drop to the cheapest thing" reflex.

| Failure | Sensible fallback |
|---|---|
| Provider returns 503 | Secondary provider (tier 1) |
| Provider rate limits you | Queue if async; secondary if interactive |
| Retrieval times out | Generate without retrieval, *and say so* |
| Retrieval returns nothing | Abstain — don't generate from nothing |
| Reranker unavailable | Use raw search order, send fewer chunks |
| Response fails validation | Repair once, then tier down |
| Cost budget exhausted | Tier down, shed batch work first |
| Model refuses | Escalate to a human, don't retry harder |

Two of those deserve emphasis.

**Retrieval returning nothing is not the same as retrieval failing.** If the corpus has no
answer, generating anyway produces a hallucination. Abstain.

**A refusal is not a transient error.** Retrying a refused request usually gets refused
again. Route it to a human.

---

## Say when you've degraded

If the user gets a worse answer and doesn't know, they conclude your product is worse. If
they know, they calibrate.

```python
@dataclass
class Response:
    text: str
    tier: str
    degraded: bool
    reason: str | None      # user-facing, honest, short
```

```
Normal:    "Your refund window is 30 days from delivery. [policy-4.2]"

Degraded:  "I couldn't reach our document search just now, so this answer is
            from general knowledge and may be out of date. Your refund window
            is usually 30 days. [unverified]"

Tier 3:    "I can't generate an answer right now, but these documents look
            relevant to your question: ..."
```

The middle one is much better than silently answering without retrieval. It's also better
than an error, because the user gets something.

**Internally, tier must be a dimension on every metric.** Otherwise your quality dashboard
averages tier 0 and tier 3 together and you can't see either.

---

## Failing over between providers

The most common fallback, and the one with the most hidden traps.

```python
async def complete_with_failover(prompt, providers, allow_failover=True):
    last_error = None
    for provider in providers:
        try:
            result = await provider.complete(prompt)
            result.provider = provider.name      # ALWAYS record which one
            return result
        except PermanentError:
            raise                    # the request is wrong; another won't fix it
        except (TransientError, CircuitOpen) as exc:
            last_error = exc
            if not allow_failover:
                raise
            log.warning("failing over from %s: %s", provider.name, exc)
    raise AllProvidersFailed(last_error)
```

Four things that decide whether this actually works:

**Only fail over on transient errors.** A 400 fails everywhere. Failing over on it wastes
time and money on a guaranteed failure.

**Record which provider served each request.** Without this, a failover event silently
contaminates your quality metrics — you see a quality dip and can't attribute it.

**`allow_failover=False` for some calls.** Anything whose output is stored and later
compared, or where a contract names a model. A silent model change is not always
acceptable.

**Keep real traffic on the secondary.** This is the one people skip and it's the one that
matters most:

```python
def route(prompt):
    if random.random() < 0.03:        # 3% permanently on the secondary
        return secondary.complete(prompt)
    return primary.complete(prompt)
```

A fallback you never exercise is a fallback that doesn't work. Prompts drift, the provider
deprecates the model, credentials expire, quotas were never raised — and you find all of
that out during the incident. 3% of traffic costs almost nothing and keeps the path alive.

### Prompts are not portable

The uncomfortable part of provider failover: a prompt tuned for one model may perform
noticeably worse on another. Failover being a config toggle implies a substitutability
that isn't real.

Two honest responses:

- Maintain and evaluate prompts per provider, and switch both together.
- Or accept a measured quality drop on the fallback path, and know what it is.

What you shouldn't do is assume they're equivalent because both are language models.

---

## Cost-based degradation

When a budget is the constraint, degrade in tiers rather than hitting a cliff.

```python
def pick_tier(budget_used_fraction, request):
    if budget_used_fraction < 0.70:
        return TIER_0
    if budget_used_fraction < 0.85:
        return TIER_0 if request.interactive else TIER_2   # shed batch first
    if budget_used_fraction < 0.95:
        return TIER_1                                       # cheaper model
    return TIER_4 if request.interactive else REJECT        # cached or nothing
```

Three rules:

**Shed batch and internal work before customer traffic.**

**Degrade quality before availability.**

**Never let a cap silently take down customer traffic.** Someone gets paged before the last
tier fires, and there's a documented override with an owner.

See [cost-control.md](cost-control.md).

---

## Graceful degradation inside the pipeline

Not every fallback is "use a different model." Often the right move is to drop a stage.

```python
async def retrieve_with_degradation(query, budget_ms=800):
    docs, degraded = [], []

    try:
        docs = await asyncio.wait_for(hybrid_search(query, k=50), timeout=0.4)
    except asyncio.TimeoutError:
        degraded.append("hybrid_search")
        try:
            docs = await asyncio.wait_for(vector_search(query, k=20), timeout=0.3)
        except asyncio.TimeoutError:
            degraded.append("vector_search")
            return [], degraded            # abstain rather than invent

    try:
        docs = await asyncio.wait_for(rerank(query, docs), timeout=0.3)
    except asyncio.TimeoutError:
        degraded.append("rerank")
        docs = docs[:5]                    # raw search order, fewer chunks

    return docs, degraded
```

The pattern: each stage has a timeout, and failing one drops to a simpler version rather
than failing the request. The `degraded` list flows into your response metadata and your
metrics.

---

## Fallbacks that make things worse

**The fallback is unmeasured.** You degrade to something nobody has evaluated, at the exact
moment quality matters most.

**The fallback has the same dependency.** Your "backup" retrieval path uses the same vector
database that just went down. Map the shared dependencies.

**The fallback isn't provisioned.** A secondary provider account with default rate limits
cannot absorb your primary's traffic. Provision it for real load, not for existence.

**Failing over hides a problem.** Everything silently degrades and nobody notices for weeks
because there's no alert on tier occupancy. Alert when you're not at tier 0.

**The fallback amplifies the outage.** Everyone fails over simultaneously to the same
secondary, which then falls over. Jitter, and don't assume the secondary has infinite
headroom.

**Degrading is silent to the user**, so they conclude the product is bad rather than
temporarily degraded.

---

## Testing them

A fallback that's never been exercised is a hypothesis.

```python
async def test_fallback_when_primary_fails():
    with primary_returning(503):
        result = await complete_with_failover("test prompt", providers)
        assert result.provider == "secondary"
        assert result.text


async def test_no_failover_on_bad_request():
    with primary_returning(400):
        with pytest.raises(PermanentError):
            await complete_with_failover("bad", providers)
        assert secondary.call_count == 0     # must not waste a call


async def test_abstains_when_retrieval_empty():
    with retrieval_returning([]):
        result = await answer("question")
        assert result.abstained
        assert not result.text_contains_invented_facts
```

Beyond unit tests, **exercise the real path deliberately**: run game days where you disable
the primary provider in a controlled window and watch what happens. Most teams discover
something — an unraised quota, an expired credential, a prompt that doesn't work on the
secondary.

---

## What to monitor

```python
FALLBACK_METRICS = [
    "tier_occupancy",              # % of traffic at each tier — alert if tier 0 < 99%
    "failover_events",             # count, by cause
    "secondary_provider_share",    # must be > 0 in steady state
    "degraded_response_rate",
    "quality_by_tier",             # so you know what degrading costs
    "time_in_degraded_state",
    "circuit_state_by_dependency",
]
```

**Tier occupancy is the headline.** If you're spending 8% of requests at tier 2 and nobody
noticed, that's a slow quality leak.

**Secondary provider share must be non-zero.** If it hits zero, your fallback has stopped
being exercised and is decaying.

---

## Interview questions

**1. "Your LLM provider goes down. What happens?"**

Walk the tiers: secondary provider, then reduced pipeline, then retrieval-only, then
cached, then an honest error. Say that each tier's quality is measured beforehand, and
that you keep continuous traffic on the secondary so it actually works.

**2. "Why keep traffic on the fallback in normal operation?"**

Because an unexercised path decays — prompts drift, models get deprecated, credentials
expire, quotas were never raised. 3% of traffic costs almost nothing and is the difference
between a working fallback and a theoretical one.

**3. "Your secondary provider scores 4 points lower. Do you fail over automatically?"**

Depends what a wrong answer costs and whether the caller can tolerate a silent model
change. Automatic for most traffic, with the provider recorded on every result so quality
metrics stay attributable; opt-out for calls where a model change isn't acceptable.

**4. "Retrieval times out. What do you do?"**

Drop to a simpler retrieval, and if that fails too, abstain rather than generating from
nothing. Tell the user the answer is unverified. Generating without retrieval and not
saying so is how you ship a confident hallucination.

**5. "What's the risk of a good fallback?"**

That it hides the problem. Everything quietly degrades and nobody notices for weeks. Alert
on tier occupancy, not just on total failure.

**6. "How do you test fallbacks?"**

Unit tests for the routing logic, plus deliberate game days that disable the primary in a
controlled window. The game day is where you find the unraised quota and the expired
credential.

---

## What to remember

- Degrade quality before availability. A worse answer beats no answer.
- Write the tiers down, and **measure each tier's quality before you need it.**
- Different failures deserve different fallbacks — don't always drop to the cheapest.
- Empty retrieval means abstain, not generate anyway.
- Tell the user when you've degraded, and tag the tier on every metric.
- Record which provider served each request, or failover contaminates your quality data.
- Keep real traffic on the secondary permanently, or it rots.
- Prompts are not portable between models. Evaluate the fallback path.
- Alert on tier occupancy — a silent slow degradation is the failure mode of good
  fallbacks.

---

**Next:** [rate-limits.md](rate-limits.md) — living inside someone else's quota.
