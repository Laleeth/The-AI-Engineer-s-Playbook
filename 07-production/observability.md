# Observability

Knowing what your AI system is doing, and finding out when it stops working properly.

The uncomfortable thing about AI systems is that **most of their failures return HTTP
200**. Your error rate stays flat while quality degrades, costs triple, or the system
starts refusing legitimate requests. Standard monitoring is blind to all of it.

This file is about the signals that aren't blind.

---

## Why normal monitoring isn't enough

A traditional service is healthy if it responds, quickly, without errors. An AI system can
do all three and be badly broken.

| Failure | Errors | Latency | Detected by normal monitoring? |
|---|---|---|---|
| Quality dropped after a prompt change | Flat | Flat | No |
| Retrieval is returning wrong documents | Flat | Flat | No |
| Cost tripled from a retry loop | Flat | Flat | No |
| Responses truncating and retrying | Flat | Slight rise | Barely |
| Model refusing legitimate requests | Flat | Flat | No |
| Cache hit rate collapsed | Flat | **Rises** | Only as a symptom |
| Agent looping and doing damage | Flat | Rises | Only as a symptom |

Every row is a real production incident. Only two are visible at all, and both only as
symptoms pointing nowhere useful.

---

## The signals that actually catch things

Ordered by how much they've earned their place.

### 1. Attempts per logical request

**The single most under-instrumented metric in AI systems.**

Retries that eventually succeed don't appear in your error rate at all. A pipeline making
3.4 attempts per document instead of 1.06 has tripled your bill while every dashboard
looks healthy.

```python
@dataclass
class RequestRecord:
    request_id: str
    attempts: int              # not just success/failure
    final_outcome: str
    total_cost_usd: float
```

Alert on the *mean*, not on errors. A rise from 1.05 to 1.4 is a problem you want to know
about.

### 2. Finish reason distribution

Why did generation stop?

```
stop            → completed normally
length          → TRUNCATED — the response was cut off
tool_call       → wants to call a tool
content_filter  → refused
```

A rising `length` rate means truncation, which means downstream validation failures, which
means retries, which means cost. There's a documented case of someone lowering
`max_tokens` to save money and tripling the bill this way.

`length` should be a few percent at most. Alert if it climbs.

### 3. Cache hit rate

If you use prompt caching, this is a first-class SLI.

It drops from 87% to 4% the moment someone edits the top of a system prompt, and cost and
latency follow immediately — with **no deploy** to point at. Latency alerts tell you
something is wrong; this one tells you what.

### 4. Token distributions, not averages

```python
TOKEN_METRICS = [
    "input_tokens_p50", "input_tokens_p95", "input_tokens_p99",
    "output_tokens_p50", "output_tokens_p95", "output_tokens_p99",
]
```

Averages hide the requests that cost 50× the median. A p99 input of 40,000 tokens against
a p50 of 2,000 tells you where your money goes and where your context limits will break.

### 5. Cost per unit of work

Not total spend — spend per *thing you actually care about*:

```python
def unit_costs(records):
    return {
        "per_request": total(records) / len(records),
        "per_resolved_ticket": total(records) / count_resolved(records),
        "per_active_user": total(records) / count_users(records),
    }
```

Cost per request can fall while cost per *resolved* ticket rises — because quality dropped
and users now need three attempts. Only the second number tells you that.

### 6. Refusal and abstention rate

How often does the system decline, or say "I don't have that information"?

A rise means something upstream broke — usually retrieval. Users don't report this; they
just leave. It's one of the cheapest early-warning signals available and almost nobody
tracks it.

### 7. Quality proxies from user behaviour

You can't grade every production response, but users tell you things:

| Signal | Meaning |
|---|---|
| Rephrased a similar question quickly | The first answer failed |
| Copied the output | It was useful |
| Escalated to a human | It failed |
| Heavily edited a draft | The draft wasn't good enough |
| Abandoned mid-response | Too slow, or clearly wrong |

Rephrase rate is the most useful and the least tracked. See
[../05-evaluation/online-evaluation.md](../05-evaluation/online-evaluation.md).

---

## A minimal set that catches most things

If you can only instrument a handful:

```python
CORE_METRICS = [
    # Cost
    "cost_per_request",
    "input_tokens_p50_p95",
    "output_tokens_p50_p95",

    # Hidden failures
    "attempts_per_request",        # invisible retries
    "finish_reason_length_rate",   # truncation
    "cache_hit_rate",              # cost/latency cliff

    # Quality proxies
    "refusal_rate",
    "abstention_rate",
    "thumbs_down_rate",

    # The usual
    "error_rate",
    "ttft_p95",                    # what users feel
    "total_latency_p95",
]
```

Three of those — attempts, finish reason, cache hit rate — cost almost nothing and catch
failures nothing else sees.

---

## Segment everything

An aggregate number hides the thing you need to find.

```python
SEGMENTS = ["model", "prompt_version", "route", "tenant_tier",
            "language", "feature", "environment"]
```

A 4% overall thumbs-down rate might be 2% in English and 19% in German. A flat cost per
request might be hiding one tenant tripling while everyone else improves.

**The rule: if you can't break a metric down, you can't act on it.**

---

## Version everything, on every record

This is what makes any of it attributable.

```python
@dataclass
class RequestContext:
    trace_id: str
    # what produced this behaviour
    model: str
    model_version: str
    prompt_version: str
    embedding_model_version: str
    retrieval_config_hash: str
    corpus_version: str
    code_sha: str
```

When a metric moves, the first question is *what changed*. Without these fields you're
guessing, and half of AI incidents are caused by things that never went through a deploy:
a prompt edited in a UI, documents imported, a provider updating a model, another team
changing a shared config.

---

## Alert on change, not on level

`thumbs_down_rate = 6%` means nothing at 3am. `thumbs_down_rate doubled in two hours` is
actionable.

```python
def anomaly_alert(metric, current, baseline_mean, baseline_std, z=3.0):
    """Alert when a metric moves far from its own recent baseline."""
    if baseline_std == 0:
        return False
    return abs(current - baseline_mean) / baseline_std > z
```

Two things worth alerting on in **both** directions:

**Cost.** A sudden drop can mean requests are failing early, not that you got efficient.

**Quality scores.** An unexplained *improvement* is as much a bug as a regression — usually
your measurement changed (a judge model updated, a dataset regenerated). Nobody
investigates good news, which is why it quietly poisons every later comparison.

---

## What to log, and the privacy problem

You want the prompt and the response, because "what exactly did the model see?" is the
first question in every investigation.

You also can't just store them. They contain user data, sometimes regulated data, and your
observability system probably has broader access than your production database.

A workable split:

```python
def log_request(ctx, prompt, response, outcome):
    # Structured attributes — safe, queryable, keep for a long time
    metrics.record({
        "trace_id": ctx.trace_id,
        "input_tokens": count(prompt),
        "output_tokens": count(response),
        "cost_usd": cost(ctx, prompt, response),
        "finish_reason": outcome.finish_reason,
        "attempts": outcome.attempts,
        "prompt_version": ctx.prompt_version,
        "model": ctx.model,
    })

    # Payloads — sampled, scrubbed, short retention, access-controlled
    if should_store_payload(outcome):
        payload_store.put(
            ctx.trace_id,
            {"prompt": scrub(prompt), "response": scrub(response)},
            ttl_days=14,
        )


def should_store_payload(outcome):
    """Sample by OUTCOME, not uniformly. Rare failures are what you need."""
    if outcome.failed or outcome.negative_feedback:
        return True                      # 100% of the interesting cases
    return random.random() < 0.02        # 2% of the boring ones
```

Three decisions in there:

**Sample by outcome, not uniformly.** At 1% uniform sampling, a 0.3% failure mode is
nearly invisible. Keep every failure and a small slice of successes.

**Scrub before storing.** Redact obvious secrets and known-sensitive fields. See
[../08-ai-security/pii.md](../08-ai-security/pii.md).

**Get retention from legal, not from engineering.** This is a data-governance decision and
choosing it yourself is how you find out during an audit.

---

## Cost attribution

If you can't say which team, feature, or customer spent the money, you can't manage it.

```python
def tag_request(ctx):
    return {
        "team": ctx.team,
        "feature": ctx.feature,
        "tenant_id": ctx.tenant_id,
        "environment": ctx.environment,
        "route": ctx.route,          # which model tier handled it
    }
```

Attach these at the gateway where every call passes through. Retrofitting attribution
across a dozen call sites is much harder than adding it once at the boundary.

The distribution matters more than the total. Cost per tenant is usually extremely
lopsided — a small number of users generating most of the spend — and you can only see
that if it's attributed.

---

## Dashboards worth having

**Health (for on-call).** Error rate, TTFT p95, total latency p95, attempts per request,
finish-reason distribution.

**Cost (weekly review).** Cost per request and per resolved unit, spend by team and
feature, token distributions, cache hit rate, burn rate against budget.

**Quality (weekly review).** Thumbs-down and rephrase rates, refusal and abstention rates,
escalation rate — all segmented by language, tenant tier, and route.

**Change tracking.** A timeline of prompt versions, model versions, corpus updates, and
config changes overlaid on your quality and cost metrics. This one is unusual and it turns
"when did this start?" from an investigation into a glance.

---

## Interview questions

**1. "What would you monitor for an LLM feature?"**

Cost per request, token distributions, attempts per request, finish-reason distribution,
cache hit rate, refusal and abstention rates — plus the usual latency and errors. Then the
key point: most AI failures return 200, so error rate alone tells you almost nothing.

**2. "Your error rate is flat but customers are unhappy. Where do you look?"**

Quality proxies — rephrase rate, escalation rate, thumbs-down — segmented. Then check
whether anything changed that wasn't a deploy: prompt edits, corpus changes, a provider
model update.

**3. "Costs tripled with no deploy. What's your first check?"**

Attempts per request and finish-reason distribution, then cache hit rate. Truncation
causing retries, or a cache invalidated by a prompt edit, are the two most common causes
and both are invisible in error rate.

**4. "How do you handle logging prompts when they contain user data?"**

Split structured attributes from payloads. Attributes are safe and kept; payloads are
sampled by outcome, scrubbed, short-retention, and access-controlled. Retention is a legal
decision, not an engineering one.

**5. "Why sample by outcome rather than uniformly?"**

Because rare failures are what you need and uniform sampling makes them invisible. Keep
100% of errors and negative feedback, a small percentage of successes.

**6. "Your quality score jumped 9 points with no deploy. Good news?"**

No — that's a measurement change, almost certainly a judge model update or a regenerated
dataset. Unexplained improvements are as much a bug as regressions and get investigated
far less.

---

## What to remember

- Most AI failures return HTTP 200. Error rate is nearly blind.
- Attempts per request, finish-reason distribution, and cache hit rate are cheap and catch
  failures nothing else sees.
- Track cost per *unit of work*, not total spend.
- Refusal and abstention rates are early warnings for retrieval problems.
- Segment everything. Aggregates hide the thing you need.
- Put every version identifier on every record — half of AI incidents have no deploy.
- Alert on change, not level. And investigate improvements too.
- Split structured attributes from payloads; sample payloads by outcome.
- Attribute cost at the gateway, before you need to.

---

**Next:** [tracing.md](tracing.md) — following one request through the whole system.
