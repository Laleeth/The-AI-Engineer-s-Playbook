# Cost Control

> **This file is about governance** — attribution, budgets, controls, and the
> conversations. The *optimization* levers (routing, model choice, caching, reranking) are
> in [../02-model-selection/cost-quality-latency.md](../02-model-selection/cost-quality-latency.md)
> and [caching.md](caching.md).

Two ideas underneath everything here:

> **You cannot control what you cannot attribute.** If you can't say which team, feature,
> or customer spent the money, every cost conversation is guesswork.
>
> **A one-time cut against compounding growth is a delay, not a fix.** Say so when you
> present it, or you'll be asked to do it again in six months with the easy levers already
> spent.

---

## Attribution first

Before any optimization, know where the money goes. Tag every call at the gateway — the one
place every request passes through.

```python
def cost_tags(ctx):
    return {
        "team":        ctx.team,
        "feature":     ctx.feature,
        "tenant_id":   ctx.tenant_id,
        "environment": ctx.environment,     # prod vs staging vs eval
        "route":       ctx.route,           # which model tier served it
        "request_type": ctx.request_type,
    }
```

Retrofitting this across a dozen call sites is far harder than adding it once at the
boundary. Do it before you need it.

**`environment` is the one people forget.** Evaluation runs and staging traffic can be a
surprising share of spend, and they're invisible if everything is tagged "production."

---

## Measure per unit of work, not total

Total spend tells you almost nothing. It rises when you grow, which is fine, and it also
rises when you break something, which isn't. The two look identical on a monthly chart.

```python
def unit_costs(records):
    return {
        "per_request":         total(records) / len(records),
        "per_resolved_ticket": total(records) / count_resolved(records),
        "per_active_user":     total(records) / count_active_users(records),
        "per_document":        total(records) / count_documents(records),
    }
```

**Cost per request can fall while cost per *resolved* ticket rises** — because quality
dropped and users now need three attempts to get an answer. Only the second number shows
that, and it's the one that matters to the business.

Pick the unit that matches what you sell.

---

## The distribution, not the average

Cost per user is almost always extremely lopsided.

```python
def cost_distribution(per_user_costs):
    values = sorted(per_user_costs.values(), reverse=True)
    total = sum(values)
    return {
        "mean":   total / len(values),
        "p50":    values[len(values) // 2],
        "p95":    values[int(len(values) * 0.05)],
        "p99":    values[int(len(values) * 0.01)],
        "top_1pct_share": sum(values[:len(values) // 100]) / total,
    }
```

A typical result: the top 3% of users generate 60% of the spend. The mean tells you nothing
useful and every decision made from it will be wrong.

**Bring the histogram to the pricing conversation, not the average.** It's the artefact
that lets finance make a good decision.

---

## Before assuming user behavior, look for a bug

When you find a wildly skewed distribution, check the top consumers' traces before
concluding anything about how people use your product.

In practice a distribution like that is often:

- ~40% genuine power users
- ~30% automated or scripted use
- ~30% **a product defect** — a retry loop, a component re-firing on every render, an
  auto-refresh, a background job that runs on every page view

Fixing one defect can remove 20% of spend in a week with zero product impact. It's always
worth checking first, and it's routinely skipped in favor of designing a rate limit.

---

## Budgets that degrade instead of breaking

A hard cap is a reliability trade: you're choosing to fail on budget rather than on cost.
Make that trade graded, not binary.

```python
def budget_tier(fraction_used, request):
    if fraction_used < 0.70:
        return "normal"
    if fraction_used < 0.85:
        return "normal" if request.interactive else "defer"      # shed batch first
    if fraction_used < 0.95:
        return "cheap_model"                                     # degrade quality
    return "cached_only" if request.interactive else "reject"
```

Three rules:

**Shed batch and internal work before customer traffic.**

**Degrade quality before availability.** A worse answer beats an error for almost every
product.

**Never let a cap silently take down customer traffic.** Someone gets paged before the last
tier fires, and there's a documented break-glass override with a named owner.

### Budget per workload, not globally

A single pool means a runaway batch job consumes the interactive budget. Same isolation
argument as queue lanes:

```python
BUDGETS = {
    "interactive":  {"monthly_usd": 40_000, "priority": 0},
    "batch":        {"monthly_usd": 15_000, "priority": 2},
    "evaluation":   {"monthly_usd":  3_000, "priority": 1},
    "internal":     {"monthly_usd":  2_000, "priority": 2},
}
```

---

## Predict, don't just account

A cap that tells you after you've spent the money is a receipt, not a control.

```python
def burn_rate_projection(spend_so_far, days_elapsed, days_in_month, budget):
    daily = spend_so_far / days_elapsed
    projected = daily * days_in_month
    return {
        "projected_month_end": projected,
        "over_budget_by":      max(0, projected - budget),
        "days_until_exhausted": (budget - spend_so_far) / daily if daily else None,
    }
```

Alert on **trajectory at day 6**, not on the cap at day 27. That's the difference between a
planning conversation and an incident.

---

## Where the money actually is

Before optimizing, decompose. The answer is usually not where people assume.

```python
def spend_breakdown(records):
    return {
        "input_tokens":  sum(r.input_tokens * r.price_in for r in records),
        "output_tokens": sum(r.output_tokens * r.price_out for r in records),
        "by_route":      group_sum(records, "route"),
        "by_feature":    group_sum(records, "feature"),
        "by_step":       group_sum(records, "pipeline_step"),
        "wasted_on_retries": sum(r.cost for r in records if r.attempt > 1),
        "wasted_on_failures": sum(r.cost for r in records if r.failed),
    }
```

Two lines there are worth having explicitly:

**Spend on retries.** Invisible in error rate. A pipeline at 3.4 attempts per document
instead of 1.06 has tripled its bill while every dashboard looks healthy.

**Spend on failures.** Money paid for requests that produced nothing usable.

And the split that decides your strategy: **input tokens are usually 70–80% of the bill**,
because a typical request sends far more than it produces. If your bill is too high, look
at input first — prompt caching, sending fewer retrieved chunks, trimming the system
prompt. "Use a smaller model" is the riskiest lever and the one people reach for first.

---

## Cost bugs that don't look like cost bugs

Real patterns, all of which triple a bill with a flat error rate:

**Truncation loops.** Someone lowers `max_tokens` to save money. Responses truncate, fail
validation, and retry — each retry re-sending the full input, which is where the cost is.
There's a documented case of this tripling a bill while looking like a cost optimization.

**Prefix cache invalidation.** A prompt edit through a UI. Hit rate 87% → 4%. No deploy to
point at.

**Eval runs in production.** An unattributed evaluation suite running nightly against
production models.

**A retry loop nobody counted.** Provider SDK retries 3× inside your decorator's 3×.

**Context accumulation.** An agent or conversation that never compresses history, so cost
grows with the square of the turn count.

**Over-retrieval.** `top_k=20` with no reranking, sending 20 chunks when 5 would answer
better and cost a quarter.

Monitoring for these: `attempts_per_request`, `finish_reason_length_rate`,
`cache_hit_rate`, and cost per unit of work. See
[observability.md](observability.md).

---

## The conversation, not just the number

The most valuable thing an engineer contributes to a cost discussion is usually a reframe
rather than a saving.

**Price the quality drop.** Before switching to a cheaper model, compute what the errors
cost:

```python
def cost_of_quality_drop(requests_per_month, quality_drop, cost_per_error):
    return requests_per_month * quality_drop * cost_per_error

# 1M requests, 3-point drop, $2 per extra error in handling time
# = 30,000 errors = $60,000/month
```

If that exceeds the model saving, the cheaper model is more expensive. Nobody can see that
without the calculation.

**Distinguish COGS from CAC.** Free-tier AI spend with zero revenue isn't a cost overrun —
it's customer acquisition, and it should be evaluated against conversion, not cut on
principle. If AI-active free users convert at 7.8% against 2.2% for everyone else, that
spend may be underfunded rather than wasteful.

**Present a curve, not a number.** Give the options with their quality and latency
attached, and let the business pick a row:

```
option          quality   p95      $/month
frontier only    0.92     4.1s     $140,000
routed           0.91     1.9s      $38,000     ← usually the answer
mid only         0.89     1.9s      $32,000
small only       0.85     0.8s       $9,000
```

Routing is within a point of the best quality at 27% of the cost *and* faster. That's not a
compromise, it's a better design — and it's invisible until someone builds the table.

**Say how durable the saving is.** A 55% one-time cut against 11% monthly growth is
consumed in about seven months. Present that up front rather than declaring victory.

---

## What to monitor

```python
COST_METRICS = [
    "cost_per_request",
    "cost_per_resolved_unit",       # the business number
    "cost_by_team_feature_tenant",
    "cost_distribution_p50_p95_p99",
    "input_vs_output_share",
    "attempts_per_request",         # invisible retry spend
    "finish_reason_length_rate",    # truncation -> retries
    "cache_hit_rate",
    "burn_rate_projection",
    "budget_tier_occupancy",        # how long are you degraded?
    "invoice_vs_internal_delta",    # accounting drift
]
```

That last one catches a specific problem: if your internal token accounting drifts from the
provider's invoice, either you're missing a call path (someone bypassing the gateway) or
your token counting is wrong. Reconcile monthly.

---

## Interview questions

**1. "Cut our AI costs by 60%."**

Ask where the money goes first — input or output, which feature, which tenant. Then work
the ranked levers: caching and reranking (free), routing, then model choice last. Say
you'd build the quality dashboard *before* shipping the cuts, because cost is visible and
quality isn't.

**2. "Costs tripled with no deploy. First check?"**

Attempts per request and finish-reason distribution, then cache hit rate. Truncation
causing retries and prefix-cache invalidation are the two most common causes, and both are
invisible in error rate.

**3. "Design a hard monthly cost cap."**

Tiered degradation, not a cliff. Per-workload budgets so batch can't consume interactive.
Quality degrades before availability. Burn-rate projection so you're warned at day 6. A
documented override with an owner.

**4. "Your top 3% of users are 60% of spend. What do you do?"**

Look at their traces before assuming it's user behavior — a meaningful share is usually a
product defect. Then segment the rest into legitimate power users and automated use, and
bring the distribution to a pricing conversation rather than unilaterally throttling.

**5. "The free tier costs $147k/month with no revenue. Cut it?"**

That's customer acquisition, not cost of goods. Evaluate it against conversion — if
AI-active free users convert several times better, the spend may be underfunded. That
reframe is worth more than any optimization.

**6. "Which matters more, cost or quality?"**

Wrong question. Price the quality drop — errors × cost per error — and compare it to the
saving. Then present the options table and let the business choose the row.

---

## What to remember

- Attribute at the gateway, before you need to. Include `environment`.
- Measure cost per *unit of work*, not total. Per resolved unit is the business number.
- Bring the distribution, not the average — spend is extremely lopsided.
- Check for a product defect before assuming user behavior.
- Budgets degrade in tiers: batch first, quality before availability, never silent.
- Budget per workload, not globally.
- Project burn rate; alert at day 6, not day 27.
- Input tokens are usually 70–80% of the bill. Look there first.
- Watch attempts-per-request and finish-reason — invisible retry spend is common.
- Price the quality drop before switching models. Sometimes cheaper is more expensive.
- Say how durable a saving is. One-time cuts against compounding growth are delays.

---

**Next:** [incident-response.md](incident-response.md) — when it breaks anyway.
