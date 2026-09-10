# GPU Economics

Self-hosting is cheaper than an API at high volume. That sentence is true and it is also
where most of the money gets lost, because it quietly assumes a utilization number that
almost nobody hits.

This file is the arithmetic behind the claim, and the three ways it goes wrong.

---

## The unit that matters

Not dollars per month, not dollars per GPU. **Dollars per million tokens**, so it can be
compared with the API price you're replacing.

```python
def cost_per_million_output_tokens(tokens_per_second, utilization,
                                   gpu_hourly=2.00):
    """The only number that compares directly to an API price."""
    effective_tps = tokens_per_second * utilization
    tokens_per_hour = effective_tps * 3600
    return gpu_hourly / (tokens_per_hour / 1e6)

cost_per_million_output_tokens(2_000, utilization=1.00)   # ≈ $0.28
cost_per_million_output_tokens(2_000, utilization=0.45)   # ≈ $0.62
cost_per_million_output_tokens(2_000, utilization=0.20)   # ≈ $1.39
```

Against an API at $1.50 per million output tokens:

| Utilization | Your cost per 1M output tokens | vs. API at $1.50 |
|---|---|---|
| 100% (theoretical) | $0.28 | 5.4× cheaper |
| 75% (a well-run fleet) | $0.37 | 4.1× cheaper |
| 45% (common in practice) | $0.62 | 2.4× cheaper |
| 20% (spiky traffic, no autoscaling) | $1.39 | break-even |

**Utilization is the whole business case.** The hardware, the model and the serving stack
are identical across every row; only the fraction of purchased GPU-time that does useful
work changes. A build-versus-buy analysis that doesn't state its utilization assumption
has not made an argument.

And none of the rows include engineering. Two engineers on serving infrastructure is
about $50,000/month fully loaded — the figure used in
[api-vs-open-model.md](../02-model-selection/api-vs-open-model.md) — which at 200M
requests/month adds meaningfully to every row and dominates the small ones.

---

## Where utilization goes

Three leaks, in the order they usually cost the most.

**The diurnal gap.** Interactive traffic peaks 2–3× above its daily average. Provision for
peak and you own idle GPUs for most of the day:

```python
def diurnal_utilization(peak_hours, peak_multiple):
    """Fraction of a peak-provisioned fleet actually used over 24h."""
    off_peak_hours = 24 - peak_hours
    peak_work = peak_hours * peak_multiple
    off_peak_work = off_peak_hours * 1.0
    provisioned = 24 * peak_multiple
    return (peak_work + off_peak_work) / provisioned

diurnal_utilization(peak_hours=6, peak_multiple=2.5)   # ≈ 0.55
```

Roughly **55%** before anything else goes wrong. Autoscaling recovers part of this — see
[autoscaling.md](autoscaling.md) — and cold-start time bounds how much.

**Headroom you can never use.** Latency targets require running below capacity, failure
domains require surviving a zone loss, and deploys require room to roll. That's real,
necessary, permanently idle capacity. It is a cost of the latency SLO, and it should be
attributed there rather than treated as waste.

**Serving inefficiency.** A low batch size means the GPU is busy and unproductive, which
utilization dashboards report as *high* utilization. This is the leak that hides: you can
be at 95% GPU utilization and 20% of achievable throughput. Measure tokens/sec per GPU
against a measured ceiling, not the vendor's utilization metric. See
[continuous-batching.md](continuous-batching.md).

---

## Buying capacity three ways

| | Price | What you're really buying |
|---|---|---|
| **On-demand** | ~$3–4/GPU-hour | Flexibility, at a premium. Correct for unpredictable or early-stage load |
| **Reserved / committed** | ~$2/GPU-hour | A discount in exchange for a 1–3 year commitment on a hardware generation |
| **Spot / preemptible** | ~$0.60–1.00/GPU-hour | Capacity that can be taken away with a couple of minutes' notice |

The blended strategy is the one to describe in an interview, because it maps onto the
traffic shape rather than to a preference:

- **Reserved for the floor** — the load that exists at 4am. It's always there, so commit
  to it.
- **On-demand for the predictable peak.** More expensive per hour, but only for the hours
  you need it.
- **Spot for anything interruptible** — batch scoring, evaluation runs, backfills,
  embedding an entire corpus. Never for interactive serving unless you can drain a replica
  within the eviction warning, which for inference usually means finishing in-flight
  streams and refusing new ones.

```python
def blended_hourly(reserved_gpus, ondemand_gpus, spot_gpus,
                   reserved=2.00, ondemand=3.50, spot=0.80):
    total = reserved_gpus + ondemand_gpus + spot_gpus
    spend = (reserved_gpus * reserved + ondemand_gpus * ondemand + spot_gpus * spot)
    return {"blended_rate": spend / total, "hourly_spend": spend}

blended_hourly(100, 50, 50)
# blended_rate ≈ $2.08/GPU-hour, hourly_spend = $415
```

The commitment risk is worth naming out loud: a multi-year reservation on a specific GPU
generation is a bet that your model, your traffic and the hardware market all stay put.
Hardware generations turn over, and a better model that needs different memory can strand
a commitment. Shorter commitments cost more per hour and are frequently the right call.

---

## The comparison people get wrong

Self-hosting is not compared against the API price. It's compared against the API price
*for equivalent quality*, and those are often different models.

If a 70B open model matches a mid-tier API model on your evaluation set, compare against
the mid-tier price, not the frontier price. Comparing your self-hosted 70B against a
frontier API model makes self-hosting look brilliant and answers a question nobody asked.

The honest comparison also includes what the API price contains and your build does not:

- Capacity management and the ability to absorb a traffic spike that wasn't forecast.
- Model upgrades. Every new model version is a project for you and a config change for
  them.
- Multi-region availability, and someone on call for the inference layer at 3am.
- The option to change your mind, which self-hosting spends.

None of that makes self-hosting wrong at scale — the arithmetic in
[build-vs-buy.md](../12-senior-scenarios/build-vs-buy.md) reaches an 85% saving on a
high-volume workload. It means the comparison must include them to be believed by anyone
who has done it before.

---

## Attribution, before optimization

You cannot reduce a cost you can't attribute. Before optimizing anything, be able to
answer: which product surface, which customer, and which model version spent this money.

For a self-hosted fleet the accounting is different from an API bill — you pay for GPU
hours, not tokens, so you must allocate:

```python
def allocate_gpu_cost(monthly_gpu_spend, tokens_by_tenant):
    """Attribute a fixed fleet bill by measured usage."""
    total = sum(tokens_by_tenant.values())
    if total == 0:
        return {t: 0.0 for t in tokens_by_tenant}
    return {
        tenant: monthly_gpu_spend * tokens / total
        for tenant, tokens in tokens_by_tenant.items()
    }
```

Note what this hides and say so: idle capacity gets spread across whoever happened to use
the fleet, so a tenant with steady off-peak traffic subsidizes the one that causes the
peak. If that matters commercially, allocate peak capacity to the tenants who drive peak.
The general discipline is the same as
[cost-control.md](../07-production/cost-control.md); only the denominator changes.

---

## What to monitor

- **Cost per million tokens**, computed monthly and trended. Everything else is an input.
- **Utilization, three ways**: fraction of GPUs provisioned vs. running, GPU-time busy vs.
  idle, and achieved throughput vs. measured ceiling. They fail differently and you need
  all three.
- **Cost per unit of business work** — per resolved ticket, per document, per session —
  which is the number a finance conversation actually turns on.
- **Spot eviction rate and its cost**, if you use spot. Evictions that force re-runs can
  erase the discount.
- **Reserved capacity coverage**: what fraction of your floor is on committed pricing.

---

## Interview questions

**"We're spending $250k/month on GPUs. Where do you start?"**

Attribution before optimization: cost per million tokens, and utilization measured three
ways. If utilization is low, the fix is scheduling and autoscaling, not faster inference.
If utilization is high but tokens/sec per GPU is far below a measured ceiling, it's a
serving problem — batch size first. If both are healthy, then it's a demand question:
which surface generates this traffic, and is it worth what it costs? I'd also recompute
the API comparison at the current volume, because the answer moves.

**"Finance asks why total spend went up 30% this quarter. Usage is up 45%."**

That's a success story that looks like a problem. Cost per unit of work fell about 10%,
which means the efficiency work landed while the product grew. Lead with cost per resolved
ticket or per session, then show total spend as the product of unit cost and volume. It's
the same argument as tracking cost per unit of work rather than total spend in
[cost-control.md](../07-production/cost-control.md).

**"Would you commit to a three-year GPU reservation for a 40% discount?"**

For the floor — the load that exists at 4am — probably yes, if the floor is genuinely
stable. For peak capacity, no. The risk is a bet on hardware generation and model
requirements staying put for three years; a better model with different memory needs can
strand it. I'd size the commitment to the trough, not the average, and buy the rest
on-demand and spot.

---

## What to remember

- Cost per million tokens is the only number that compares to an API price. Compute it,
  and state your utilization assumption when you do.
- At $2/GPU-hour and 2,000 output tokens/sec, output costs about $0.28 per million at
  full utilization and about $1.39 at 20% — the API price.
- Utilization leaks in three places: the diurnal gap, unusable headroom, and serving
  inefficiency that dashboards report as high utilization.
- Blend reserved for the floor, on-demand for peak, spot for interruptible work.
- Compare against the API model that matches your quality, and include the things the API
  price buys that your build doesn't.
