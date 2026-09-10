# Autoscaling and Cold Starts

Web services autoscale in seconds. Inference services do not, and every autoscaling
mistake in AI infrastructure comes from applying web reflexes to a system where adding a
replica takes minutes.

The constraint is cold start: loading tens of gigabytes of weights onto a GPU that may not
exist yet. Until you know that number for your deployment, you cannot design the scaling
policy — and the number is usually worse than people guess.

---

## Where cold-start time goes

Four phases, and only one of them is the part people think about:

```python
def cold_start_seconds(weights_gb, read_gb_per_s, node_provision_s=0,
                       image_pull_s=45, warmup_s=20):
    load_s = weights_gb / read_gb_per_s
    return {
        "node_provision": node_provision_s,   # 0 if a warm node is waiting
        "image_pull": image_pull_s,           # 0 if cached on the node
        "weight_load": load_s,
        "warmup": warmup_s,                   # kernel autotuning, graph capture, first batch
        "total": node_provision_s + image_pull_s + load_s + warmup_s,
    }

# Weights on a network filesystem at ~1 GB/s, node already running
cold_start_seconds(26, 1.0)["total"]        # ≈ 91 s

# Same weights on local NVMe at ~8 GB/s
cold_start_seconds(26, 8.0)["total"]        # ≈ 68 s

# Cold node from the cloud provider, image not cached
cold_start_seconds(26, 1.0, node_provision_s=240)["total"]   # ≈ 331 s
```

**Between one and six minutes.** Note which term dominates: on a cold node it is waiting
for the GPU instance, not loading the model. Optimizing weight-load time when your real
problem is instance provisioning is a common and expensive mistake — measure the phases
separately before optimizing any of them.

Note also that quantization cuts weight-load time proportionally. A model at INT4 is a
quarter the bytes of BF16 (see
[quantization.md](../01-llm-internals/quantization.md)), so it loads in a quarter of the
time. That is a real operational argument for quantization that has nothing to do with
memory capacity.

---

## The arithmetic that decides your policy

Autoscaling works when you can add capacity faster than demand arrives. Compare the two
directly:

```python
def scaling_is_viable(cold_start_s, traffic_doubling_time_s):
    """Can you add capacity faster than load grows?"""
    return cold_start_s < traffic_doubling_time_s

scaling_is_viable(cold_start_s=300, traffic_doubling_time_s=1_800)   # True — a slow ramp
scaling_is_viable(cold_start_s=300, traffic_doubling_time_s=120)     # False — a spike
```

If traffic doubles in two minutes and a replica takes five, reactive autoscaling cannot
help you. It will still *try*, arriving with capacity after the incident is over, and then
scale back down in time for the next one. That pattern — permanently late — is what
autoscaling looks like when the arithmetic doesn't work.

When it doesn't work, you have three honest options, and none of them is a better scaling
policy:

1. **Provision for peak** and accept the utilization. Simple, expensive, and correct for
   spiky interactive workloads with hard latency targets.
2. **Queue the overflow** and degrade gracefully — shed load, or fall back to a smaller
   model or an API. See [fallbacks.md](../07-production/fallbacks.md).
3. **Predict instead of react.** Scale on the clock for known patterns rather than on the
   metric. Most consumer traffic is highly predictable day to day.

---

## Scale on queue depth, not GPU utilization

The default autoscaling metric is CPU or GPU utilization, and for inference it is close to
useless.

GPU utilization during decode reflects memory-bandwidth saturation, not spare capacity. A
deployment can sit at 95% "utilization" with a healthy batch and room for more sequences,
or at 40% while every request queues for four seconds. The metric doesn't distinguish
these, so a policy built on it scales at the wrong times in both directions.

Scale on the signals that mean something:

| Signal | What it tells you |
|---|---|
| **Time-in-queue (p95)** | Users are waiting for admission. The most direct measure of "not enough capacity." |
| **Queue depth per replica** | Same thing, earlier, and less noisy. Best default. |
| **KV cache utilization** | Memory-bound saturation. High utilization means batch size is about to fall. |
| **Running batch size vs. its ceiling** | Whether replicas have room for more concurrent work. |

Queue depth per replica is the best single default because it is a leading indicator: the
queue grows *before* latency does, which buys you exactly the minutes that cold start
costs.

---

## Scale up fast, scale down slowly

The costs are asymmetric, so the policy should be too.

Scaling up late costs you an incident. Scaling down early costs you a cold start on the
way back, during which you are under-provisioned — the worst of both. Under noisy traffic,
a symmetric policy will oscillate: scale down at a trough, cold-start back up two minutes
later, repeat. Every cycle spends real money and delivers a latency spike.

```python
POLICY = {
    "scale_up_threshold_queue_depth": 5,
    "scale_up_evaluation_window_s": 30,      # react quickly
    "scale_up_step": "double",               # add meaningful capacity at once

    "scale_down_threshold_queue_depth": 1,
    "scale_down_evaluation_window_s": 900,   # 15 minutes of calm before releasing
    "scale_down_step": 1,                    # one replica at a time
    "min_replicas": 4,                       # never scale to zero for interactive traffic
}
```

The 15-minute scale-down window looks wasteful and is not: it costs one replica-hour of
idle GPU, against a latency spike for every user during the next cold start. Compute both
sides before arguing about it — the idle GPU is $2, and being under-provisioned during a
traffic recovery is a customer-visible incident.

---

## Scale-to-zero and the floor

Scaling to zero is attractive — no traffic, no bill — and it is the right choice for a
narrow set of workloads: internal tools, demos, batch endpoints, anything where the first
request after idle can wait minutes.

For anything user-facing it is a trap, because *someone* pays the cold start and it is
always a real user, always at the worst moment: the first user after a quiet period, and
the first user of the morning.

Keep a floor of warm replicas sized to your quiet-period traffic, and let autoscaling work
above it. The floor is not waste; it is the thing that makes the p99 possible.

A useful middle option is a **warm pool**: nodes provisioned with the image pulled and, if
you can afford it, weights already resident, but not receiving traffic. That removes the
two largest cold-start terms and leaves only warmup. You pay for idle hardware to buy
minutes of response time, which is the same trade as the floor, priced differently.

---

## What multi-model deployments change

Serving several models from one pool multiplies the problem: each scale-up event may need
a *different* set of weights, and swapping models on a live GPU means unloading and
reloading.

- **Pin high-traffic models** to dedicated replicas. Their cold start becomes a
  deploy-time problem instead of a request-time one.
- **Share a pool for the long tail** and accept cold starts there. Route with awareness of
  which replica already holds which weights — see
  [load-balancing.md](load-balancing.md).
- **Watch the swap rate.** If replicas are swapping models more than a few times an hour,
  the pool is too small and you're spending most of your GPU-time loading weights rather
  than serving.

---

## What to monitor

- **Cold-start duration, broken into its four phases.** One aggregate number tells you
  nothing about what to fix.
- **Scaling event frequency**, and specifically up-then-down cycles within an hour. That's
  oscillation and it is pure cost.
- **Time from threshold breach to replica serving traffic** — the real end-to-end lag,
  which is always longer than the cold start alone.
- **Requests served during scale-up**, and their latency, separately. This is the
  population that is actually harmed.
- **Warm floor utilization** during quiet periods, to check the floor is sized right and
  not merely inherited.

---

## Interview questions

**"Traffic triples in 90 seconds every weekday at 9am. Design the scaling."**

Don't scale reactively — a replica takes minutes and the spike is over in one. This
pattern is predictable, so scale on the clock: warm up before 9am and release afterward.
Reactive autoscaling stays as the backstop for the unpredictable case. Then say what
you'd measure to size it: the peak-to-quiet ratio, and cold-start time end to end.

**"Your autoscaler is thrashing — up and down every few minutes. Why, and what do you
change?"**

Symmetric thresholds and windows against noisy traffic. Scale-down is firing during
troughs that reverse before the replica is gone. Widen the scale-down evaluation window to
10–15 minutes, keep scale-up fast, add a floor. Also check the metric: if it's GPU
utilization, replace it with queue depth, which is less noisy and actually correlates with
user pain.

**"Would you scale to zero?"**

For internal tools, batch endpoints and demos, yes. For anything user-facing, no — the
cold start is always paid by a real user, and the first request of the morning is the one
that pays it. Keep a warm floor sized to quiet-period traffic instead.

---

## What to remember

- Cold start is one to six minutes, and on a cold node it is dominated by instance
  provisioning rather than weight loading. Measure the phases.
- Autoscaling only works if cold start is shorter than your traffic doubling time. If it
  isn't, provision for peak, degrade, or predict — don't tune the policy.
- Scale on queue depth. GPU utilization does not mean what it means for web services.
- Scale up fast, scale down slowly. The costs are asymmetric, so the thresholds should be.
- Scale-to-zero is fine when a machine pays the cold start and wrong when a user does.
