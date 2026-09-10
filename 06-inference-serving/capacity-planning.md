# Capacity Planning

How many GPUs do you need? It is the most common quantitative question in an AI
infrastructure interview, and the answer is arithmetic — which is exactly why it is asked.
An engineer who can size a fleet out loud has run one. An engineer who says "it depends"
and stops has not.

It does depend. Say what on, then compute anyway.

---

## The four numbers you need

Before any arithmetic, ask for these. Asking is part of the answer:

| Number | Why |
|---|---|
| **Requests per second**, at peak, not average | You size for peak; the ratio of the two is your utilization problem |
| **Input tokens per request** (p50 and p95) | Drives prefill load and KV cache size |
| **Output tokens per request** (p50 and p95) | Drives decode load, which is usually the constraint |
| **Latency target**, split into time-to-first-token and total | Decides how much headroom you must leave unused |

Two more if you can get them: the traffic shape over a day, and whether the workload is
interactive or batch. Those decide autoscaling and whether you can use spot capacity.

---

## The calculation

Prefill and decode are different workloads with different throughputs, so size them
separately and add. This is the same split as
[prefill-vs-decode.md](../01-llm-internals/prefill-vs-decode.md), turned into a fleet.

```python
def size_fleet(rps, input_tokens, output_tokens,
               prefill_tps=20_000, decode_tps=2_000):
    """GPUs needed for steady-state load, before any headroom.

    prefill_tps: prompt tokens/sec one GPU sustains
    decode_tps:  output tokens/sec one GPU sustains across its whole batch
    """
    prefill_load = rps * input_tokens        # tokens/sec of prompt processing
    decode_load = rps * output_tokens        # tokens/sec of generation

    return {
        "prefill_gpus": prefill_load / prefill_tps,
        "decode_gpus": decode_load / decode_tps,
    }

size_fleet(rps=500, input_tokens=2_000, output_tokens=500)
# {'prefill_gpus': 50.0, 'decode_gpus': 125.0}
```

**175 GPUs** for 500 requests/sec — and decode needs two and a half times what prefill
needs, despite processing a quarter as many tokens. That inversion surprises people, and
being unsurprised by it is a strong signal. It happens because decode is sequential and
memory-bandwidth-bound while prefill is parallel and compute-bound.

Then the money:

```python
def monthly_cost(gpus, hourly=2.00, hours=730):
    return gpus * hourly * hours

monthly_cost(175)     # $255,500/month at 100% utilization — which you will not get
```

At $2/GPU-hour reserved, 175 GPUs is about **$256,000/month**, and that figure assumes
every GPU is busy every second of every month. It won't be. See
[gpu-economics.md](gpu-economics.md) for what the real number becomes.

---

## Headroom is not optional, and it is not 10%

The number above is the load. What you provision is the load plus headroom for four
things, and they stack:

- **Peak-to-average ratio.** Diurnal traffic commonly runs 2–3× average at peak. If you
  sized on average, you are already short at 10am.
- **Failure domains.** Losing a node, a rack, or a zone must not take you below peak. With
  three zones, that alone means provisioning ~1.5× so any one zone can fail.
- **Latency headroom.** A queueing system's wait time rises sharply as utilization
  approaches 1. Running at 95% of theoretical capacity produces terrible tail latency;
  70–80% is the usual target for interactive workloads. This is queueing theory, not
  caution.
- **Deploy and drain.** Rolling a new model version means running two versions briefly.

None of these multiply cleanly, and stacking them naively over-provisions badly. Size for
*peak, with one failure domain gone, at your target utilization*, and treat deploys as a
scheduling problem rather than a capacity one.

```python
def provisioned_gpus(steady_state_gpus, peak_multiple=2.5,
                     target_utilization=0.75, zones=3):
    peak = steady_state_gpus * peak_multiple
    with_utilization_headroom = peak / target_utilization
    # survive losing one zone
    return with_utilization_headroom * zones / (zones - 1)

provisioned_gpus(175)     # ≈ 875 GPUs
```

That is five times the steady-state number, and it is why capacity planning conversations
end in autoscaling: holding 875 GPUs to serve an average load of 175 is indefensible if
the peak lasts four hours a day. [autoscaling.md](autoscaling.md) is the other half of
this file.

---

## Memory sizing, which is a separate constraint

Compute tells you how many GPUs. Memory tells you whether a request fits at all, and it
binds first surprisingly often.

```python
def memory_budget(gpu_gb, weights_gb, kv_bytes_per_token):
    free = (gpu_gb - weights_gb) * 1e9
    return {
        "kv_tokens_total": free / kv_bytes_per_token,
        "concurrent_at_2500": free / (kv_bytes_per_token * 2_500),
        "concurrent_at_32k": free / (kv_bytes_per_token * 32_000),
    }

memory_budget(80, 26, 131_072)
# kv_tokens_total    ≈ 411,987
# concurrent_at_2500 ≈ 165
# concurrent_at_32k  ≈ 13
```

Same GPU, same model. At 2,500-token contexts it holds 165 sequences; at 32k contexts it
holds 13. Your throughput per GPU falls by roughly the same factor, and so your fleet size
rises by it.

**This is the number that ruins capacity plans.** A product change that adds conversation
history, or a retrieval change that sends 20 chunks instead of 5, doesn't look like a
capacity decision. It is one. Anyone sizing a fleet should ask what the context length
roadmap is, and anyone answering should say that context length is a capacity commitment.

---

## Sanity checks before you present a number

Run these in your head; each has caught a real plan:

1. **Cost per request.** Divide monthly cost by monthly requests. If that number is larger
   than the API price for the same work, self-hosting doesn't pay and no amount of
   engineering fixes it — check against
   [api-vs-open-model.md](../02-model-selection/api-vs-open-model.md).
2. **Tokens per GPU-second.** Should land in the hundreds to low thousands for decode. If
   your plan implies 50, you've assumed batch size 1 somewhere. If it implies 50,000,
   you've assumed a batch that won't fit in memory.
3. **Does the peak fit?** Multiply concurrent sequences per GPU by fleet size and compare
   with peak concurrent users. These are two different constraints and both must hold.
4. **What happens at 2×?** If the answer is "we buy twice the GPUs," fine. If it's "the
   architecture changes," say so now rather than at 1.8×.

---

## What to monitor

- **Tokens/sec per GPU**, split by prefill and decode. The denominator of everything.
- **GPU memory utilization and KV cache utilization** — different numbers, both needed.
- **Queue depth and time-in-queue** as the leading indicator. Queues grow before latency
  does.
- **Peak-to-average ratio**, weekly. It drifts as the product changes and nobody notices
  until a capacity plan is wrong.
- **Average context length**, as a first-class metric. It moves your capacity plan and it
  moves for product reasons rather than infrastructure ones.

---

## Interview questions

**"Size a deployment for 500 requests/sec, 2,000 tokens in, 500 out."**

Say the four questions you'd normally ask, then compute anyway: 1M prompt tokens/sec at
~20k/sec/GPU is 50 GPUs for prefill; 250k output tokens/sec at ~2k/sec/GPU is 125 for
decode; 175 steady state, and I'd provision several times that for peak, failure domains
and latency headroom. Then flag the assumptions — those two throughput figures dominate
the answer and both need measuring on your model and sequence profile.

**"Your fleet is sized correctly and p95 latency is still bad."**

Utilization. A correctly sized fleet at 95% utilization has bad tail latency by
construction; queueing delay rises steeply as you approach capacity. Check time-in-queue
against generation time. If queueing dominates, you need headroom or admission control,
not faster inference.

**"Product wants to add conversation memory. What does that do to your capacity?"**

It raises average context length, which cuts concurrent sequences per GPU roughly
proportionally, which cuts throughput per GPU, which raises fleet size. Ask for the target
context length and recompute — going from 2,500 to 10,000 tokens is roughly a 4× hit to
sequences-per-GPU. Then offer the mitigations: summarize old turns rather than resending
them, and check the retrieval budget while you're there.

---

## What to remember

- Size prefill and decode separately, then add. Decode usually dominates despite
  processing far fewer tokens.
- 500 rps at 2k in / 500 out is roughly 175 GPUs of steady-state load, and several times
  that provisioned.
- Headroom comes from peak-to-average, failure domains, and queueing — not from a flat
  percentage.
- Memory is a separate constraint from compute, and context length is the variable that
  moves it. Long contexts cut concurrency, and concurrency is throughput.
- Always divide back to cost per request and compare with the API. The comparison is the
  whole reason you're doing the arithmetic.
