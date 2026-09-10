# 06 — Inference & Serving

Running models yourself, in production. [01-llm-internals](../01-llm-internals/) explains
the mechanics; this section is about operating them. Plain language throughout.

The fact underneath every file here:

> **The gap between owning a GPU and using one is about 5×.** Same weights, same hardware,
> same output — a well-batched deployment produces output tokens for roughly $0.28 per
> million, a poorly batched one for about $1.39, which is the API price it was meant to
> beat.

That gap is what this section is about. Self-hosting is a serving-efficiency problem
wearing a hardware costume.

## The files

| File | What it covers |
|---|---|
| [serving-stacks.md](serving-stacks.md) | What the layer between your endpoint and the GPU actually does, and when you don't need one |
| [continuous-batching.md](continuous-batching.md) | **The technique that makes self-hosting viable.** Read this one. |
| [capacity-planning.md](capacity-planning.md) | Sizing a fleet from traffic, and the headroom that isn't optional |
| [autoscaling.md](autoscaling.md) | Cold starts, and why web autoscaling reflexes fail here |
| [multi-gpu.md](multi-gpu.md) | Sharding a model that doesn't fit — and why it isn't a throughput technique |
| [load-balancing.md](load-balancing.md) | Routing across replicas when requests differ in cost by 20× |
| [gpu-economics.md](gpu-economics.md) | Utilization, cost per million tokens, and the comparison people get wrong |

## Read in this order

File order works, and the dependencies are real: batching decides throughput, throughput
decides fleet size, fleet size decides the bill.

**If you're deciding whether to self-host at all**, read
[gpu-economics.md](gpu-economics.md) first, then
[api-vs-open-model.md](../02-model-selection/api-vs-open-model.md). The decision is
arithmetic and it usually goes the other way than people expect.

**If you already run a deployment and it's expensive**, start at
[continuous-batching.md](continuous-batching.md) and check your average running batch
size. That one number explains most bad unit economics.

**If you're sizing something new**, [capacity-planning.md](capacity-planning.md), then
[autoscaling.md](autoscaling.md) — cold-start time constrains the architecture more than
throughput does.

## The short version

**Batching amortizes the weight read.** Producing one token means reading the whole model
out of memory, whether you're serving one sequence or sixty. That's why throughput scales
with batch size at almost no cost in step time, and why a deployment running at batch 1
wastes most of what it pays for.

**Batch size is an outcome, not a setting.** It falls out of how much KV cache memory is
free. Shorter contexts mean more concurrent sequences, which means more throughput — so
trimming retrieved context buys throughput on top of the token saving.

**Static batching wastes GPU-time in proportion to output-length variance.** A realistic
mix of 30-token and 800-token responses runs under 30% useful. Continuous batching
re-forms the batch every step so finished sequences free their memory immediately, and the
gain is proportional to that same variance.

**Decode needs more GPUs than prefill** despite processing far fewer tokens. 500 rps at
2,000 in and 500 out is about 50 GPUs for prefill and 125 for decode. Size the two
separately and add.

**Context length is a capacity commitment.** The same 80 GB card holds 165 concurrent
sequences at 2,500 tokens and 13 at 32k. A product decision to add conversation history is
an infrastructure decision, and nobody labels it as one.

**GPU utilization does not mean what it means for web services.** Decode is
memory-bandwidth-bound, so you can sit at 95% "utilization" while delivering 20% of
achievable throughput. Watch running batch size, queue depth and tokens/sec per GPU
instead.

**Cold start is one to six minutes**, and on a cold node it's dominated by waiting for the
instance, not loading weights. Autoscaling only works if cold start is shorter than your
traffic doubling time; when it isn't, provision for peak, degrade, or predict.

**Sharding is for fitting, not throughput.** A 70B model is 140 GB at BF16 and 35 GB at
INT4 — quantization frequently removes the need to shard at all, and two independent
replicas beat one two-way sharded replica on every axis.

**Round-robin is the wrong default.** Inference requests differ in cost by more than an
order of magnitude. Balance on outstanding work or replica-reported queue depth, and route
shared prefixes to the replica that already holds them.

**Utilization is the entire business case for self-hosting.** Every cost row uses the same
hardware and the same model; only the fraction of purchased GPU-time doing useful work
changes.

## The arithmetic worth doing out loud

```python
# Fleet size, prefill and decode separately
prefill_gpus = rps * input_tokens / 20_000
decode_gpus = rps * output_tokens / 2_000

# Concurrent sequences on one GPU
(gpu_gb - weights_gb) * 1e9 / (kv_bytes_per_token * avg_context)

# Cost per million output tokens
gpu_hourly / (tokens_per_second * utilization * 3600 / 1e6)

# Is autoscaling even viable?
cold_start_seconds < traffic_doubling_seconds
```

## Related

- [01-llm-internals/kv-cache.md](../01-llm-internals/kv-cache.md) — where concurrency
  comes from, and the arithmetic this section spends
- [01-llm-internals/prefill-vs-decode.md](../01-llm-internals/prefill-vs-decode.md) — the
  asymmetry that makes decode the constraint
- [02-model-selection/api-vs-open-model.md](../02-model-selection/api-vs-open-model.md) —
  whether to be in this section at all
- [07-production/](../07-production/) — operating the service around the serving layer:
  retries, fallbacks, cost control
- [12-senior-scenarios/scaling.md](../12-senior-scenarios/scaling.md) — this material as
  interview scenarios, with the constraints changing round by round
