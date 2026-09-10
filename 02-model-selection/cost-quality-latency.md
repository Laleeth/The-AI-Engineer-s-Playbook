# Cost, Quality, Latency

Three things you want, and improving one usually costs you another. Every model decision
is a position on this triangle, whether you chose it deliberately or not.

Most teams optimize one of them by accident and discover the damage later.

---

## Where the money and the time actually go

Before optimizing anything, know which lever moves which number. This trips people up
constantly:

> **Input tokens mostly drive cost. Output tokens mostly drive latency.**

Why:

**Cost.** A typical request sends far more input than it produces output — a system
prompt, retrieved documents, conversation history, then a short answer. Even though
output tokens are priced higher per token, the sheer volume of input usually dominates
the bill.

**Latency.** Output tokens are generated one at a time, in sequence. Input is processed
in parallel. So a 2,000-token input might take 200ms to process, while a 500-token
output takes 5 seconds.

The practical consequence: **if your bill is too high, look at input first. If it's too
slow, look at output first.** Optimizing the wrong one is the most common wasted week in
this work.

```python
def where_does_it_go(requests, in_tok, out_tok, price_in, price_out):
    cost_in = requests * in_tok / 1e6 * price_in
    cost_out = requests * out_tok / 1e6 * price_out
    total = cost_in + cost_out
    return {
        "input_share": round(cost_in / total, 2),
        "output_share": round(cost_out / total, 2),
    }

# Typical RAG request: 3,000 in, 300 out, at $0.50/M in and $1.50/M out
# → input is about 77% of the cost
```

Run that on your own numbers before you plan anything.

---

## The levers, ranked

Not all optimizations are equal. Ranked by value-per-risk:

| Lever | Attacks | Typical saving | Quality risk |
|---|---|---|---|
| Prompt caching | Input cost + latency | 20–50% | **None** |
| Reranking → fewer chunks | Input cost | 30–60% | Often *improves* quality |
| Trim the system prompt | Input cost | 10–30% | Low |
| Shorter outputs | Latency + cost | 20–50% | Depends |
| Response caching | Whole requests | 10–40% | Medium — correctness risk |
| Deduplication | Whole requests | 5–25% | None |
| Routing / cascades | Blended price | 30–70% | Medium |
| Batch API tier | Price | ~50% | None (latency cost) |
| Smaller model everywhere | Price | 50–90% | **High** |
| Replace a model call with code | Whole requests | 100% of that call | Low, and testable |

Two things stand out:

**The top of the list is free.** Prompt caching and reranking cost nothing in quality —
reranking usually *improves* it, because you removed distracting context. Do these
before anything risky.

**"Use a smaller model" is at the bottom.** It's the first thing people reach for and
the riskiest. Everything above it should be exhausted first.

---

## Prompt caching: do this first

If a chunk of your prompt is identical on every request — system prompt, tool
definitions, policy text — most providers can cache the processing of it.

Two conditions:

**1. Put stable content first.** Caching works on a prefix. Order your prompt:

```
[system prompt]        ← never changes    ┐
[tool definitions]     ← rarely changes   │ cached
[policy text]          ← rarely changes   ┘
[retrieved documents]  ← changes per request
[conversation history] ← changes per request
[user message]         ← changes per request
```

**2. Don't break the prefix.** Anything that varies — a timestamp, a user ID, a request
ID — must go *after* the stable part. One variable token at the top invalidates
everything.

This is the source of a real production incident pattern: someone edits a word in the
system prompt, the cache breaks for every request, and latency and cost jump 3× with no
code deploy. If you cache, monitor your cache hit rate as a first-class metric and alert
on drops.

---

## Cutting input tokens

After caching, this is where the money is.

**Rerank and send fewer documents.** Retrieve 40, rerank, send the best 5. Counter-
intuitive but true: fewer, better documents beat more, worse ones. See
[../03-rag/reranking.md](../03-rag/reranking.md).

**Audit the system prompt.** Long-lived prompts accumulate rules. Ablate them — remove
one at a time and check whether quality drops. A meaningful fraction is usually dead
weight from a problem that no longer exists.

**Scope the context to the task.** Does "summarize this paragraph" really need the whole
document? Often it sends the whole document because that was easier to write.

**Summarize conversation history.** After a few turns, compress older turns instead of
resending them verbatim.

---

## Cutting latency

Different techniques entirely.

**Stream.** The single biggest perceived-latency win, and it changes nothing about your
actual speed. Users see the first token in a few hundred milliseconds instead of staring
at a spinner for 5 seconds. Most "our system is too slow" complaints are solved by
streaming.

Worth being precise about which metric you're targeting:

- **Time to first token** — what users feel. Streaming fixes this.
- **Total completion time** — matters for batch work and anything a machine consumes.

If someone says "p95 latency must be under 2 seconds," ask which one they mean. Very
often they mean the first, and the whole problem dissolves.

**Shorten outputs.** Generation is sequential, so output length is nearly linear in
time. Ask for concise answers, use structured output, cap `max_tokens` sensibly.

Careful with that last one — capping `max_tokens` too low causes truncation, which
causes retries, which costs *more*. There's a real incident where someone lowered
`max_tokens` to save money and tripled the bill because every response truncated and
retried. Set it from the actual distribution of output lengths, not by guessing.

**Do independent work in parallel.** If you retrieve from two sources, do it at the same
time.

**Use a smaller model for the fast path.** Especially for autocomplete-style features
where a large model simply cannot meet the latency bar.

---

## Making the trade explicit

The useful move in most of these conversations is to stop arguing about which matters
and start pricing them.

```python
def compare_options(options, volume_per_month):
    """options: [{"name", "quality", "cost_per_req", "p95_ms"}]"""
    for o in options:
        o["monthly_cost"] = o["cost_per_req"] * volume_per_month
    return sorted(options, key=lambda o: -o["quality"])
```

Then put it in front of the people who own the decision:

```
option          quality   p95      $/month
frontier only    0.92     4.1s     $140,000
routed           0.91     1.9s      $38,000     ← usually the answer
mid only         0.89     1.9s      $32,000
small only       0.85     0.8s       $9,000
```

Notice what the table shows: routing gets you within 1 point of the best quality at 27%
of the cost, *and* it's faster. That's not a compromise, it's a better design — and it's
invisible until someone builds the table.

**The engineer's job here is to produce this table, not to pick a row.** Which row is
right depends on what a quality point is worth, and that's a business question. But
nobody can answer it without the table.

---

## Turning quality into money

The conversation gets much easier when quality has a price attached.

Ask: what does a bad answer cost?

- **A human reviews everything.** Cost = extra editing time. Measurable, and usually
  modest. Cheap models are viable here.
- **It goes to a customer.** Cost = a complaint, maybe churn. Higher.
- **It affects a real decision.** Cost = the decision being wrong. Could be enormous.

```python
def cost_of_quality_drop(requests, quality_drop, cost_per_error):
    extra_errors = requests * quality_drop
    return extra_errors * cost_per_error
```

Example: 1M requests/month, 3-point quality drop, each error costs $2 in extra handling.
That's 30,000 extra errors, $60,000/month — which may well exceed the model saving.

Run this before switching models. Sometimes the cheaper model is more expensive.

---

## Budgets that degrade instead of breaking

If you're asked for a hard spending cap, don't build a cliff. Build tiers:

```python
def pick_model(budget_used_fraction, request):
    if budget_used_fraction < 0.70:
        return normal_routing(request)
    if budget_used_fraction < 0.85:
        return normal_routing(request, disable_batch_jobs=True)
    if budget_used_fraction < 0.95:
        return cheap_model(request)          # degrade quality, stay up
    return cached_or_error(request)          # last resort
```

Two rules:

**Degrade quality before you degrade availability.** A worse answer beats no answer for
almost every product.

**Never let a cap silently take down customer traffic.** Shed batch and internal work
first, and make sure someone gets paged before the last tier fires.

Also: track burn rate and project the month-end total. A cap that only tells you after
you've spent the money is a receipt, not a control.

---

## What to watch

```python
TRACK = [
    "cost_per_request",           # and per resolved task, which differs
    "input_tokens_p50_p95",
    "output_tokens_p50_p95",
    "cache_hit_rate",             # a drop here is a 3x cost incident
    "time_to_first_token_p95",    # what users feel
    "total_latency_p95",
    "quality_score_by_segment",
    "finish_reason_length_rate",  # truncation → retries → cost
    "attempts_per_request",       # invisible retries, invisible in error rate
]
```

Two of these are early warnings that most teams don't have:

**Cache hit rate.** Drops from 85% to 4% when someone edits the top of a prompt. Costs
and latency follow immediately.

**Attempts per request.** Retries that eventually succeed don't show up in your error
rate at all. A retry loop can triple your bill while every dashboard looks healthy.

---

## Interview questions

**1. "Cut our LLM costs by 60%."**

Start by asking where the money goes — input or output? Then work down the ranked list:
caching and reranking first (free), then routing, then model choice last. Say explicitly
that you'd measure quality per segment before and after, because a cost cut with no
quality dashboard is a cost cut you'll regret.

**2. "Which matters more, cost or quality?"**

Wrong question. Ask what a bad answer costs, then price the quality drop and compare it
to the saving. Produce the options table and let the business pick the row.

**3. "Your p95 latency is 5 seconds and the requirement is 2. What do you do?"**

First ask whether it's time-to-first-token or total time, and whether the product
streams. That question often ends the problem. If not: shorter outputs, prompt caching
to cut prefill, parallel retrieval, smaller model for the fast path.

**4. "Someone lowered max_tokens to save money and the bill tripled. Explain."**

Responses truncated, failed validation, and retried. Each retry re-sends the full input,
which is where the cost is. Error rate stayed flat because the retries eventually
succeeded. Then say what you'd monitor: `finish_reason: length` and attempts per
request.

**5. "Design a hard monthly cost cap."**

Tiered degradation, not a cliff. Quality degrades before availability. Batch work sheds
first. Burn-rate projection so you get warned at day 6, not day 27. And a documented
override with an owner.

---

## What to remember

- Input tokens drive cost; output tokens drive latency. Check which you're fixing.
- Prompt caching and reranking are free wins. Do them before anything risky.
- "Use a smaller model" is the last resort, not the first.
- Streaming changes the metric that matters. Ask whether the requirement is TTFT.
- Capping `max_tokens` too low costs more, not less.
- Price the quality drop before switching models — sometimes cheaper is expensive.
- Build the options table; let the business pick the row.
- Budgets should degrade quality before availability.
- Watch cache hit rate and attempts-per-request. Both hide expensive problems.

---

**Back to:** [the model selection index](README.md) · **Next section:**
[RAG](../03-rag/README.md)
