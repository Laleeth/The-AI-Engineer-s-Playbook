# Tracing

Following a single request through every stage of your system: retrieval, reranking,
generation, tool calls, retries.

Metrics tell you *something* is wrong. Traces tell you *where*. For AI systems the gap
between those two is unusually large, because a request touches many stages and any of
them can be the problem.

---

## What a trace is

A trace is one request. Spans are the steps inside it, nested.

```
trace: req_a4f9c2
├── span: http.request                          3,240ms
│   ├── span: query.rewrite                       180ms
│   ├── span: retrieval                           310ms
│   │   ├── span: embed.query                      45ms
│   │   ├── span: vector.search                   180ms
│   │   └── span: permission.filter                85ms
│   ├── span: rerank                              240ms
│   ├── span: context.build                         8ms
│   └── span: llm.generate                      2,480ms
│       ├── attempt 1  (429, retried)             120ms
│       └── attempt 2  (ok)                     2,360ms
└── span: feedback.record                          12ms
```

That picture answers "why was this slow?" in one glance. Without it you're guessing which
stage to instrument next.

---

## Use your existing tracing

An important and money-saving point: **most of what you need is normal distributed
tracing with the right attributes on the spans.**

Token counts, costs, model versions, retry counts — these are just span attributes. Nested
tool calls are just nested spans. Your existing tracing system already does all of that.

What's genuinely new is two things:

1. **Payload storage** — prompts and responses are large and contain user data
2. **Quality signals** — joining user feedback back to the trace that produced it

Everything else is a schema decision, not a tooling decision. Teams who reach for a
separate AI observability product before asking this question often end up with two
tracing systems and have to correlate between them manually during an incident — which is
exactly the problem they were trying to solve.

---

## The span schema

Use OpenTelemetry's GenAI semantic conventions where they fit, so you're not inventing
names that diverge from the ecosystem. Add your own for what they don't cover.

```python
# The model call
span.set_attributes({
    # standard-ish
    "gen_ai.system":                "provider-name",
    "gen_ai.request.model":         "model-id",
    "gen_ai.response.model":        "model-id-actually-served",
    "gen_ai.request.temperature":   0.0,
    "gen_ai.request.max_tokens":    1024,
    "gen_ai.usage.input_tokens":    3_412,
    "gen_ai.usage.output_tokens":   287,
    "gen_ai.usage.cached_tokens":   2_800,
    "gen_ai.response.finish_reason": "stop",

    # yours
    "app.prompt_version":   "v14",
    "app.cost_usd":         0.0041,
    "app.attempts":         2,
    "app.route":            "standard",
    "app.tenant_id":        "t_8891",
    "app.payload_ref":      "s3://traces/req_a4f9c2/llm-1",   # pointer, not payload
})
```

```python
# Retrieval
span.set_attributes({
    "app.index_id":                "docs-v3",
    "app.embedding_model_version": "embed-model-2",
    "app.query_rewritten":         True,
    "app.candidates_retrieved":    50,
    "app.results_returned":        5,
    "app.score_top1":              0.81,
    "app.score_spread":            0.19,      # flat spread = nothing relevant
    "app.sources":                 "handbook,policies",
    "app.payload_ref":             "s3://traces/req_a4f9c2/retrieval",
})
```

```python
# Tool call (agents)
span.set_attributes({
    "app.tool_name":       "get_order_status",
    "app.tool_version":    "3",
    "app.effect":          "read_only",
    "app.idempotency_key": "run_x:9f2a...",
    "app.error_kind":      None,             # terminal | transient | denied | invalid
    "app.output_truncated": False,
    "app.iteration":       4,
})
```

Three deliberate choices in there.

**`payload_ref` is a pointer, not the payload.** Prompts are large and sensitive. The span
carries a reference; the content lives in storage with its own retention and access
control.

**Version attributes on every span.** `prompt_version`, `embedding_model_version`,
`index_id`. When a metric moves, you filter traces by version and see it immediately.

**`score_spread` on retrieval.** A flat score distribution means nothing was clearly
relevant. It's a one-number signal that retrieval failed, available before you look at any
text.

---

## Retries as child spans

Make each attempt its own span. This is what makes invisible retries visible.

```python
async def call_with_retries(fn, max_attempts=3):
    with tracer.start_span("llm.generate") as parent:
        for attempt in range(max_attempts):
            with tracer.start_span(f"llm.attempt.{attempt + 1}") as child:
                try:
                    result = await fn()
                    child.set_attribute("app.outcome", "ok")
                    parent.set_attribute("app.attempts", attempt + 1)
                    return result
                except TransientError as exc:
                    child.set_attribute("app.outcome", "retry")
                    child.set_attribute("app.error_kind", "transient")
                    child.record_exception(exc)
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(backoff(attempt))
```

The parent span succeeds. The children show three attempts. Without the child spans your
trace says "this took 6 seconds" and doesn't say why — and your error rate says everything
is fine.

---

## One trace ID, everywhere

The single most important operational decision here.

```python
@dataclass
class RequestContext:
    trace_id: str          # follows the request everywhere
    span_id: str
    tenant_id: str
    user_id: str
```

Rules that make it useful:

**Generate at the edge**, on the first service that touches the request.

**Propagate through everything** — including into agent sub-runs, background jobs, and
async work triggered by the request.

**Return it to the client.** Put it in a response header or show it in the UI. Then a
support ticket saying "trace_id: req_a4f9c2" turns a two-hour investigation into a
two-minute one.

**Store it with feedback.** When a user clicks thumbs-down, record the trace ID. Now "show
me the traces for the 200 worst-rated responses this week" is a query rather than a
project.

```python
def record_feedback(trace_id, rating, comment=None):
    feedback_store.put({
        "trace_id": trace_id,      # the join key that makes everything work
        "rating": rating,
        "comment": comment,
        "timestamp": time.time(),
    })
```

That join is what turns observability from "here are some numbers" into "here are the
exact requests that made users unhappy, with everything the model saw."

---

## Tracing agents

Agents need a nested structure, because a single run has many steps.

```
trace: agent_run_x8821
├── span: agent.run                      stop_reason=no_progress, cost=$0.84
│   ├── span: agent.iteration.1
│   │   ├── span: llm.generate           tool_calls=[get_alerts]
│   │   └── span: tool.get_alerts        ok, 3 results
│   ├── span: agent.iteration.2
│   │   ├── span: llm.generate           tool_calls=[get_runbook]
│   │   └── span: tool.get_runbook       error_kind=terminal (404)
│   ├── span: agent.iteration.3
│   │   ├── span: llm.generate           tool_calls=[get_runbook]   ← repeat
│   │   └── span: tool.get_runbook       error_kind=terminal (404)
│   └── ...
```

Reading that trace, the loop is obvious: the same tool with the same arguments returning
the same terminal error. Reading a flat log of the same run, it isn't.

Attributes worth putting on the `agent.run` span:

```python
span.set_attributes({
    "app.stop_reason":      "no_progress",   # or final_answer, max_iterations, budget
    "app.iterations":       7,
    "app.total_cost_usd":   0.84,
    "app.tools_called":     12,
    "app.distinct_tools":   3,
    "app.repeat_calls":     4,               # loop indicator
    "app.side_effects":     1,               # mutating calls made
})
```

`stop_reason` is the free quality signal from
[../04-agents/agent-failure-modes.md](../04-agents/agent-failure-modes.md) — put it on the
span and you can query the distribution without any extra instrumentation.

---

## Sampling

You cannot store every trace at volume. Sample by *outcome*, not uniformly.

```python
def should_sample(request, outcome):
    # Always keep the interesting ones
    if outcome.failed:
        return True
    if outcome.attempts > 1:
        return True
    if outcome.negative_feedback:
        return True
    if outcome.latency_ms > SLOW_THRESHOLD:
        return True
    if outcome.cost_usd > EXPENSIVE_THRESHOLD:
        return True
    if request.tenant_tier == "enterprise":
        return True

    # A small slice of normal traffic, for baselines
    return random.random() < 0.02
```

Uniform sampling at 1% makes a 0.3% failure mode nearly invisible — which is precisely the
failure mode you need traces for.

**Decide late where you can.** Tail-based sampling (deciding after the request finishes,
when you know the outcome) is much better than head-based for this, at the cost of
buffering.

---

## Payload storage

Prompts and retrieved documents are the most useful thing in a trace and the most awkward
to keep.

```python
class PayloadStore:
    """Prompts and responses live here, not on the span."""

    def put(self, trace_id, span_id, payload, ttl_days=14):
        scrubbed = redact_secrets(payload)
        key = f"traces/{trace_id}/{span_id}.json"
        self.storage.put(key, scrubbed, ttl_days=ttl_days)
        self.audit.record_write(key, size=len(scrubbed))
        return key

    def get(self, key, requester):
        # Reading a payload is an audited event — these contain user data
        self.audit.record_read(key, requester)
        return self.storage.get(key)
```

Four properties that matter:

- **Separate retention** from your metrics. Metrics can live for a year; payloads usually
  shouldn't.
- **Access controlled and audited.** Reading a user's prompt is a privileged action.
- **Scrubbed on write**, not on read.
- **Referenced by the span**, so the trace UI can link out to it.

---

## What a good trace answers

Test your setup against these. If any takes more than a minute, something's missing.

- Why was this request slow? *(span durations)*
- What did the model actually see? *(payload)*
- Which documents were retrieved, and did they contain the answer? *(retrieval span)*
- Did this retry? Why? *(attempt spans)*
- What did this cost? *(cost attribute)*
- Which prompt and model version produced it? *(version attributes)*
- Why did the agent stop? *(stop_reason)*
- Show me all traces for users who rated us badly this week. *(feedback join)*

That last one is the highest-value query in the whole system and it only works if trace
IDs are stored with feedback.

---

## Interview questions

**1. "How would you trace an AI request?"**

Spans per stage — retrieval, rerank, generation, tool calls — with token counts, cost, and
version attributes. Retries as child spans. Payloads stored by reference. Then the key
point: most of this is normal tracing with the right attributes, not a new tool.

**2. "A customer complains about an answer from three days ago. Can you investigate?"**

Only if you have the trace ID, the retrieved documents, and payload retention that covers
three days. Then describe returning the trace ID to the client so support tickets carry
it.

**3. "Why store payloads separately from spans?"**

Size and sensitivity. Prompts are large and contain user data, so they need their own
retention, access control, and audit trail. The span carries a reference.

**4. "How do you sample when you can't keep everything?"**

By outcome, not uniformly — 100% of failures, retries, slow requests, expensive requests
and negative feedback; a small percentage of normal traffic. Uniform sampling hides rare
failures, which are exactly what you need.

**5. "How do you connect a thumbs-down to what caused it?"**

Store the trace ID with the feedback event. That join turns "our quality is 4% worse" into
"here are the 200 requests that caused it, with the retrieved documents."

**6. "How would you trace an agent?"**

Nested spans per iteration, each containing the model call and its tool calls, with
`stop_reason`, cost, iteration count and repeat-call count on the parent. A loop is
visually obvious in that structure and invisible in a flat log.

---

## What to remember

- Most of AI tracing is normal tracing with the right span attributes.
- The genuinely new parts are payload storage and joining quality signals.
- One trace ID, generated at the edge, propagated everywhere, returned to the client.
- Store the trace ID with user feedback — that join is the highest-value thing here.
- Retries must be child spans, or invisible retries stay invisible.
- Payloads go in separate storage: scrubbed, short retention, access-controlled, audited.
- Sample by outcome. Uniform sampling hides rare failures.
- Version attributes on every span make "what changed?" a filter instead of a guess.

---

**Next:** [retries.md](retries.md) — the operational policy around retrying.
