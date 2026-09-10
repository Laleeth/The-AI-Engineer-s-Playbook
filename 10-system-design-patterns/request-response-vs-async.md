# Request/Response, Async Jobs, and Streaming

The first architectural decision in an AI product is how the work is delivered, and it is
usually made by accident — someone writes an HTTP handler that calls a model, and the
shape of the system is set.

The choice matters because inference is slow and variable. A request that takes 40 seconds
breaks assumptions that a 200-millisecond request never tested.

---

## The three shapes

| | Client waits | Good for | Breaks when |
|---|---|---|---|
| **Request/response** | Yes, for the whole answer | Short, bounded work — classification, extraction, short answers | Latency exceeds a timeout somewhere in the chain |
| **Streaming** | Yes, but sees partial output | Anything a person reads as it appears | The consumer isn't a person, or the client can't hold a connection |
| **Async job** | No — submits, polls or is notified | Long or batch work, anything over ~30 seconds | You've added state, storage and a completion path to maintain |

The mistake is treating this as one decision for the whole product. Most systems need at
least two of these, and the useful question is per surface: *who is waiting, and what are
they waiting for?*

---

## Request/response, and where it fails

Simplest, and correct more often than architecture discussions suggest. Use it when the
work is genuinely short and bounded.

The failure is that "bounded" is an assumption, and inference output length is variable by
nature. A handler tested on 200-token answers meets a 2,000-token answer in production and
sits for 40 seconds, holding a connection through every layer between the client and the
model.

```python
def timeout_budget(client_timeout, hops):
    """Every hop needs a shorter timeout than the one outside it."""
    return [round(client_timeout * (0.9 ** i), 1) for i in range(hops)]

timeout_budget(60, hops=4)     # [60.0, 54.0, 48.6, 43.7]
```

The rule: **timeouts must decrease inward.** If a load balancer times out at 30 seconds
while your application waits 60, the client gets a 504 while your server happily continues
generating — paying for tokens nobody will read, and logging a success. That mismatch is
one of the most common and most confusing production bugs in AI systems, because the
application logs and the client's experience disagree completely.

Bound output explicitly with `max_tokens`, and treat that bound as part of the API
contract rather than a safety net.

---

## Streaming, and what it doesn't fix

Streaming changes perceived latency, not actual latency. Time to first token becomes the
number the user feels, and total time stops mattering as much — a large improvement in
experience for exactly zero improvement in throughput.

What it costs:

- **A held connection per active request**, which changes your concurrency model. Server-
  sent events or websockets, and both need thought about proxies, timeouts and reconnects.
- **Error handling after output has begun.** A failure 300 tokens in cannot be turned into
  a clean 500. You need an error event in the stream and a client that renders it.
- **Cancellation that must actually free the GPU.** A client disconnecting should stop
  generation; if it doesn't, you pay for the full response for a user who left. That
  requires the serving layer to honor cancellation —
  [06-inference-serving](../06-inference-serving/README.md) — and it is worth explicitly
  testing rather than assuming.
- **Anything that needs the whole response first.** Validating structured output,
  filtering, or scanning for policy violations can't stream — or streams optimistically
  and retracts, which is worse.

That last one is a real design constraint. If your response must be validated as JSON
before the user sees it, you are not streaming to the user; you are streaming to your own
validator and then delivering. Say so, and set expectations on latency accordingly.

---

## Async jobs, and the state they bring

Once work takes minutes, the client cannot wait and the shape has to change: submit,
receive an identifier, and collect the result later.

```python
def submit(request, store, queue):
    job_id = new_id()
    store.put(job_id, {"state": "queued", "submitted_at": now()})
    queue.push({"job_id": job_id, "request": request})
    return job_id            # 202 Accepted, with a status URL
```

What you have now built, and must maintain: durable job state, a queue, workers, a
completion path, a retention policy for results, and an answer for every failure mode the
client can no longer observe directly.

Design decisions worth making explicitly rather than by default:

- **Polling or notification?** Polling is simpler and works everywhere; a webhook or
  websocket is better at scale and needs delivery guarantees, retries and an endpoint the
  customer maintains. Offer polling first; add notification when the polling volume
  justifies it.
- **How long do results live?** Storage cost against a client that comes back tomorrow.
  Pick a number and put it in the API documentation.
- **What does the client do about a failed job?** They can't retry a stream they never saw.
  The job state needs a failure reason specific enough to act on.
- **Idempotency on submit.** A client that retries submission must not create a second
  job. Take an idempotency key, and honor it — this is the same discipline as
  [ingestion-pipelines.md](../09-data-pipelines/ingestion-pipelines.md).

Queue depth is your capacity signal here, and it's a good one: it's a leading indicator,
it's what [autoscaling](../06-inference-serving/autoscaling.md) should scale on, and it
tells the user something honest if you surface it as a position or an estimate.

---

## Choosing, in one question

Work down this list:

1. **Is a person watching the output appear?** → Streaming.
2. **Is the work reliably under a few seconds, with output bounded?** →
   Request/response.
3. **Anything else** — long work, batch work, work whose duration you can't bound, or a
   consumer that is another system → Async job.

And a fourth case worth naming because it's common and often mishandled: **a person
watching, but the work takes minutes.** An agent doing a multi-step task, a long report.
Neither streaming tokens nor a bare job id serves them well. Stream *progress* — steps
completed, current activity, partial results — over the async job. The user gets feedback,
you get the async architecture, and nobody holds a connection for six minutes hoping.

---

## Mixing shapes without duplicating logic

Most products end up with all three, and the failure is implementing the work three times.
Keep one execution path and vary the delivery:

```python
def execute(request):
    """One implementation. Delivery is a wrapper, not a fork."""
    return pipeline.run(request)

def handle_sync(request):
    return execute(request)

def handle_stream(request):
    yield from execute(request).stream()

def handle_async(request, store, queue):
    return submit(request, store, queue)      # worker calls execute()
```

When the three paths are separate implementations they diverge — a prompt fix lands in
two of them, a guardrail in one. The differences that are real (cancellation, partial
output, result storage) belong in the delivery wrappers, not in the work.

---

## What to monitor

- **Time to first token and total time**, separately, for anything streaming.
- **Timeout mismatches**: requests where the server completed after the client gave up.
  This should be near zero and usually isn't.
- **Cancellation rate, and whether cancellation actually stopped the work** — measured by
  tokens generated after disconnect.
- **Queue depth and time-in-queue** for async work, plus age of the oldest queued job.
- **Job completion rate and failure reasons**, since the client can't see the failure
  directly.
- **Result retrieval rate** — jobs that complete and are never collected are wasted spend
  and a signal that a notification path is missing.

---

## Interview questions

**"Your API returns 504s but the logs show requests succeeding. What's happening?"**

A timeout mismatch: something in front of the application — load balancer, gateway, CDN —
gives up before the application does, so the client sees a gateway error while your server
finishes normally and logs a 200. Fix by making timeouts decrease inward from the client's
budget, and bound output length so the tail can't exceed it. It's worth checking what
happens to the generation after the client disconnects, too, because you're likely paying
for it.

**"When would you not stream?"**

When the consumer isn't a person, when the whole response must be validated or filtered
before anyone sees it, or when the client can't hold a connection. Streaming improves
perceived latency and costs you clean error handling, cancellation correctness and any
whole-response processing. For a machine consumer it's complexity with no benefit.

**"A user runs an agent task that takes four minutes. What shape is that?"**

An async job, with progress streamed over it. A bare job id and a polling loop gives a
person nothing to look at for four minutes; holding an HTTP connection open for four
minutes fails at some proxy. Submit, return an id, and stream step-level progress and
partial results so the user can see it working — and can cancel, which needs to genuinely
stop the work.

---

## What to remember

- Pick the shape per surface, asking who is waiting and for what. Most products need
  more than one.
- Timeouts must decrease inward from the client. The mismatch produces 504s alongside
  success logs, and you pay for output nobody reads.
- Streaming fixes perceived latency and costs error handling, cancellation and any
  whole-response validation.
- Async jobs bring durable state, a completion path, retention and submit-idempotency.
  Budget for them.
- A person waiting on minutes of work wants progress streamed over an async job, not
  either extreme.
- One execution path, three delivery wrappers. Forked implementations drift.
