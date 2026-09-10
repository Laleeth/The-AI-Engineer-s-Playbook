# Model Routing

Routing means sending different requests to different models. Easy questions go to a
cheap fast model, hard ones go to an expensive strong model.

Done well it's the single biggest cost lever in most LLM products — 50–70% savings with
quality held or improved. Done badly it's a source of quiet, hard-to-find failures.

---

## Why it works

Look at any real traffic and you find it's lopsided:

```
"what's your refund policy"        ← a small model handles this fine
"reset my password"                ← same
"where's my order"                 ← same
"I was charged twice, then the
 refund went to a closed card,
 and now my account is locked"     ← this one needs the strong model
```

Typical split: 70–90% of requests are straightforward, and a small model answers them at
near-identical quality. The remaining 10–30% genuinely need more.

Paying frontier prices for all of it means overpaying on most of your traffic. Using the
cheap model for all of it means failing on the hard part — which is usually the part
where failure costs the most.

---

## Two ways to route

Everything else is a variation on these.

### Route on the input (decide before)

Look at the request, pick a model, call it once.

```python
def route_by_input(request):
    if is_simple(request):
        return small_model(request)
    return large_model(request)
```

**Fast and cheap** — one call, no extra latency.

**But** you're predicting difficulty from the question alone, and difficulty often
depends on things you can't see yet. "What's the status of my order?" is easy if the
order exists and hard if it's in a weird state.

### Route on the output (decide after)

Try the cheap model. Check the result. Escalate if it's not good enough. This is often
called a cascade.

```python
def cascade(request):
    draft = small_model(request)

    if good_enough(draft, request):
        return draft                    # ~80% of the time

    return large_model(request)         # the rest
```

**Much more reliable** — you're judging the actual answer, not guessing from the
question.

**But** escalated requests pay for both calls and take both models' latency. If 20% of
requests escalate, your p95 latency is now cheap-call + expensive-call.

That latency point catches people out. **Your hardest queries become your slowest ones**
— from users who are already having a bad time. Compute this against your latency
budget before committing.

---

## Use both

The best designs combine them, and the reason is worth stating: some routing decisions
should never be statistical.

```python
def route(request, user):
    # 1. Hard rules first — never route these down, whatever a classifier says
    if mentions_cancellation(request) or mentions_legal(request):
        return large_model(request)
    if user.tier == "enterprise":
        return large_model(request)
    if conversation_length(request) > 5:      # long chats are going badly
        return large_model(request)

    # 2. Try cheap
    draft = small_model(request)

    # 3. Check the actual output
    if passes_checks(draft, request):
        return draft

    # 4. Escalate
    return large_model(request)
```

The hard rules matter because **optimizing cost on an account-security question or a
legal threat is a business risk, not an engineering trade-off.** Use deterministic rules
where the stakes are high and statistics where they're not.

---

## The escalation check

Everything rests on this. If it's good, routing works. If it's bad, you either escalate
everything (no savings) or escalate nothing (quality drops silently).

Good signals, roughly in order of usefulness:

### 1. Build an explicit "I'm not sure" option

The best trick here. Design the model's output format so abstaining is a first-class
answer:

```python
SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["answered", "needs_specialist", "insufficient_information"],
        },
    },
    "required": ["answer", "status"],
}

def needs_escalation(result):
    return result["status"] != "answered"
```

This turns "is the model confident?" — which you can't reliably measure — into "which
branch did it pick?", which is a simple check.

### 2. Deterministic checks

Free, exact, and they catch real failures:

```python
def passes_checks(draft, request):
    if not valid_json(draft):
        return False
    if not all_citations_real(draft):
        return False
    if contains_placeholder(draft):        # "[INSERT NAME]"
        return False
    if breaks_policy(draft):
        return False
    return True
```

### 3. Retrieval signals

For RAG systems, weak retrieval predicts a weak answer:

```python
def weak_retrieval(docs):
    if not docs:
        return True
    if docs[0].score < THRESHOLD:
        return True
    # Flat score distribution = nothing clearly relevant
    if docs[0].score - docs[-1].score < 0.05:
        return True
    return False
```

**Warning:** score thresholds break when you change the embedding model, because scores
shift. Re-calibrate on every retrieval change, or this quietly stops working.

### 4. Log-probabilities, if available

Low token probabilities can indicate uncertainty. Useful, but noisy, and not available
from every provider.

### What doesn't work: asking the model how confident it is

"Rate your confidence 1–10" produces numbers that don't correlate well with correctness.
Models are poorly calibrated at self-reporting confidence, and they're *particularly*
bad at it when they're confidently wrong — which is exactly the case you need to catch.

---

## Where the router lives

Three options.

**In your application.** Simple, no extra network hop, easy to change. Fine when one
team is doing this.

**In a shared library.** Every team imports the same routing code. Consistent, still no
network hop. Good middle ground.

**In a gateway service.** All calls go through one service that routes. Central control,
one place for cost tracking and quota — but it's now in the critical path of every
request, and it costs a network round trip.

A useful split: **the mechanism (calling models, handling fallback, logging decisions)
is generic and belongs in a shared library. The policy (which model for which request)
is specific to each product and belongs in the application.**

That distinction resolves most "should the platform team own routing?" arguments.

---

## What to measure

Routing that isn't measured drifts, and you find out from your invoice.

```python
ROUTING_METRICS = [
    "route_distribution",      # % to each model — alert if it shifts
    "escalation_rate",         # % that escalate — the health signal
    "escalation_precision",    # of those, how many actually needed it
    "cost_per_request",        # by route and blended
    "p95_latency",             # by route — the escalated path is the slow one
    "quality_by_route",        # graded, per route
    "hard_rule_hit_rate",      # how often the deterministic rules fire
]
```

**Escalation rate is the metric that tells you most.** If it drifts from 20% to 50%,
something upstream broke — retrieval got worse, a prompt changed, or traffic shifted.
You'll learn it from this number before you learn it from customers.

**Escalation precision** takes more work but is worth it: sample escalated requests,
have both models answer, and grade both. If the small model's answer was fine most of
the time, your check is too eager and you're leaving savings on the table.

---

## Things that go wrong

**The traffic mix shifts.** You built for 80/20. A big customer joins with much harder
queries and it's now 50/50. Your costs double and quality assumptions no longer hold.
Alert on route distribution.

**Threshold drift after an upgrade.** You change the embedding model; retrieval scores
shift; your escalation threshold now means something different. Version thresholds with
the thing they depend on.

**The cheap model learns to abstain.** If you tune its prompt against a metric that
rewards escalation, escalation creeps up and savings evaporate. Watch the rate.

**Latency blows the budget.** 20% of requests taking cheap+expensive latency may exceed
your p95 target. Options: run the expensive model in parallel for high-risk segments
(pay double, save time), or negotiate a segment-specific latency target.

**Nobody can explain a decision.** A customer complains; you can't say why their request
got the small model. Log the routing decision and its inputs on every request.

**Quality drops and nobody notices for six weeks.** Cost is on a dashboard; quality
isn't. Build the quality dashboard *before* you ship the routing.

---

## Rolling it out safely

```
1. Log what a router WOULD do, without acting on it.  ← shadow mode
   Check the distribution looks sane.

2. Route 5% of traffic. Watch quality and escalation rate.

3. Ramp: 25% → 50% → 100%, with rollback at each step.

4. Keep watching route distribution forever.
```

Step 1 is cheap and skipped constantly. It tells you whether your rules do what you
think before any customer is affected.

---

## Interview questions

**1. "How would you cut LLM costs by half without hurting quality?"**

Routing is the headline answer, but the strong version starts elsewhere: check input
tokens first (prompt caching, trimming retrieved context), because that's often a bigger
and safer lever. Then routing. Then explain the escalation check, because that's the
part that decides whether it works.

**2. "Route on the input or the output?"**

Input is cheaper and faster; output is more reliable because you judge the actual
answer. Use hard rules on the input for high-stakes cases and a cascade for the rest.
Then mention the latency cost of cascading.

**3. "How do you know if a request needs the bigger model?"**

Build an explicit abstain option into the output schema, plus deterministic checks and
retrieval signals. Say clearly that asking the model to rate its own confidence doesn't
work well, especially when it's confidently wrong.

**4. "Your escalation rate went from 20% to 55% overnight. What happened?"**

Something upstream. Retrieval degraded, a prompt changed, the traffic mix shifted, or a
threshold became stale after an embedding change. The point is that escalation rate is a
*detector* for upstream problems, not just a cost number.

**5. "What's the risk of routing?"**

Silent quality loss on the segment you routed down, invisible because cost is monitored
and quality often isn't. And routing on statistics where you should have used a rule —
account security, legal, cancellations.

---

## What to remember

- Most traffic is easy. Paying top prices for all of it is the waste.
- Hard rules for high-stakes requests; statistics for the rest.
- Routing on the output beats routing on the input, at the cost of latency.
- Build an explicit "not sure" branch into the output schema.
- Model self-reported confidence is unreliable — don't route on it.
- Escalation rate is your early warning for upstream problems.
- Score thresholds break when you change the embedding model.
- Shadow-mode the router before it decides anything real.
- Cost is easy to see; quality isn't. Build that dashboard first.

---

**Next:** [small-vs-large-model.md](small-vs-large-model.md) — when a small model is
genuinely enough.
