# Online Evaluation

Offline evaluation tells you whether a change looks better on your test set. Online
evaluation tells you whether it *is* better for real users.

They disagree more often than people expect, and when they do, online wins.

---

## Why offline isn't enough

Your test set is a sample of what you *think* production looks like. Production is:

- messier inputs than you tested
- a different mix of topics and languages than you sampled
- users who adapt to your system in ways you didn't predict
- longer conversations than your dataset
- cases you never imagined

A change can improve your test set and hurt real users. That's not a rare edge case —
it's the most common way an AI feature quietly gets worse.

The framing that keeps this straight:

> **Offline eval prevents disasters cheaply. Online measurement tells you what's
> actually better.**

Different jobs. Trying to make offline do the second job is where teams get stuck.

---

## Signals you can collect without labels

You don't need humans grading production traffic. Users generate signal constantly if
you record it.

### Explicit feedback

Thumbs up/down. Simple, direct, and heavily biased — people rate when they're annoyed,
rarely when things work.

```python
@dataclass
class Feedback:
    trace_id: str            # ties back to the exact request
    rating: int              # 1 = up, -1 = down
    comment: str | None
    timestamp: float
```

Use it as a **trend**, not an absolute. "Thumbs-down rate went from 3% to 7%" is
meaningful. "Our satisfaction is 94%" is not, because most people never rate.

The `trace_id` is the important field. Without it you have a complaint you can't
investigate. With it you can pull the exact prompt, retrieved documents, and output.

### Implicit signals — usually better

What users *do* tells you more than what they say.

| Signal | What it suggests |
|---|---|
| Copied the answer | It was useful |
| Rephrased and asked again | First answer was wrong or unclear |
| Asked a follow-up | Could be good (engaged) or bad (incomplete) |
| Abandoned mid-response | Too slow, or clearly going wrong |
| Escalated to a human | Failed |
| Edited the draft heavily | Draft wasn't good enough |
| Closed immediately after | Got what they needed, or gave up |

```python
@dataclass
class Interaction:
    trace_id: str
    copied_output: bool
    asked_followup: bool
    rephrased_similar: bool      # very strong negative signal
    escalated: bool
    edit_distance: float | None  # for draft-and-review products
    time_to_next_action_s: float
```

**Rephrasing is the most useful single signal** in most products. If someone asks
almost the same thing again in different words, the first answer failed. It's easy to
detect and rarely tracked:

```python
def is_rephrase(prev_query, this_query, threshold=0.85):
    """Similar question, asked again quickly = the first answer didn't work."""
    return cosine(embed(prev_query), embed(this_query)) > threshold
```

### Free labels from your workflow

Some products generate ground truth as a side effect. If a human reviews every AI
output before it goes out, their edits are labels:

```python
def draft_quality(draft, sent):
    """How much did the human change it?"""
    return 1.0 - (levenshtein(draft, sent) / max(len(draft), len(sent)))
```

This is the best signal available — continuous, free, real, and it covers your actual
traffic distribution rather than a sample of it.

One caution: a tired agent under queue pressure accepts a mediocre draft rather than
rewriting it. Low edit distance can mean "good draft" or "busy reviewer." Validate the
proxy occasionally against sampled human grading.

### The business metric

The one that actually justifies the feature:

- Support: tickets resolved without a human, handling time per ticket
- Sales: conversion, time to first response
- Internal tools: time saved per task
- Search: did they find it and stop searching

Slow and noisy, but it's the only one that answers "is this worth the money?"

---

## Rolling out safely

Never switch everyone at once. Ramp, and watch at each step.

```python
STAGES = [
    {"traffic": 0.01, "hold_hours": 2,  "watch": ["error_rate", "p95_latency"]},
    {"traffic": 0.05, "hold_hours": 12, "watch": ["error_rate", "p95_latency",
                                                  "thumbs_down_rate", "cost"]},
    {"traffic": 0.25, "hold_hours": 24, "watch": ["...", "escalation_rate"]},
    {"traffic": 0.50, "hold_hours": 48, "watch": ["...", "resolution_rate"]},
    {"traffic": 1.00, "hold_hours": 0,  "watch": ["everything"]},
]
```

Notice the watch list grows. Early stages watch fast signals — errors, latency, cost —
because those appear in minutes. Later stages watch slow signals — resolution rate,
satisfaction — because they need volume and time.

### Automatic rollback

```python
ROLLBACK_RULES = [
    # Fast and unambiguous — trigger immediately
    {"metric": "error_rate",      "worse_than": 0.02, "window_min": 5},
    {"metric": "p95_latency_ms",  "worse_than": 1.5,  "window_min": 10},  # 1.5x

    # Quality — needs more data
    {"metric": "thumbs_down_rate", "worse_than": 1.4, "window_min": 120},
    {"metric": "escalation_rate",  "worse_than": 1.3, "window_min": 240},
]
```

Automate the fast ones. Errors and latency are unambiguous and need no judgement — if
they spike, roll back and investigate afterwards. Quality signals need more time and
usually a human looking at examples.

---

## A/B tests, and their limits

Split traffic, compare. The standard tool, with real caveats for AI features.

```python
def assign(user_id, experiment):
    """Stable assignment — the same user always gets the same version.
    Randomising per request makes multi-turn conversations incoherent."""
    h = hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest()
    return "treatment" if int(h[:8], 16) % 100 < 50 else "control"
```

**Assign by user, not by request.** If a user gets version A for turn 1 and version B
for turn 2, the conversation breaks and you've measured neither version.

### How long it takes

This is the number people don't compute before starting.

```python
def days_needed(baseline_rate, min_effect, daily_users, power=0.8):
    """Roughly how long to detect a change of this size."""
    p1, p2 = baseline_rate, baseline_rate + min_effect
    pooled = (p1 + p2) / 2
    z_a, z_b = 1.96, 0.84
    n_per_arm = ((z_a + z_b) ** 2 * 2 * pooled * (1 - pooled)) / (min_effect ** 2)
    return math.ceil(2 * n_per_arm / daily_users)
```

Example: 5% thumbs-down rate, want to detect a 1-point change, 2,000 users a day.
That's around **three weeks**.

Three weeks per change is a real constraint on how fast you can ship. Which is why
offline eval exists — it's the fast, cheap filter, and A/B is for decisions that matter
enough to wait for.

### Things that go wrong

**Peeking.** Checking daily and stopping when it looks significant inflates your false
positive rate badly. Decide the sample size up front, or use a method designed for
continuous monitoring.

**Novelty effects.** Users react to *change*, not just quality. A new behaviour gets
more engagement for a week regardless of whether it's better. Run long enough to see it
settle.

**Contamination.** Support agents talk to each other. If half your agents get the new
version, they discuss it, and the control group's behaviour changes too.

**Measuring the wrong thing.** Engagement goes up 6%. Good? Could mean more useful, or
more verbose, or *worse* — users having to ask follow-ups. Engagement without a
completion or satisfaction metric alongside it is not evidence of quality.

---

## Shadow mode

Run the new version on real traffic without showing anyone the output. Log both, compare
offline.

```python
async def serve_with_shadow(request, current, candidate):
    result = await current(request)      # this is what the user sees

    # Fire and forget — must never affect the user's request
    asyncio.create_task(run_shadow(request, candidate, result))

    return result


async def run_shadow(request, candidate, live_result):
    try:
        shadow = await asyncio.wait_for(candidate(request), timeout=30)
        log_comparison(request.trace_id, live_result, shadow)
    except Exception:
        log.warning("shadow failed", exc_info=True)   # never propagate
```

**This is the most underrated technique here.** You get a full-scale comparison on real
production traffic at zero risk to users. It's the standard way to de-risk a model
migration, and its absence in a migration plan is a sign of inexperience.

Costs: you pay for both, and you can't measure user reaction — only whether outputs
differ and how. For anything with side effects, the shadow must run against a sandbox,
not production systems.

---

## When offline and online disagree

They will. Treat the disagreement as information about your offline metric.

| Offline | Online | Likely explanation |
|---|---|---|
| Better | Worse | Eval set doesn't match production. **Most common.** |
| Better | No change | Change too small to matter, or you measured a proxy nobody cares about |
| No change | Better | You're not measuring what matters offline |
| Worse | Better | Your offline metric penalises something users like |

**Believe online**, provided the experiment is properly randomised and powered. Then go
and fix the offline metric — every disagreement is a bug in your cheap proxy, and
fixing it makes the next hundred decisions better.

Teams that shrug at disagreements end up with an eval suite nobody believes, which is
the same as having none.

Worth tracking as a metric on your evaluation itself:

```python
def offline_online_agreement(history):
    """Of changes where both were measured, how often did they agree on
    direction? This is a quality score for your offline suite."""
    both = [c for c in history if c.offline_delta and c.online_delta]
    agree = sum(1 for c in both
                if (c.offline_delta > 0) == (c.online_delta > 0))
    return agree / len(both) if both else None
```

---

## What to watch every day

If you have room for one dashboard:

```python
DAILY = {
    # Health
    "error_rate", "p95_latency_ms", "timeout_rate",

    # Quality proxies
    "thumbs_down_rate", "rephrase_rate", "escalation_rate",

    # Economics
    "cost_per_request", "mean_input_tokens", "mean_output_tokens",

    # Behaviour drift — cheap, and moves first
    "refusal_rate", "abstention_rate", "mean_output_length",

    # Business
    "resolution_rate", "tasks_completed",
}
```

**Segment all of it.** By customer tier, language, and topic. An overall thumbs-down
rate of 4% might be 2% for English and 19% for German, and you'd never know.

**Alert on rate of change, not absolute values.** "Thumbs-down doubled in two hours" is
actionable. "Thumbs-down is 6%" needs context you don't have at 3am.

---

## Interview questions

**1. "How do you measure quality in production?"**

Implicit signals first — rephrasing, escalation, copying, edit distance — because
explicit feedback is sparse and biased. Then business metrics. Mention the trace ID
that connects a complaint back to the exact request.

**2. "How do you roll out a model change safely?"**

Shadow mode first for a full-scale risk-free comparison, then a ramp with automatic
rollback on fast signals and human review of quality signals. Explain why the watch
list grows at each stage.

**3. "Offline says neutral, online says 6% better engagement. Ship it?"**

Probably, but ask what engagement means here. It could mean more useful, more verbose,
or users needing more follow-ups. Ask for a completion or satisfaction metric
alongside. Then treat the disagreement as a bug in the offline metric.

**4. "Your A/B test needs three weeks. How do you ship weekly?"**

Offline eval and shadow mode as fast filters; A/B only for changes big enough to
justify the wait. Ramp with automatic rollback so most changes are safe without a full
experiment. Be honest that not everything gets a properly powered test.

**5. "A customer complains about an answer from three days ago. Can you investigate?"**

Only if you logged trace IDs, prompts, retrieved documents, and model versions with
retention long enough. If not, that's the finding — and it's a common one.

**6. "What's the single best production signal?"**

Depends on the product. For draft-and-review, edit distance — continuous, free, real.
For chat, rephrase rate. Both beat thumbs because they cover all traffic, not the
sliver that bothers to rate.

---

## What to remember

- Offline prevents disasters. Online tells you what's actually better.
- Implicit signals beat explicit ones. Rephrasing is the most underused.
- If humans review outputs, their edits are free labels.
- Trace IDs on everything, or complaints are uninvestigable.
- Shadow mode gives a full-scale comparison at zero user risk. Use it before migrations.
- Assign A/B by user, not per request.
- Compute how long the test takes *before* starting it.
- When offline and online disagree, believe online and fix the offline metric.
- Alert on rate of change, and segment everything.

---

**Back to:** [the evaluation index](README.md)
