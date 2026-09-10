# Human-in-the-Loop Patterns

"A human reviews it" is the most common answer to "what if the model is wrong," and on its
own it is not a design. It becomes one when you can say *which* outputs a human sees, what
they see, what happens while they decide, and what the review capacity is.

Get those wrong and you build a system whose safety property is a queue nobody drains.

---

## Review is a budget, not a principle

Human review has a throughput, and it is small. Compute it before designing around it:

```python
def review_capacity(reviewers, minutes_per_item, hours_per_day=6):
    """6 productive hours, not 8 — review is attention-heavy work."""
    per_reviewer = hours_per_day * 60 / minutes_per_item
    return reviewers * per_reviewer

review_capacity(reviewers=4, minutes_per_item=3)      # 480 items/day

def review_share(daily_volume, capacity):
    return capacity / daily_volume

review_share(daily_volume=50_000, capacity=480)       # ≈ 0.0096
```

Four reviewers can look at **roughly 1% of 50,000 daily items.** That number is the design
constraint, and it decides everything downstream: you are not choosing whether to review,
you are choosing which 1%.

Teams that skip this arithmetic build "human reviews low-confidence outputs," discover
that 30% of outputs are low-confidence, and end up with a queue that grows forever. The
queue then does what unmanaged queues do — it gets bypassed, batch-approved, or quietly
ignored, and the safety property was never real.

---

## Four places a human can go

**Before the action — approval.** The model proposes, a human approves, then it executes.
The strongest control and the most expensive. Correct when the action is irreversible and
consequential: sending money, deleting data, contacting a customer, changing production.

**After the action — audit.** It executes, a human reviews a sample afterwards. Cheap,
scales, and catches systematic problems rather than individual ones. Correct when actions
are reversible and individual errors are tolerable.

**Alongside — the model assists a human who was doing the work anyway.** Draft a reply, the
agent reviews and sends. Often the highest-value pattern and the one that gets dismissed
as unambitious; it converts a quality problem into a speed improvement, and quality
problems are the ones that hurt.

**On escalation — the model hands off.** It handles what it can and routes the rest. The
design question here is what triggers the handoff, and it is harder than it looks.

Choose per action class, not per system. The same product usually needs approval for
refunds, audit for classifications, and assistance for drafting.

---

## Confidence is the wrong trigger

The obvious escalation rule is "escalate when the model is unsure," and self-reported
confidence does not work. [02-model-selection](../02-model-selection/README.md) states it
plainly: asking a model to rate its confidence produces numbers that don't track
correctness, and they're worst exactly when the model is confidently wrong — which is the
case you built the escalation for.

Triggers that do work, roughly in order of reliability:

- **Stakes.** Route by what the action *does*, not by how the model feels. Refunds over
  $500 get approved by a person, always. This is a rule, it's auditable, and it can't be
  argued out of.
- **Explicit abstention.** Build an "I don't know" branch into the output schema and
  escalate on it. The model choosing not to answer is far more reliable than the model
  scoring itself.
- **Disagreement.** Two methods — a rule engine and a model, or two different models —
  reaching different conclusions is a good escalation signal and needs no calibration.
- **Novelty.** Inputs unlike anything in your evaluation set. You don't know the system
  works there because you never measured it there.
- **Retrieval failure.** Nothing relevant retrieved, so any answer is unsupported. This
  should abstain, not generate — the rule from
  [07-production](../07-production/README.md).

The general shape: **hard rules where the stakes are high, statistics for the rest.** The
same division section 02 applies to model routing.

---

## What happens while the human decides

The part that is almost always underspecified, and the source of the ugliest failures.

- **What does the user see?** "Pending review" is honest. Silence looks like a bug and
  generates support tickets that cost more than the review.
- **What is the timeout, and what happens at it?** A review request with no expiry becomes
  a permanently pending action. Decide the default — approve, reject, or escalate — and
  make it explicit. For anything consequential the safe default is reject, because failing
  closed is the rule that survives contact with an unstaffed weekend.
- **Is the work held or discarded?** Holding costs storage and a state machine; discarding
  means redoing it, and for an agent mid-task the intermediate state may not be
  reproducible.
- **Who is on the hook overnight?** A queue with a four-hour SLA and no weekend coverage
  has a 60-hour SLA on Friday evening.

```python
def review_request(action, reviewer_queue, timeout_hours, on_timeout):
    """on_timeout must be an explicit choice, not an accident."""
    assert on_timeout in {"reject", "escalate", "approve"}
    if on_timeout == "approve" and action.is_irreversible:
        raise ValueError("irreversible actions must not auto-approve on timeout")
    return reviewer_queue.enqueue(action, expires_in_hours=timeout_hours,
                                  default=on_timeout)
```

---

## Design the review, not just the queue

A reviewer approving 200 items a day under time pressure will rubber-stamp unless the
interface makes the decision cheap and the error visible. Rubber-stamping is the normal
outcome of a badly designed review, not a failing of the reviewer.

What makes review effective:

- **Show the evidence, not just the conclusion.** The retrieved passages, the tool calls,
  the input. A reviewer who has to go find context will stop doing it by item 30.
- **Make rejection as easy as approval.** If approve is one click and reject opens a form,
  you have designed an approval machine and your approval rate will tell you so.
- **Say what's unusual about this item.** "Escalated because the amount exceeds $500" tells
  the reviewer where to look. An undifferentiated queue makes every item equally
  suspicious, which means none of them are.
- **Capture the reason for rejection.** Free text is fine; the rejections are your best
  source of evaluation cases, and they cost nothing extra to collect.
- **Watch approval rate as a health metric.** Approaching 100% means either the escalation
  trigger is miscalibrated or review has become a formality. Both are worth knowing, and
  the metric is free.

---

## Review output is training data and eval data

The most under-used asset in these systems. Every human decision is a labeled example
produced by someone whose judgment you trust, generated by the traffic distribution you
actually serve.

Feed it back:

- **Into the golden dataset**, especially rejections — which are, by definition, cases
  where the system was wrong on real traffic. This is exactly the "sample from production,
  not from your imagination" requirement in
  [golden-datasets.md](../05-evaluation/golden-datasets.md).
- **Into judge calibration.** Human decisions are the ground truth an LLM judge is checked
  against, and [llm-as-judge.md](../05-evaluation/llm-as-judge.md) asks for about fifty
  cases a month. Review already produces them.
- **Into the escalation trigger.** If 95% of escalated items are approved, the trigger is
  too broad and you're spending scarce review capacity on the wrong 1%.

A review process that produces no data is doing a third of its job.

---

## What to monitor

- **Queue depth and age of the oldest item.** The oldest item is where the SLA breaks
  first.
- **Review capacity utilization**, against the arithmetic above. Above ~80% sustained, the
  queue will run away on any bad day.
- **Approval rate**, trended and segmented by trigger.
- **Time-to-decision** distribution, not mean.
- **Timeout rate and what the default did**, which is the number that tells you whether
  your safety property is real.
- **Post-review error rate** — items approved by a human that turned out wrong. This
  measures whether review is working at all, and almost nobody tracks it.

---

## Interview questions

**"Add human review to a system processing 50,000 items a day."**

First the arithmetic: with four reviewers at three minutes an item, that's around 480
items a day, about 1%. So the design question isn't whether to review, it's which 1% — and
I'd route by stakes rather than model confidence, because self-reported confidence doesn't
track correctness. Then specify what happens while the item waits: what the user sees, the
timeout, and the default at timeout, which for anything irreversible has to be reject.

**"Your review queue has 12,000 items and grows daily. What went wrong?"**

The escalation trigger admits more than capacity can absorb, and it was never checked
against the arithmetic. Two fixes, and both are needed: narrow the trigger to what
genuinely needs a person — usually meaning stakes-based rules rather than a confidence
threshold — and give the queue a timeout with an explicit default so items can't
accumulate forever. I'd also look at approval rate: if it's near 100%, most of that queue
never needed review.

**"How do you know human review is actually improving things?"**

Post-review error rate — how often an item a human approved turned out to be wrong. If
that's close to the unreviewed error rate, review is a formality. Alongside it, approval
rate by trigger to see whether escalation is well-targeted, and I'd be feeding rejections
into the eval set, which is the return on review nobody collects.

---

## What to remember

- Compute review capacity first. It's usually about 1% of volume, and that fraction is the
  design constraint.
- Route by stakes, not by model confidence. Self-reported confidence fails worst exactly
  when you need it.
- Approval before, audit after, assistance alongside, escalation on trigger — choose per
  action class, not per system.
- Specify what happens while the human decides: user-visible state, timeout, and an
  explicit default that fails closed for irreversible actions.
- A queue nobody drains is not a safety property. Watch depth, oldest item, and approval
  rate.
- Review output is your best eval data. Rejections especially.
