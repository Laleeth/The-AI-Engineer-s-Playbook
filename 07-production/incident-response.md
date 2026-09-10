# Incident Response

> **This file is the process.** For practice — incidents with evidence revealed step by
> step — see
> [../12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md).
> For the debugging *technique*, see
> [../00-interview-framework/how-to-answer-debugging.md](../00-interview-framework/how-to-answer-debugging.md).

AI incidents differ from ordinary service incidents in three ways that change how you run
them:

1. **Most failures return HTTP 200.** You often find out from a customer, not an alert.
2. **Half of them have no deploy.** A prompt edit, a document import, a provider updating a
   model.
3. **Some are actively doing damage while you diagnose.** An agent with write access
   doesn't wait for your investigation.

---

## Severity for AI systems

Standard severity ladders don't capture "it's up and answering wrongly." Add the axes that
matter.

| Level | Meaning | Examples |
|---|---|---|
| **SEV1** | Data exposure, or ongoing external damage | Cross-tenant leak; agent taking destructive actions |
| **SEV2** | Down, or badly wrong at scale | Provider outage with no fallback; quality collapse |
| **SEV3** | Degraded, or expensive | Latency 3×; cost 3×; one segment failing |
| **SEV4** | Isolated or cosmetic | One customer, one query type |

Two AI-specific rules:

**Silent wrongness at scale is SEV2, not SEV3.** A system confidently answering wrongly for
50,000 users is worse than one that's down, because nobody knows to stop trusting it.

**Anything with ongoing external side effects is SEV1** regardless of scale. An agent
posting messages or opening pull requests is doing damage every minute you diagnose.

---

## The first ten minutes

Order matters, and it's not the order most people follow.

### 1. Stop ongoing damage — before diagnosing

If the system is actively doing something harmful, kill it first.

```python
KILL_SWITCHES = {
    "agent_writes":     "disable the agent's service account",
    "ai_feature":       "feature flag off, fall back to non-AI path",
    "batch_processing": "pause the queue consumer",
    "cache_reads":      "bypass cache (for suspected poisoned entries)",
}
```

The agent incident that ran overnight — 4,200 iterations, 312 Slack messages, 6 duplicate
pull requests — kept doing damage the whole time because nobody disabled it first. Every
minute of diagnosis cost more messages.

**If mitigation is cheap and reversible, do it before you understand the problem.** You can
always investigate from a stable state; you cannot un-send 300 messages.

### 2. Preserve evidence — before rolling back

For anything that might be a data-exposure incident, snapshot logs, traces, and the
specific requests *before* you redeploy. You may need to prove scope later, to a customer's
legal team or a regulator, and a rollback can destroy the state that proves it.

### 3. Escalate the ones that aren't yours to decide

Cross-tenant data exposure has notification obligations in many jurisdictions and
contracts. That's a legal and security decision, not an engineering one. Escalating while
you investigate — rather than after — is the difference between an incident and a career
event.

### 4. Then scope

- What's the symptom? Wrong, slow, erroring, expensive?
- Everyone or a subset? *Which* subset?
- What percentage?
- **When did it start?**

The subset question is the highest-value one. Failures concentrated in one language, one
tenant, one document type, or multi-turn conversations cut the problem enormously in one
question.

---

## "When did it start?" and the no-deploy list

The cheapest high-value question in debugging. If it worked last week, something changed —
and the list of things that change is short and enumerable.

**Things that change production behavior without a deploy:**

```python
NON_DEPLOY_CHANGES = [
    "prompt edited in a management UI",
    "documents added, removed, or re-indexed",
    "provider silently updated a model version",
    "another team changed a shared config (embedding model, retrieval params)",
    "traffic mix shifted — new customer, marketing campaign",
    "a quota, rate limit, or certificate changed",
    "data volume crossed a threshold (index no longer fits in memory)",
    "a feature flag rolled out",
]
```

> **"Nothing was deployed" is not the same as "nothing changed."**

This is why a change timeline overlaying prompt versions, corpus updates, model versions
and config changes onto your metrics is worth building. It turns this question from an
investigation into a glance.

---

## The split that halves the problem

For any quality incident, ask first:

> **Was the right information in the prompt?**

- **No** → retrieval problem. Chunking, search, ranking, filtering.
- **Yes** → generation problem. Prompt, model, context length.

Completely different fixes. Teams routinely spend a week tuning prompts on what turns out
to be a retrieval failure.

For other symptoms:

| Symptom | The splitting question |
|---|---|
| Slow | Which stage? Get latency decomposed by component |
| Expensive | Input or output tokens? Requests or tokens per request? |
| Errors | Ours, the provider's, or a dependency's? |
| Agent misbehaving | Did it fail, or succeed at the wrong thing? |

---

## Narrowing, out loud

The habit that makes investigations fast:

```
hypothesis → what would disprove it → ask for exactly that → update → repeat
```

Say what you're ruling out. Narrowing is progress:

> "Retrieval latency is flat, so this isn't the vector database, the embedding model, or
> corpus size. Output tokens are flat, so it isn't decoding. That leaves prefill or
> something provider-side."

And **do the arithmetic** — numbers kill hypotheses faster than intuition:

> "Input tokens are up 15% but LLM latency is up 300%. Prefill is roughly linear in input
> length, so 15% can't explain 300% — unless something crossed a threshold. That points at
> prefix cache invalidation, not gradual growth."

---

## Roles

Even a small team benefits from naming these, because during an incident people default to
all doing the same thing.

| Role | Does | Does not |
|---|---|---|
| **Commander** | Decides, tracks state, delegates | Debug personally |
| **Investigator** | Forms hypotheses, asks for data | Communicate externally |
| **Comms** | Status page, support, stakeholders | Debug |
| **Scribe** | Timeline of what was tried and found | Anything else |

The scribe matters more than it looks. The postmortem is only as good as the timeline, and
nobody remembers accurately afterwards — especially the things that were tried and didn't
work, which are exactly what a future engineer needs.

---

## Communicating

**Internally**, at a fixed cadence (every 20–30 minutes), even when there's nothing new:

```
[14:52] SEV2 — assistant returning wrong answers, ~15% of requests
Impact:  all tenants, concentrated in multi-turn conversations
Status:  investigating; retrieval ruled out, looking at prompt version
Next:    checking whether the 13:45 prompt change is implicated
Update:  15:20
```

**Externally**, three rules that matter more for AI incidents than most:

**Don't state scope before you've proven it.** "No other customers were affected" is not
something to say hopefully. For data-exposure incidents especially, engineering supplies
facts and legal decides what's communicated.

**Tell support before customers call them.** They're your front line and they'll be asked.

**Describe impact in user terms**, not system terms. "Some answers may have been based on
outdated documents" is more useful than "our retrieval index was stale."

---

## Mitigation before root cause

The reflex to fight: understanding the problem is *not* the priority when a cheap
reversible mitigation exists.

```python
def mitigation_options(incident):
    """Ranked by (speed × reversibility), not by how well you understand the cause."""
    return [
        ("revert the recent config/prompt change", "seconds", "fully reversible"),
        ("feature flag off",                       "seconds", "fully reversible"),
        ("fail over to secondary provider",        "minutes", "reversible"),
        ("degrade to a lower tier",                "minutes", "reversible"),
        ("roll back the deploy",                   "minutes", "reversible"),
        ("hotfix",                                 "hours",   "risky under pressure"),
    ]
```

> "The 13:45 prompt change is a one-field rollback with no dependencies. I'd revert it now
> and keep diagnosing from a stable state. If I'm wrong, I've lost a copy edit."

**One caution:** revert *one* thing, then wait and measure. Changing a second variable while
the first is still settling destroys your ability to attribute anything — and partial
recovery usually means two causes, not one.

---

## Recovery is its own phase

When the dependency comes back or the fix lands, doing nothing gives you a second outage:
queued work drains at once, every client's circuit breaker closes simultaneously, and every
user who's been refreshing arrives together.

- Ramp: half-open probing, controlled queue drain, jitter everywhere.
- Prioritize interactive over batch.
- Watch for second-order failures — your database may not be sized for the burst either.

**Many incidents have two outages.** Treating recovery as "and then it was fine" is how you
get the second one.

---

## Data integrity: the part that outlives the incident

For AI incidents specifically, ask a question that ordinary service incidents don't need:

> **Did we write anything wrong to durable storage?**

- Truncated extractions accepted as complete
- Wrong classifications persisted to a database
- Answers cached and still being served
- Actions an agent actually took
- Downstream systems that consumed bad output

```python
def integrity_check(incident_window):
    return {
        "records_written":     count_writes(incident_window),
        "cached_entries":      count_cache_writes(incident_window),
        "side_effects":        list_mutating_calls(incident_window),
        "downstream_consumed": trace_downstream(incident_window),
    }
```

The dollars are usually the smaller cost. A quality incident that silently wrote wrong data
for six weeks is a far bigger problem than the same incident detected in an hour, and it's
the question most incident processes never ask.

---

## The postmortem

Blameless, and specific about the *system* rather than the person.

The framing that produces useful outcomes:

> "The person who edited the prompt did a reasonable thing. Our system made a
> cheap-looking action have an expensive, invisible consequence. That's the finding."

Sections worth having:

- **Timeline** — from the scribe, including what was tried and didn't work
- **Impact** — users, requests, duration, and data written
- **Root cause** — and separately, the **mechanism**
- **Detection** — how you found out, and how long it took
- **What would have caught it sooner** — the highest-value section
- **Actions** — with owners and dates

### Mechanism versus root cause

Distinguishing these is what turns a postmortem into a systemic fix:

- **Mechanism:** editing the prompt invalidated the prefix cache.
- **Root cause:** a PM can change production behavior through a UI with no review, no
  canary, and no visibility into the consequence.

Fix the mechanism and this incident won't recur. Fix the root cause and a whole class
won't.

Then generalize: *what else can change production behavior without going through a deploy
pipeline?* Prompts, feature flags, corpus imports, retrieval configs, provider-side model
updates. That question, asked once, is worth more than most action items.

---

## Detection is usually the real finding

For AI incidents, "how long until we knew?" is frequently the most damning number.

```python
DETECTION_METRICS = [
    "time_to_detect",           # incident start -> someone knew
    "time_to_mitigate",
    "time_to_resolve",
    "detected_by",              # alert | customer | engineer | audit
]
```

**If `detected_by` is "customer" more than occasionally, your monitoring is the problem**,
not the individual incidents. The signals that catch AI failures early —
attempts-per-request, finish-reason distribution, cache hit rate, abstention rate, tier
occupancy — are cheap and most teams have none of them. See
[observability.md](observability.md).

---

## A runbook worth having

```markdown
## AI quality incident

1. Is anything actively doing damage? → disable it now
2. Might this be data exposure? → preserve evidence, escalate to security/legal
3. Scope: symptom, who, how much, when did it start?
4. Check the change timeline — including non-deploy changes
5. Split: was the right information in the prompt?
6. Cheap reversible mitigation available? → do it, then wait and measure
7. Narrow: hypothesis → disproof → data → update
8. Did we write anything wrong? → integrity check
9. Ramp recovery
10. Postmortem: mechanism AND root cause; what would have detected it sooner
```

---

## Interview questions

**1. "An agent has been posting to Slack and opening PRs for four hours. Go."**

Disable it first — it's actively doing damage and every minute of diagnosis costs more
side effects. Then contain: close the PRs, note the channel. Then diagnose. Getting the
order right is most of the answer.

**2. "A customer screenshots your assistant quoting another customer's data."**

Preserve evidence before rolling back, contain by disabling the feature, escalate to
security and legal immediately, and don't state scope until you've proven it. This is a
data-breach process, not a quality investigation.

**3. "It worked last week, nothing was deployed."**

Enumerate what changes without a deploy: prompt edits, corpus imports, provider model
updates, shared config changes, traffic shifts, thresholds crossed. "Nothing deployed" is
not "nothing changed."

**4. "You mitigate and it only partially recovers."**

Two causes, or an effect that doesn't reverse instantly — cache warm-up, a draining queue,
a second concurrent change. Wait and re-measure before touching anything else; changing a
second variable destroys attribution.

**5. "What's different about an AI postmortem?"**

Three things: ask whether wrong data was written to durable storage, distinguish mechanism
from root cause, and treat detection time as a first-class finding — because most AI
failures return 200 and are found by customers.

**6. "Your `detected_by` is 'customer' for most incidents. What does that tell you?"**

That monitoring is the actual problem, not the individual incidents. Then name the cheap
AI-specific signals most teams lack: attempts per request, finish reason, cache hit rate,
abstention rate.

---

## What to remember

- Stop ongoing damage before diagnosing. Agents don't wait.
- Preserve evidence before rolling back if data exposure is possible.
- Escalate legal/security decisions early — they aren't yours to make.
- "When did it start?" is the cheapest high-value question.
- "Nothing was deployed" ≠ "nothing changed." Keep the non-deploy list.
- Split quality incidents with: was the right information in the prompt?
- Mitigate with cheap reversible actions before you understand the cause — but change one
  thing at a time.
- Recovery is its own phase. Ramp it.
- Ask whether wrong data was written. That outlives the incident.
- Separate mechanism from root cause, and generalize the root cause.
- If customers detect your incidents, monitoring is the finding.

---

**Back to:** [the production index](README.md) · **Next section:**
[08 — AI Security](../08-ai-security/README.md)
