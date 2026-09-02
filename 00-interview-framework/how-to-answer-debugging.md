# How to Answer Debugging Questions

Debugging interviews are different from design interviews, and people who are good at one
are often bad at the other.

Design rewards breadth: cover the components, name the trade-offs. Debugging rewards
**narrowing**: every question you ask should eliminate possibilities. Breadth here looks
like flailing.

---

## The one rule

> **State a hypothesis. Say what would disprove it. Ask for exactly that data.**

Everything else is detail.

Weak debugging sounds like a list of things that might be wrong. Strong debugging sounds
like a series of questions that each cut the space in half.

**Weak:**
> "It could be the model, or retrieval, or maybe the prompt changed, or there's a caching
> issue, or the traffic mix shifted…"

**Strong:**
> "My first hypothesis is that this is retrieval rather than generation, because the
> symptom is wrong facts rather than bad writing. What would disprove it: if the correct
> documents were in the prompt and the answer was still wrong. So — can I see the
> retrieved chunks for a few of the failing requests?"

Same knowledge. Completely different signal.

---

## The shape

```
1. Scope        What exactly is broken, and for whom?
2. Timeline     When did it start? What changed?
3. Split        Which half of the system is it in?
4. Narrow       Hypothesis → disproof → data → update
5. Mitigate     Stop the bleeding before you fix the cause
6. Prevent      What monitoring would have caught this?
```

Steps 5 and 6 are frequently skipped and carry a lot of signal, especially at senior
level.

---

## 1. Scope

Before any hypothesis, find out what "broken" means.

- What's the actual symptom? Wrong answers, slow, errors, expensive?
- Is it everyone or a subset? Which subset?
- All requests or some? What do the failing ones have in common?
- How bad — 1% or 40%?

The subset question is the highest-value one. If failures are concentrated in one
language, one customer, one document type, or multi-turn conversations, you've cut the
problem enormously in one question.

> "Are the failures spread evenly, or concentrated? If they're all multi-turn
> conversations, or all from one tenant, that points somewhere very different from a
> uniform failure rate."

---

## 2. Timeline

> **"When did it start?"** is the highest-value question in debugging, and it's free.

If it worked last week and doesn't now, something changed, and the list of things that
change is short and enumerable. That's a much faster investigation than debugging from
first principles.

Things that change *without a deploy* — the list worth having memorised:

- Documents added, removed, or re-indexed
- A prompt edited in a management UI
- A provider silently updating a model
- Another team changing a shared config (embedding model, retrieval parameters)
- Traffic mix shifting — a new customer, a marketing campaign
- A certificate, quota, or rate limit changing
- Data volume crossing a threshold (an index no longer fitting in memory)

> "Nothing was deployed" is not the same as "nothing changed," and saying so is a strong
> signal.

---

## 3. Split

Cut the system in half with one question. For AI systems, the highest-value split is
almost always:

> **Was the right information in the prompt?**

- **No** → retrieval problem. Chunking, search, ranking, filtering.
- **Yes** → generation problem. Prompt, model, context length.

These have completely different fixes, and teams routinely spend a week tuning prompts on
what turns out to be a retrieval failure.

Other useful splits by symptom:

| Symptom | The splitting question |
|---|---|
| Wrong answers | Was the right information in the prompt? |
| Slow | Which stage? Get the latency decomposed by component |
| Expensive | Input tokens or output tokens? Requests or tokens per request? |
| Errors | Are they from us, the provider, or a dependency? |
| Agent misbehaving | Did it fail, or succeed at the wrong thing? |

---

## 4. Narrow

Now the loop. Each cycle should eliminate something.

```
hypothesis → what would disprove it → ask for that → update → repeat
```

Two things make this go well.

**Say what you're ruling out.** Narrowing out loud is progress, and interviewers score it:

> "Retrieval latency is flat, so this isn't the vector database, the embedding model, or
> corpus size. Output tokens are flat too, so it isn't decoding. That leaves prefill or
> something provider-side."

**Do the arithmetic.** Numbers eliminate hypotheses faster than intuition:

> "Input tokens are up 15% but LLM latency is up 300%. Prefill is roughly linear in input
> length, so a 15% increase can't explain a 300% rise on its own — unless something crossed
> a threshold. That points at prefix cache invalidation rather than a gradual increase."

That's the reasoning that solves the latency incident in
[../12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md),
and it comes from doing the division rather than pattern-matching on "tokens went up."

**Update visibly when you're wrong.** Interviewers often feed you data that contradicts
your hypothesis on purpose:

> "That rules out my theory — if the provider's status page is clean and our other
> endpoint is unaffected, it's not provider-side. So it's something specific to this
> code path. What changed on it?"

Being wrong and updating cleanly scores better than being right slowly.

---

## 5. Mitigate before you fix

Senior candidates separate *stopping the damage* from *finding the cause*. Many
candidates go straight to root cause while the system is still on fire.

> "Before I keep investigating — the prompt change at 13:45 is a one-field rollback with
> no dependencies. I'd revert it now, confirm recovery, and carry on diagnosing from a
> stable state. If I'm wrong, I've lost a copy edit."

This matters even more when there are ongoing side effects:

> "This agent has been posting to Slack and opening pull requests for four hours. First
> thing is to disable it — it's actively doing damage. Diagnosis comes second."

The general principle: **if mitigation is cheap and reversible, do it first.** You can
always investigate afterwards; you can't un-send 300 messages.

---

## 6. Prevent

Close by saying what would have caught this sooner. This is where senior answers land.

> "This ran for 40 minutes before the latency alert fired, and the alert pointed at a
> symptom rather than the cause. Prefix cache hit rate would have dropped at 13:47,
> immediately after the prompt change — that alert would have fired 30 minutes earlier and
> pointed straight at it.
>
> The deeper issue is that a PM can change production behaviour through a UI with no canary
> and no review. That's the actual root cause; the cache is just the mechanism. I'd put
> prompt changes behind the same canary as code deploys."

Two moves in there worth copying: distinguishing the **mechanism** from the **root cause**,
and framing the fix as a system change rather than blaming a person.

---

## Signals that don't show up in error rate

Worth knowing, because these come up constantly and catch people out.

**Retries that eventually succeed.** Error rate is flat while you're making three times
the calls. Track *attempts per logical request* separately.

**Truncation.** `finish_reason: length` means the response was cut off. It's a 200 OK. It
fails validation downstream, retries, and triples your bill while every dashboard looks
healthy.

**Cache hit rate collapse.** Costs and latency jump with no code change and no errors.

**Silent quality drops.** A cheaper model or a smaller context returns fluent, confident,
wrong answers. Nothing errors. You find out from churn.

**Abstention rate rising.** "I don't have that information" going from 4% to 15% means
retrieval broke. Users don't report it — they leave.

Naming any of these unprompted is a strong signal you've operated a system rather than
read about one.

---

## Worked example

> **"p95 latency went from 1.8s to 9s over 40 minutes. Error rate normal. Traffic slightly
> below average. Go."**

**Scope:** "Is it all requests or a subset? And is p50 affected or just the tail — if p50
is fine and only p95 moved, that's a different problem from everything getting slower."

**Timeline:** "40 minutes is fast for organic degradation, so something changed. What
deployed today, and does that include non-code changes — prompts, configs, documents?"

*(Two deploys: a frontend change at 09:00, a prompt tweak at 13:45.)*

**Split:** "Can I get latency decomposed by stage? I want to know whether this is
retrieval, generation, or our own code."

*(Retrieval flat. LLM call up 300%. Input tokens up 15%. Output tokens flat.)*

**Narrow:** "That's informative. Retrieval being flat rules out the vector database and
the embedding model. Output tokens flat rules out decoding — the extra time isn't in
generation.

So it's prefill or provider-side. Input up 15% can't cause latency up 300% linearly, so
something crossed a threshold rather than growing. The prompt change at 13:45 fits: if it
edited the top of the prompt, it invalidated the prefix cache, and every request now
processes the full input from scratch.

Can I see prefix cache hit rate before and after?"

*(87% → 4%.)*

**Mitigate:** "Revert the prompt change. It's one field, no dependencies. I'd watch cache
hit rate rather than latency to confirm, because it recovers first."

**Prevent:** "Alert on cache hit rate — it would have fired 30 minutes earlier and pointed
at the cause. And put prompt edits behind a canary, because right now a UI change can alter
production behaviour with no review."

---

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Listing possible causes without narrowing | Reads as flailing |
| Not asking when it started | Skips the fastest path |
| Jumping to a fix before diagnosing | You'll fix the wrong thing |
| Ignoring data that contradicts you | The worst signal available |
| Not doing arithmetic | Misses that 15% can't cause 300% |
| Investigating while damage continues | Especially bad with agents |
| No prevention at the end | Misses the senior signal |
| Treating "nothing deployed" as "nothing changed" | Half of incidents are non-deploy changes |

---

## Practising this

The incident scenarios in
[../12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md)
are built for this and work best with a partner.

One person plays the dashboards and **only reveals evidence when asked for a specific
metric.** The other must:

1. State a hypothesis before asking for anything.
2. Say what would disprove it.
3. Ask for exactly that data.
4. Say out loud what's now ruled out.

The habit being trained is not knowing the answer. It's narrowing with every question.
That's what the interview is measuring, and it's trainable in a way that knowledge isn't.

---

## What to remember

- Hypothesis → what would disprove it → ask for exactly that.
- "When did it start?" is the cheapest high-value question there is.
- "Was the right information in the prompt?" splits most AI quality problems in half.
- Say what you're ruling out. Narrowing is progress.
- Do the arithmetic — it kills hypotheses faster than intuition.
- Mitigate before diagnosing when mitigation is cheap and reversible.
- Stop ongoing damage immediately, especially with agents.
- Finish with what monitoring would have caught it sooner.
- Distinguish the mechanism from the root cause.

---

**Back to:** [the interview framework index](README.md) · **Next section:**
[01 — LLM Internals](../01-llm-internals/README.md)
