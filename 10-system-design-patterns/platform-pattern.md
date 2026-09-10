# The Platform Pattern

Once more than two teams in a company are calling models, the same question arrives:
should there be a shared AI platform, or should each team own its own integration?

The answer that survives contact with reality is a **thin waist** — a narrow layer every
team goes through, owning the few things that must be consistent, and getting out of the
way of everything else. The failure mode on one side is five teams each discovering prompt
injection separately. The failure mode on the other is a platform team that has become a
ticket queue between engineers and the model they need.

---

## What the thin waist owns

The test for putting something in the platform: **would you accept each team implementing
this differently, or getting it wrong once?** If no, it belongs in the waist. Everything
else does not.

That test admits a short list:

| In the waist | Why |
|---|---|
| **Credentials and provider access** | Nobody should hold raw provider keys, and key rotation must be one operation |
| **Cost attribution** | You cannot allocate a bill you didn't tag at the call site — [cost-control.md](../07-production/cost-control.md) |
| **Rate limit and quota management** | The provider quota is one shared resource; uncoordinated teams exhaust it — [rate-limits.md](../07-production/rate-limits.md) |
| **Logging, tracing and version metadata** | "What changed?" must be answerable across teams, so the identifiers have to be uniform |
| **PII handling and data residency routing** | Legal requirements that cannot be per-team best-effort |
| **Provider fallback and retry policy** | Retries are load amplification; every team implementing their own multiplies it |

And, just as importantly, what does *not* belong:

- **Prompts.** They are product logic. A platform team reviewing prompt changes becomes
  the bottleneck on every product iteration in the company.
- **Model choice per use case.** Give teams a menu and the data to choose from; don't
  choose for them.
- **Evaluation content.** The platform provides harnesses and infrastructure; the teams
  own what "good" means for their surface, because only they know.
- **Business logic of any kind**, however tempting the reuse.

The line is: **the platform owns what must be uniform, teams own what must be theirs.**
Every platform that fails does so by crossing that line in one direction or the other.

---

## Why the waist has to be thin

A thick platform layer fails predictably, and the mechanism is worth naming because it
looks like success at first:

1. The platform adds an abstraction over provider APIs, to be provider-agnostic.
2. A team needs a feature the abstraction doesn't expose — structured output, a new
   parameter, a model-specific capability.
3. They file a ticket. The platform team prioritizes it against everything else.
4. Meanwhile the team ships by calling the provider directly, around the platform.
5. Now you have a platform *and* direct calls, and the platform's guarantees — cost
   attribution, PII handling, quota coordination — cover only part of the traffic. Which
   is the same as covering none of it, for compliance purposes.

Step 4 is not a discipline failure; it is what any competent team does when blocked. The
design implication is that **the platform must not be able to block a team**, which means
it must be thin enough that it rarely needs changing, and it must pass through what it
doesn't understand:

```python
def call_model(request, provider_params=None):
    """Own the cross-cutting concerns. Pass through everything else."""
    check_quota(request.team)
    redact = pii_policy(request.team)
    response = provider.call(
        model=request.model,
        messages=redact(request.messages),
        **(provider_params or {}),        # unknown params pass through untouched
    )
    record(team=request.team, tokens=response.usage, versions=response.versions)
    return response
```

The `**provider_params` passthrough is the single most important line. Without it, every
new provider capability becomes a platform ticket, and every platform ticket is a reason
to bypass the platform.

---

## The value is measurable, and you should measure it

Platform teams struggle to justify their existence because their output is other teams'
velocity. Make the case with numbers that already exist:

- **Cost attribution coverage.** What share of spend can be traced to a team, a surface
  and a customer. This alone usually funds the platform.
- **Time from "we want to try a model" to a call in production**, per team. The number the
  platform exists to reduce.
- **Duplicate implementations avoided** — retry logic, token counting, PII redaction —
  counted honestly as teams onboard.
- **Incidents caused by an inconsistency** the waist would have prevented, before and
  after.
- **Bypass rate**: traffic reaching providers without going through the platform. The
  platform's actual health metric, and the one nobody publishes.

That last one deserves attention. A platform with a rising bypass rate is failing
regardless of how good its adoption numbers look, because the traffic that bypasses it is
exactly the traffic doing something new.

---

## When not to build one

Most companies build this too early. Reasons to wait:

- **Fewer than three teams calling models.** Two teams coordinate in a channel. A platform
  for two teams is overhead with a roadmap.
- **Nobody will own it.** An unowned platform decays into a shared library nobody upgrades
  and everyone is afraid to change.
- **The use cases are genuinely unlike each other.** A batch classification pipeline and an
  interactive agent share credentials and cost attribution, and almost nothing else. The
  waist for them is very thin indeed — possibly just a proxy.

The honest starting point is usually not a platform at all: a shared library for the
cross-cutting concerns, plus a convention for cost tags. Promote it to a service when the
library's version skew starts causing incidents, which is the actual signal that a service
is needed.

This is a build-vs-buy decision with the same structure as
[12-senior-scenarios/build-vs-buy.md](../12-senior-scenarios/build-vs-buy.md), and the
same trap: the build cost is visible and the ongoing ownership cost is not.

---

## Migration, and the reason the waist earns its keep

The strongest argument for a thin waist arrives the day you change providers — a price
change, an outage, a new model, or a residency requirement that forces self-hosting.

With a waist: change the routing in one place, canary it, and every team's traffic moves.
With direct integrations everywhere: a coordination project across every team's roadmap,
which in practice means it doesn't happen and you stay on the provider you meant to leave.

That optionality is the platform's real product, and it is worth saying explicitly when
justifying one — not "consistency," which sounds like tidiness, but "we can change our
mind about a provider in a week instead of a quarter."

The caveat, and it's a real one: a waist that abstracts *too* much creates a false
portability. Prompts are tuned per model, and moving providers means re-evaluating quality
regardless of how clean the interface is. The waist makes the plumbing swap cheap; it does
not make the models equivalent, and anyone claiming otherwise hasn't run the evaluation.

---

## What to monitor

- **Bypass rate.** Traffic reaching providers without going through the platform.
- **Cost attribution coverage**, as a share of total spend.
- **Onboarding time** for a new team or a new use case.
- **Platform-caused incidents**, tracked separately from provider incidents. A platform
  that becomes a single point of failure needs to know it.
- **Feature request queue depth and age.** A growing queue predicts a rising bypass rate.
- **Quota utilization per team against reserved floors**, which is the coordination the
  waist exists to provide.

---

## Interview questions

**"Five teams are each calling model providers directly. What would you do?"**

First find out what's actually broken, because "five teams" isn't a problem by itself. The
usual findings are that nobody can attribute the bill, quota exhaustion by one team takes
down another, and PII handling is inconsistent. Those three justify a thin waist: shared
credentials, cost tagging, quota coordination, logging and PII policy. I'd start as a
shared library rather than a service, and I'd deliberately leave prompts, model choice and
evals with the teams — the fastest way to kill a platform is to make it the approver of
prompt changes.

**"Your platform team is a bottleneck. Diagnose it."**

The waist is too thick — it's abstracting things it should pass through, so every new
provider capability becomes a ticket. Check the feature queue age and the bypass rate;
rising bypass means teams are already routing around it, which is rational and which means
the platform's guarantees now cover only part of traffic. The fix is to pass unknown
parameters through untouched and to give back anything that's product logic rather than a
cross-cutting concern.

**"How would you justify a platform team to a VP?"**

Optionality and attribution, with numbers. Cost attribution coverage usually pays for it
outright — you can't optimize a bill you can't allocate. Then the migration argument:
changing provider is a one-week routing change instead of a quarter-long coordination
project across every roadmap, which matters the first time a price changes or a residency
requirement lands. I'd avoid arguing "consistency," which sounds like tidiness and loses.

---

## What to remember

- The test for the waist: would you accept every team doing this differently, or getting
  it wrong once?
- In: credentials, cost attribution, quota, logging and versions, PII and residency,
  fallback policy. Out: prompts, model choice, evals, business logic.
- Thin, because a platform that can block a team gets bypassed — and partial coverage is
  no coverage for compliance.
- Pass unknown provider parameters straight through. It's the line that prevents the
  bypass spiral.
- Bypass rate is the platform's real health metric.
- The product is optionality: changing provider in a week rather than a quarter. But the
  models still aren't equivalent, and prompts still need re-evaluating.
