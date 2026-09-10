# Multi-Agent Systems

Several agents working together. It sounds like the obvious next step after one agent,
and it's usually a mistake.

This file is mostly about when *not* to do it, because that's the part people skip.

---

## Start here: you probably don't need this

The honest position first.

Most "multi-agent" systems are one agent that got split up for no reason. The split
adds cost, latency, and new failure modes, and the thing it was supposed to fix
usually wasn't fixed.

Before adding a second agent, ask: **what exactly is one agent failing at?**

If the answer is "nothing yet, but this seems more scalable," stop. Build the single
agent. Measure it. Come back when you have a real problem.

Real reasons to split:

- **Different permissions.** One part needs write access to production, another
  doesn't. Splitting means the risky capability lives in one small, tightly controlled
  place. This is the strongest reason.
- **Genuinely parallel work.** Three independent research tasks that don't need to see
  each other's results.
- **Different owners.** Two teams maintain two things and shouldn't have to coordinate
  every prompt change.
- **Context is too big.** One agent's context won't fit, and the work splits cleanly.

Bad reasons:

- It matches how a human team is organized. (Human teams are shaped by human
  constraints — sleep, expertise, hiring. Your software has none of those.)
- Specialized prompts sound better than one general prompt. (Usually you can just
  write a better single prompt.)
- It's more modular. (It's more *distributed*, which is not the same thing.)

---

## What splitting actually costs

Worth being concrete, because the costs are invisible on a whiteboard.

### Information is lost at every handoff

When agent A passes to agent B, B gets a summary, not everything A saw. A knew the
customer sounded angry, mentioned a competitor, and had contacted support twice
before. The summary says "customer wants a refund."

Every handoff loses detail. Three agents in a chain and the last one is working from a
summary of a summary.

### Cost multiplies

Each agent has its own system prompt, its own tool list, its own context. Three agents
means three lots of overhead — and the handoff messages themselves cost tokens.

A rough comparison:

```
One agent, 10 steps:              ~110,000 tokens (see memory.md)

Three agents, 4 steps each:
  agent A: ~20,000
  agent B: ~20,000 + A's summary
  agent C: ~20,000 + B's summary
  coordinator overhead: ~15,000
  total: ~80,000 tokens

...but only if each really needs 4 steps and the split is clean.
If they need 8 steps each, you're well past the single agent.
```

Sometimes splitting is cheaper. Often it isn't. **Measure, don't assume.**

### Latency adds up

Agents in a chain run one after another. Three agents at 8 seconds each is 24 seconds.
Only genuinely parallel work avoids this.

### Debugging gets much harder

One agent: read one transcript.

Three agents: read three transcripts, plus the handoffs, and work out which agent
introduced the error — bearing in mind that agent C may have behaved perfectly given
the bad summary it received from B.

---

## The patterns that work

If you do need multiple agents, these are the shapes worth knowing.

### 1. Router

One cheap model decides which specialist handles the request. No coordination after
that.

```python
def route(request):
    kind = classify(request)          # small, fast model
    return {
        "billing":   billing_agent,
        "technical": technical_agent,
        "account":   account_agent,
    }.get(kind, general_agent).run(request)
```

**This is the best multi-agent pattern**, and it barely counts as one. There's no
handoff, no shared state, no coordination. Each agent is independent and testable.

Use when requests fall into clear categories that need different tools or different
permissions.

Watch out for: routing mistakes. Have a fallback, and let a specialist hand back if it
gets something it can't handle.

### 2. Manager and workers

One agent breaks the job up, workers do the pieces, the manager combines.

```python
def research(topic):
    subtasks = manager.plan(topic)                 # "research A", "research B", "research C"

    # Workers run at the same time — this is the whole point
    results = await asyncio.gather(*[
        worker.run(task) for task in subtasks
    ])

    return manager.combine(topic, results)
```

**Only worth it when the subtasks are truly independent.** If worker B needs what
worker A found, you can't parallelize, and you've added coordination for nothing.

Watch out for:
- Workers duplicating each other's work (all three search the same thing).
- The combining step being the hard part and getting the least attention.
- One slow worker holding up everything.

### 3. Pipeline

Fixed stages, each doing one thing.

```
extract → validate → enrich → format
```

Here's the thing though: **this is a workflow, not a multi-agent system.** If the
stages are fixed and always run in order, you don't need agents at all — you need
functions that happen to call a model. See
[agent-vs-workflow.md](agent-vs-workflow.md).

Calling it multi-agent adds nothing except confusion.

### 4. Reviewer

One agent produces, another checks.

```python
draft = writer.run(task)
review = reviewer.check(draft, task)

if not review.ok:
    draft = writer.revise(draft, review.feedback)
```

Sometimes this genuinely helps — a fresh look catches things. Often it doesn't, and
you should test rather than assume.

Two things to watch:

**Is the reviewer better than the writer at reviewing?** Same model, same weaknesses.
If the writer missed something because the model has a blind spot, the reviewer has
the same blind spot.

**Does the reviewer need a model at all?** Very often the check is deterministic — is
the JSON valid, do the numbers add up, are the citations real, does it break a policy
rule. Those are code, they're free, they're exact, and they never have a bad day.
Using a model to check a model is expensive and less reliable than a validator, when
a validator is possible.

### 5. Debate

Several agents argue, then something picks a winner.

Occasionally useful for genuinely subjective judgement calls. Expensive (N× the
calls), slow, and hard to evaluate. **Try simpler things first** — a better prompt,
a better model, a validator — and only reach for this if you've measured that they
don't work.

---

## Handoffs: the part that breaks

Most multi-agent bugs live in the handoff. Make it explicit rather than hoping a
summary carries enough.

```python
@dataclass
class Handoff:
    task: str                     # what B should do
    context: str                  # background B needs
    facts: dict[str, str]         # IDs, numbers — structured, not prose
    already_tried: list[str]      # so B doesn't redo A's dead ends
    from_agent: str
    trace_id: str                 # ties the whole run together in logs
```

Three details that matter more than they look:

**`facts` is structured.** Prose summaries lose IDs. A dict doesn't. Same lesson as
[memory.md](memory.md).

**`already_tried` prevents the most common waste** — agent B repeating the searches
agent A already did and found nothing in.

**`trace_id` is how you debug this at all.** Without one shared ID across all agents,
you cannot reconstruct a single run from your logs. Add it on day one; retrofitting it
is miserable.

---

## New ways things break

Multi-agent systems have failure modes single agents don't have.

**Loops between agents.** A hands to B, B decides it's A's job, hands back. Fix: a
global step budget across the whole system, not per agent, and track which agents have
already seen this request.

```python
if agent_name in request.handled_by:
    return escalate_to_human("agents are passing this back and forth")
request.handled_by.append(agent_name)
```

**Duplicated work.** Three workers all search the same phrase. Fix: share a cache of
tool results keyed by (tool, arguments) across the run.

**Contradictions.** Agent A says the refund window is 30 days, agent B says 14. Both
read different documents. The combining step has to notice and resolve it — and it
usually doesn't, unless you tell it to.

**Cost with no ceiling.** One agent with a step limit is bounded. Five agents that can
call each other is not, unless you budget globally.

```python
@dataclass
class RunBudget:
    max_total_steps: int = 30       # ACROSS all agents
    max_total_cost: float = 2.00    # ACROSS all agents
    steps_used: int = 0
    cost_used: float = 0.0
```

Per-agent limits don't add up to a system limit. Budget the system.

**Blame is unclear.** The output is wrong. Which agent? Often it's "agent C did the
right thing with a bad summary from B, which was reasonable given A's incomplete
search." Nobody is at fault, which makes it hard to fix.

---

## How to tell if it's helping

Compare against one agent. Actually compare — don't assume.

Measure, for both versions:

- **Quality** on the same test set.
- **Cost** per completed task.
- **Latency**, end to end.
- **Success rate** — did it finish the job?
- **Debugging time** — how long to work out why a failure happened. This one is real
  and never gets measured.

If the multi-agent version isn't clearly better on quality or success, go back to one
agent. "It's more modular" is not a result.

---

## Interview questions

**1. "When would you use multiple agents?"**

Different permissions, genuinely parallel work, different team owners, or context that
won't fit. Then say clearly that most cases don't need it and that you'd measure the
single-agent version first. The willingness to say "probably not" is the signal here.

**2. "What's the main cost of splitting into agents?"**

Information loss at handoffs. Each one is a summary, and detail disappears. Then
mention cost, latency, and much harder debugging.

**3. "Two agents keep handing the same request back and forth. Fix it."**

Track which agents have handled it, budget steps globally across the system rather than
per agent, and escalate to a human when it bounces.

**4. "You have a writer agent and a reviewer agent. How do you know the reviewer
helps?"**

Test with and without. Also ask whether the check could be code instead — schema
validation, citation checks, policy rules — because that's cheaper and exact. And note
that same-model reviewers share the writer's blind spots.

**5. "How do you debug a 3-agent system?"**

One trace ID across all agents, structured handoffs you can inspect, and per-agent
logs. Then say honestly that it's much harder than one agent, which is part of the
cost of the design.

---

## What to remember

- Most multi-agent systems should be one agent.
- Best real reasons: different permissions, real parallelism, different owners.
- Bad reason: it mirrors a human org chart.
- Every handoff loses information. Make handoffs structured, with facts and
  already-tried.
- Budget steps and cost **across the system**, not per agent.
- A "reviewer agent" is often better as a plain validator.
- One trace ID across everything, from day one.
- Compare against the single-agent version before you commit.

---

**Next:** [mcp.md](mcp.md) — a standard way to connect tools to models.
