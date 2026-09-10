# 04 — Agents

How agents actually work, and the ways they break. Written in plain language — terms
are explained when they first appear, and there's no jargon for its own sake.

The theme running through all eight files: **agents fail because something is
unbounded.** Unbounded steps, unbounded context, unbounded retries, unbounded side
effects. Find the thing without a limit and you've usually found the bug.

## The files

| File | What it covers |
|---|---|
| [agent-vs-workflow.md](agent-vs-workflow.md) | The first decision, and the one most teams get wrong |
| [tool-calling.md](tool-calling.md) | How the model runs your code, and why descriptions matter more than code |
| [planning.md](planning.md) | How agents decide what to do — and when planning makes things worse |
| [memory.md](memory.md) | What gets sent back each step, and why cost grows so fast |
| [multi-agent.md](multi-agent.md) | When several agents help, and when they multiply your problems |
| [mcp.md](mcp.md) | The Model Context Protocol — a standard way to connect tools |
| [permissions.md](permissions.md) | Giving an agent real access without giving it too much |
| [agent-failure-modes.md](agent-failure-modes.md) | Twelve specific ways agents break, with fixes |

## Read in this order

If you're new to this, go in file order. The first file decides whether you need any of
the rest.

If you're debugging something now, jump to
[agent-failure-modes.md](agent-failure-modes.md) — it's organized by symptom.

If you're about to give an agent write access to something real, read
[permissions.md](permissions.md) first. It's the file with the highest cost of getting
wrong.

## The short version

A few things worth knowing even if you read nothing else:

**If you can write the steps down, write them down.** A workflow beats an agent for
anything predictable. Most "agents" are workflows with extra cost and less
predictability.

**Cost grows with the square of the steps.** Every step resends the whole conversation,
so 10 steps costs about 55× one step, not 10×. This surprises people and it's the main
reason agent bills blow up.

**Tool descriptions are documentation.** The model reads them to decide what to call.
Say what a tool is for, and — more importantly — what it is *not* for.

**Terminal errors must say "do not retry."** A 404 returned as `"Error: not found"`
reads to the model like "maybe try again." That's the mechanism behind the runaway loop
that spent $2,800 overnight.

**Prompts are preferences. Credentials are rules.** You cannot prompt your way out of
prompt injection. A read-only database connection cannot be talked into a `DELETE`,
whatever any text says.

**Detect "no progress," don't just cap steps.** A step limit tells you *that* it
failed. No-progress detection tells you *why*, and stops earlier.

**Watch the stop-reason distribution.** It's free, needs no labels, and moves before
any other metric when something breaks.

## Related

- [05-evaluation/agent-evaluation.md](../05-evaluation/agent-evaluation.md) — how to
  test the things described here
- [11-coding-rounds/agent-loop.md](../11-coding-rounds/agent-loop.md) — build an agent
  loop from scratch, with tested code
- [12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md)
  — the runaway agent incident, as an interview exercise
