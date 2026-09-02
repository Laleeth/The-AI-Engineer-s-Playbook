# Agent or Workflow? Pick the Right One

This is the first decision you make, and most teams get it wrong in the same
direction: they build an agent when a workflow would have worked better.

Getting this right saves you months. Getting it wrong means you spend those months
debugging something that didn't need to exist.

---

## The difference in one sentence

**Workflow:** *you* decide the steps. The model does the thinking inside each step.

**Agent:** *the model* decides the steps. It picks what to do next, over and over,
until it thinks it's done.

That's it. Everything else follows from this.

---

## Same task, both ways

Say a support email arrives and you want to reply to it.

### As a workflow

You write the steps in code:

```python
def handle_email(email):
    # Step 1: what kind of email is this?
    category = classify(email)          # LLM call

    # Step 2: find relevant help articles
    docs = search_knowledge_base(email.body)   # normal search, no LLM

    # Step 3: write a reply
    draft = write_reply(email, docs, category)  # LLM call

    # Step 4: check it before sending
    if breaks_policy(draft):            # normal code, no LLM
        return escalate_to_human(email)

    return draft
```

Three LLM calls. Always the same three. Always in that order. You know exactly what
will happen before you run it.

### As an agent

You give the model tools and let it work it out:

```python
tools = [search_knowledge_base, look_up_customer, check_order_status,
         read_policy, draft_reply, escalate]

agent.run("Handle this support email: " + email.body)
```

The model might search, then look up the customer, then search again, then check an
order, then draft. Or it might do something completely different next time on a
similar email.

---

## Which one should you use?

Here's the honest rule:

> **If you can write down the steps, write down the steps.**

If you sat down and could sketch the flow on paper, you don't need an agent. You need
a workflow. The model is doing the hard thinking (classify this, write this) and your
code is doing the easy thinking (do A, then B, then C).

Use an agent when you genuinely **cannot** know the steps ahead of time.

### Signs a workflow is right

- The task has the same shape every time.
- You can list the steps.
- The order rarely changes.
- You need it to be predictable — same input, roughly same behaviour.
- It has to be fast (a workflow with 3 calls beats an agent with 12).
- It has to be cheap.
- You need to explain to someone exactly what it does.

### Signs an agent is right

- The number of steps depends on what you find along the way.
- "Look into X and tell me what's going on" is the actual request.
- Each step's result changes what makes sense to do next.
- A human doing this job would also be improvising.
- You're willing to pay more and wait longer for that flexibility.

### A good test

Ask: **"Could I draw this as a flowchart?"**

If yes → workflow.
If the flowchart would need a box saying *"figure out what to do next"* → agent.

---

## Why teams pick agents when they shouldn't

A few reasons, and they're all understandable:

**Agents demo better.** Watching a model decide things looks impressive in a meeting.
A workflow looks like normal software, because it is.

**Agents feel more future-proof.** "What if the requirements change?" But requirements
change for workflows too, and changing a workflow is a code edit you can review. With
an agent, you change a prompt and hope.

**The first version is easier.** Writing `agent.run(task)` is faster than writing the
steps out. The cost shows up later, when it does something weird in production and you
have to work out why.

**Everyone else is building agents.** Fair, but most of them are also finding out that
the agent part is the smallest and least reliable bit of the system.

---

## What agents actually cost you

Worth being clear about this, because the costs are real and they arrive later.

### Cost in money

A workflow makes 3 calls. An agent might make 12. Each call sends the *entire
conversation so far*, so the cost isn't 4× — it's worse than that.

Rough example. Say each step adds 2,000 tokens of context:

```
Workflow, 3 calls:
  call 1: 2,000 tokens
  call 2: 4,000 tokens
  call 3: 6,000 tokens
  total: 12,000 tokens

Agent, 12 steps:
  call 1: 2,000
  call 2: 4,000
  ...
  call 12: 24,000
  total: 156,000 tokens
```

**13× the cost, not 4×.** This surprises people. Context grows every step, and you
resend all of it every time.

### Cost in speed

Agent steps happen one after another. Each step waits for the model. Twelve steps at
2 seconds each is 24 seconds, and there's nothing you can do about most of it — step 7
genuinely can't start before step 6 finishes.

### Cost in debugging

When a workflow breaks, you look at which step failed. When an agent breaks, you read
a transcript of twelve steps and try to work out where it went off the rails — and
the answer is often "step 3 got a slightly odd result and everything after that was
reasonable given that."

### Cost in predictability

Same input, different behaviour. Sometimes fine, sometimes a support ticket. If
someone asks "what will this do?", the honest answer for an agent is "probably
roughly this."

---

## The middle ground (this is usually the answer)

You don't have to pick one. The best systems are mostly workflow with a small agent
part inside.

```python
def handle_ticket(ticket):
    # Fixed steps — no agent needed
    category = classify(ticket)
    customer = load_customer(ticket.customer_id)

    if category in SIMPLE_CATEGORIES:
        # 80% of tickets: pure workflow, fast and cheap
        docs = search_kb(ticket.body)
        return draft_reply(ticket, docs, customer)

    # 20% of tickets: genuinely need investigation
    # Give the agent a small box to work in
    return investigation_agent.run(
        ticket=ticket,
        customer=customer,
        max_steps=8,
        tools=[search_kb, check_orders, check_shipping, read_policy],
    )
```

This gets you:

- Most traffic goes through the cheap, fast, predictable path.
- The hard cases get flexibility.
- The agent has a small set of tools and a step limit, so it can't wander far.
- When something breaks you know which half to look at.

**Keep the agent part as small as you can.** The bigger the box you give it, the more
ways there are for it to do something surprising.

---

## Rewriting an agent as a workflow

If you already have an agent and you're not sure it needs to be one, do this:

1. **Log what it actually does.** Record the tool calls for a few hundred real runs.
2. **Count the paths.** Group runs by the sequence of tools they used.
3. **Look at the top few.**

You will usually find something like this:

```
search → draft                        41% of runs
search → check_order → draft          23%
search → search → draft               14%
check_order → search → draft           8%
everything else                       14%
```

Nearly 80% of runs follow four fixed paths. Those four paths are workflows. Write them
as workflows, and keep the agent only for the 14% that's genuinely varied.

This one exercise often cuts cost by 70% and latency by half, with no quality loss.

---

## Common mistakes

**Mistake: using an agent for one tool call.**

If your "agent" has one tool and always calls it, that's a function call with extra
steps and extra cost. Just call the function.

**Mistake: no step limit.**

Every agent needs a maximum number of steps. Not because you expect to hit it, but
because the day you do hit it, you want it to stop rather than run all night. (See
[agent-failure-modes.md](agent-failure-modes.md).)

**Mistake: giving it every tool you have.**

More tools = worse choices. A model picking from 40 tools makes worse decisions than
one picking from 6. Filter the tool list down to what's relevant for this task.

**Mistake: assuming the agent will be consistent.**

It won't. If consistency matters — pricing, legal, anything a customer will compare
against last week's answer — use a workflow.

**Mistake: not measuring the workflow baseline.**

Before you build an agent, build the dumb workflow version. Measure it. Often it's
90% as good for 10% of the cost, and that ends the discussion.

---

## Quick comparison

| | Workflow | Agent |
|---|---|---|
| Who picks the steps | You | The model |
| Cost | Low, predictable | Higher, varies a lot |
| Speed | Fast | Slow (steps are sequential) |
| Same input → same behaviour | Mostly yes | Often no |
| Debugging | Easy — check each step | Hard — read the whole transcript |
| Handles surprises | No | Yes |
| Good for | Known tasks | Open-ended investigation |
| Testing | Normal testing works | Needs a whole approach ([agent-evaluation](../05-evaluation/agent-evaluation.md)) |

---

## Interview questions

These come up a lot. Worth having real answers.

**1. "When would you not use an agent?"**

Good answer: whenever you can write the steps down. Then give a concrete example —
document processing with a fixed pipeline, or a classification task. Bad answer:
listing agent benefits without naming a case where they don't apply.

**2. "Your agent costs $2.40 per run. The workflow version costs $0.18. The agent is
5% more accurate. Which ships?"**

There's no fixed answer, and that's the point. It depends on what a wrong answer
costs. If it's a support reply a human checks, ship the cheap one. If it's a medical
triage suggestion, 5% matters a lot. The interviewer is checking whether you ask what
the errors cost before answering.

**3. "How would you turn this agent into a workflow?"**

Describe the logging exercise above. Log real runs, group by tool sequence, find the
common paths, write those as workflows, keep an agent for the tail.

**4. "Your agent takes 40 seconds. Product says that's too slow. Options?"**

Several, in rough order of value:
- Find the fixed steps and pull them out into a workflow (biggest win).
- Run independent tool calls at the same time instead of one by one.
- Stream progress so the user sees something happening.
- Use a smaller model for the "which tool next" decisions and a bigger one for the
  final answer.
- Cut the number of tools so it makes fewer wrong turns.

**5. "How do you stop an agent looping forever?"**

Step limits are the backstop, not the answer. The real answer is detecting *no
progress* — if three steps in a row produce no new information, stop. Plus a cost
limit in dollars, and telling the model explicitly when it repeats a call. Full
details in [agent-failure-modes.md](agent-failure-modes.md).

---

## What to remember

- Workflow = you pick the steps. Agent = the model picks.
- If you can write the steps down, write them down.
- Agents cost more than you think, because context grows every step.
- Most good systems are a workflow with a small agent inside, not an agent.
- Before building an agent, build the workflow and measure it. Sometimes that's the
  whole project.

---

**Next:** [tool-calling.md](tool-calling.md) — how the model actually calls your code,
and the things that go wrong.
