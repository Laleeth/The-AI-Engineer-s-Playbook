# Planning

Planning is how an agent decides what to do next. There are a few common approaches,
they suit different jobs, and — importantly — **more planning is not always better.**

A lot of teams add a planning step because it sounds sophisticated, then find it made
things slower, more expensive, and no more accurate.

---

## The simplest approach: just react

No plan at all. Look at where you are, pick one action, do it, look again.

```
think → act → see result → think → act → see result → ...
```

This is what most agent loops do by default. It's often called ReAct (reason + act).

```python
while not done:
    response = model(messages, tools)     # model thinks and picks one tool
    if response.is_final_answer:
        return response.content
    result = run_tool(response.tool_call)
    messages.append(result)               # model sees the result next time round
```

**Good because:** it adapts immediately. If step 1 returns something surprising, step 2
takes that into account. No wasted planning of steps you'll never take.

**Bad because:** it can wander. There's nothing keeping it pointed at the goal, so it
can drift, or take a long path when a short one existed.

**Use it when:** the task is short (under about 10 steps), and each result meaningfully
changes what to do next. This covers most real agents.

---

## Plan first, then execute

Ask the model to write the whole plan up front, then work through it.

```python
plan = model(f"""
Break this task into steps. Output JSON:
{{"steps": [{{"n": 1, "action": "...", "tool": "..."}}]}}

Task: {task}
""")

for step in plan["steps"]:
    result = run_step(step)
```

**Good because:** you can see the plan before anything runs. You can show it to a
person for approval. You can spot a bad plan early and stop. Steps that don't depend
on each other can run at the same time.

**Bad because:** the plan is written *before* you know anything. Step 4 was planned
without knowing what step 2 returned. When reality doesn't match, the plan is wrong
and you're following it anyway.

**Use it when:** you need human approval before acting, or the task genuinely is
predictable, or you want to run steps in parallel.

---

## Plan, then adjust as you go

The middle ground, and usually the best of the three for longer tasks. Make a plan,
follow it, and re-plan when something surprises you.

```python
plan = make_plan(task)
step_index = 0

while step_index < len(plan) and steps_used < max_steps:
    result = run_step(plan[step_index])

    if surprising(result, plan[step_index]):
        # Something didn't match expectations — rethink the rest
        plan = make_plan(task, done_so_far=plan[:step_index+1], latest=result)
        step_index = 0
        continue

    step_index += 1
```

The whole design rests on one question: **what counts as surprising?** Get that wrong
and you either re-plan constantly (expensive, and it never finishes) or never re-plan
(and you're back to a rigid plan).

Reasonable triggers:

```python
def surprising(result, step):
    return (
        result.failed                          # the step didn't work
        or result.is_empty                     # found nothing
        or result.contradicts(step.assumption) # we assumed X, found not-X
        or steps_since_replan > 5              # long enough that things may have changed
    )
```

**Use it when:** tasks run more than about 10 steps and the environment can surprise
you.

---

## Breaking a task into smaller tasks

For genuinely big jobs, split the work into sub-tasks and handle each one separately.

```
"Research our top 3 competitors and write a summary"
  ├── research competitor A   ← own mini-agent run
  ├── research competitor B   ← own mini-agent run
  ├── research competitor C   ← own mini-agent run
  └── combine into a summary
```

**Good because:** each sub-task has a small, clean context instead of one giant
growing one. The three research tasks can run at the same time. If one fails you
retry just that one.

**Bad because:** sub-tasks can't see each other's findings. If researching A turns up
something relevant to B, that's lost. And you need something to combine the results,
which is its own job.

**Use it when:** the parts really are independent. If they aren't, forcing this split
makes the output worse.

---

## When planning makes things worse

This is the part that gets skipped, so it's worth being direct.

**For short tasks, planning is pure overhead.** If the task takes 3 steps, a planning
call adds one more call and some latency to produce a plan that a reactive agent would
have followed anyway. You paid for nothing.

**Plans made without information are guesses.** The model writes step 4 having no idea
what step 2 will return. That's not a plan, it's a hopeful sketch. And the model will
then *follow* it, even when the results say otherwise, because it's in the context and
it looks authoritative.

**Planning can make things less flexible.** A reactive agent naturally adapts. Give it
a plan and it tends to stick to it, including when sticking to it is wrong.

**It's another thing that can be wrong.** Now you have two failure modes: bad
execution, and a bad plan followed correctly. The second is harder to spot, because
every individual step looks fine.

A useful rule:

> **Add planning when you've measured a problem it solves.** Not before.

If your reactive agent is wandering, taking too many steps, or forgetting the goal —
plan. If it's working, don't add planning because it seems more advanced.

---

## Keeping the goal in view

The most common failure in long agent runs isn't a bad plan. It's **drift** — the
agent slowly forgets what it was asked and ends up somewhere adjacent.

It happens because the original request is at the top of a context that keeps growing.
After 15 steps it's buried under thousands of tokens of tool output.

Cheap fix: put the goal back in front of it.

```python
def build_messages(task, history, step):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
        *history,
        # Reminder right before the model responds — the most recent thing it reads
        {"role": "user", "content": (
            f"Reminder — your task is: {task}\n"
            f"Step {step} of max {MAX_STEPS}. "
            f"If you have enough information, give your final answer now."
        )},
    ]
```

That last message costs about 40 tokens and prevents a real problem. The "give your
answer now if you can" nudge is worth including — agents often keep gathering
information past the point where they could answer.

---

## Knowing when to stop

Planning and stopping are the same problem seen from two sides. An agent that can't
tell when it's done will keep going.

Three signals, and you want all three:

```python
def should_stop(state):
    # 1. It said it's done
    if state.gave_final_answer:
        return "done"

    # 2. It's stuck — no new information for several steps
    if state.steps_without_new_info >= 3:
        return "stuck"

    # 3. Hard limits (the backstops)
    if state.steps >= MAX_STEPS:
        return "hit_step_limit"
    if state.cost >= MAX_COST:
        return "too_expensive"

    return None
```

The middle one is the important one and it's the one people leave out. A step limit
tells you *that* it failed. "Stuck" tells you *why*, and it stops earlier — cheaper,
and much more useful when someone reads the logs.

More on this in [agent-failure-modes.md](agent-failure-modes.md).

---

## Should you show the plan to users?

Usually yes, and it's underrated.

Agents are slow. A user staring at a spinner for 30 seconds assumes it's broken.
Showing the plan turns dead time into progress:

```
Looking into your billing question...
  ✓ Found your account
  ✓ Checked recent invoices
  → Checking payment method...
    Reviewing refund policy
```

Three benefits, and only one is about speed:

- The wait feels shorter because something is happening.
- The user can stop it early if it's clearly going the wrong way.
- **You get free feedback.** If users regularly cancel at the same step, that step is
  wrong, and you'd never have learned that from your metrics.

---

## Interview questions

**1. "When would you add a planning step?"**

When you've measured a problem it fixes — wandering, too many steps, goal drift on
long tasks. Not by default. Say clearly that for short tasks planning is overhead.

**2. "Your agent makes a plan and then reality doesn't match. What happens?"**

Without re-planning it follows a wrong plan and every step looks individually fine.
Describe re-planning triggers, and be honest that defining "surprising" is the hard
part.

**3. "The agent forgets what it was asked halfway through. Why, and how do you fix
it?"**

The task is buried under growing context. Re-inject the goal near the end of the
message list, keep context under control, and nudge it to answer when it can.

**4. "Plan-first or react-as-you-go?"**

Depends on task length and whether you need approval before acting. Short and
adaptive → react. Needs human sign-off, or has parallel steps → plan first. Long and
unpredictable → plan and adjust.

**5. "How do you know an agent is stuck rather than just slow?"**

Look for new information, not elapsed time. If the last three steps produced no new
tool results and no state change, it's stuck. A step limit alone can't tell the
difference.

---

## What to remember

- Reactive (think-act-look) handles most real agents fine.
- Planning helps for long tasks, human approval, and parallel work.
- Planning hurts for short tasks — it's an extra call and an extra thing to be wrong.
- A plan made before you have information is a guess, and the agent will follow it
  anyway.
- Drift is the real failure in long runs. Re-inject the goal.
- Detect "stuck," don't just cap steps.
- Show the plan to users. It buys patience and gives you feedback.

---

**Next:** [memory.md](memory.md) — what the agent remembers, and what that costs you.
