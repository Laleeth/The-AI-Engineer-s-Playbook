# Evaluating Agents

Agents are harder to evaluate than single calls, for three reasons:

1. They take many steps, and only the end result is easy to check.
2. They do things — so the output isn't text, it's a change in the world.
3. There are many valid ways to do the same task, so "did it follow the right path?"
   has no single right answer.

The most important idea here: **check the world, not the transcript.**

---

## Why reading the final message isn't enough

An agent finishes and says:

> "I've created ticket TICK-4471 and emailed the customer."

Did it? Maybe. Or maybe it drafted both and treated writing them as doing them. Or a
tool failed and it summarized optimistically.

Text-based scoring passes this. It reads like a correct answer. It says the right
things.

**Check the actual state instead:**

```python
async def task_succeeded(case, run):
    """Look at the world, not at what the agent said about the world."""
    if case.expected["ticket_created"]:
        if not await tickets.exists_for(case.customer_id):
            return False, "claimed a ticket was created; none exists"

    if case.expected["email_sent"]:
        if not await emails.sent_to(case.customer_email):
            return False, "claimed an email was sent; none was"

    return True, ""
```

This requires a test environment you can inspect and reset. **That's the main practical
obstacle to agent evaluation**, and it's worth building early — everything else depends
on it.

---

## The four things to measure

Agents are multi-objective. Collapsing them to one score destroys the information you
need, because these trade off against each other.

| Axis | Question | How |
|---|---|---|
| **Outcome** | Did it do the job? | Check end state |
| **Efficiency** | How much did it cost? | Steps, tokens, money, time |
| **Path** | Did it work sensibly? | Compare tools used |
| **Safety** | Did it stay in bounds? | Check for violations |

### Outcome

The main one. Did the world end up how it should?

```python
@dataclass
class AgentCase:
    id: str
    task: str
    setup: dict                     # world state before the run
    expected_end_state: dict        # world state after
    required_tools: list[str]       # tools it must use
    forbidden_tools: list[str]      # tools it must NOT use
    optimal_steps: int              # for efficiency scoring
    segments: dict[str, str]
```

### Efficiency

An agent that succeeds in 30 steps when 4 would do is failing, even though it
succeeded.

```python
def step_efficiency(case, run):
    if not case.optimal_steps or not run.steps:
        return None
    # Capped at 1.0 — being faster than optimal usually means it skipped
    # a verification step, which isn't better
    return min(1.0, case.optimal_steps / run.steps)
```

Track cost separately in actual money. Steps are a proxy; dollars are the resource.

### Path

Tempting to compare against a reference sequence of tool calls. **Don't do it
strictly** — there are many valid paths, and an exact-sequence match punishes an agent
that found a better one. You'd be rewarding imitation of your reference solution rather
than solving the problem.

Score coverage plus waste instead:

```python
def path_score(case, run):
    called = [c.name for c in run.tool_calls]
    called_set = set(called)

    # Hard failure: it used something it must not
    if set(case.forbidden_tools) & called_set:
        return 0.0

    # Did it use what it needed?
    required = set(case.required_tools)
    coverage = len(required & called_set) / len(required) if required else 1.0

    # Penalize repeated calls (a sign of looping)
    waste = (len(called) - len(called_set)) / max(1, len(called))

    return max(0.0, coverage - 0.5 * waste)
```

### Safety

**Safety is not averaged.** A 99% safety score means one run in a hundred did something
it wasn't allowed to do. For an agent with real permissions that's a failing grade, not
a good one.

```python
def safety_violations(run):
    """Return a list of violations. Any violation is a failure, not a low score."""
    problems = []
    for call in run.tool_calls:
        if call.effect == "destructive" and not call.approved:
            problems.append(f"destructive {call.name} without approval")
        if call.error_kind == "denied":
            problems.append(f"attempted {call.name} without permission")
    return problems
```

Report violations as **counts**, gate on zero, and treat each one as an incident rather
than a metric that moved.

---

## The free signal nobody uses

Stop-reason distribution. No labels needed, no judge, no reference answers.

```python
def stop_reasons(runs):
    counts = Counter(r.stop_reason for r in runs)
    total = sum(counts.values()) or 1
    return {reason: n / total for reason, n in counts.most_common()}
```

A healthy agent:

```
final_answer      0.91      ← finished properly
no_progress       0.05      ← got stuck, escalated (fine)
max_iterations    0.03      ← task was too hard
error             0.01
```

An unhealthy one:

```
final_answer      0.44
max_iterations    0.38      ← something is badly wrong
no_progress       0.15
error             0.03
```

The second agent "completes" most runs and is broken. Nobody notices, because the runs
technically finish.

**Put this on a dashboard before anything else.** It costs nothing, it moves before
other metrics do, and a shift in the distribution is usually the first sign that a tool
or a prompt change broke something.

---

## Building a test environment

The hard practical part. You need a world you can set up, inspect, and reset.

```python
class TestEnvironment:
    """Fake versions of the systems the agent touches."""

    def __init__(self):
        self.tickets = {}
        self.emails = []
        self.orders = {}
        self.tool_calls = []          # everything the agent did

    def setup(self, state):
        self.reset()
        self.orders.update(state.get("orders", {}))
        self.tickets.update(state.get("tickets", {}))

    def reset(self):
        self.tickets.clear()
        self.emails.clear()
        self.orders.clear()
        self.tool_calls.clear()

    # --- tools the agent can call ---

    def create_ticket(self, subject, body):
        self.tool_calls.append(("create_ticket", {"subject": subject}))
        tid = f"TICK-{len(self.tickets) + 1}"
        self.tickets[tid] = {"subject": subject, "body": body}
        return tid

    def send_email(self, to, body):
        self.tool_calls.append(("send_email", {"to": to}))
        self.emails.append({"to": to, "body": body})
        return "sent"
```

Three things this gives you:

**Repeatability.** Same setup, same starting point, every time.
**Inspection.** You can check what actually happened.
**Safety.** Testing doesn't email real customers. (This matters more than it sounds —
running agent evals against production systems is how you send 400 test emails to real
people.)

### Make failures happen on purpose

Most of your agent's problems are in error handling. Test it deliberately:

```python
class FlakyEnvironment(TestEnvironment):
    def __init__(self, failures=None):
        super().__init__()
        self.failures = failures or {}      # {"get_order": "timeout"}

    def get_order(self, order_id):
        mode = self.failures.get("get_order")
        if mode == "timeout":
            raise TimeoutError("upstream timed out")
        if mode == "not_found":
            raise FileNotFoundError(f"order {order_id} not found")
        if mode == "huge":
            return "x" * 5_000_000          # does the agent handle this?
        return super().get_order(order_id)
```

Cases worth having:

- A tool returns 404 forever → does it stop, or loop? (This is the incident in
  [agent-failure-modes.md](../04-agents/agent-failure-modes.md).)
- A tool times out once, then works → does it retry sensibly?
- A tool returns 5 MB → does it cope?
- A tool returns nothing → does it try something else, or give up?
- The task is impossible → does it say so, or invent an answer?
- Tool output contains an injection attempt → does anything bad happen?

---

## Handling variability

Agents are non-deterministic. The same case can succeed and fail across runs.

```python
async def evaluate_case(case, agent, env, runs=3):
    """Run each case several times — one run tells you very little."""
    results = []
    for _ in range(runs):
        env.setup(case.setup)
        run = await agent.run(case.task)
        ok, why = await task_succeeded(case, run, env)
        results.append({"ok": ok, "why": why, "steps": run.steps,
                        "cost": run.cost_usd, "stop": run.stop_reason})

    return {
        "success_rate": sum(r["ok"] for r in results) / runs,
        "consistent": len({r["ok"] for r in results}) == 1,
        "mean_steps": statistics.mean(r["steps"] for r in results),
        "runs": results,
    }
```

The `consistent` flag is worth having. A case that passes twice and fails once is more
worrying than one that fails every time — a consistent failure is a bug you can fix, an
inconsistent one is a reliability problem that will show up randomly in production.

**Cost note:** three runs triples your evaluation bill. A reasonable compromise is one
run for routine CI, three for release checks.

---

## What a report should look like

```python
def agent_report(results):
    return {
        # Outcome
        "success_rate":      mean(r["success_rate"] for r in results),
        "success_range":     confidence_interval(...),
        "consistent_cases":  fraction(r["consistent"] for r in results),

        # Safety — counts, not averages
        "safety_violations": total_violations(results),

        # Efficiency
        "mean_steps":        mean(...),
        "p95_steps":         percentile(..., 95),
        "mean_cost_usd":     mean(...),
        "p95_cost_usd":      percentile(..., 95),

        # Free signal
        "stop_reasons":      stop_reasons(all_runs(results)),

        # Path
        "path_score":        mean(...),
        "forbidden_tool_use": count(...),
    }
```

**Never reduce this to one number.** Success and cost trade off against each other, and
the right balance is a product decision. An agent that's 3 points more successful and
twice as expensive may or may not be better — and you can only have that conversation
if you can see both.

---

## Interview questions

**1. "How do you evaluate an agent that takes 12 steps?"**

Check end state, not the transcript. Measure four axes — outcome, efficiency, path,
safety. Mention that you need a test environment you can reset and inspect, and that
building it is the main practical obstacle.

**2. "The agent says it sent an email but didn't. How does your evaluation catch
that?"**

By checking the fake email system, not by reading the final message. This is the whole
argument for state-based checking.

**3. "Two agents both succeed but take completely different paths. How do you score
path?"**

Loosely. Coverage of required tools, hard failure on forbidden tools, penalty for
repeated calls. Not exact-sequence matching — that punishes agents that find better
paths and rewards imitating your reference.

**4. "Your agent succeeds 82% of the time but uses 3× the optimal steps. Which do you
fix?"**

Ask what each costs. If runs are cheap and fast enough, success rate matters more. If
cost or latency is the complaint, efficiency. The point is that it's a trade-off, not a
single score — and it's a product decision informed by engineering.

**5. "How would you evaluate an agent whose task takes two days and needs human
approval?"**

Separate the machine time from the human time so waiting isn't scored as slowness.
Evaluate the decision points rather than end-to-end runs — did it ask for approval at
the right moment, with the right information? And use a simulated approver for
automated runs.

**6. "How do you handle agents being non-deterministic?"**

Run each case several times, report success rate rather than pass/fail, and flag
inconsistent cases separately. Note the cost implication and the CI-versus-release
compromise.

---

## What to remember

- Check the world, not the transcript. Agents claim things they didn't do.
- You need a resettable, inspectable test environment. Build it first.
- Four axes: outcome, efficiency, path, safety. Never one number.
- Safety is counted, not averaged. Any violation is a failure.
- Stop-reason distribution is free and moves first. Dashboard it.
- Test failure handling on purpose — 404s, timeouts, huge results, injections.
- Run each case several times. Flag inconsistent ones.
- Score paths loosely, or you punish agents for being smarter than your reference.

---

**Next:** [regression-testing.md](regression-testing.md) — catching it when things get
worse.
