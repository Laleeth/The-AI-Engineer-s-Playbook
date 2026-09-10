# 12 — Senior Scenarios

Open-ended system-design and judgment problems for senior, staff, and principal AI
engineering interviews. Every scenario is built around a realistic situation with
numbers, constraints, and competing pressures — the kind where there is no single right
answer and the interview signal is in the reasoning.

## What's here

| File | What it tests | Scenarios |
|---|---|---|
| [architecture-tradeoffs.md](architecture-tradeoffs.md) | Choosing between defensible options and knowing when your choice stops being right | 5 |
| [production-incidents.md](production-incidents.md) | Hypothesis discipline under pressure; debugging with partial evidence | 7 + 8 rapid-fire |
| [cost-vs-quality.md](cost-vs-quality.md) | Unit economics, routing, and refusing false dichotomies | 5 |
| [build-vs-buy.md](build-vs-buy.md) | Total cost of ownership, reversibility, and honest scoping | 5 |
| [model-selection.md](model-selection.md) | Turning "which model" into a measurable engineering decision | 5 |
| [scaling.md](scaling.md) | Quantitative reasoning about inference, memory, and throughput | 6 |
| [staff-level.md](staff-level.md) | Problem selection, influence without authority, organizational design | 6 |

## Format

Each scenario follows the same structure:

- **Situation** — the business and engineering context, with real numbers.
- **Your Task** — what you're being asked to solve.
- **Constraints** — traffic, latency, budget, security, team size, deadlines.
- **How Would You Approach It?** — a senior-level reasoning path, behind a spoiler so you
  can attempt it first. It deliberately does *not* start with the answer.
- **Strong Answer** — a detailed response focused on decisions and their justification.
- **Trade-offs** — competing approaches, with the conditions under which each wins.
- **What Could Go Wrong?** — realistic failure modes, including failures of the fix.
- **Metrics** — what to measure, technical and business.
- **Follow-up Questions** — the interviewer changes the constraints, not the definitions.
- **What Separates Senior From Staff?** — the same problem at two levels.
- **Interviewer Notes** — what's actually being evaluated, with red and green flags.

Many scenarios **evolve in rounds**: you design the system, then traffic grows 10×, then
costs triple, then security changes the requirements, then accuracy drops. That's how
these interviews actually run.

## How to practice

1. Read only **Situation**, **Your Task**, and **Constraints**. Close the file.
2. Spend 10–15 minutes writing your approach — *including the numbers you'd ask for*.
3. Open the reasoning path. Note where you jumped to a solution before measuring.
4. Read the strong answer and trade-offs, then ask: **what number would have to change
   for the other option to win?** If you can't answer that, you haven't understood the
   trade-off.
5. Answer the follow-ups out loud. That's where the interview actually happens.

The incident scenarios work best with a partner: one person reveals evidence tables only
when asked for specific metrics, and the other must state a hypothesis, say what would
disconfirm it, and ask for exactly that data.

## Recurring ideas

A few things show up across almost every file, because they show up across almost every
real system:

- **Measure before you design.** "p95 is 5 seconds" is not actionable until it's
  decomposed by stage.
- **Quality is never a scalar.** Segment it, or you'll optimize the average into a tail
  disaster.
- **Input tokens usually dominate cost; output tokens usually dominate latency.**
- **Version everything.** Model, prompt, embedding model, corpus, judge. Every hard
  question in these files becomes answerable when this metadata exists, and unanswerable
  when it doesn't.
- **The thing that changed is usually not the thing that broke.** Model behavior is
  downstream of prompts, retrieval, data, and traffic mix.
- **A plan with no "no" in it is a wish list.**

## Related

The [coding rounds](../11-coding-rounds/) implement many of the mechanisms discussed
here — rate limiting, retries and circuit breaking, retrieval, agent termination, and
evaluation harnesses. Several incidents in `production-incidents.md` are the direct
consequence of getting one of those implementations wrong.
