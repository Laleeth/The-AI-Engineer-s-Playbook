# 05 — Evaluation

How to tell whether your AI system is any good, and whether a change made it better or
worse. Plain language throughout.

Evaluation is the thing teams skip first and regret most. You can't add it
retroactively — without a baseline from before a change, you can't measure the change.

## The files

| File | What it covers |
|---|---|
| [evaluation-framework.md](evaluation-framework.md) | The overview: four things you need, and how to build them in a week |
| [golden-datasets.md](golden-datasets.md) | Building test data that looks like real traffic |
| [rag-evaluation.md](rag-evaluation.md) | Measuring retrieval and generation separately |
| [llm-as-judge.md](llm-as-judge.md) | Using a model to grade, without fooling yourself |
| [agent-evaluation.md](agent-evaluation.md) | Testing something that takes twelve steps and changes the world |
| [regression-testing.md](regression-testing.md) | Telling a real drop from normal noise |
| [online-evaluation.md](online-evaluation.md) | Measuring what real users actually experience |

## Read in this order

Start with [evaluation-framework.md](evaluation-framework.md) — it's the overview and
everything else assumes it.

Then pick by what you're building: RAG, agents, or neither.
[regression-testing.md](regression-testing.md) and
[online-evaluation.md](online-evaluation.md) apply to everything.

## The short version

**Never report one number.** A single "quality score" hides the change that improved
five things and broke two. Segment by conversation length, language, topic, and
difficulty.

**Never report a bare number either.** On 200 examples, 0.81 really means 0.75–0.86. So
when a change moves it to 0.83, that's noise. Put the range in your output format and
nobody can claim an improvement that isn't there.

**Sample from production, not from your imagination.** Invented test cases are too
clean. The classic failure: eval set is all single-turn, production is 62% multi-turn,
a change improves single-turn and wrecks multi-turn, the aggregate goes up, and
complaints triple.

**Compare paired, not as two averages.** Both versions ran the same cases, so compare
case by case. Only the cases where they disagree carry information. This makes your
test far more sensitive — often the difference between detecting a real improvement and
discarding it.

**Free checks before expensive ones.** Valid JSON, real citations, banned phrases,
required fields — all checkable in code, instantly, for nothing. Use a model judge only
for the genuinely subjective part, and note that judge costs often exceed the cost of
the system you're testing.

**Check the judge against humans.** Fifty cases a month. Until you've done it, you have
a number, not a measurement.

**A failed scorer means "not scored," not "scored zero."** Averaging in a zero when
your judge timed out silently punishes the system for your infrastructure failing.

**Investigate improvements too.** An unexplained 9-point jump with no deploy is a
measurement bug — usually a judge or dataset change — and it poisons every comparison
after it.

**Offline and online do different jobs.** Offline prevents disasters cheaply. Online
tells you what's actually better. When they disagree, believe online and fix the
offline metric.

## A worked example worth knowing

Prompt A scores 0.79. Prompt B scores 0.83. Both on the same 200 cases. Ship B?

Break it down:

```
both correct     144
both wrong        20
B fixed it        22    ← only these
B broke it        14    ← and these carry information
                 ───
                 200
```

Only 36 cases disagree. McNemar's test gives **p ≈ 0.24**. Not significant. You'd need
around 880 cases to detect a 4-point difference reliably.

The useful answer isn't "we can't tell" — it's *"this is within noise; I can expand the
eval set, or we ship behind a canary and measure online."* Statistical honesty is only
valuable when it comes with a path forward.

(Every number here is re-computed by CI — the breakdown, the p-value and the sample
size. See [`scripts/verify.py`](../scripts/verify.py) and the
[coding rounds README](../11-coding-rounds/README.md#the-code-is-tested).)

## Related

- [11-coding-rounds/evaluation-harness.md](../11-coding-rounds/evaluation-harness.md) —
  build the harness, with tested code for the statistics
- [04-agents/agent-failure-modes.md](../04-agents/agent-failure-modes.md) — what you're
  testing for
- [12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md)
  — "it works in staging," as an interview exercise
