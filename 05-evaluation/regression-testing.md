# Regression Testing

A regression is when something that worked stops working. In normal software you catch
them with tests that pass or fail. With AI systems there's no clean pass/fail — scores
move a bit every run, and you have to decide which movements matter.

That decision is the whole problem.

---

## The core difficulty

You run your eval. Last week: 0.81. This week: 0.78.

Is that a regression?

**You cannot answer that without knowing how much the score moves on its own.** If
running the identical system twice gives you 0.81 and 0.77, then 0.78 is normal. If it
gives you 0.81 and 0.810, then 0.78 is a real problem.

So the first thing to measure is not your system. It's **your measurement.**

```python
async def measure_noise(dataset, system, runs=5):
    """Run the SAME system several times. How much does the score move?"""
    scores = [await evaluate(dataset, system) for _ in range(runs)]
    values = [s.mean("correctness") for s in scores]
    return {
        "mean": statistics.mean(values),
        "spread": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }
```

Run this once. Whatever spread you find is your noise floor — **any change smaller than
that is not detectable by your setup**, and treating it as real is how teams ship on
noise for months.

Sources of the wobble: model non-determinism (even at temperature 0), judge
inconsistency, retrieval ties broken differently, and ordinary sampling variation.

---

## What counts as a regression

Three things have to be true. Miss any one and you'll either miss real problems or
chase ghosts.

**1. The change is bigger than your noise floor.**

**2. It's outside the confidence range.** With 200 examples, a score of 0.81 has a range
of roughly 0.75–0.86. A move to 0.78 sits inside that. Not detectable.

**3. It's a paired comparison.** Both versions ran on the same cases, so compare them
case by case, not as two averages. This matters more than people expect — see below.

---

## Compare paired, always

This is the single most useful technical point in this file.

Both versions ran the same examples. Some examples are hard for everyone; some are
easy. If you compare two averages, all that variation ("some cases are just harder")
sits in the middle and drowns out the difference you're looking for.

Compare case by case and it disappears:

```python
def compare(scores_old, scores_new):
    """Paired comparison. Only cases both versions scored."""
    shared = sorted(set(scores_old) & set(scores_new))
    old = [scores_old[k] for k in shared]
    new = [scores_new[k] for k in shared]

    got_better = sum(1 for o, n in zip(old, new) if n > o)
    got_worse  = sum(1 for o, n in zip(old, new) if n < o)
    unchanged  = sum(1 for o, n in zip(old, new) if n == o)

    return {
        "n": len(shared),
        "old_mean": statistics.mean(old),
        "new_mean": statistics.mean(new),
        "better": got_better,
        "worse": got_worse,
        "same": unchanged,
        "p_value": mcnemar(old, new),
    }
```

Notice `unchanged` is usually the biggest number, and it carries **no information**
about which version is better. Only the cases where the two versions disagree tell you
anything. That's exactly what McNemar's test uses:

```python
def mcnemar(old, new):
    """Only disagreements matter. Cases both got right, or both got wrong,
    tell you nothing about which version is better."""
    new_fixed  = sum(1 for o, n in zip(old, new) if o == 0 and n == 1)
    new_broke  = sum(1 for o, n in zip(old, new) if o == 1 and n == 0)
    total = new_fixed + new_broke

    if total == 0:
        return 1.0

    if total < 25:
        # Exact test for small numbers — the approximation is unreliable here
        k = min(new_fixed, new_broke)
        tail = sum(math.comb(total, i) for i in range(k + 1)) / (2 ** total)
        return min(1.0, 2 * tail)

    chi2 = (abs(new_fixed - new_broke) - 1) ** 2 / total
    return math.erfc(math.sqrt(chi2 / 2))
```

### A worked example

200 cases. Old scores 0.79, new scores 0.83. Ship it?

Break it down:

```
Both correct:         144
Both wrong:            20
New fixed it:          22    ← only these
New broke it:          14    ← and these matter
                      ───
                      200

old = 144 + 14 = 158 correct → 0.79
new = 144 + 22 = 166 correct → 0.83
```

Only 36 cases disagree. McNemar on 22 vs. 14 gives **p ≈ 0.24**. Not significant. You'd
need around 880 cases to reliably detect a 4-point difference.

The honest answer to "should we ship?" is: *"the difference is within noise. I can
expand the eval set, or we ship it behind a small canary and measure online."*

Offering that second path is what makes statistical honesty useful rather than
obstructive.

---

## How many examples you need

| Difference to detect | Examples needed (roughly) |
|---|---|
| 15 points | ~60 |
| 10 points | ~140 |
| 5 points | ~550 |
| 2 points | ~3,500 |
| 1 point | ~14,000 |

These come from the paired formula, assuming about 18% of cases disagree between the
two versions (so the standard deviation of the per-case difference is around 0.42):

```python
def examples_needed(effect, sd_of_differences=0.42, power=0.8, alpha=0.05):
    z_alpha, z_beta = 1.96, 0.84
    return math.ceil(((z_alpha + z_beta) * sd_of_differences / effect) ** 2)
```

If your two versions disagree on far more or far fewer cases than 18%, recompute with
your own number — measure it from a real comparison rather than trusting the table.

Two consequences worth internalizing:

**Small eval sets can only catch big breaks.** That's genuinely fine — a 200-case set
that catches every 10-point disaster is doing useful work. Just don't claim it can
detect a 2-point improvement.

**Detecting a 1-point improvement offline is usually not worth it.** 14,000 labeled
examples costs more than measuring it online. Know when to stop.

---

## What to run, and when

Not everything needs to run on every commit.

```python
TIERS = {
    "smoke": {
        "size": 40,
        "when": "every commit",
        "time": "under 2 minutes",
        "blocks": "obvious breakage — crashes, empty outputs, invalid JSON",
    },
    "standard": {
        "size": 400,
        "when": "every pull request touching prompts, models or retrieval",
        "time": "10-15 minutes",
        "blocks": "regressions over 5 points, any segment dropping over 10",
    },
    "full": {
        "size": 2000,
        "when": "before release, and nightly",
        "time": "1-2 hours",
        "blocks": "release, if any segment regresses significantly",
    },
}
```

The smoke tier is the one that earns its keep. It's fast enough that nobody minds, and
it catches the embarrassing failures — a prompt template with a broken variable, a
model name typo, a retrieval config that returns nothing.

---

## Gating a release

```python
@dataclass
class Gate:
    metric: str
    max_drop: float          # how much regression is tolerated
    min_n: int               # need at least this many cases to judge
    blocking: bool

GATES = [
    # Safety: no regression allowed at all
    Gate("policy_violations", max_drop=0.0,  min_n=200, blocking=True),
    Gate("citation_validity", max_drop=0.02, min_n=200, blocking=True),

    # Quality: small movements tolerated
    Gate("correctness",       max_drop=0.03, min_n=400, blocking=True),
    Gate("groundedness",      max_drop=0.03, min_n=400, blocking=True),

    # Per-segment: catches the aggregate-hides-a-regression problem
    Gate("correctness:multi_turn", max_drop=0.05, min_n=100, blocking=True),
    Gate("correctness:de",         max_drop=0.05, min_n=100, blocking=False),

    # Non-quality
    Gate("p95_latency_ms",    max_drop=-0.20, min_n=400, blocking=True),
    Gate("cost_per_request",  max_drop=-0.30, min_n=400, blocking=False),
]
```

Three design points:

**Segment gates are the important ones.** A change can improve the overall score while
destroying multi-turn conversations. Only a per-segment gate catches that, and it's
exactly the incident that keeps happening.

**`min_n` prevents nonsense.** With 30 examples in a segment, don't block a release on
its score.

**Not everything blocks.** Too many blocking gates and people start disabling them.
Warn on the ones that need a human decision; block on the ones that are never
acceptable.

---

## Regressions that don't show up in the score

Some real problems leave the quality score untouched.

**Cost.** Same quality, 3× the tokens. Gate on cost per request.

**Latency.** Same quality, twice as slow.

**Length.** Answers creep longer over time — a classic side effect of optimizing
against a judge that prefers long answers. Track mean output length.

**Refusal rate.** The system starts declining things it used to handle. Scores fine on
the cases it does answer.

**Distribution shift.** It changes *which* things it gets right — better on easy cases,
worse on hard ones. Net score unchanged, users worse off. Only visible if you segment
by difficulty.

Track all of these alongside quality:

```python
ALWAYS_TRACK = [
    "correctness", "groundedness",
    "cost_per_request", "p95_latency_ms",
    "mean_output_tokens", "refusal_rate",
    "error_rate", "abstention_rate",
]
```

---

## When something regresses

A process that works:

**1. Confirm it's real.** Re-run. Is it outside the noise floor? Is it paired?

**2. Find which cases changed.**

```python
def newly_broken(old_results, new_results, scorer):
    """Cases the old version got right and the new one gets wrong."""
    return [
        case_id for case_id in old_results
        if old_results[case_id].scores[scorer] == 1.0
        and new_results.get(case_id, {}).scores.get(scorer) == 0.0
    ]
```

**3. Read them.** Ten broken cases usually reveal a pattern in a couple of minutes.
This is where the actual diagnosis happens — not in the numbers.

**4. Check the segments.** Is it one language, one topic, one difficulty band? A
concentrated regression is much easier to explain than a diffuse one.

**5. Check what actually changed.** Model version, prompt version, retrieval config,
corpus, judge version, dataset version. If more than one changed, you can't attribute —
which is the argument for shipping changes one at a time.

**6. Decide.** Fix it, accept it with a reason written down, or roll back.

---

## Making changes attributable

The most common cause of "we don't know why quality dropped" is shipping several
changes together.

```python
@dataclass
class RunRecord:
    timestamp: float
    dataset_version: str
    dataset_hash: str
    model: str
    prompt_version: str
    retrieval_config_hash: str
    judge_version: str
    code_sha: str
    scores: dict
```

Store this with every run. Then when the score moves, you can see exactly what was
different. Without it, you're guessing — and the guessing is expensive.

**Ship one change at a time** whenever you can. Two prompt changes in one release means
you learn nothing from the result.

---

## The unexplained improvement

Worth its own section, because nobody investigates it.

Your score jumps 9 points. Nothing was deployed. The team is pleased.

This is a bug, not good news. Likely causes:

- The judge model was updated by the provider.
- The dataset was regenerated.
- A scoring bug now marks more answers correct.
- Cases got filtered out somewhere.

An unexplained improvement means your measurement changed. If you accept it, every
comparison after it is against a moving baseline.

**Rule: investigate score movements in both directions.** Re-run yesterday's exact
config today. If the number differs, you have a measurement problem, not a quality one.

---

## Interview questions

**1. "Your eval score dropped from 0.81 to 0.78. Is that a regression?"**

Not necessarily. Ask about the noise floor, the confidence range, and the number of
examples. Then say you'd compare paired and run McNemar. On 200 cases, that difference
is probably noise.

**2. "Why compare paired rather than two averages?"**

Because both versions ran the same cases, so case difficulty is a controllable source
of variance. Removing it makes the test much more sensitive. Only the cases where the
versions disagree carry information.

**3. "How do you stop regressions reaching production?"**

Tiered gates: fast smoke test on every commit, fuller suite on relevant PRs, full run
before release. Per-segment gates, because aggregates hide segment regressions. And a
canary, because offline eval is a proxy.

**4. "Your eval passes but customers complain. What went wrong?"**

The eval set doesn't match production traffic. Or you're measuring the wrong thing. Or
the regression is in something you don't score — cost, latency, length, refusals. Then
talk about sampling from production and tracking non-quality metrics.

**5. "Scores jumped 9 points with no deploy. What do you do?"**

Treat it as a bug. Something in the measurement changed — judge version, dataset,
scoring code. Re-run the old config and compare. Say that unexplained improvements are
as suspicious as regressions and get investigated far less.

**6. "How do you gate CI when eval takes an hour?"**

Tier it. A 40-case smoke test in under two minutes on every commit, the full suite
nightly and pre-release. Most regressions worth blocking on are big enough for the
small set to catch.

---

## What to remember

- Measure your noise floor first. Changes smaller than it aren't detectable.
- Always compare paired. Only disagreements carry information.
- 200 cases catches 10-point breaks, not 2-point ones. Know which you have.
- Gate per segment, not just overall.
- Track cost, latency, length, and refusals — regressions hide there.
- Record every version with every run, or you can't attribute anything.
- Ship one change at a time.
- Investigate improvements too. An unexplained jump is a measurement bug.

---

**Next:** [online-evaluation.md](online-evaluation.md) — measuring what real users
actually experience.
