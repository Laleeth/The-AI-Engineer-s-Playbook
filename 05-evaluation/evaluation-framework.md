# Building an Evaluation Setup

Evaluation means: **can you tell whether a change made things better or worse?**

That's it. If you can't answer that, you're guessing, and you'll keep guessing however
sophisticated your setup looks.

This file is the overview. The rest of the section goes deeper on each part.

---

## Why teams skip it, and why that fails

Most teams ship AI features on vibes at first. Someone tries ten examples, it looks
good, it ships. This works fine — right up until it doesn't:

- Someone changes a prompt. Did it help? Nobody knows.
- A customer complains. Is this new or has it always been like this? Nobody knows.
- You want to try a cheaper model. Is it worse? Nobody knows.
- Quality drops over three months. When did it start? Nobody knows.

Every one of those is the same missing thing. And you can't add it retroactively —
without a baseline from before the change, you can't measure the change.

---

## The four things you need

Most of the complexity people add is optional. These four are not.

**1. A dataset.** Examples that look like real traffic.
**2. A way to score.** Turn an output into a number.
**3. A way to compare.** Old version vs. new version, honestly.
**4. Somewhere to keep results.** So a score from three months ago still means
something.

That's a working evaluation setup. You can build it in a week. Everything else —
dashboards, annotation tools, judge frameworks — is an improvement on top of it, not a
prerequisite.

---

## Start with what you'll measure

Before writing any code, answer: **what does "good" mean for this feature?**

This is harder than it sounds and it's where most of the thinking is. "The answer is
good" isn't measurable. Break it down:

For a support assistant, "good" might be:

- The answer is factually correct.
- It only says things supported by the documents it retrieved.
- It cites real sources.
- It says "I don't know" when it should.
- It doesn't break policy (no promises about refunds it can't make).
- It's fast enough.
- It's cheap enough.

Seven things, and they fail independently. A single "quality score" would hide all of
that.

**Rule: never report one number.** The moment you collapse quality to one score, you
lose the ability to see that a change improved five things and broke two.

---

## Cheap scorers beat expensive ones

Before reaching for a model to grade things, check whether code can do it. Code is
free, instant, and gives the same answer every time.

Things you can check with plain code:

```python
def check_json_valid(output):
    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0

def check_citations_real(output, retrieved_ids):
    """Every [id] in the answer must be a document we actually retrieved."""
    cited = re.findall(r"\[([^\]]+)\]", output)
    if not cited:
        return None                      # nothing cited — not scored
    return sum(1 for c in cited if c in retrieved_ids) / len(cited)

def check_no_banned_phrases(output, banned):
    return 0.0 if any(p in output.lower() for p in banned) else 1.0

def check_has_required_fields(output, fields):
    data = json.loads(output)
    return sum(1 for f in fields if f in data) / len(fields)
```

These catch a surprising amount. Broken JSON, made-up citations, policy violations,
missing fields — all real production failures, all free to detect.

Use a model to grade only the genuinely subjective part: is this answer actually
correct and helpful? See [llm-as-judge.md](llm-as-judge.md), and be aware that judge
costs often exceed the cost of the system you're testing.

---

## Always report uncertainty

This is the single most useful habit in evaluation, and almost nobody does it.

If you score 200 examples and get 0.81, the true value is roughly 0.75 to 0.86. So when
someone changes a prompt and the score goes to 0.83, that means **nothing**. It's
noise.

Make it impossible to hide:

```python
def score_with_range(values):
    """Return the mean and a 95% range. Never return a bare number."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    # Wilson interval for 0/1 scores — behaves correctly near 0 and 1,
    # which is exactly where eval sets live
    z = 1.96
    denom = 1 + z*z/n
    centre = (mean + z*z/(2*n)) / denom
    margin = z * math.sqrt(mean*(1-mean)/n + z*z/(4*n*n)) / denom
    return {
        "score": round(mean, 3),
        "range": (round(max(0, centre-margin), 3), round(min(1, centre+margin), 3)),
        "n": n,
    }
```

Output `0.81 (0.75–0.86)` and nobody can claim a 2-point improvement. Output `0.81` and
they will.

**How many examples do you need?** Roughly:

| Difference you want to detect | Examples needed |
|---|---|
| 10 points (0.70 → 0.80) | ~150 |
| 5 points | ~600 |
| 2 points | ~3,500 |

If you have 200 examples, you cannot detect small improvements. That's fine — but know
it, and say so when someone asks about a 2-point change. More in
[regression-testing.md](regression-testing.md).

---

## Segment everything

The most damaging evaluation failure isn't a wrong score. It's a **right score that
hides a problem.**

Real example from the senior scenarios: a team shipped a prompt change. Offline eval
went from 0.84 to 0.87. Customer complaints tripled.

Why? The eval set was all single-turn questions. Production was 62% multi-turn. The
change made single-turn better and multi-turn much worse. The aggregate went up. The
product got worse.

The fix is to tag every example and always report by segment:

```python
@dataclass
class EvalCase:
    id: str
    inputs: dict
    expected: dict
    segments: dict[str, str]      # {"turns": "multi", "lang": "de",
                                  #  "topic": "billing", "difficulty": "hard"}
```

Then:

```
Overall:        0.87 (0.83–0.90)

By turns:
  single        0.86 (0.81–0.90)  n=180
  multi         0.71 (0.64–0.77)  n=120   ← there it is

By language:
  en            0.89 (0.85–0.92)  n=240
  de            0.74 (0.65–0.82)  n=60
```

Useful segments: conversation length, language, topic, customer tier, difficulty,
whether the answer exists in your documents at all.

---

## Record what produced the score

A number without its configuration is not comparable to anything.

```python
provenance = {
    "timestamp":       time.time(),
    "dataset_hash":    hash_dataset(cases),    # did the data change?
    "dataset_size":    len(cases),
    "model":           "model-name-and-version",
    "prompt_version":  "v14",
    "temperature":     0.0,
    "retrieval_config": {"k": 5, "reranker": "v2"},
    "judge_model":     "judge-model-version",  # if you used one
    "scorer_versions": {"exact": "1", "judge": "2"},
    "code_version":    git_sha(),
}
```

Two runs with different `dataset_hash` values are **not comparable**, and your reports
should say so loudly.

There's a specific failure this prevents: your eval score jumps 9 points and everyone
celebrates. Nothing was deployed. What actually happened is the judge model was updated
by the provider, or someone regenerated the dataset. An unexplained *improvement* is as
much a bug as an unexplained regression — and far less likely to be investigated.

---

## Where the dataset comes from

Short version: **from production, not from your imagination.**

Curated examples drift away from real traffic. Real traffic has typos, one-word
queries, multiple questions in one message, other languages, and things your product
doesn't handle.

The process:

1. Sample real requests from logs.
2. Stratify so segments are represented in roughly the right proportion.
3. Have humans write the correct answer for each.
4. Refresh every quarter, keeping a frozen slice so you can still compare over time.

Full treatment in [golden-datasets.md](golden-datasets.md).

---

## Offline and online do different jobs

Two kinds of measurement, and people conflate them constantly.

**Offline eval** — run a dataset through the system, score it.

- Fast (minutes)
- Cheap
- Repeatable
- Runs before you ship
- Measures a *proxy* for quality

**Online measurement** — watch what real users do.

- Slow (days or weeks)
- Measures what you actually care about
- Only works after shipping
- Noisy, and hard to attribute

They have different jobs:

> **Offline eval prevents disasters cheaply. Online measurement tells you what's
> actually better.**

Use offline as a gate — block a release if it regresses badly. Use online to decide
whether a change was genuinely good. Trying to make offline eval do the second job is
where teams get stuck. See [online-evaluation.md](online-evaluation.md).

---

## A minimal setup that works

Here's the whole thing, small enough to actually build:

```python
async def evaluate(cases, system, scorers, concurrency=10):
    """Run cases through the system and score them."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(case):
        async with sem:
            start = time.monotonic()
            try:
                output = await asyncio.wait_for(system(case), timeout=120)
            except Exception as e:
                # A failure is a RESULT, not an exception. Record it and move on.
                return CaseResult(case.id, None, {}, error=str(e),
                                  segments=case.segments)

            scores = {}
            for scorer in scorers:
                try:
                    scores[scorer.name] = await scorer.score(case, output)
                except Exception:
                    scores[scorer.name] = None      # not scored ≠ scored zero

            return CaseResult(
                case.id, output.text, scores,
                latency_ms=(time.monotonic()-start)*1000,
                cost_usd=output.cost_usd,
                segments=case.segments,
            )

    for coro in asyncio.as_completed([one(c) for c in cases]):
        results.append(await coro)

    return Report(results, provenance=collect_provenance(cases, system, scorers))
```

Two details in there that matter more than they look:

**A failure is a result, not an exception.** One broken case shouldn't kill a
1,000-case run. And the error rate is itself a quality signal.

**A scorer that fails returns `None`, not `0`.** If your judge times out, the system
didn't score zero — it wasn't scored. Averaging in a zero silently punishes the system
for your infrastructure failing. Exclude it and report how many were actually scored.

---

## Reading the results

The most useful thing you can do with a report is not look at the number. It's **read
the twenty worst cases**.

```python
def worst(results, scorer, n=20):
    scored = [r for r in results if r.scores.get(scorer) is not None]
    return sorted(scored, key=lambda r: r.scores[scorer])[:n]
```

The aggregate tells you *whether* to investigate. Reading failures tells you *what to
fix*. Almost all the real value of an eval suite comes from a human reading failures
and noticing a pattern.

---

## Common mistakes

**Waiting until you "have time."** The right time is before you need it. You cannot
measure a regression without a baseline from before it.

**One number.** Hides everything. Segment.

**Bare numbers with no range.** Invites shipping on noise.

**A dataset built from imagination.** Diverges from production, and you won't know.

**Scoring a failed scorer as zero.** Punishes the system for your infrastructure.

**Never re-checking the eval itself.** Your suite is a measuring instrument. Does it
agree with human judgement? Does it agree with what happens online? If you've never
checked, you're trusting a number nobody validated.

**Optimizing against the same 400 examples for six months.** You end up with a system
that's excellent at those 400 examples. Keep a held-out set you don't tune against.

---

## Interview questions

**1. "How would you evaluate an LLM feature?"**

Start with what "good" means, broken into separate measurable things. Then dataset from
production, cheap deterministic scorers first, model judge only for the subjective
part, always segmented, always with confidence ranges. Mention offline vs. online doing
different jobs.

**2. "Your eval score went up but customers complain more. What happened?"**

Almost certainly the eval set doesn't match production traffic. Give the multi-turn
example. Then talk about sampling from production and segmenting.

**3. "How many examples do you need?"**

Depends on the difference you want to detect. ~150 for 10 points, ~600 for 5, ~3,500
for 2. The important part is knowing your set *can't* resolve small differences and
saying so rather than shipping on noise.

**4. "How do you know your evaluation is any good?"**

Measure it: agreement with human graders, agreement with online outcomes, coverage of
real traffic, stability across repeated runs of an unchanged system. If a suite has
never been checked against reality, it's a number nobody has validated.

**5. "A team says their feature can't be evaluated automatically. Response?"**

Ask what a bad output looks like. There's nearly always something checkable — format,
citations, policy violations, refusals. Start with the cheap deterministic checks and a
small human-graded sample, rather than accepting that nothing can be measured.

---

## What to remember

- Evaluation answers one question: did this change help or hurt?
- Four parts: dataset, scorer, comparison, storage.
- Define "good" as several separate things, not one.
- Free code checks first. Model judges only for the subjective part.
- Never report a bare number — always the range.
- Always segment. Aggregates hide regressions.
- Record what produced each score, or you can't compare anything.
- A failed scorer is "not scored," not zero.
- Read the worst 20 cases. That's where the value is.

---

**Next:** [golden-datasets.md](golden-datasets.md) — building the dataset properly.
