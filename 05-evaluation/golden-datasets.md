# Golden Datasets

A golden dataset is your set of examples with known-correct answers. Everything else in
evaluation depends on it, and it's where most evaluation goes wrong.

The failure isn't usually "the dataset is small." It's "the dataset doesn't look like
what real users send."

---

## The one rule

> **Sample from production. Don't invent examples.**

When you write test cases from your head, you write clean, well-formed, reasonable
questions. Real users send:

- typos and half-sentences
- one-word queries ("refund?")
- three questions in one message
- other languages
- questions your product doesn't handle at all
- follow-ups that only make sense given the previous turn ("what about the other one?")
- copy-pasted error messages, logs, and whole documents

If your dataset is all clean questions, your evaluation measures how well the system
handles clean questions. Then you ship a change that helps clean questions, breaks
messy ones, and your score goes up while your product gets worse.

---

## Building it

### 1. Pull real requests

Take a few thousand recent requests from your logs. Not the ones people complained
about — a random sample, or you'll build a dataset of only hard cases and your scores
will look artificially bad and never move.

```python
def sample_production(logs, n=1000, days=30):
    recent = [r for r in logs if r.timestamp > now() - days*86400]
    return random.sample(recent, min(n, len(recent)))
```

### 2. Tag each one

Before labeling answers, tag what kind of request it is. This is what lets you
segment later, and it's much easier to do now than to retrofit.

```python
@dataclass
class Case:
    id: str
    inputs: dict
    expected: dict | None = None
    segments: dict[str, str] = field(default_factory=dict)
```

Segments worth having:

| Segment | Values | Why it matters |
|---|---|---|
| `turns` | single / multi | Biggest hidden regression source |
| `language` | en / de / ja / ... | Quality varies a lot by language |
| `topic` | billing / technical / account | Different tools, different failure modes |
| `difficulty` | easy / medium / hard | Where changes actually show up |
| `answerable` | yes / no | Tests whether it says "I don't know" |
| `tier` | free / paid / enterprise | Business impact differs |

The `answerable: no` cases are the ones people forget, and they're important. If every
example has an answer in your documents, you never test whether the system correctly
says "I don't have that information." A system that always answers is a system that
hallucinates whenever retrieval fails.

### 3. Make sure hard cases are represented

If you sample randomly, 90% of your set will be easy cases where everything already
works. Changes won't move the score, because most of the score is saturated.

Better: sample so you have enough of each segment to measure it.

```python
def stratified_sample(cases, per_segment=100):
    """Take up to N from each segment, so small segments aren't invisible."""
    groups = defaultdict(list)
    for c in cases:
        key = (c.segments.get("topic"), c.segments.get("difficulty"))
        groups[key].append(c)

    out = []
    for key, group in groups.items():
        out.extend(random.sample(group, min(per_segment, len(group))))
    return out
```

**Important:** if you over-sample hard cases, your overall score is no longer
representative of production. Either keep the weights and report a weighted overall
score, or report per-segment only. Don't quietly report a stratified average as if it
were the production rate.

### 4. Write the correct answers

The expensive part. A few ways to make it cheaper without wrecking quality:

**Use your existing humans.** If support agents already answer these questions, their
sent replies are labels. Free, real, and written by people who know the domain.

**Have a model draft, a human check.** Much faster than writing from scratch. But be
careful: if the model drafts and a human rubber-stamps, you've encoded the model's
mistakes as ground truth. Make the human's job "is this right?" with real authority to
say no, and measure how often they change it. If they change almost nothing, they're
rubber-stamping.

**Label the easy part with code.** For extraction tasks, existing structured data is
often already the answer.

**Start small.** 100 well-labeled cases beat 1,000 sloppy ones. You can grow it.

### 5. Check your labels

Your evaluation cannot be more accurate than your labels. If 5% of your answers are
wrong, your ceiling is 95%, and you'll spend weeks chasing failures that aren't
failures.

Have two people label the same 50 cases and compare:

```python
def agreement(labels_a, labels_b):
    shared = set(labels_a) & set(labels_b)
    same = sum(1 for k in shared if labels_a[k] == labels_b[k])
    return same / len(shared)
```

If two humans agree less than about 80% of the time, the problem isn't the labellers —
it's that your definition of "correct" is unclear. Fix the definition before labeling
more.

This is also the honest answer to "why is our accuracy stuck at 91%?" Sometimes the
task is genuinely ambiguous and 91% is near the human ceiling.

---

## How big?

Depends on what you need to detect.

| Purpose | Size | Notes |
|---|---|---|
| Smoke test in CI | 30–50 | Catches obvious breakage, runs in a minute |
| Normal regression testing | 200–500 | Catches 5–10 point changes |
| Comparing two models | 500–1,500 | Enough for small differences |
| Per-segment measurement | 100+ **per segment** | The constraint people forget |

That last row is the one that bites. If you have 500 cases across 6 segments, some
segments have 40 examples, and you cannot say anything useful about those.

---

## Keeping it honest over time

Two problems appear after a few months.

### Problem 1: the dataset goes stale

Your product changes. Users change. Six-month-old examples stop looking like current
traffic.

Fix: refresh quarterly. Sample new production data, re-label, add it in.

### Problem 2: you overfit to it

You've been optimizing against the same 400 examples for six months. Your system is now
excellent at those 400 examples. That is not the same as being good.

Fix: **keep three sets.**

```python
@dataclass
class Datasets:
    dev: list[Case]        # you iterate against this, look at it, tune on it
    holdout: list[Case]    # you NEVER look at individual cases; scores only
    frozen: list[Case]     # never changes, so old scores stay comparable
```

- **dev** — use freely. Read failures, tune prompts against it.
- **holdout** — run it, record the score, don't read the cases. If dev improves and
  holdout doesn't, you're overfitting.
- **frozen** — never changes. This is what lets you say "we're better than we were a
  year ago" and mean it. If you refresh everything every quarter, you lose the ability
  to compare across time.

The frozen set is the one people skip, and then they can't answer whether quality has
improved over the year.

### Version it

```python
@dataclass
class DatasetVersion:
    version: str            # "2026-Q3"
    hash: str               # content hash — the thing you compare on
    n_cases: int
    created: float
    notes: str              # "added 120 German cases, removed 40 stale billing cases"
```

Scores from different dataset versions are **not comparable**. Your reports should
refuse to compare them, or at minimum print a loud warning. This prevents the "our
score jumped 9 points and nothing changed" confusion — which is nearly always a dataset
or judge change, not a quality change.

---

## Growing the dataset from failures

The best source of new cases is your own production failures.

```python
def add_from_incident(incident):
    """Every real failure becomes a permanent test case."""
    return Case(
        id=f"incident-{incident.id}",
        inputs=incident.request,
        expected={"answer": incident.correct_answer},
        segments={**incident.segments, "source": "incident"},
    )
```

Do this every time. Over a year you build a set that specifically covers the ways your
system actually breaks — which is far more valuable than a set covering the ways you
imagined it might.

A useful metric on your evaluation itself: **what fraction of production incidents had
a matching test case before they happened?** If it's low, your suite isn't covering
reality.

---

## Synthetic data: use with care

You can generate examples with a model. Sometimes useful, often misleading.

**Reasonable uses:**

- Bootstrapping when you have no traffic yet.
- Filling a segment you have too little real data for.
- Testing specific edge cases you can describe but haven't seen.

**Where it goes wrong:**

- Generated questions look like what a model thinks users ask. Real users are messier.
- If you generate questions *from* your documents, every question is answerable —
  so you never test the "I don't know" path.
- Using the same model family to generate and to answer creates a suspicious
  circularity.

If you use synthetic data, **tag it** and report scores separately:

```python
segments={"source": "synthetic"}
```

Then you can see whether your system is only good on the synthetic half.

---

## Common mistakes

**Only including questions that have answers.** No test of abstention. Add
`answerable: no` cases.

**Only including failures.** Score looks terrible, never moves, and you can't tell
whether normal cases still work.

**Labels written by the same model you're testing.** You measure agreement with
itself.

**No segments.** You can't find out that multi-turn regressed.

**Never refreshing.** Slowly stops matching production.

**Refreshing everything.** Loses comparability. Keep a frozen slice.

**Not checking label quality.** Your accuracy ceiling is your label accuracy, and you
won't know where it is.

---

## Interview questions

**1. "Where do your evaluation examples come from?"**

Production logs, sampled and stratified. Then say why invented examples fail — they're
too clean, and they cause the "eval up, complaints up" problem.

**2. "How do you stop overfitting to your eval set?"**

Three sets: dev to tune against, holdout you only get scores from, frozen for
comparing across time. Watch for dev improving while holdout doesn't.

**3. "How do you know your labels are right?"**

Double-label a sample and measure agreement. Under ~80% means the definition of
"correct" is unclear, not that the labellers are bad. Also note that label accuracy
caps your measurable accuracy.

**4. "Your dataset is 9 months old. What do you do?"**

Refresh from recent production, keep a frozen slice for comparability, version both,
and be explicit that scores across versions aren't directly comparable.

**5. "Can you use synthetic data?"**

For bootstrapping and thin segments, yes. Tag it and report separately. Warn about
questions generated from documents always being answerable, which hides the abstention
failure mode.

---

## What to remember

- Sample from production. Invented examples are too clean.
- Tag every case with segments before you label answers.
- Include cases where the right answer is "I don't know."
- 100 good labels beat 1,000 sloppy ones.
- Check label agreement — it caps everything downstream.
- Three sets: dev, holdout, frozen.
- Version the dataset. Scores across versions aren't comparable.
- Every production incident becomes a permanent test case.

---

**Next:** [rag-evaluation.md](rag-evaluation.md) — evaluating retrieval separately from
generation.
