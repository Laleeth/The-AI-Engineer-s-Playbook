# LLM as Judge

Using a model to grade another model's output. It's cheap, it scales, and it works
reasonably well — as long as you know what it's bad at and check it against reality
occasionally.

The risk isn't that judges are useless. It's that they produce confident numbers that
*look* like measurement, so nobody checks them, and the whole team optimizes toward
something nobody validated.

---

## When to use one, and when not to

**Don't use a judge if code can do the job.** Code is free, instant, and gives the same
answer every time.

Things code should check:

```python
json.loads(output)                          # is it valid JSON?
required_fields <= set(data.keys())         # are the fields there?
all(c in retrieved_ids for c in citations)  # are citations real?
not any(p in output.lower() for p in BANNED)  # policy violations
len(output) < MAX_LENGTH                    # is it too long?
expected_value == extracted_value           # exact match on a known field
```

**Use a judge for the genuinely subjective part:** is this answer correct, complete,
and helpful? Is the tone right? Does this summary capture the important points?

Worth knowing: on a large eval set, **the judge often costs more than the system you're
testing**, because it reads both the question and the answer and sometimes the context
too. Budget for it.

---

## Writing a rubric that works

The difference between a useful judge and a useless one is almost entirely the rubric.

### Bad

```
Rate this answer from 1 to 10.
```

You'll get 7s and 8s for everything. There's no shared meaning for the numbers, so
scores drift between runs and mean nothing between examples.

### Good

```python
RUBRIC = """Grade the ANSWER against the REFERENCE for the QUESTION.

Score 0-4:
4 - Fully correct and complete. No claims beyond the reference.
3 - Correct, but missing a minor detail from the reference.
2 - Partially correct, OR correct but adds a claim not in the reference.
1 - Mostly incorrect, but on topic.
0 - Incorrect, or refuses when the reference contains the answer.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Reply with JSON only:
{{"score": <0-4>, "reason": "<one sentence>"}}"""
```

Four things make this work:

**A small scale.** 0–4 or 0–3. Anything finer is fake precision — the judge can't
reliably tell a 7 from an 8, so you're adding noise.

**Each level described.** The judge needs to know what a 2 means. So do you, when
you're reading the results.

**A reference answer.** Grading against something concrete is much more stable than
grading in the abstract.

**A reason.** Costs a few tokens and makes the results readable. When you look at the
20 worst cases, the reasons are what tell you the pattern.

### Split the axes

Don't ask for one score covering several things. Ask separately:

```python
AXES = {
    "correctness":  "Is the factual content right?",
    "completeness": "Does it cover everything the reference covers?",
    "grounding":    "Is every claim supported by the provided context?",
    "tone":         "Is the tone appropriate for a customer?",
}
```

A change might improve correctness and hurt tone. One number hides that. This is the
same argument as segmenting your dataset, applied to the scoring side.

---

## What judges are bad at

Every one of these is a real, documented bias. Know them, because they shape what your
system will drift toward if you optimize against a judge blindly.

### They prefer longer answers

A thorough-looking answer scores higher than a concise correct one. If you optimize
against a judge, your answers get longer over time — and your costs go up while your
users get more to read.

Counter it in the rubric:

```
Length is not quality. A short complete answer scores the same as a long one.
Penalise padding, repetition, and restating the question.
```

Then check: is your mean answer length creeping up release over release? That's the
signal.

### They prefer confident writing

A hedged, correct answer scores lower than a confident, wrong one. Which is exactly
backwards for anything where being wrong matters.

```
Confidence is not correctness. An answer that says "I'm not certain, but..."
and is right scores higher than a confident answer that is wrong.
```

### Position bias in comparisons

If you show two answers and ask which is better, the judge favors one position — often
the first.

**Always randomize, and ideally run both orders:**

```python
async def compare(a, b, judge):
    """Run both orders and only count it if they agree."""
    forward  = await judge(PAIRWISE.format(first=a, second=b))
    backward = await judge(PAIRWISE.format(first=b, second=a))

    if forward == "first" and backward == "second":
        return "a"
    if forward == "second" and backward == "first":
        return "b"
    return "tie"        # judge is inconsistent — treat as no preference
```

That "tie" case is informative on its own. If 30% of comparisons are inconsistent, your
judge isn't reliable enough for pairwise use on this task.

### They favor their own style

A judge tends to prefer outputs from the same model family — similar phrasing,
structure, formatting. If you're comparing two different models with one judge, that's
a thumb on the scale.

Mitigation: use a different model family as judge, or validate against human grading
for that specific comparison.

### They're inconsistent

Same input, same judge, different score. Reduce it:

```python
score = await judge(prompt, temperature=0.0)   # always
```

Temperature 0 doesn't guarantee identical output, but it helps a lot. Then measure
what's left:

```python
async def judge_stability(cases, judge, runs=3):
    """Run the same cases several times and see how much the score moves."""
    all_runs = [await score_all(cases, judge) for _ in range(runs)]
    per_case = zip(*all_runs)
    return statistics.mean(statistics.pstdev(scores) for scores in per_case)
```

If that number is large relative to the differences you're trying to detect, your judge
can't resolve them.

---

## The step almost everyone skips

**Check the judge against humans.**

Until you do, you have a number, not a measurement.

```python
def judge_agreement(cases, judge_scores, human_scores):
    """How often does the judge agree with a person?"""
    shared = set(judge_scores) & set(human_scores)

    exact = sum(1 for k in shared if judge_scores[k] == human_scores[k])
    close = sum(1 for k in shared if abs(judge_scores[k] - human_scores[k]) <= 1)

    return {
        "exact_agreement": exact / len(shared),
        "within_one": close / len(shared),
        "n": len(shared),
    }
```

**How to do it without much effort:**

1. Take 50 cases, at random, from a normal eval run.
2. Have a person grade them with the same rubric, without seeing the judge's scores.
3. Compare.
4. Repeat monthly.

**What to look for:**

- **Within-one agreement above ~80%** is usually workable for tracking changes.
- **Exact agreement above ~60%** is decent for a 0–4 scale.
- **Systematic bias** matters more than random disagreement. If the judge is
  consistently one point higher than humans, that's fine for comparing versions — it
  cancels out. If it's higher on long answers specifically, that's a bias that will
  distort your decisions.

Look at the disagreements, not just the number. That's where you find out your rubric
is ambiguous.

### Pin the judge, or your history breaks

```python
@dataclass
class JudgeConfig:
    model: str              # exact version, not "latest"
    rubric_version: str
    temperature: float = 0.0

    def version_string(self):
        return f"{self.model}/{self.rubric_version}"
```

Record it with every score. When the provider updates the judge model, every score
shifts and none of your history is comparable. That's the "our eval scores jumped 9
points and nothing changed" incident — and unexplained improvements get investigated
far less than unexplained regressions, which is why this one can quietly poison months
of data.

When you do change the judge, re-score a sample of old runs with the new judge so you
have a translation between the two.

---

## Reference-free judging

Sometimes you have no reference answer. You can still judge, but expect it to be less
reliable.

```python
REFERENCE_FREE = """Assess whether the ANSWER is supported by the CONTEXT.
You do not have a reference answer. Judge only support, not correctness.

CONTEXT: {context}
ANSWER: {answer}

Reply with JSON:
{{"supported": <0-4>, "unsupported_claims": ["..."]}}"""
```

Reference-free works reasonably for **grounding** (is this supported by the given
text?) because that's a comparison the judge can actually make. It works much worse for
**correctness**, because the judge is then relying on its own knowledge, with its own
gaps and mistakes.

Rule: reference-free for grounding, reference-based for correctness.

---

## Costs, and how to cut them

A judge call per case per run adds up, especially with 1,000 cases in CI.

**Cheap wins:**

- **Run free checks first.** If the JSON is invalid or a citation is fake, you already
  know it failed. Don't pay a judge to confirm it.

  ```python
  async def score(case, output, judge):
      if not passes_cheap_checks(output):
          return 0.0                    # no judge call needed
      return await judge(case, output)
  ```

- **Use a smaller judge model.** Test whether it agrees with the bigger one on your
  task. Often it does, at a fraction of the cost.

- **Sample.** Judge 200 of 1,000 cases for the routine run; judge everything for a
  release. Report `n` so nobody mistakes a sample for the full set.

- **Cache by content.** Same output, same rubric, same judge version → same score. Key
  on the hash of all three.

---

## Interview questions

**1. "How do you know your LLM judge is any good?"**

Compare against human grading on a sample, monthly. Report exact and within-one
agreement. Look at the disagreements to find rubric ambiguity. Say clearly that until
you've done this, you have a number, not a measurement.

**2. "What biases do judges have?"**

Longer answers, confident phrasing, position in pairwise comparisons, their own model
family's style. Then say what you'd do about each — rubric wording, randomized order,
different judge family.

**3. "Your provider updated the judge model. What happens?"**

Every score shifts and your history is no longer comparable. Pin versions, record them
with each score, and re-score a sample of old runs to build a translation. Mention that
unexplained improvements are as suspicious as regressions.

**4. "Judge or human?"**

Judges for scale and speed, humans for calibrating the judge and for anything
high-stakes. Not either/or — humans validate the judge, the judge does the volume.

**5. "The judge scores 0.85 and users complain. Which do you believe?"**

Users. Then investigate: does the eval set match production traffic, and does the judge
agree with humans on the cases users complained about? An offline/online disagreement
is a bug in the offline metric and should be investigated as one.

---

## What to remember

- Use code where you can. Judges only for the subjective part.
- Small scale (0–4), each level described, with a reference and a reason.
- Split axes — correctness, completeness, grounding, tone.
- Known biases: length, confidence, position, own-family style.
- Randomize order in pairwise comparisons and run both ways.
- Temperature 0, and measure the remaining wobble.
- Check against humans on 50 cases a month. Without this it's not a measurement.
- Pin the judge version and record it with every score.
- Reference-free for grounding, reference-based for correctness.

---

**Next:** [agent-evaluation.md](agent-evaluation.md) — evaluating something that takes
twelve steps.
