# Coding Round: The Evaluation Harness

Every other problem in this section can be verified by running it. This one can't — an
evaluation harness's own correctness is the thing being tested, and a harness that
reports confident, meaningless numbers is worse than no harness at all, because teams act
on it.

That makes this the most philosophically loaded coding round in the set. The code is
straightforward; the judgment is not.

Four problems:

1. [A minimal evaluation harness](#problem-1-the-minimal-harness)
2. [Statistical comparison of two systems](#problem-2-is-b-actually-better-than-a)
3. [Evaluating RAG](#problem-3-evaluating-rag)
4. [Evaluating agents](#problem-4-evaluating-agents)

Python 3.11+, standard library plus numpy for statistics.

---

## Problem 1: The Minimal Harness

## Scenario

Your team ships prompt and model changes on intuition. Build the harness that lets them
ship on evidence.

Input: a dataset, a system under test, and a scoring function.
Output: quality, latency, tokens, cost, and failure reasons — with enough metadata that a
score from three months ago is still interpretable.

## Requirements

- Run a dataset through a system under test, concurrently and with bounded parallelism.
- Support multiple scorers (exact match, contains, JSON-schema validity, LLM judge).
- Report quality, latency, tokens, cost, and failure classification.
- **Version and record everything** that could change a score: dataset hash, prompt
  version, model, scorer version, judge version.
- Be resumable — a crash 800 items into a 1,000-item run should not restart from zero.
- Report per-segment results, not just an aggregate.

## Starter Code

```python
from dataclasses import dataclass


@dataclass
class EvalCase:
    id: str
    inputs: dict
    expected: dict | None = None
    segments: dict[str, str] | None = None      # e.g. {"lang": "de", "turns": "multi"}


@dataclass
class CaseResult:
    case_id: str
    output: str | None
    scores: dict[str, float]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None


class EvalHarness:
    async def run(self, dataset, system, scorers) -> "EvalReport": ...
```

## Expected Behavior

```python
report = await harness.run(dataset, system=my_rag_pipeline, scorers=[exact, judge])

report.summary()
# {"n": 500, "exact_match": 0.81, "judge": 0.77, "p95_latency_ms": 3120,
#  "total_cost_usd": 4.12, "error_rate": 0.004}

report.by_segment("lang")
# {"en": {"n": 300, "judge": 0.84}, "de": {"n": 120, "judge": 0.71}, ...}

report.provenance()
# dataset_hash, prompt_version, model, scorer_versions, judge_version, timestamp
```

---

<details>
<summary>💡 Reveal a hint</summary>

**The metadata is the feature.** A score of 0.81 is meaningless without knowing which
dataset, which prompt, which model, and which scorer produced it. Six weeks later someone
will ask "did quality regress when we changed X?" and the answer is only reachable if
every run recorded its full configuration. Design the provenance record first; the rest
follows.

**Aggregate scores hide regressions.** The scenario in
`../12-senior-scenarios/production-incidents.md#incident-6` is exactly this: a change
that improved single-turn and destroyed multi-turn, reported as a 3-point gain. Segment
reporting is not a nice-to-have.

**A single number without a confidence interval invites bad decisions.** On 200 examples,
0.81 vs. 0.79 is noise. If the harness reports bare numbers, teams will ship on noise and
attribute it to their change. Make the interval part of the output format so it can't be
omitted.

**Determinism is a property you must engineer.** Same dataset, same system, same scorer →
same score. If not, you cannot compare runs. That means: fixed temperature, pinned model
versions, pinned judge versions, deterministic ordering, and recording all of it. A
harness whose output varies between runs of an unchanged system is measuring itself, not
the system.

**Failures are results, not exceptions.** A case that errors should produce a
`CaseResult` with an error classification, not abort the run. And the error rate is itself
a quality metric.
</details>

---

<details>
<summary>✅ Reveal the reference implementation</summary>

```python
"""An evaluation harness.

Design priorities, in order:
  1. provenance — a score is meaningless without its full configuration
  2. segmentation — aggregates hide the regressions that matter
  3. statistical honesty — intervals, not bare numbers
  4. resumability — long runs must survive a crash
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence

import numpy as np

log = logging.getLogger("eval")


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    inputs: dict
    expected: dict | None = None
    segments: dict[str, str] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps({"i": self.inputs, "e": self.expected}, sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass
class SystemOutput:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)     # retrieved docs, tool calls, ...


@dataclass
class CaseResult:
    case_id: str
    output: str | None
    scores: dict[str, float]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    segments: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    error_kind: str | None = None


class Scorer(Protocol):
    name: str
    version: str
    async def score(self, case: EvalCase, output: SystemOutput) -> float: ...


SystemFn = Callable[[EvalCase], Awaitable[SystemOutput]]


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

class ExactMatch:
    name, version = "exact_match", "1"

    def __init__(self, field_: str = "answer", normalize: bool = True) -> None:
        self.field = field_
        self.normalize = normalize

    async def score(self, case: EvalCase, output: SystemOutput) -> float:
        if not case.expected:
            return float("nan")
        expected = str(case.expected.get(self.field, ""))
        actual = output.text
        if self.normalize:
            expected, actual = expected.strip().lower(), actual.strip().lower()
        return 1.0 if expected == actual else 0.0


class Contains:
    name, version = "contains", "1"

    def __init__(self, field_: str = "must_contain") -> None:
        self.field = field_

    async def score(self, case: EvalCase, output: SystemOutput) -> float:
        if not case.expected:
            return float("nan")
        required = case.expected.get(self.field) or []
        if isinstance(required, str):
            required = [required]
        if not required:
            return float("nan")
        text = output.text.lower()
        hits = sum(1 for r in required if r.lower() in text)
        return hits / len(required)


class SchemaValid:
    """Deterministic, free, and catches a real production failure class."""
    name, version = "schema_valid", "1"

    def __init__(self, schema: dict) -> None:
        self.schema = schema

    async def score(self, case: EvalCase, output: SystemOutput) -> float:
        try:
            obj = json.loads(output.text)
        except json.JSONDecodeError:
            return 0.0
        for key in self.schema.get("required", []):
            if key not in obj:
                return 0.0
        return 1.0


class LLMJudge:
    """Model-graded scoring.

    Three things make an LLM judge trustworthy enough to use:
      - a rubric with concrete criteria, not "is this good?"
      - a pinned judge model version (an upgrade silently shifts every score)
      - measured agreement with human graders, tracked over time
    Without the third, you have a number, not a measurement.
    """
    name = "llm_judge"

    RUBRIC = """Grade the ANSWER against the REFERENCE for the QUESTION.

Score 0-4:
4 - fully correct, complete, no unsupported claims
3 - correct, minor omissions
2 - partially correct, or correct with one unsupported claim
1 - mostly incorrect but on topic
0 - incorrect, or refuses when the reference contains the answer

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Reply with JSON only: {{"score": <0-4>, "reason": "<one sentence>"}}"""

    def __init__(self, judge_fn, judge_model: str) -> None:
        self.judge_fn = judge_fn
        self.judge_model = judge_model
        self.version = f"rubric1/{judge_model}"      # version includes the model

    async def score(self, case: EvalCase, output: SystemOutput) -> float:
        if not case.expected:
            return float("nan")
        prompt = self.RUBRIC.format(
            question=case.inputs.get("question", ""),
            reference=case.expected.get("answer", ""),
            answer=output.text,
        )
        try:
            raw = await self.judge_fn(prompt, model=self.judge_model, temperature=0.0)
            return json.loads(raw)["score"] / 4.0
        except Exception:                            # noqa: BLE001
            log.warning("judge failed on %s", case.id)
            return float("nan")                      # excluded, not counted as 0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

@dataclass
class EvalReport:
    results: list[CaseResult]
    provenance_: dict
    duration_s: float

    # ---- aggregate ----

    def summary(self) -> dict:
        ok = [r for r in self.results if r.error is None]
        out: dict[str, Any] = {
            "n": len(self.results),
            "n_scored": len(ok),
            "error_rate": (len(self.results) - len(ok)) / max(1, len(self.results)),
        }
        for name in self._scorer_names():
            values = self._values(ok, name)
            if values:
                lo, hi = wilson_interval(values)
                out[name] = round(float(np.mean(values)), 4)
                out[f"{name}_ci95"] = (round(lo, 4), round(hi, 4))
                out[f"{name}_n"] = len(values)
        lat = [r.latency_ms for r in ok]
        if lat:
            out["p50_latency_ms"] = round(float(np.percentile(lat, 50)), 1)
            out["p95_latency_ms"] = round(float(np.percentile(lat, 95)), 1)
        out["total_cost_usd"] = round(sum(r.cost_usd for r in self.results), 4)
        out["mean_input_tokens"] = int(np.mean([r.input_tokens for r in ok] or [0]))
        out["mean_output_tokens"] = int(np.mean([r.output_tokens for r in ok] or [0]))
        return out

    # ---- segmentation ----

    def by_segment(self, key: str) -> dict[str, dict]:
        """Per-segment scores. An aggregate number can hide a large regression
        in a segment that matters; this is the method that finds it."""
        groups: dict[str, list[CaseResult]] = {}
        for r in self.results:
            groups.setdefault(r.segments.get(key, "__unset__"), []).append(r)

        out = {}
        for value, rows in sorted(groups.items()):
            ok = [r for r in rows if r.error is None]
            entry: dict[str, Any] = {"n": len(rows)}
            for name in self._scorer_names():
                values = self._values(ok, name)
                if values:
                    lo, hi = wilson_interval(values)
                    entry[name] = round(float(np.mean(values)), 4)
                    entry[f"{name}_ci95"] = (round(lo, 4), round(hi, 4))
            out[value] = entry
        return out

    def worst(self, scorer: str, n: int = 20) -> list[CaseResult]:
        """The cases to actually read. Aggregate scores tell you whether to
        investigate; these tell you what to fix."""
        scored = [r for r in self.results
                  if r.error is None and not _isnan(r.scores.get(scorer))]
        return sorted(scored, key=lambda r: r.scores[scorer])[:n]

    def failures(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            if r.error_kind:
                counts[r.error_kind] = counts.get(r.error_kind, 0) + 1
        return counts

    def provenance(self) -> dict:
        return dict(self.provenance_)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "provenance": self.provenance_,
            "duration_s": self.duration_s,
            "summary": self.summary(),
            "results": [asdict(r) for r in self.results],
        }, indent=2))

    # ---- internals ----

    def _scorer_names(self) -> list[str]:
        names: list[str] = []
        for r in self.results:
            for k in r.scores:
                if k not in names:
                    names.append(k)
        return names

    @staticmethod
    def _values(rows: list[CaseResult], name: str) -> list[float]:
        return [r.scores[name] for r in rows
                if name in r.scores and not _isnan(r.scores[name])]


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def wilson_interval(values: Sequence[float], z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion, via the Wilson score interval.

    Wilson rather than the normal approximation because it behaves correctly
    near 0 and 1 and at small n — exactly where eval sets live. Values that are
    not 0/1 are treated as a mean with a normal interval.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if all(v in (0.0, 1.0) for v in values):
        p = sum(values) / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))
    mean = statistics.fmean(values)
    if n < 2:
        return (mean, mean)
    se = statistics.stdev(values) / math.sqrt(n)
    return (mean - z * se, mean + z * se)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

class EvalHarness:
    def __init__(self, *, concurrency: int = 10, timeout_s: float = 120.0,
                 checkpoint_path: str | Path | None = None) -> None:
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    async def run(
        self,
        dataset: list[EvalCase],
        system: SystemFn,
        scorers: list[Scorer],
        *,
        system_metadata: dict | None = None,
    ) -> EvalReport:
        started = time.monotonic()
        done = self._load_checkpoint()
        pending = [c for c in dataset if c.id not in done]
        log.info("eval: %d cases, %d already complete", len(dataset), len(done))

        sem = asyncio.Semaphore(self.concurrency)
        results: list[CaseResult] = list(done.values())

        async def one(case: EvalCase) -> CaseResult:
            async with sem:
                return await self._run_case(case, system, scorers)

        # as_completed so a long run reports progress and checkpoints
        # incrementally rather than only at the end.
        for coro in asyncio.as_completed([one(c) for c in pending]):
            result = await coro
            results.append(result)
            self._checkpoint(result)
            if len(results) % 50 == 0:
                log.info("eval progress %d/%d", len(results), len(dataset))

        provenance = {
            "timestamp": time.time(),
            "dataset_hash": dataset_hash(dataset),
            "dataset_size": len(dataset),
            "scorers": {s.name: s.version for s in scorers},
            "harness_version": "1.0",
            **(system_metadata or {}),
        }
        # Preserve dataset order for reproducible reports.
        order = {c.id: i for i, c in enumerate(dataset)}
        results.sort(key=lambda r: order.get(r.case_id, 1 << 30))
        return EvalReport(results, provenance, time.monotonic() - started)

    async def _run_case(self, case, system, scorers) -> CaseResult:
        started = time.monotonic()
        try:
            output = await asyncio.wait_for(system(case), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return CaseResult(case.id, None, {}, (time.monotonic() - started) * 1000,
                              0, 0, 0.0, case.segments,
                              error="timeout", error_kind="timeout")
        except Exception as exc:                     # noqa: BLE001
            return CaseResult(case.id, None, {}, (time.monotonic() - started) * 1000,
                              0, 0, 0.0, case.segments,
                              error=str(exc), error_kind=type(exc).__name__)

        latency = (time.monotonic() - started) * 1000

        scores: dict[str, float] = {}
        for scorer in scorers:
            try:
                scores[scorer.name] = await scorer.score(case, output)
            except Exception as exc:                 # noqa: BLE001
                log.warning("scorer %s failed on %s: %s", scorer.name, case.id, exc)
                scores[scorer.name] = float("nan")   # excluded from means

        return CaseResult(
            case.id, output.text, scores, latency,
            output.input_tokens, output.output_tokens, output.cost_usd,
            case.segments,
        )

    # ---- checkpointing ----

    def _load_checkpoint(self) -> dict[str, CaseResult]:
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return {}
        done: dict[str, CaseResult] = {}
        for line in self.checkpoint_path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                done[data["case_id"]] = CaseResult(**data)
        return done

    def _checkpoint(self, result: CaseResult) -> None:
        if self.checkpoint_path:
            with self.checkpoint_path.open("a") as fh:
                fh.write(json.dumps(asdict(result)) + "\n")


def dataset_hash(dataset: list[EvalCase]) -> str:
    """Content hash of the dataset.

    Two runs with the same hash are comparable; two with different hashes are
    not, and the report must make that obvious. This one line prevents a whole
    class of false conclusions.
    """
    h = hashlib.sha256()
    for case in sorted(dataset, key=lambda c: c.id):
        h.update(case.id.encode())
        h.update(case.fingerprint().encode())
    return h.hexdigest()[:16]
```

</details>

---

<details>
<summary>📖 Reveal the explanation</summary>

### The decisions

**Provenance first.** `dataset_hash`, scorer versions (including the judge model in the
judge's version string), and system metadata are recorded with every run. This is what
makes a score from three months ago interpretable, and it's what prevents the most common
false conclusion in evaluation: comparing two numbers produced by different
configurations. The rapid-fire incident "eval scores jumped 9 points with no change" in
`../12-senior-scenarios/production-incidents.md` is exactly this failure.

**Confidence intervals in the output format, not as an option.** `summary()` always
reports `_ci95` alongside every score. A team that sees `0.81 (0.77–0.85)` next to
`0.79 (0.75–0.83)` cannot claim an improvement; a team that sees `0.81` and `0.79` will.
The format enforces the honesty.

**Wilson intervals rather than the normal approximation.** For binary scores near 0 or 1,
or at small n, the normal approximation produces intervals that extend beyond [0, 1] and
are badly calibrated. Eval sets are usually small and scores are often high, which is
precisely where the normal approximation is worst.

**NaN for "not scored," never 0.** A judge that fails to respond has not scored 0 — it has
not scored. Coercing it to 0 silently penalizes the system for the scorer's failure and
is one of the more insidious ways an eval suite lies. NaN values are excluded from means
and the scored-n is reported separately so the exclusion is visible.

**Failures are results.** A timeout produces a `CaseResult` with `error_kind="timeout"`,
not an exception that kills the run. The error rate is itself a quality signal, and
`failures()` gives the breakdown.

**Segmentation is a first-class method.** `by_segment` with per-segment intervals is how
you catch the change that improves the aggregate and destroys a segment. Any harness
whose primary output is a single number will eventually ship that regression.

**`worst()` is the most-used method in practice.** Aggregate scores tell you *whether* to
investigate; reading the twenty worst cases tells you *what to fix*. Most of the value of
an eval suite is realized by a human reading failures, and the harness should make that
easy.

**Checkpointing as append-only JSONL.** Crash-safe, trivially resumable, and inspectable
with `wc -l` while the run is in progress. A 1,000-case run against a slow system can take
an hour; restarting from zero after a transient failure is how eval suites stop being run.

**Determinism is engineered, not assumed.** Temperature 0 for the judge, pinned model
versions in provenance, dataset sorted before hashing, results sorted into dataset order
before reporting. Two runs of an unchanged system should produce the same report.

### Complexity

- Time: `O(n / concurrency × per_case_latency)`.
- Memory: `O(n)` — all results are held to compute the report. For very large datasets,
  stream to disk and aggregate incrementally.
- Cost: `n × (system_cost + judge_cost)`. Judge cost is frequently *larger* than system
  cost, which is a real budget consideration and an argument for cheap deterministic
  scorers wherever they suffice.

</details>

---

## Production Improvements

- **A results registry.** Post every run's summary and provenance to a central store so
  quality is trendable across time and comparable across teams (see
  `../12-senior-scenarios/build-vs-buy.md#3-the-evaluation-platform`).
- **CI integration** that blocks on regression beyond a configured threshold — with the
  threshold set from the confidence interval, not from intuition.
- **A held-out set** the team cannot tune against, to detect benchmark overfitting.
- **Judge/human agreement tracking.** Sample 50 judged cases per month, grade them by
  hand, and report agreement. A judge whose agreement drifts is silently changing your
  quality bar.
- **Cost and latency budgets as pass/fail criteria**, not just reported numbers.
- **Dataset refresh from production traffic** on a schedule, with a frozen slice retained
  for historical comparability.
- **Pairwise comparison mode** (Problem 2) for A/B decisions.
- **Failure clustering** — group the worst cases by embedding similarity so a human reads
  patterns rather than 200 individual cases.

## Edge Cases

| Case | Correct behavior |
|---|---|
| Judge model fails on 5% of cases | NaN, excluded from means, `n_scored` shows the reduction |
| A scorer raises | Logged, NaN, run continues |
| System times out | `error_kind="timeout"`, counted in error rate, excluded from quality means |
| Dataset changed between runs | Different `dataset_hash` — reports must not be compared |
| All cases in a segment error | Segment reports `n` with no scores; don't report 0.0 |
| Duplicate case IDs | Checkpoint dedupes by ID; the dataset should be validated at load |
| n = 1 | Interval collapses to the point estimate; don't pretend otherwise |
| Scores outside [0, 1] | Falls back to a normal interval; document the scorer's range |
| Resumed run with a changed dataset | Stale checkpoint entries — the checkpoint file should record the dataset hash and refuse to resume across a change |

## Follow-up Questions

1. **How do you evaluate RAG?** (Problem 3.)
2. **How do you evaluate agents?** (Problem 4.)
3. **How do you detect regressions** without blocking every PR for an hour?
4. **How do you prevent benchmark overfitting?**
5. **When should humans be involved,** and how do you make that affordable?
6. **How do you compare two models statistically?** (Problem 2.)
7. **Your judge model is upgraded by the provider.** What happens to your history?

## Senior-Level Discussion

**An eval suite is a measurement instrument and it needs its own quality metrics.** The
useful ones: agreement with human grading, agreement with online outcomes, coverage of
the production distribution, and stability across repeated runs of an unchanged system.
A team that never measures these is trusting a number nobody has validated.

**The eval set must be a sample of production, not a curation of imagination.** The
single most common failure — documented in
`../12-senior-scenarios/production-incidents.md#incident-6` — is an eval set built from
clean, curated examples that diverges from real traffic. Sample from production, stratify,
refresh, and *measure the distributional match* rather than assuming it.

**Offline eval and online measurement do different jobs.** Offline eval's job is to
prevent catastrophes cheaply and quickly — it should be fast, deterministic, and run on
every change. Online measurement decides what's actually better, because it measures the
thing you care about. Conflating them leads to either shipping on noise or never shipping.

**LLM judges are useful and must be validated.** They correlate reasonably with human
judgment on many tasks, they're cheap, and they scale. They also have systematic biases:
a preference for longer answers, for confident phrasing, for their own model family's
style, and position bias in pairwise comparisons. Pin the judge version, measure
agreement with humans periodically, and randomize position in pairwise mode.

**Deterministic scorers beat model-graded ones wherever they apply.** Schema validity,
citation presence and validity, exact match on extractable fields, latency, cost, and
refusal rate are free, instant, and exactly reproducible. Reach for a judge only for the
genuinely subjective part, and note that judge cost often exceeds system cost on a large
suite.

---

## Problem 2: Is B Actually Better Than A?

## Scenario

Prompt A scores 0.79 on 200 cases. Prompt B scores 0.83. A PM asks whether to ship B.

## Requirements

Answer properly. Implement the comparison.

<details>
<summary>💡 Reveal a hint</summary>

Two errors to avoid:

1. **Treating 0.83 > 0.79 as an answer.** On 200 binary cases, the 95% CI for 0.79 is
   roughly ±0.057. The intervals overlap substantially. This difference is consistent
   with noise.

2. **Using an unpaired test when the data is paired.** Both systems were run on *the same
   cases*. That's paired data, and a paired test is far more powerful — it removes the
   variance from "some cases are harder than others," which is usually the dominant
   source of variance. The right tool is McNemar's test for binary outcomes, or a paired
   bootstrap for continuous scores.

The practical consequence: with paired analysis you can often detect a real 4-point
difference on 200 cases that an unpaired test would call inconclusive. Using the right
test is not pedantry; it's the difference between shipping an improvement and discarding
it.

And a third point worth making: **statistical significance is not practical
significance.** A change that is real, +0.5 points, and costs 40% more is not obviously
worth shipping.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""Statistical comparison of two systems on the same dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Comparison:
    n: int
    mean_a: float
    mean_b: float
    difference: float
    ci95: tuple[float, float]
    p_value: float
    significant: bool
    a_better: int          # cases where A won
    b_better: int          # cases where B won
    ties: int
    verdict: str


def compare_paired(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Paired comparison on the cases both systems scored.

    Paired, because the same cases were run through both systems. Pairing
    removes case difficulty as a source of variance, which is usually the
    largest one — an unpaired test on the same data has substantially less
    power and will call real differences inconclusive.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        raise ValueError("no overlapping cases")

    a = np.array([scores_a[k] for k in shared], dtype=float)
    b = np.array([scores_b[k] for k in shared], dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = len(a)
    if n == 0:
        raise ValueError("no cases scored by both systems")

    diffs = b - a
    observed = float(diffs.mean())

    # Bootstrap CI on the paired difference: assumption-free, works for binary
    # and continuous scores alike, and directly answers "how uncertain is this
    # difference?" rather than "is it different from zero?"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
    if binary:
        p = mcnemar_p(a, b)
    else:
        # Two-sided bootstrap p-value: the fraction of resamples on the
        # opposite side of zero.
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        p = float(min(1.0, p))

    a_better = int((a > b).sum())
    b_better = int((b > a).sum())
    ties = int((a == b).sum())
    significant = bool(p < alpha)

    if not significant:
        verdict = (f"No significant difference (p={p:.3f}). The observed "
                   f"{observed:+.3f} is consistent with noise at n={n}. "
                   f"To detect a difference this size reliably you would need "
                   f"about {required_n(observed, float(a.std() or 0.5)):,} cases.")
    elif observed > 0:
        verdict = (f"B is better by {observed:+.3f} "
                   f"(95% CI {lo:+.3f} to {hi:+.3f}, p={p:.4f}).")
    else:
        verdict = (f"A is better by {-observed:.3f} "
                   f"(95% CI {lo:+.3f} to {hi:+.3f}, p={p:.4f}).")

    return Comparison(n, float(a.mean()), float(b.mean()), observed,
                      (float(lo), float(hi)), float(p), significant,
                      a_better, b_better, ties, verdict)


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """McNemar's test for paired binary outcomes.

    Only the DISCORDANT pairs carry information: cases both systems got right,
    or both got wrong, tell you nothing about which is better. This is why the
    test is so much more powerful than comparing two proportions.
    """
    b_only = int(((a == 0) & (b == 1)).sum())     # B fixed it
    a_only = int(((a == 1) & (b == 0)).sum())     # B broke it
    n_disc = a_only + b_only
    if n_disc == 0:
        return 1.0
    if n_disc < 25:
        # Exact binomial for small discordant counts; the chi-square
        # approximation is unreliable here.
        from math import comb
        k = min(a_only, b_only)
        tail = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc)
        return float(min(1.0, 2 * tail))
    chi2 = (abs(b_only - a_only) - 1) ** 2 / n_disc      # continuity correction
    return float(math.erfc(math.sqrt(chi2 / 2)))


def required_n(effect: float, sd: float, power: float = 0.8,
               alpha: float = 0.05) -> int:
    """Rough sample size needed to detect an effect of this size.

    Useful for the conversation that matters: 'this eval set cannot answer the
    question you are asking.' Better to say that than to ship on noise.
    """
    if effect == 0:
        return 10 ** 9
    z_a, z_b = 1.96, 0.84                                 # alpha=.05, power=.8
    return int(math.ceil(((z_a + z_b) * sd / abs(effect)) ** 2))


def compare_by_segment(results_a, results_b, segment_key: str) -> dict:
    """Per-segment comparison.

    Necessary because an aggregate win can hide a segment regression — and
    because with many segments you are running many tests, so apply a
    correction or treat segment results as exploratory rather than confirmatory.
    """
    segments = {r.segments.get(segment_key) for r in results_a} | \
               {r.segments.get(segment_key) for r in results_b}
    out = {}
    for seg in sorted(s for s in segments if s):
        a = {r.case_id: r.scores.get("judge", float("nan"))
             for r in results_a if r.segments.get(segment_key) == seg}
        b = {r.case_id: r.scores.get("judge", float("nan"))
             for r in results_b if r.segments.get(segment_key) == seg}
        try:
            out[seg] = compare_paired(a, b)
        except ValueError:
            out[seg] = None
    return out
```

**Applied to the scenario:** 0.79 vs. 0.83 on 200 cases. If the systems agree on most
cases and the discordant pairs favour B 22–14, McNemar gives p ≈ 0.24 — **not
significant**. The honest answer to the PM is: *"the difference is within noise; we'd need
roughly 900 cases to detect a 4-point difference reliably. I can expand the eval set, or
we can ship it behind a canary and measure online."*

That answer — including a concrete path forward rather than just "we can't tell" — is
what makes statistical honesty useful rather than obstructive.
</details>

## Follow-up Questions

1. The difference is significant at p=0.03 but the effect is +0.4 points and B costs 40%
   more. Ship it?
2. You compare 12 prompt variants against a baseline. What's wrong with taking the best
   p-value?
3. Your eval set has 200 cases and you need to detect 2-point differences. What do you
   do?
4. A/B testing online takes 3 weeks to reach significance. How do you ship weekly?
5. B wins overall but loses on German-language cases. What now?

---

## Problem 3: Evaluating RAG

## Scenario

Evaluating a RAG system as a single number tells you it got worse and nothing about why.
Build evaluation that separates the stages.

<details>
<summary>💡 Reveal a hint</summary>

The decomposition that matters:

```
retrieval quality   — did the right documents come back?
groundedness        — is the answer supported by the retrieved context?
citation validity   — do the citations point at text that supports the claim?
answer correctness  — is the answer right?
```

These fail independently and the diagnosis is completely different:

- Retrieval bad, groundedness good → the retriever is the problem.
- Retrieval good, groundedness bad → the model is ignoring its context.
- Both good, answer wrong → your reference answer may be wrong, or the corpus lacks the
  information.
- Groundedness good, citations invalid → an attribution problem that destroys user trust
  independently of correctness.

A single "accuracy" number cannot distinguish any of these, which is why teams spend
weeks bisecting. The rapid-fire "citations point at the wrong documents" incident is a
case where groundedness is high and citation validity is 0.61 — invisible to any harness
that measures only the former.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""Stage-separated RAG evaluation."""

from __future__ import annotations

import numpy as np


class RetrievalRecall:
    """Did the retriever find the documents a human marked relevant?

    Measured at several k values because recall@5 and recall@50 answer
    different questions: recall@5 is what the model sees; recall@50 is the
    ceiling a reranker could reach.
    """
    name, version = "retrieval_recall", "1"

    def __init__(self, ks: tuple[int, ...] = (5, 10, 50)) -> None:
        self.ks = ks

    async def score(self, case, output) -> float:
        relevant = set(case.expected.get("relevant_doc_ids", []))
        if not relevant:
            return float("nan")
        retrieved = [d["id"] for d in output.metadata.get("retrieved", [])]
        return len(relevant & set(retrieved[:self.ks[0]])) / len(relevant)

    async def score_all_k(self, case, output) -> dict[str, float]:
        relevant = set(case.expected.get("relevant_doc_ids", []))
        retrieved = [d["id"] for d in output.metadata.get("retrieved", [])]
        return {
            f"recall@{k}": len(relevant & set(retrieved[:k])) / len(relevant)
            for k in self.ks
        } if relevant else {}


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """nDCG rewards putting the most relevant documents first, which matters
    because context is budgeted: rank 1 gets read, rank 20 gets truncated."""
    gains = [relevance.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


class Groundedness:
    """Is every claim in the answer supported by the retrieved context?

    Crucially INDEPENDENT of whether the answer is correct. A system can be
    perfectly grounded and completely wrong — that is the signature of a
    retrieval failure, and separating the two is the whole point.
    """
    name = "groundedness"

    RUBRIC = """Determine whether every factual claim in the ANSWER is
supported by the CONTEXT. Ignore whether the answer is correct — only whether
it is supported by what is provided.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only:
{{"supported_claims": <int>, "total_claims": <int>,
  "unsupported": ["<claim>", ...]}}"""

    def __init__(self, judge_fn, judge_model: str) -> None:
        self.judge_fn = judge_fn
        self.version = f"rubric1/{judge_model}"
        self.judge_model = judge_model

    async def score(self, case, output) -> float:
        context = "\n\n".join(
            d["text"] for d in output.metadata.get("retrieved", [])
        )
        if not context:
            return float("nan")
        raw = await self.judge_fn(
            self.RUBRIC.format(context=context, answer=output.text),
            model=self.judge_model, temperature=0.0,
        )
        try:
            data = json.loads(raw)
            total = data["total_claims"]
            return data["supported_claims"] / total if total else 1.0
        except Exception:                            # noqa: BLE001
            return float("nan")


class CitationValidity:
    """Do the cited chunks exist, and does the cited text actually appear?

    Deterministic and free. Measures something groundedness does not: an answer
    can be fully grounded while attributing claims to the wrong source, which
    users experience as the system being untrustworthy even when it is right.
    """
    name, version = "citation_validity", "1"

    async def score(self, case, output) -> float:
        cited = re.findall(r"\[([^\]]+)\]", output.text)
        if not cited:
            return float("nan")                      # nothing cited: not a score
        available = {d["id"] for d in output.metadata.get("retrieved", [])}
        valid = sum(1 for c in cited if c in available)
        return valid / len(cited)


class AbstentionCorrectness:
    """Does the system say 'I don't know' exactly when it should?

    Two failure modes with very different costs:
      - answering when it should abstain  -> hallucination
      - abstaining when it should answer  -> uselessness
    Report both; a single score hides the trade-off you are actually making.
    """
    name, version = "abstention", "1"

    ABSTAIN_MARKERS = ("don't have enough information", "cannot answer",
                       "no information", "unable to determine")

    async def score(self, case, output) -> float:
        should_abstain = bool(case.expected.get("should_abstain"))
        did_abstain = any(m in output.text.lower() for m in self.ABSTAIN_MARKERS)
        return 1.0 if should_abstain == did_abstain else 0.0


def diagnose(report) -> dict[str, str]:
    """Turn stage scores into a diagnosis.

    This is the output people actually use: not 'quality is 0.74' but
    'retrieval is the bottleneck, work there.'
    """
    s = report.summary()
    retrieval = s.get("retrieval_recall", float("nan"))
    grounded = s.get("groundedness", float("nan"))
    correct = s.get("llm_judge", float("nan"))
    citations = s.get("citation_validity", float("nan"))

    findings = {}
    if retrieval < 0.7:
        findings["retrieval"] = (
            "Recall is low: the correct documents are not being retrieved. "
            "Investigate chunking, embeddings, hybrid search, or reranking "
            "before touching prompts."
        )
    if grounded < 0.85 and retrieval >= 0.7:
        findings["generation"] = (
            "Retrieval is fine but answers are not grounded: the model is "
            "ignoring or contradicting its context. Investigate the prompt, "
            "context ordering, and context length."
        )
    if citations < 0.9:
        findings["attribution"] = (
            "Citations point at chunks that were not retrieved or do not "
            "support the claim. Users will lose trust independently of "
            "correctness."
        )
    if correct < 0.7 and grounded >= 0.85 and retrieval >= 0.7:
        findings["corpus_or_labels"] = (
            "Retrieval and grounding are healthy but answers are wrong. Either "
            "the corpus lacks the information or the reference answers are "
            "wrong. Read the failures."
        )
    return findings or {"ok": "No stage is clearly the bottleneck."}
```

</details>

## Follow-up Questions

1. Building `relevant_doc_ids` labels requires human annotation. How do you make that
   affordable at 500 cases?
2. Groundedness is 0.94 and correctness is 0.62. Diagnose.
3. How do you evaluate retrieval when several different documents could support a correct
   answer?
4. Your corpus changes weekly. How do you keep the labels valid?
5. How do you evaluate multi-turn RAG, where turn 3 depends on the system's own turn 2?

---

## Problem 4: Evaluating Agents

## Scenario

An agent takes 12 steps, calls 6 tools, and produces an answer. Only the final answer is
easily checkable, but most failures happen in the middle.

<details>
<summary>💡 Reveal a hint</summary>

Outcome-only evaluation is necessary and insufficient. An agent can reach the right answer
by luck after 11 wasted steps, or fail for a reason invisible in the final output.

Evaluate on four axes:

1. **Outcome** — right answer / correct end state.
2. **Efficiency** — steps, tokens, cost, wall time. An agent that takes 30 steps to do a
   3-step job is failing even when it succeeds.
3. **Trajectory** — did it call sensible tools in a sensible order? Compare against a
   reference trajectory, but *loosely*: many valid paths exist, and over-specifying the
   trajectory penalizes better solutions than the one you imagined.
4. **Safety** — did it stay within its permissions, avoid destructive actions, terminate
   correctly?

And the axis people forget: **termination behavior**. What fraction of runs end in
`final_answer` vs. `max_iterations` vs. `no_progress`? That distribution is a direct,
cheap quality signal that requires no labels at all.
</details>

<details>
<summary>✅ Reveal the implementation</summary>

```python
"""Multi-axis agent evaluation."""

from __future__ import annotations


class TaskSuccess:
    """Did the agent achieve the goal? Checked against END STATE where
    possible, not against the text of the answer — an agent that says it
    filed the ticket and did not is a failure that text matching misses."""
    name, version = "task_success", "1"

    def __init__(self, verify_fn) -> None:
        self.verify_fn = verify_fn       # inspects the world, not the transcript

    async def score(self, case, output) -> float:
        return 1.0 if await self.verify_fn(case, output) else 0.0


class StepEfficiency:
    """Steps used relative to a reference solution. Capped at 1.0 so being
    faster than the reference is not rewarded — it usually means it skipped
    verification."""
    name, version = "step_efficiency", "1"

    async def score(self, case, output) -> float:
        optimal = case.expected.get("optimal_steps")
        actual = output.metadata.get("iterations", 0)
        if not optimal or not actual:
            return float("nan")
        return min(1.0, optimal / actual)


class TrajectoryMatch:
    """Did it call the tools it needed to call?

    Deliberately measured as SET coverage of required tools plus a penalty for
    obviously wasted calls — not as an exact sequence match. Exact-sequence
    scoring penalizes agents that find a better path, which is precisely the
    behavior you want.
    """
    name, version = "trajectory", "2"

    async def score(self, case, output) -> float:
        required = set(case.expected.get("required_tools", []))
        forbidden = set(case.expected.get("forbidden_tools", []))
        called = [c["name"] for c in output.metadata.get("tool_calls", [])]
        called_set = set(called)

        if forbidden & called_set:
            return 0.0                    # a forbidden tool is a hard failure

        coverage = len(required & called_set) / len(required) if required else 1.0
        waste = (len(called) - len(called_set)) / max(1, len(called))  # repeats
        return max(0.0, coverage - 0.5 * waste)


class SafetyCompliance:
    """Hard constraints. Any violation is a zero — safety is not averaged."""
    name, version = "safety", "1"

    async def score(self, case, output) -> float:
        calls = output.metadata.get("tool_calls", [])
        for call in calls:
            if call.get("effect") == "destructive" and not call.get("approved"):
                return 0.0
            if call.get("error_kind") == "denied":
                return 0.0                # attempted something it may not do
        return 1.0


def termination_distribution(results) -> dict[str, float]:
    """The cheapest agent quality signal there is — no labels required.

    A healthy agent ends in final_answer most of the time. A rising
    no_progress rate means the tools or the prompt have degraded; a rising
    max_iterations rate means tasks are getting harder or the agent is looping.
    """
    counts: dict[str, int] = {}
    for r in results:
        reason = (r.output or {}).get("stop_reason", "unknown") \
            if isinstance(r.output, dict) else "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in sorted(counts.items())}


def agent_report(report) -> dict:
    """Agents are multi-objective. A single score is not just lossy here — it
    is actively misleading, because success and efficiency trade off against
    each other and the right balance is a product decision."""
    s = report.summary()
    return {
        "success_rate": s.get("task_success"),
        "success_ci95": s.get("task_success_ci95"),
        "safety_violations": 1 - (s.get("safety") or 1.0),
        "mean_steps": s.get("mean_steps"),
        "p95_steps": s.get("p95_steps"),
        "mean_cost_usd": s.get("total_cost_usd", 0) / max(1, s["n"]),
        "trajectory_score": s.get("trajectory"),
        "termination": termination_distribution(report.results),
    }
```

</details>

## Follow-up Questions

1. Your agent succeeds 82% of the time but uses 3× the optimal steps. Which do you
   optimize and how do you decide?
2. Two agents both succeed but take completely different paths. How do you score
   trajectory without punishing creativity?
3. How do you build a reproducible test environment for an agent that calls real systems?
4. `no_progress` terminations rose from 4% to 19% after a prompt change. What do you do?
5. How would you evaluate an agent whose task takes two days and involves human approval?

## Senior-Level Discussion

**Agents are multi-objective and collapsing them to one number destroys the information
you need.** Success rate, cost, steps, latency, and safety trade off against each other —
an agent that is 3 points more successful and twice as expensive may or may not be better,
and that's a product decision that requires seeing both numbers.

**Verify the end state, not the transcript.** An agent that reports "I've filed the
ticket" without filing it passes any text-based check. Evaluation for agents means
inspecting the world afterwards, which means a test environment you can reset and
inspect — the single largest practical obstacle to agent evaluation and the thing to
build first.

**Termination distribution is free quality signal.** It requires no labels, no judge, and
no reference answers, and it moves immediately when something breaks. Any team running
agents in production should have this on a dashboard before they have anything else.

**Safety is not averaged.** A 99% safety score means one run in a hundred did something it
was not permitted to do. For any agent with real capabilities, that's a failing grade, not
a good one. Report violations as counts, gate on zero, and treat any violation as an
incident rather than as a metric that moved.

**Trajectory scoring must not over-specify.** If you score against an exact reference
sequence, you penalize agents that find shorter or more robust paths — and you build a
suite that rewards imitating your reference solution rather than solving the problem. Set
coverage plus a waste penalty plus hard constraints on forbidden tools captures what you
actually care about.
