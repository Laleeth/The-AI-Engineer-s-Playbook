# Contributing

The most useful thing you can send is **"this scenario is unrealistic, here's why."**

That's the failure mode this material is most exposed to. Everything here was written to
match production reality, and a scenario that quietly doesn't is worse than no scenario —
someone will rehearse it and walk into an interview with it. It's also the hardest thing
to catch alone, which is why the repo is public before it's finished.

Close behind, in order of how much they help:

1. A production incident you've had that isn't represented here.
2. A number in the docs that doesn't match your experience — say what you measured, and
   on what hardware or traffic.
3. A real interview question these sections don't prepare you for.
4. A section you'd reorder, merge, or cut.

You don't need to open a PR for any of those. An issue with the details is worth more
than a pull request that fixes wording.

---

## Before you open a pull request

Run the verifier. CI runs it on every push and pull request anyway, so you'll find out
either way — finding out locally is faster:

```console
$ python scripts/verify.py

Verifying The AI Engineer's Playbook

PASS  475 python blocks parse
PASS  587 internal links resolve
PASS  numeric claims match the prose
PASS  the README describes the repo that exists

All checks passed.
```

No dependencies beyond the standard library. Python 3.11 or newer.

---

## If you're adding content

**Scenarios need real numbers and competing constraints.** If a question has one
memorized answer, it doesn't belong here. The test every scenario had to pass: could an
experienced AI engineer learn something from this?

**Follow the structure of the file you're adding to.** In `12-senior-scenarios/` that
means Situation, Your Task, Constraints, the reasoning path behind a spoiler, Strong
Answer, Trade-offs, What Could Go Wrong, Metrics, Follow-ups, What Separates Senior From
Staff, and Interviewer Notes. In `11-coding-rounds/` it means Scenario, Requirements,
Starter Code, Expected Behavior, Hint, Reference Implementation, Explanation, Complexity,
Production Improvements, Edge Cases, Follow-ups, and Senior-Level Discussion.

The trade-offs, failure modes and interviewer notes are usually worth more than the
answer itself. Don't skip them to save time.

**Run any code you add.** The reference implementations here weren't just written — they
were executed against the claims made about them, and doing that found a bug in one of
the explanations. `verify.py` checks that every Python block *parses*; it can't check
that your code does what you say it does. That part is on you.

**If you quote a number, add it to the claims check** in [`scripts/verify.py`](scripts/verify.py).
The point of that file is that a figure in the prose and the arithmetic behind it can't
drift apart without CI noticing. A number nobody re-computes is a number that quietly
goes wrong.

---

## House style

- **Numbers over adjectives.** "Expensive" is not information. "$140,000/month at 200M
  requests" is.
- **Plain language in the reference sections.** `00`–`08` explain terms on first use.
  `11`–`12` are written at interview register, because that's the register you'll be
  answering in.
- **Say when something is unknown.** Where a figure is genuinely model- or
  workload-dependent, write "measure it" rather than inventing a rule.
- **US spelling**, for consistency with the rest of the repo. The verifier doesn't check
  this; a reviewer will mention it.
- Wrap prose at roughly 95 characters. Tables and code are exempt.

---

## Writing one of the planned sections

`06`, `09` and `10` are unwritten and [sketched in the roadmap](ROADMAP.md). Open an issue
before starting one — the sketches are sketches, not commitments, and it's worth agreeing
the scope before you spend a weekend on it.

---

## Licensing

Contributions are accepted under the [MIT License](LICENSE), the same terms as the rest of
the repository.
