# 00 — Interview Framework

How AI engineering interviews actually work, and how to answer them. Read this before the
technical sections if you're preparing for interviews — it changes what you get out of
everything else.

Four files. Plain language, no filler.

## The files

| File | What it covers |
|---|---|
| [what-interviewers-actually-test.md](what-interviewers-actually-test.md) | The question behind the question, and the five things being assessed |
| [junior-vs-mid-vs-senior.md](junior-vs-mid-vs-senior.md) | The same question answered at four levels, and what separates them |
| [how-to-answer-system-design.md](how-to-answer-system-design.md) | A structure for "design a system that…" |
| [how-to-answer-debugging.md](how-to-answer-debugging.md) | A different structure, for "it's broken, find out why" |

## The short version

**The question is rarely about the topic it names.** "How would you build search over our
docs?" is not checking whether you know what RAG is. It's checking whether you ask about
the data before designing, whether you know what breaks, and whether you mention how you'd
know it worked.

**Design and debugging reward opposite behaviors.** Design rewards breadth — cover the
components, name the trade-offs. Debugging rewards *narrowing* — every question should
eliminate possibilities. Breadth in a debugging question reads as flailing.

**The one debugging rule:** state a hypothesis, say what would disprove it, ask for exactly
that data. Everything else is detail.

**The mid → senior jump is not about knowing more.** It's about letting the situation pick
the design instead of applying a known-good one. Three habits mark it: asking before
designing, doing arithmetic out loud, and naming specific failure modes.

**Three things most candidates never do**, and all three are scored:

- Say how they'd measure whether it works
- Say what they'd *skip* and why
- Say what would change their mind

**Updating your answer when given new information is a strength.** Interviewers often push
back on a correct answer just to see what you do. Defending your first idea against
evidence is the worst signal available.

## A useful self-check

After answering any scenario in this repo:

- [ ] Did I ask anything before designing?
- [ ] Did I compute anything?
- [ ] Did I name a failure mode with its mechanism?
- [ ] Did I say how I'd know it worked?
- [ ] Did I say what I'd skip?
- [ ] Did I name what would change my mind?

Most people are strong on technical content and missing three or four of these. That gap
is usually what's holding the level back — not knowledge.

## Related

- [12-senior-scenarios](../12-senior-scenarios/) — every scenario ends with **Interviewer
  Notes** saying what's being assessed. The fastest way to calibrate.
- [11-coding-rounds](../11-coding-rounds/) — the same, for implementation problems.
- [01-llm-internals](../01-llm-internals/) — the arithmetic that makes "do the numbers out
  loud" possible.
