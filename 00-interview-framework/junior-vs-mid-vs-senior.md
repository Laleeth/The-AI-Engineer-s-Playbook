# Junior, Mid, Senior, Staff

The same question gets asked at every level. What changes is the answer.

Understanding the difference matters for two reasons: you can aim your answers at the
level you're interviewing for, and you can stop over-preparing on things that don't move
you up a band.

---

## The short version

| Level | Core question | What they're trusted with |
|---|---|---|
| **Junior** | Can you build it? | A well-defined task |
| **Mid** | Can you build it *well*? | A feature, end to end |
| **Senior** | Can you decide *what* to build, and operate it? | A system, and its failures |
| **Staff** | Can you decide *what the org should build*? | A direction, across teams |

The jump people find hardest is mid → senior, and it isn't about knowing more. It's about
shifting from *solving the problem given* to *working out which problem to solve*.

---

## The same question, four answers

> **"We want to add AI-powered search to our documentation site. How would you build it?"**

### Junior

> "I'd chunk the documents, embed them with an embedding model, store the vectors in a
> vector database, and then for each query embed it, find the nearest chunks, and put them
> in a prompt for the model to answer from."

**This is correct.** It's a working design and it demonstrates real knowledge.

What's missing: no questions asked, no failure modes, no measurement, no cost. It's the
textbook answer applied without inspection.

### Mid

> "I'd start with the standard pipeline — chunk, embed, retrieve, generate. A few things
> I'd think about:
>
> Chunking depends on the docs. If they're markdown with headings I'd split on those
> rather than fixed character counts, and I'd keep 10–15% overlap so answers don't fall
> between chunks.
>
> I'd add reranking, because raw vector search ordering isn't reliable — retrieve 50,
> rerank, send 5.
>
> And I'd make sure the prompt tells the model to only use the provided context and to say
> when it doesn't know, otherwise it'll make things up when retrieval fails."

**Good.** Knows the components, knows why each is there, anticipates a failure.

What's missing: still no questions about the actual situation. No numbers. No mention of
how you'd know it works. No cost. Assumes the standard design fits.

### Senior

> "Before designing, a few things I'd want to know:
>
> What do the docs look like — are there code samples, API references, error codes? That
> changes retrieval a lot. If people search for `E-4471`, dense retrieval alone will fail,
> because embeddings put `E-4471` and `E-4472` in nearly the same place. That's a
> hybrid-search requirement, not a nice-to-have.
>
> How big is the corpus? Under a million chunks I wouldn't use a vector database at all —
> brute force in memory is exact, faster, and one less thing to run. People reach for
> infrastructure before doing that arithmetic.
>
> What's the latency requirement, and does the UI stream? If it streams, the number that
> matters is time-to-first-token, and a lot of apparent latency problems disappear.
>
> Assuming a normal docs site: chunk on headings, prepend the document title and section
> to each chunk before embedding — that's an hour of work and it fixes a whole class of
> 'the chunk had no context so nobody could find it' failures. Hybrid search if there are
> identifiers. Reranking, which cuts context tokens by about 75% and makes it cheaper as
> well as better.
>
> The part I'd build first, though, is evaluation. A couple of hundred real questions with
> the documents that should come back. Without it I can't tell whether any of the above
> helped, and I can't tell a retrieval failure from a generation failure — which have
> completely different fixes.
>
> What I'd skip in v1: query rewriting, multimodal handling, anything clever. Ship the
> simple version, measure, then add the piece that fixes a measured problem."

**What changed.** Asks first. Ties design decisions to properties of the data. Does
arithmetic. Names a specific failure mode with a mechanism. Puts measurement first. Says
what to skip.

### Staff

Everything above, plus a different frame:

> "One thing before the design — is this the only search system we're building? Because if
> three other teams are going to want the same thing in six months, I'd rather find that
> out now.
>
> The expensive, hard-to-reverse decisions here are the embedding model and its
> dimensions, because changing them later means re-embedding everything and holding two
> indexes at once. If we don't design for that from the start, we can't migrate without
> downtime, so we won't, and we'll be stuck on whatever we picked today.
>
> I'd also want to know whether the docs have access controls. If they do, that's a
> retrieval-layer requirement and it's much harder to add later than to build in — and
> it's the difference between a bug and a data breach.
>
> On sequencing: I'd want the evaluation set to be something other teams can reuse, and
> I'd write down what we're *not* standardising so nobody thinks this is a platform
> mandate. Otherwise every team builds their own and in eighteen months we have eight of
> these with different embedding models and nobody can say what any of them cost."

**What changed again.** Thinks past this project. Identifies which decisions are
irreversible. Anticipates organisational consequences. Considers reuse without
overreaching into a platform mandate.

---

## What actually separates the levels

### Junior → Mid: knowing why

Junior knows *what* the components are. Mid knows *why* each one is there and what
happens without it.

To move: for every component you'd use, be able to say what breaks if you leave it out.
Why overlap in chunking? Why reranking? Why an abstention instruction?

### Mid → Senior: the situation decides the design

This is the big one, and it's a change in posture rather than knowledge.

Mid applies a known-good design. Senior works out which design this situation needs, by
asking about the data, the traffic, and the constraints — then justifies each choice
against what they learned.

Three habits that mark the transition:

**Ask before designing.** Not one clarifying question, but the two or three whose answers
would change your architecture.

**Do arithmetic.** Corpus size, storage, tokens, cost, throughput. Being willing to
compute in front of someone is a strong signal, even approximately.

**Name failure modes.** Not "it might not work well" but "adding documents can outrank
existing correct answers, so I'd run the eval suite on corpus changes."

### Senior → Staff: the problem isn't the one you were given

Senior solves the problem well. Staff asks whether it's the right problem, and considers
the consequences past this project.

- Which decisions here are irreversible, and can we defer them?
- Who else will need this, and what happens if they each build their own?
- What are we *not* going to do, and can I defend that?
- What does this look like in two years, and who maintains it?

Staff answers frequently include a refusal — something they'd decline to build, with a
reason a non-engineer would accept.

---

## Calibrating your answer

If you're interviewing for senior, you can be *too* junior in your answer even knowing
everything. The fix is structural.

**Aim your answer at the level:**

| If you're interviewing for | Make sure you |
|---|---|
| Mid | Explain why each component exists; name what breaks without it |
| Senior | Ask 2–3 questions first; do arithmetic out loud; name failure modes; say what you'd skip |
| Staff | All of the above, plus: which decisions are irreversible, who else is affected, and what you'd refuse |

**A trap worth knowing:** answering above your level can hurt too. If you're interviewing
for mid and you spend the whole time on organisational strategy without demonstrating you
can build the thing, that reads as avoidance. Show you can do the work, *then* show the
judgement.

---

## What doesn't move you up

Things candidates over-invest in:

**More technique names.** Knowing five reranking approaches doesn't move you from mid to
senior. Knowing when *not* to add reranking does.

**Deeper theory.** Being able to derive attention doesn't make you senior. Knowing that
KV cache size limits concurrency, which limits your GPU count, which decides your cost —
that does, because it's load-bearing.

**Newer tools.** The senior signal is being able to say why you *wouldn't* adopt something
yet.

**Longer answers.** Senior answers are often shorter, because they're targeted. A
five-minute answer that starts with two questions beats a fifteen-minute tour.

---

## Self-check

Read a scenario in [section 12](../12-senior-scenarios/), answer it, then check yourself:

- [ ] Did I ask anything before designing? *(mid → senior)*
- [ ] Did I compute anything? *(mid → senior)*
- [ ] Did I name a specific failure mode with its mechanism? *(mid → senior)*
- [ ] Did I say how I'd know it worked? *(mid → senior)*
- [ ] Did I say what I'd skip? *(senior)*
- [ ] Did I name what would change my mind? *(senior)*
- [ ] Did I identify which decision is hardest to reverse? *(senior → staff)*
- [ ] Did I consider who else is affected? *(senior → staff)*
- [ ] Did I refuse something, with a reason? *(staff)*

Most people find they're strong on the technical content and missing three or four of
these. That gap is usually what's holding the level back — not knowledge.

---

## What to remember

- Junior knows the components. Mid knows why they're there.
- Senior lets the situation pick the design — which means asking first.
- Staff asks whether it's the right problem, and what it costs the organisation.
- Arithmetic, failure modes, and measurement are the three habits that read as senior.
- Saying what you'd skip is one of the fastest level signals available.
- More technique names won't move you up. Knowing when not to use them will.

---

**Next:** [how-to-answer-system-design.md](how-to-answer-system-design.md) — a structure
for open-ended design questions.
