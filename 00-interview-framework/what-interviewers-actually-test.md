# What Interviewers Actually Test

Most candidates prepare for the wrong thing. They memorize definitions, then get asked
to debug a latency spike with incomplete information and freeze.

This file is about what's really being assessed, which is usually not what the question
appears to be about.

---

## The question behind the question

An interviewer asks: *"How would you build a system to answer questions over our
documentation?"*

They are not checking whether you know what RAG is. They're finding out:

- Do you ask about the data before designing?
- Do you know what breaks at scale?
- Can you name trade-offs, or do you present one design as obviously correct?
- Do you mention how you'd know it works?
- Do you notice the parts that aren't the fun parts — permissions, updates, parsing?

You can give a technically complete answer and still fail, if you gave it without asking
a single question.

---

## The five things being assessed

Nearly every AI engineering interview is testing some mix of these.

### 1. Do you reason from evidence, or from patterns?

The strongest signal available, and the cheapest to demonstrate.

**Pattern-matching sounds like:** "For RAG I'd use hybrid search, reranking, and a
frontier model."

**Reasoning sounds like:** "Before I pick anything — what do the documents look like, and
what do people actually ask? If it's product codes and error IDs, dense search alone will
fail and I'd want keyword search. If it's prose questions, maybe not."

The second answer might arrive at the same architecture. It's a much better answer,
because it shows the architecture came from somewhere.

### 2. Do you know what actually breaks?

Anyone can describe a happy path. The signal is in failure modes.

Things that separate people who've operated these systems:

- Retries amplify load exactly when a dependency is struggling
- Adding documents to a corpus can make existing answers worse
- An agent that loops is cheap; an agent that loops *with side effects* is an incident
- Upgrading an embedding model without rebuilding the index fails **silently**
- A cache key missing a tenant is a data breach, not a bug

You don't need all of them. Naming two or three unprompted changes how the rest of the
interview goes.

### 3. Can you hold a trade-off in your head?

Weak answers pick a side. Strong answers say what would change their mind.

> "I'd self-host, because at 200 million requests a month the API bill is $1.7M a year.
> But that assumes high GPU utilization — if traffic is spiky, the economics collapse and
> I'd stay on the API. What does the daily traffic curve look like?"

That's the shape. A position, the reasoning, the condition under which it flips, and a
question that would resolve it.

### 4. Do you measure?

An enormous number of candidates design systems with no way of knowing whether they work.

Saying "I'd build an eval set from production traffic, segmented by query type, and gate
releases on it" puts you ahead of most people, because most people don't mention
measurement at all.

The follow-on signal: do you know what your measurement *can't* tell you? "200 examples
catches 10-point regressions, not 2-point ones" is a senior sentence.

### 5. Do you know what you'd skip?

Every plan has a budget. Interviewers want to know you can prioritize.

> "I wouldn't build reranking in the first version. I'd ship the simple pipeline,
> measure retrieval recall, and add reranking when I can point at the queries it fixes."

Saying what you'd *not* do — and why — is one of the fastest ways to sound senior.

---

## What is not being tested

Worth knowing so you stop spending preparation time on it.

**Memorized definitions.** Nobody cares if you can recite how attention works, unless
it's load-bearing for the answer. (Knowing *why* KV cache size limits your concurrency —
that's load-bearing. See [../01-llm-internals/kv-cache.md](../01-llm-internals/kv-cache.md).)

**Current model names and prices.** These change monthly. Interviewers know that. What
they want is whether you can *do the arithmetic* once you have the numbers.

**Framework APIs.** "Which LangChain class does X" is not a senior question. If you get
one, it's a signal about the team.

**Getting the exact right answer.** Most good questions don't have one. The interviewer
is watching how you get there.

---

## What a strong answer sounds like

Same question, three answer levels. The difference isn't knowledge — it's structure.

> **Question:** "Our RAG system returns wrong answers about 15% of the time. How would you
> investigate?"

**Weak:**
> "I'd try a better model, improve the prompt, and maybe add reranking."

Jumps to solutions. No diagnosis. Every suggestion might be irrelevant.

**Decent:**
> "First I'd look at whether it's a retrieval or generation problem. If the right
> documents aren't being retrieved, that's chunking or search. If they are and the answer
> is still wrong, that's the prompt or the model."

Correct structure. This is a solid mid-level answer.

**Strong:**
> "The first thing I'd want is a sample of the failures with the retrieved documents
> logged alongside them — do we have that? Because the question I need to answer is
> whether the right information was in the prompt.
>
> If it wasn't, it's retrieval, and I'd look at chunking and search before anything else.
> If it *was* in the prompt and the answer is still wrong, that's generation.
>
> I'd also want to know whether 15% is new or whether it's always been like this. If it
> changed, something changed — documents added, an embedding model upgraded, a prompt
> edited. That's a much faster investigation.
>
> And I'd check what the 15% have in common. If they're all multi-turn, or all in one
> language, that's a different problem from a uniform 15%."

What makes it strong: asks for data, states the discriminating question, splits the
problem, considers whether it's a regression, and looks for structure in the failures.
Notice it doesn't propose a single fix.

---

## Thinking out loud, properly

You've been told to think out loud. Most people do it badly — narrating rather than
reasoning.

**Narrating (weak):**
> "So I'm thinking about the architecture... we need retrieval... probably a vector
> database... and then a model..."

**Reasoning (strong):**
> "Two things will decide this: how big the corpus is, and whether the data has
> identifiers in it. If it's under a million chunks I don't need a vector database at all
> — brute force in memory is exact and simpler. Let me assume it's bigger and come back
> to that.
>
> On identifiers — if people search by product code, dense retrieval alone will fail,
> because embeddings can't distinguish E-4471 from E-4472. Do you know what the queries
> look like?"

The second names the decisions, states what would resolve them, and asks. That's the
behavior being assessed.

---

## Questions worth asking

Asking good questions is scored. These work in almost any AI system design interview:

**About the data**
- What do the documents look like? Format, length, how often they change?
- Are there identifiers, codes, or names people search by?
- Can everyone see everything, or are there permissions?

**About the traffic**
- How many requests, and what's the peak-to-average ratio?
- What's the distribution of input and output length?
- What do people actually ask? Is it uniform or lopsided?

**About the constraints**
- What's the latency requirement — and is that time-to-first-token or total?
- What's the budget?
- Any data residency or compliance rules?

**About success**
- What does a wrong answer cost?
- How would we know it's working?
- What's the current baseline, if there is one?

That last group is the one most candidates skip and interviewers most notice.

---

## Red flags interviewers watch for

From the interviewer notes across this repo:

| Behavior | What it signals |
|---|---|
| Designing before asking anything | Won't gather requirements on the job |
| Naming a vendor as the answer | Reads marketing, not systems |
| No measurement mentioned | Ships things nobody can verify |
| Treating quality as one number | Will hide a regression in an average |
| "We'd use a bigger model" as the fix | Doesn't diagnose |
| No rollback story | Hasn't operated anything |
| Can't say what they'd skip | Can't prioritize |
| Defends the first idea against evidence | Hard to work with |

That last one is worth dwelling on. Interviewers often push back on a correct answer just
to see what you do. Updating when given new information is a *strength*, not a
concession.

---

## Green flags

| Behavior | What it signals |
|---|---|
| Asks about the data distribution, not the average | Has been surprised by a p99 |
| Names a failure mode unprompted | Has operated something |
| Does arithmetic out loud | Doesn't guess at capacity |
| Says "I'd measure X before deciding" | Won't ship on vibes |
| Names what they'd *not* build | Can prioritize |
| Says "that changes my answer" | Reasons from evidence |
| Asks what a wrong answer costs | Thinks about the business |

---

## How this repo maps to it

If you're preparing, the shortest path:

| To practice | Read |
|---|---|
| Structuring a design answer | [how-to-answer-system-design.md](how-to-answer-system-design.md) |
| Structuring a debugging answer | [how-to-answer-debugging.md](how-to-answer-debugging.md) |
| Knowing what breaks | [../12-senior-scenarios/production-incidents.md](../12-senior-scenarios/production-incidents.md) |
| Trade-off reasoning | [../12-senior-scenarios/architecture-tradeoffs.md](../12-senior-scenarios/architecture-tradeoffs.md) |
| Measurement | [../05-evaluation/](../05-evaluation/) |
| The arithmetic | [../01-llm-internals/prefill-vs-decode.md](../01-llm-internals/prefill-vs-decode.md) · [../02-model-selection/cost-quality-latency.md](../02-model-selection/cost-quality-latency.md) |

Every scenario in section 12 ends with **Interviewer Notes** saying what's being
assessed. Reading those is the fastest way to calibrate.

---

## What to remember

- The question is rarely about the topic it names.
- Ask about the data before you design anything.
- Name failure modes. It's the clearest signal you've operated something.
- State trade-offs with the condition that would flip them.
- Say how you'd measure it, and what your measurement can't detect.
- Say what you'd skip.
- Updating your answer when given new information is a strength.

---

**Next:** [junior-vs-mid-vs-senior.md](junior-vs-mid-vs-senior.md) — the same question,
answered at three levels.
