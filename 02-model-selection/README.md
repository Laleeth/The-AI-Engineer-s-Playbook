# 02 — Model Selection

Choosing which model to use, and how to use it without overpaying. Plain language, terms
explained on first use.

The running idea: **there is rarely one right model.** Almost every good answer here is
"different models for different requests," and the interesting work is in deciding which
request goes where.

## The files

| File | What it covers |
|---|---|
| [api-vs-open-model.md](api-vs-open-model.md) | Call someone else's model, or run your own |
| [model-routing.md](model-routing.md) | Sending different requests to different models |
| [small-vs-large-model.md](small-vs-large-model.md) | When a small model is genuinely enough |
| [fine-tuning-vs-rag.md](fine-tuning-vs-rag.md) | Teaching the model versus giving it the information |
| [cost-quality-latency.md](cost-quality-latency.md) | The three-way trade-off, and which lever moves what |

## Read in this order

Start with [cost-quality-latency.md](cost-quality-latency.md) if you're trying to fix a
specific problem — it tells you which lever to pull.

Start with [api-vs-open-model.md](api-vs-open-model.md) if you're making an architecture
decision from scratch.

[model-routing.md](model-routing.md) is the one most teams get value from fastest.

## The short version

**Input tokens drive cost. Output tokens drive latency.** Most requests send far more
input than they produce output, so the bill is usually an input problem. Generation is
sequential, so slowness is usually an output problem. Optimising the wrong one is the
most common wasted week.

**Do the arithmetic before the argument.** Self-hosting versus API is a calculation, not
a preference. Below roughly $30–50k/month of API spend it rarely pays — and two engineers
cost $50k/month before you rent a single GPU.

**Most traffic is easy.** 70–90% of requests in a typical product are handled fine by a
small model. Paying frontier prices for all of it is the waste that routing fixes.

**Use hard rules where the stakes are high.** Never let a classifier decide that an
account-security question or a legal threat goes to the cheap model. Statistics for the
rest, rules for the ones that matter.

**Model self-reported confidence doesn't work.** "Rate your confidence 1–10" produces
numbers that don't track correctness — and it's worst exactly when the model is
confidently wrong. Build an explicit "I'm not sure" branch into the output schema
instead.

**Fine-tuning teaches *how*, RAG teaches *what*.** Fine-tuning on your documentation
usually fails: facts don't stick reliably, you can't update it, there are no citations,
and there are no per-user permissions. That last one alone rules it out for most
enterprise document sets.

**Prompt caching and reranking are free wins.** Both cut cost with no quality risk —
reranking usually *improves* quality, because you removed distracting context. Do these
before anything involving a smaller model.

**"Use a smaller model" is the last resort, not the first.** It's the riskiest lever and
the one people reach for first.

## A number worth remembering

Cutting from 20 retrieved chunks to 5 (via reranking) cuts your retrieved-context tokens
by 75%. Since input usually dominates the bill, that's often a bigger saving than
switching models — with quality going up rather than down.

## Related

- [03-rag/reranking.md](../03-rag/reranking.md) — the cheapest cost cut in most RAG
  systems
- [12-senior-scenarios/cost-vs-quality.md](../12-senior-scenarios/cost-vs-quality.md) —
  these decisions as interview scenarios, with numbers
- [12-senior-scenarios/model-selection.md](../12-senior-scenarios/model-selection.md) —
  harder versions, with the constraints changing round by round
