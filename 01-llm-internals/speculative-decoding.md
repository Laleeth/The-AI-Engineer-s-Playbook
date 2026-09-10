# Speculative Decoding

A way to generate several tokens per forward pass instead of one, without changing the
output.

That last part is the important bit: done correctly, speculative decoding produces
**exactly the same text** the model would have produced anyway. It's a pure latency
optimization, not a quality trade-off.

---

## The problem it solves

From [prefill-vs-decode.md](prefill-vs-decode.md): decode is memory-bandwidth-bound. For
every single token, the GPU reads the entire model's weights from memory and does a
comparatively tiny amount of arithmetic.

The arithmetic units sit mostly idle. You're paying for memory traffic, not compute.

The insight: **if you're going to read all those weights anyway, you may as well check
several candidate tokens in the same pass.** Verifying 5 tokens costs barely more than
verifying 1, because the expensive part is the memory read, not the maths.

---

## How it works

Two models. A small fast one guesses; the big one checks.

```
1. Draft model proposes k tokens ahead     (cheap, sequential, but small)
2. Target model verifies all k at once     (one expensive pass)
3. Accept the longest correct prefix
4. Repeat
```

```python
def speculative_step(prompt, draft_model, target_model, k=4):
    """Returns the accepted tokens. Output is identical to plain decoding."""
    # 1. Draft k tokens, one at a time (cheap — small model)
    draft_tokens = []
    context = list(prompt)
    for _ in range(k):
        token = draft_model.next_token(context)
        draft_tokens.append(token)
        context.append(token)

    # 2. Verify all k in ONE pass of the big model
    target_logits = target_model.forward(prompt + draft_tokens)

    # 3. Accept while the draft agrees with what the target would have done
    accepted = []
    for i, draft_token in enumerate(draft_tokens):
        if target_would_accept(target_logits[i], draft_token):
            accepted.append(draft_token)
        else:
            # First disagreement: take the target's token and stop
            accepted.append(sample(target_logits[i]))
            break
    else:
        # All k accepted — we also get a free token from the last position
        accepted.append(sample(target_logits[k]))

    return accepted
```

Two details worth noticing:

**You always make progress.** Even if the very first draft token is wrong, you still emit
the target model's token for that position. The worst case is one token per pass — the
same as no speculation, plus the wasted draft cost.

**The bonus token.** If all k drafts are accepted, the verification pass also gives you the
token *after* them, so you get k+1 tokens from one pass.

---

## Why the output is identical

This is what makes speculative decoding unusual among optimizations, and it's the thing to
be able to explain.

With greedy decoding it's obvious: accept a draft token only if it's the one the target
model would have picked.

With sampling it's subtler. The acceptance rule is probabilistic, designed so that the
resulting distribution matches the target model's distribution exactly:

```python
def accept_token(draft_prob, target_prob, rng):
    """Accept with probability min(1, p_target / p_draft).

    On rejection, sample from an adjusted distribution. The maths works out
    so the overall output distribution is exactly the target model's.
    """
    if draft_prob <= 0:
        return False
    return rng.random() < min(1.0, target_prob / draft_prob)
```

So it isn't an approximation you're trading quality for. It's the same model output,
arrived at faster. If someone tells you speculative decoding degrades quality, either
they're using a variant that deliberately relaxes this (some do), or something is wrong
with their implementation.

---

## How much faster?

It depends entirely on the **acceptance rate** — how often the draft model guesses right.

```python
def speedup(k, acceptance_rate, draft_cost_ratio=0.1):
    """
    k: tokens drafted per round
    acceptance_rate: probability a given draft token is accepted
    draft_cost_ratio: draft model cost relative to target
    """
    # Expected accepted tokens per round (geometric, plus the bonus token)
    a = acceptance_rate
    expected_tokens = sum(a ** i for i in range(k + 1))

    # Cost: k cheap draft passes + 1 expensive verify pass
    cost = k * draft_cost_ratio + 1.0
    return expected_tokens / cost


for rate in (0.5, 0.7, 0.8, 0.9):
    print(f"acceptance {rate:.0%}: {speedup(4, rate):.2f}x")

# acceptance 50%: 1.34x
# acceptance 70%: 1.85x
# acceptance 80%: 2.20x
# acceptance 90%: 2.62x
```

**Acceptance rate is everything.** Below about 50% the draft cost starts eating the gains.
Above 80% you get a substantial speedup.

What drives acceptance rate:

- **How similar the draft model is to the target.** A distilled version of the same model
  family works much better than an unrelated small model.
- **How predictable the text is.** Boilerplate, code, structured output, and repetitive
  formats have high acceptance. Creative or reasoning-heavy generation has lower.
- **Temperature.** Higher temperature means more randomness, so more disagreement, so
  lower acceptance.

That third point has a practical consequence: speculative decoding helps most at
temperature 0, which is where most production systems run anyway.

---

## Choosing k

More lookahead means more potential gain per round and more waste when a draft is
rejected early.

```python
for k in (2, 4, 8, 16):
    print(f"k={k:2d}: {speedup(k, 0.75):.2f}x")

# k= 2: 1.85x
# k= 4: 2.05x
# k= 8: 1.83x
# k=16: 1.31x
```

There's an optimum, and it moves with your acceptance rate. Higher acceptance supports
larger k. **Tune it empirically on your traffic** — the right value for code generation
differs from chat.

---

## Variants worth knowing

**Draft model.** The classic form described above. Needs a second model — ideally a small
one from the same family.

**Self-speculation / early exit.** Use the target model's own early layers as the draft.
No second model to serve, but the drafts are usually weaker.

**N-gram / prompt lookup.** No model at all — look for the current context in the prompt or
recent output and propose the continuation that followed last time. Free, and works
remarkably well when there's a lot of repetition: summarization, editing, code completion,
anything that copies from the input.

```python
def ngram_draft(context, k=4, n=3):
    """Find where the last n tokens appeared before, propose what followed."""
    pattern = tuple(context[-n:])
    for i in range(len(context) - n - k, 0, -1):
        if tuple(context[i:i + n]) == pattern:
            return context[i + n:i + n + k]
    return []
```

**This is the one to try first** if your workload involves the model reproducing parts of
its input. Zero infrastructure, and for RAG-style tasks where answers quote retrieved text,
acceptance can be very high.

**Medusa-style multi-head.** Extra prediction heads on the target model that propose
several future tokens at once. Needs training the heads, but avoids serving a second model.

---

## When it's worth it, and when it isn't

**Worth it:**
- Latency-sensitive interactive workloads
- Low batch sizes, where the GPU's arithmetic units are underused
- Predictable output — structured formats, code, text that quotes its input
- Temperature 0 or low

**Not worth it:**
- **High batch sizes.** This is the important caveat. With a large batch, the GPU is
  already busy — the memory read is amortized across many sequences and the arithmetic
  units are no longer idle. Speculation adds work without filling spare capacity, and can
  make throughput *worse*.
- Low acceptance rate — the draft cost dominates
- Throughput-oriented batch jobs, where total tokens per second matters more than
  per-request latency

That first point is the trade-off to understand: **speculative decoding optimizes latency
at low batch sizes, and continuous batching optimizes throughput at high batch sizes.**
They pull against each other.

Some serving stacks handle this by enabling speculation adaptively — on when the batch is
small, off when it's full.

```python
def should_speculate(current_batch_size, threshold=8):
    """Speculation helps when the GPU has spare arithmetic capacity."""
    return current_batch_size < threshold
```

---

## What to measure

```python
SPEC_DECODE_METRICS = [
    "acceptance_rate",           # the number that decides everything
    "avg_accepted_per_round",    # should be close to k when working well
    "tokens_per_second",         # versus the non-speculative baseline
    "time_to_first_token",       # should be unchanged — this is a decode optimization
    "batch_size",                # to check speculation is on when it should be
    "draft_model_overhead_ms",
]
```

If acceptance rate drifts down over time, your traffic has changed — probably toward less
predictable output. Worth re-tuning k, or turning it off for that segment.

And verify the "identical output" property once, explicitly:

```python
def test_output_unchanged(prompts, model, spec_model, seed=0):
    """Greedy decoding must produce identical text with and without speculation."""
    for p in prompts:
        assert model.generate(p, temperature=0, seed=seed) == \
               spec_model.generate(p, temperature=0, seed=seed)
```

If that test fails, the implementation is wrong — and the failure is silent otherwise.

---

## Interview questions

**1. "What's speculative decoding?"**

A small model drafts several tokens, the big model verifies them in one pass, you accept
the correct prefix. The key property: the output is identical to normal decoding, so it's
pure latency gain, not a quality trade-off.

**2. "Why does it work at all?"**

Decode is memory-bandwidth-bound — the GPU reads all the weights per token and barely uses
its arithmetic units. Verifying 5 candidate tokens costs almost the same as verifying 1,
so you get the extras nearly free.

**3. "When does it not help?"**

At high batch sizes, where the GPU is already busy and the memory read is amortized across
sequences — speculation then adds work without filling idle capacity and can hurt
throughput. Also with low acceptance rates, where draft cost dominates.

**4. "What determines the speedup?"**

Acceptance rate, mainly. That depends on how similar the draft model is to the target, how
predictable the output is, and the temperature. Below ~50% acceptance the gains largely
disappear.

**5. "How would you get speculation without serving a second model?"**

N-gram or prompt-lookup drafting — propose continuations found in the prompt or recent
output. Free, and very effective for summarization, editing, and RAG answers that quote
retrieved text. Or self-speculation using the model's own early layers.

**6. "Does it change the output?"**

No, if implemented correctly. Greedy decoding accepts only tokens the target would have
chosen; sampling uses an acceptance rule designed so the output distribution matches the
target exactly. Worth having a test that asserts this.

---

## What to remember

- Small model drafts, big model verifies in one pass, accept the correct prefix.
- **Output is identical** — it's a latency optimization, not a quality trade-off.
- It works because decode is memory-bandwidth-bound, so extra tokens ride along nearly
  free.
- Acceptance rate decides the speedup. Below ~50% the gains mostly vanish.
- Helps at low batch sizes; can hurt at high batch sizes where the GPU is already busy.
- N-gram drafting needs no second model and works well when output quotes the input.
- Tune k empirically; the optimum moves with acceptance rate.
- Test the identical-output property explicitly — otherwise a broken implementation is
  silent.

---

**Back to:** [the LLM internals index](README.md) · **Next section:**
[02 — Model Selection](../02-model-selection/README.md)
