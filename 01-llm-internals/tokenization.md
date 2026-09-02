# Tokenization

Models don't read characters or words. They read **tokens** — chunks of text from a fixed
vocabulary, usually 30,000 to 200,000 entries.

Tokenization looks like plumbing, and it's the source of a surprising number of production
bugs: cost estimates that are wrong for some languages, context limits hit earlier than
expected, and truncation in the middle of a character.

---

## What a token is

Roughly a common chunk of text. Frequent words are one token; rare words split into
pieces.

```
"The quick brown fox"     →  ["The", " quick", " brown", " fox"]        4 tokens
"antidisestablishment"    →  ["anti", "dis", "establish", "ment"]       4 tokens
"E-4471"                  →  ["E", "-", "44", "71"]                     4 tokens
"日本語のテキスト"          →  many more tokens than characters
```

Note the leading spaces. In most tokenizers `" quick"` and `"quick"` are **different
tokens**. That's why a trailing space in your prompt can change the output.

---

## The rule of thumb, and where it breaks

The common approximation:

```python
def estimate_tokens(text):
    return len(text) // 4        # ~4 characters per token
```

This is roughly right **for English prose** and wrong for a lot of other things:

| Content | Chars per token | Notes |
|---|---|---|
| English prose | ~4 | The rule of thumb |
| Code | ~3 | Punctuation and symbols split more |
| JSON | ~2.5 | Braces, quotes, colons all cost |
| Non-Latin scripts | ~1–2 | Often one token per character or worse |
| Random identifiers | ~2 | `ORD-88213` splits into pieces |
| Base64 / hashes | ~2 | Nothing is in the vocabulary |

**This has a real business consequence.** If your product serves Japanese, Korean, Thai,
or Arabic customers, their conversations may cost 2–3× more tokens than English ones for
the same content. Which means:

- Your cost-per-user is not uniform across markets
- Your context budget is effectively smaller for those users
- Truncation fires earlier for them
- Your quality can be worse for them *for structural reasons*, not model reasons

That last point is the one people miss. If you set a global 8,000-token context budget and
Japanese consumes tokens 2.5× faster, Japanese users get far less retrieved context, and
their answers are worse. It looks like a model problem and it's a budgeting problem.

```python
def context_budget_check(text_samples_by_language, tokenizer, budget=8000):
    """Are some languages structurally starved of context?"""
    for lang, sample in text_samples_by_language.items():
        chars = len(sample)
        tokens = len(tokenizer.encode(sample))
        ratio = chars / tokens
        effective_chars = budget * ratio
        print(f"{lang}: {ratio:.2f} chars/token, "
              f"~{effective_chars:,.0f} chars fit in {budget} tokens")
```

Run this before setting a global budget.

---

## Always use the real tokenizer for anything that matters

Estimation is fine for chunk sizing. It is **not** fine for a hard limit.

```python
# Fine — chunk size is approximate anyway
approx = len(text) // 4

# Not fine — this decides whether the request fits
exact = len(tokenizer.encode(text))
```

Being 20% under on an estimate is harmless. Being 5% over on a context limit is a failed
request or a truncated response.

```python
def fits_in_context(prompt, max_output, context_limit, tokenizer):
    """Reserve output space. This is the check people forget."""
    input_tokens = len(tokenizer.encode(prompt))
    return input_tokens + max_output <= context_limit
```

**Reserving output space is the part that gets skipped.** A prompt that fits the context
window leaves no room to generate, so the response truncates immediately. That produces
`finish_reason: length`, which downstream code often retries — and retries re-send the
whole input, which is where your cost is. That's how a context bug becomes a bill.

---

## Why identifiers tokenize badly

```
"E-4471"   →  ["E", "-", "44", "71"]
"E-4472"   →  ["E", "-", "44", "72"]
```

Three of the four tokens are identical. This is part of why embeddings struggle to
distinguish them, and it's the mechanical basis for needing
[hybrid search](../03-rag/hybrid-search.md) when your content has product codes, error
codes, or version numbers.

It also affects generation: models are worse at reproducing long identifiers exactly,
because each piece is a separate prediction and errors compound.

---

## Practical consequences

**Trailing whitespace changes output.** `"Answer:"` and `"Answer: "` end in different
tokens and can produce noticeably different completions. Be consistent in prompt
templates.

**Token limits are not character limits.** A "4,000 character" user input might be 1,000
tokens or 3,000 depending on content. Validate in tokens.

**Streaming can split characters.** A multi-byte character (an emoji, a CJK character) can
span token boundaries. Naive byte-level decoding mid-stream produces replacement
characters. Buffer incomplete sequences:

```python
class StreamDecoder:
    """Handle multi-byte characters split across streaming chunks."""

    def __init__(self):
        self._buffer = b""

    def feed(self, chunk: bytes) -> str:
        self._buffer += chunk
        try:
            text = self._buffer.decode("utf-8")
            self._buffer = b""
            return text
        except UnicodeDecodeError as exc:
            # Keep the incomplete tail for the next chunk
            safe = self._buffer[:exc.start].decode("utf-8")
            self._buffer = self._buffer[exc.start:]
            return safe
```

Most SDKs handle this. If you're parsing a raw stream yourself, you need it.

**Counting is not free.** Tokenizing a large document to count tokens costs real CPU time.
Cache counts where you can, and use estimates where precision doesn't matter.

**Different models, different tokenizers.** A prompt that is 3,000 tokens for one model may
be 3,400 for another. When comparing costs across providers, tokenize with each one's
tokenizer — the per-token price alone can mislead you.

```python
def compare_true_cost(prompt, providers):
    """Per-token price is meaningless without each model's own token count."""
    for name, (tokenizer, price_per_million) in providers.items():
        tokens = len(tokenizer.encode(prompt))
        print(f"{name}: {tokens:,} tokens, ${tokens/1e6*price_per_million:.5f}")
```

---

## Special tokens

Vocabularies include control tokens: beginning-of-sequence, end-of-sequence, padding, and
chat-format markers for turn boundaries.

Two things to know:

**They count toward your limits.** Chat formatting adds tokens per message. A conversation
with 20 short turns pays that overhead 20 times.

**Never let user input contain them.** If a user can inject a turn-boundary token into
their message, they may be able to make the model believe a new turn started — a
prompt-injection vector. Most APIs handle this, but if you're building prompts as raw
strings against a self-hosted model, check.

---

## Interview questions

**1. "How do you count tokens?"**

Use the model's real tokenizer for anything that decides behaviour; estimates are fine for
chunk sizing. Then mention reserving output space, because that's the check people skip
and it causes truncation-then-retry cost bugs.

**2. "Why do some languages cost more?"**

Tokenizers are trained mostly on English-heavy corpora, so non-Latin scripts fragment
into more tokens per character. Then give the real consequence: with a global token
budget, those users get structurally less context and worse answers — a budgeting problem
that looks like a model problem.

**3. "A user reports garbled characters in your streaming output. Why?"**

A multi-byte character split across token or chunk boundaries, decoded before the sequence
was complete. Buffer the incomplete tail.

**4. "Why does adding a space to the end of my prompt change the output?"**

`"word"` and `" word"` are different tokens. A trailing space changes which token the model
sees last and therefore what it predicts next.

**5. "You're comparing two providers on price per token. What's the catch?"**

Different tokenizers produce different token counts for the same text. Compare total cost
for your actual prompts, not price per token.

---

## What to remember

- ~4 characters per token for English prose. Much worse for code, JSON, and non-Latin
  scripts.
- Non-English users can cost 2–3× more tokens — and get less context under a global
  budget.
- Use the real tokenizer for anything that decides behaviour.
- Always reserve output space in your context budget, or you get truncate-then-retry.
- Leading spaces matter: `" word"` ≠ `"word"`.
- Identifiers fragment, which is why exact-match search needs keyword retrieval.
- Buffer incomplete multi-byte sequences when streaming.
- Price per token is not comparable across models with different tokenizers.

---

**Next:** [positional-encoding.md](positional-encoding.md) — how the model knows what
order the tokens are in.
