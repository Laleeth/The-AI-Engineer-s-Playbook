# Query Rewriting

Users don't write good search queries. They write things like "what about the other
one?" and expect it to work.

Query rewriting fixes the question before you search with it.

---

## The problem

Real user queries in a conversation:

```
User: "how long is the return window?"
Bot:  "30 days for most items."
User: "what about electronics?"        ← searching this finds nothing useful
```

"what about electronics?" as a search query is meaningless. It has no subject. The words
you'd want — "return window", "policy" — are in the *previous* turn.

Other broken query shapes:

| User writes | Problem |
|---|---|
| "broken" | Too short, no context |
| "it doesn't work" | No subject at all |
| "how do I fix the thing where it says error and then crashes when I upload" | Too long, mixed signals |
| "refund???" | No structure |
| "I need to return my order but also change my address" | Two questions in one |

Search works best on well-formed, self-contained questions. Rewriting turns messy input
into that.

---

## Rewriting a follow-up question

The most valuable rewrite, and the easiest.

```python
REWRITE_PROMPT = """Rewrite the user's latest message as a standalone question
that makes sense without the conversation.

Rules:
- Keep the user's actual intent. Do not answer the question.
- Fill in what "it", "that", "the other one" refer to.
- Keep any IDs, numbers, or product names exactly as written.
- If the message is already standalone, return it unchanged.

Conversation:
{history}

Latest message: {message}

Standalone question:"""


async def rewrite_followup(history, message, model):
    if len(history) == 0:
        return message                      # nothing to resolve

    result = await model(
        REWRITE_PROMPT.format(history=format_recent(history, turns=3),
                              message=message),
        temperature=0.0,
        max_tokens=100,
    )
    return result.strip()
```

In practice:

```
history:  "how long is the return window?" / "30 days for most items"
message:  "what about electronics?"
rewrite:  "how long is the return window for electronics?"
```

That rewritten query retrieves properly. The original doesn't.

**Only use the last few turns.** Sending the whole conversation costs tokens and makes
the rewrite worse — the model starts pulling in irrelevant earlier topics.

---

## Splitting multi-part questions

"I want to return my order and also update my billing address" is two questions. One
search will do badly on both.

```python
async def split_question(question, model):
    result = await model(f"""
If this contains multiple separate questions, split them.
If it's one question, return it unchanged.

Reply with JSON: {{"questions": ["...", "..."]}}

Question: {question}
""", temperature=0.0)
    return json.loads(result)["questions"]


async def answer_multipart(question, model):
    parts = await split_question(question, model)

    if len(parts) == 1:
        return await answer_one(parts[0])

    # Retrieve for each part separately, then answer with everything
    all_docs = []
    for part in parts:
        all_docs.extend(retrieve(part, k=3))

    return await generate(question, dedupe(all_docs))
```

Worth noting the cost: splitting means more retrieval calls. Only do it when your traffic
actually contains multi-part questions — check before building it.

---

## Generating several search queries

One question, several different searches. Helps when a single phrasing might miss.

```python
async def expand_query(question, model, n=3):
    result = await model(f"""
Write {n} different search queries that would find documents answering this
question. Vary the wording and specificity. Keep any exact identifiers.

Question: {question}

Reply with JSON: {{"queries": ["...", "..."]}}
""", temperature=0.3)      # some variety is the point here
    return json.loads(result)["queries"]


async def multi_query_retrieve(question, model, k=5):
    queries = await expand_query(question, model)
    queries.append(question)                   # always include the original

    rankings = []
    for q in queries:
        hits = search(q, k=20)
        rankings.append([h.id for h in hits])

    fused = rrf(rankings)                      # same fusion as hybrid search
    return [chunks[doc_id] for doc_id, _ in fused[:k]]
```

**Cost:** N searches instead of one, plus a model call to generate them. Worth it when
retrieval is genuinely weak, wasteful when it isn't.

---

## HyDE: search with a fake answer

A neat trick with a real basis. Instead of searching with the *question*, generate a
hypothetical *answer* and search with that.

```python
async def hyde_retrieve(question, model, k=5):
    """Generate a plausible answer, then search with it.

    Rationale: your documents contain answers, not questions. A hypothetical
    answer is more similar in shape and vocabulary to real documents than the
    question is.
    """
    fake_answer = await model(
        f"Write a short factual paragraph that would answer this question. "
        f"It's fine if details are invented — this is only for searching.\n\n"
        f"Question: {question}",
        max_tokens=150,
        temperature=0.0,
    )
    return search(embed(fake_answer), k=k)
```

Why it can work: embedding a short question and matching it against long document
passages is comparing two different shapes of text. A generated answer looks more like a
document.

**When it helps:** short or vague questions, and domains where question phrasing differs
a lot from document phrasing.

**When it doesn't:** questions containing identifiers (the fake answer may invent a
different one — check that), and anywhere you can't afford the extra model call and its
latency.

Test it. It helps meaningfully on some corpora and does nothing on others.

---

## The cheapest fixes first

Before adding model calls, try these. They cost nothing and handle a lot.

```python
def clean_query(q):
    q = q.strip()
    q = re.sub(r"[?!]{2,}", "?", q)              # "refund???" → "refund?"
    q = re.sub(r"\s+", " ", q)
    return q


def is_too_vague(q):
    """Don't search on this — ask the user instead."""
    words = q.split()
    if len(words) < 2:
        return True
    if q.lower().strip("?.! ") in {"help", "hi", "hello", "thanks", "broken", "it"}:
        return True
    return False


def needs_rewrite(message, history):
    """Only pay for a rewrite when the message actually depends on history."""
    if not history:
        return False
    pronouns = {"it", "that", "this", "they", "them", "the other", "those"}
    lower = message.lower()
    return (len(message.split()) < 6
            or any(p in lower for p in pronouns))
```

That last function matters for cost. Rewriting every message doubles your model calls. If
only 30% of messages are follow-ups that need resolving, only rewrite those.

And `is_too_vague` is worth having as a product decision: asking "could you tell me a bit
more about what's not working?" is a better response to "broken" than searching for it
and returning something random.

---

## Keep the original

Always search with the original query as well as any rewrite:

```python
async def retrieve_with_rewrite(message, history, model, k=5):
    rankings = []

    # Original always included — the rewrite might have lost something
    rankings.append([h.id for h in search(message, k=20)])

    if needs_rewrite(message, history):
        rewritten = await rewrite_followup(history, message, model)
        if rewritten != message:
            rankings.append([h.id for h in search(rewritten, k=20)])

    fused = rrf(rankings)
    return [chunks[doc_id] for doc_id, _ in fused[:k]]
```

Rewrites can go wrong — they can drop a detail, change the meaning, or invent context
that wasn't there. Keeping the original as a second ranking makes the system robust to a
bad rewrite rather than dependent on a good one.

---

## Things that go wrong

**The rewrite changes the meaning.** "What about electronics?" becomes "Tell me about
electronics" — losing the return-window context entirely. Fix: a stricter prompt, and
keep the original query in the mix.

**Identifiers get mangled.** The user wrote `ORD-88213`; the rewrite says "the order the
user mentioned." Now exact matching fails. Fix: instruct explicitly to preserve
identifiers, and verify:

```python
IDENTIFIER = re.compile(r"\b([A-Z]{2,4}-\d{3,}|v\d+\.\d+(?:\.\d+)?)\b")

def rewrite_kept_identifiers(original, rewritten):
    return IDENTIFIER.findall(original) == [] or \
           set(IDENTIFIER.findall(original)) <= set(IDENTIFIER.findall(rewritten))
```

If the check fails, use the original.

**Latency.** Every rewrite is a model call on the critical path — 200–500ms before search
even starts. Use a small fast model, and only rewrite when needed.

**Cost.** Rewriting every query doubles your calls. Gate it.

**It answers instead of rewriting.** The model helpfully replies to the question rather
than reformulating it. Fix: say "do not answer" in the prompt, cap `max_tokens` low, and
validate the output doesn't look like an answer.

**Over-rewriting a good query.** A clear, well-formed question gets "improved" into
something worse. The `needs_rewrite` gate handles most of this.

---

## What to measure

```python
REWRITE_METRICS = [
    "rewrite_rate",              # % of queries rewritten — should match your
                                 # follow-up rate, not be 100%
    "recall_with_rewrite",
    "recall_without_rewrite",    # the comparison that justifies it
    "identifier_preservation",   # % of rewrites keeping IDs intact
    "rewrite_latency_p95",
    "rewrite_cost_per_request",
]
```

**Segment by whether the query was a follow-up.** Rewriting shouldn't change standalone
queries at all, and should help follow-ups a lot. An overall number hides both.

```
query_type      n     without   with    gain
standalone     150     0.86     0.85    -0.01   ← should be roughly flat
follow_up       80     0.41     0.79    +0.38   ← this is the win
```

If standalone queries get worse, you're rewriting things that didn't need it.

---

## Interview questions

**1. "A user asks 'what about the other one?' — how does your system handle it?"**

Rewrite it into a standalone question using the last few conversation turns. Then say you
keep the original query in the mix too, because rewrites can drop details.

**2. "What's the cost of query rewriting?"**

A model call on the critical path — latency before search even starts, plus per-request
cost. Gate it: only rewrite when the message is short or contains pronouns. Use a small
fast model.

**3. "What's HyDE and when would you use it?"**

Generate a hypothetical answer and search with that instead of the question, because
documents contain answers rather than questions. Useful for short or vague queries.
Watch out for invented identifiers, and test whether it actually helps on your corpus.

**4. "Your rewrite loses order numbers. How do you catch that?"**

Extract identifiers from the original with a regex, check they survive in the rewrite,
and fall back to the original if not. Mention it as an automatic check, not a prompt
instruction.

**5. "How do you know rewriting helped?"**

Segment by follow-up versus standalone. Follow-ups should improve a lot; standalone
queries should be unchanged. An overall average hides both effects.

**6. "When would you not rewrite?"**

Single-turn products with no conversation history, or when latency is tight. Also when
you haven't measured a problem — it's a model call on every request and should earn it.

---

## What to remember

- Users write bad queries. Fix them before searching.
- Follow-up resolution is the highest-value rewrite.
- Only use the last 2–3 turns of history — more makes it worse and costs more.
- Gate it: rewrite short messages and ones with pronouns, not everything.
- Always keep the original query in the mix.
- Check identifiers survive the rewrite, automatically.
- Free fixes first: normalise punctuation, detect too-vague queries and ask instead.
- HyDE searches with a fake answer — test whether it helps on your data.
- Measure segmented by follow-up versus standalone.

---

**Next:** [contextual-retrieval.md](contextual-retrieval.md) — giving each chunk enough
context to be found.
