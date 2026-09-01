# Memory

Models don't remember anything. Every request starts blank. "Memory" is just you
deciding what to put back into the next request.

That framing matters, because it turns a vague question ("how do we give the agent
memory?") into a concrete one: **what do we send, and what do we leave out?**

---

## The one thing to understand first

Every step, you resend the whole conversation. All of it. So context doesn't just
grow — the *cost* grows faster than the context does.

```
step 1:  2,000 tokens sent
step 2:  4,000 tokens sent
step 3:  6,000 tokens sent
...
step 10: 20,000 tokens sent

Total sent across 10 steps: 110,000 tokens
```

Ten steps, but you paid for 110,000 tokens, not 20,000. Double the steps and it's
roughly four times the cost, not twice.

This is why long agent runs get expensive fast, and why managing context isn't
optimisation — it's the difference between a system that works and one that doesn't.

---

## The four kinds of memory

People use "memory" to mean four different things. Keeping them separate makes the
design obvious.

| Kind | Lives for | Example |
|---|---|---|
| Working | One run | What tools I already called this run |
| Short-term | One conversation | What we discussed 5 messages ago |
| Long-term | Forever | This user prefers metric units |
| Shared | Across users | Company facts, product docs |

Different storage, different rules, different problems. Take them one at a time.

---

## Working memory: within one run

This is the message list. Grows every step. The main job is stopping it growing
without limit.

### Set a budget and stick to it

Don't let context grow and hope. Decide how much you'll send, and split it up:

```python
BUDGET = 8_000   # tokens

def build_context(task, history, tool_results):
    parts = {
        "system":      1_200,   # fixed, never changes
        "tools":         400,   # tool descriptions
        "task":          200,   # the original request
        "summary":       600,   # compressed older steps
        "recent":      3_600,   # last few steps in full
        "latest":      2_000,   # the newest tool result
    }
    # total = 8,000
```

Now adding to one part means taking from another. That's the point — it makes the
trade-off visible in code review, instead of context quietly growing 7× over a year.

### Keep recent steps in full, compress older ones

Recent steps matter most. Older ones can be squashed.

```python
def fit(messages, budget):
    if count_tokens(messages) <= budget:
        return messages

    head = messages[:2]           # system prompt + original task: always keep
    recent = messages[-6:]        # last 3 exchanges: keep in full
    middle = messages[2:-6]

    # Compress the middle — tool results are usually the bulk
    squashed = []
    for m in middle:
        if m["role"] == "tool" and len(m["content"]) > 300:
            squashed.append({**m, "content": m["content"][:300] + "...[trimmed]"})
        else:
            squashed.append(m)

    return head + squashed + recent
```

### Keep a small structured record too

Text history is easy for the model to read but hard for your code to use. Keep facts
in a separate object as well:

```python
@dataclass
class RunState:
    task: str
    tools_called: list[str]           # for loop detection
    facts: dict[str, str]             # {"order_id": "ORD-123", "plan": "pro"}
    failed_paths: list[str]           # "runbook A-8814 does not exist"
    steps: int
    cost_usd: float
```

Two reasons this pays off:

1. **Your loop can use it.** Loop detection, budgets, and stop conditions all need
   structured data, not prose.
2. **Facts survive compression.** If you squash old messages, `order_id` might get
   trimmed away. Keeping it in `facts` means it's still there, and you can re-inject
   it cheaply.

That second point prevents a specific, very annoying bug: the agent asks the user for
their order number twice, because the first one got compressed out.

---

## Short-term memory: across a conversation

Multi-turn conversations grow the same way, just slower. Same fix: summarise older
turns, keep recent ones.

```python
class Conversation:
    def __init__(self, keep_recent=6, summarise_every=4):
        self.messages = []
        self.summary = ""
        self.keep_recent = keep_recent
        self.summarise_every = summarise_every
        self.turns_since_summary = 0

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.turns_since_summary += 1
        if (self.turns_since_summary >= self.summarise_every
                and len(self.messages) > self.keep_recent):
            self._compress()

    def _compress(self):
        old = self.messages[:-self.keep_recent]
        self.summary = model(f"""
Update this running summary with the new messages.

MUST keep: the user's actual problem, any IDs or numbers mentioned,
decisions made, and things already tried and ruled out.

Current summary: {self.summary}
New messages: {format(old)}
""")
        self.messages = self.messages[-self.keep_recent:]
        self.turns_since_summary = 0
```

### Test your summariser on the things it drops

The failure isn't "the summary is bad prose." It's "the summary lost the order
number." Those are very different, and only one of them makes users angry.

Write tests for exactly that:

```python
def test_summary_keeps_identifiers():
    convo = Conversation()
    convo.add("user", "My order ORD-88213 hasn't arrived")
    for i in range(12):                       # push it past the summarise point
        convo.add("assistant", f"reply {i}")
        convo.add("user", f"follow-up {i}")

    assert "ORD-88213" in convo.summary, "lost the order ID"
```

Things a summary must never lose: IDs, numbers, dates, the actual problem, and what
you've already ruled out. Test each one.

A safer approach than trusting the summary: pull identifiers out into structured
storage as they appear, and re-inject them separately.

```python
IDENTIFIER = re.compile(r"\b(ORD-\d+|INV-\d+|[A-Z]{2,4}-\d{3,})\b")

def extract_ids(text):
    return set(IDENTIFIER.findall(text))
```

Now the ID lives in a dict, not in prose, and no summariser can lose it.

---

## Long-term memory: across conversations

"Remember that I prefer email over phone." Stored somewhere, loaded next time.

### Store facts, not transcripts

Bad: save the whole conversation and search it later. You end up retrieving long
chunks of chat that mostly aren't relevant.

Better: pull out specific facts.

```python
@dataclass
class Fact:
    user_id: str
    key: str              # "preferred_contact"
    value: str            # "email"
    source: str           # where it came from, for auditing
    updated_at: float
```

Small, easy to show the user, easy to update, easy to delete. And you can put the
whole set in the prompt cheaply — 20 facts is maybe 200 tokens.

### The hard parts

**When do you write a fact?** Write too eagerly and you fill up with noise ("user
seemed frustrated"). Write too rarely and it never learns. A decent rule: only save
things the user stated directly about themselves, and only if they'd still be true in
a month.

**What if it changes?** Someone says "actually, call me instead." You need updates,
not just inserts. Key on `(user_id, key)` and overwrite.

**What if it's wrong?** A wrong stored fact is worse than no memory, because it
silently affects every future conversation and nobody knows why. Show users what
you've stored and let them delete it.

**What shouldn't you store?** Anything sensitive. Health details, financial details,
anything a person wouldn't want in a settings page. If in doubt, don't. The
convenience is small and the downside isn't.

---

## Shared memory: knowledge for everyone

This is retrieval — company docs, product info, policies. It's covered properly in the
RAG material; the only agent-specific point is:

**Retrieve fresh each step. Don't accumulate.**

```python
# BAD: context grows and grows
context = []
for step in steps:
    context.extend(retrieve(step.query))     # never removed
    respond(context)

# GOOD: fresh each time, bounded
for step in steps:
    context = retrieve(step.query, k=5)      # replaces, doesn't add
    respond(context)
```

The bad version is how you get an agent whose 10th step sends 50 retrieved documents,
most of which were relevant to step 2 and nothing since.

---

## Memory and privacy

Two things that catch teams out.

**Memory crosses conversation boundaries, so it crosses privacy boundaries too.** If
one user's stored fact can appear in another user's conversation, that's a data
breach, not a bug. Key everything by user and test it:

```python
def test_no_memory_leak_between_users():
    store.save(Fact("alice", "company", "Acme Corp", "chat", now()))
    facts = store.load("bob")
    assert not any(f.value == "Acme Corp" for f in facts)
```

**Users should be able to see and delete what you remember.** Partly because it's the
right thing, partly because in several places it's the law, and partly because it's
your best debugging tool — when someone says "it keeps getting this wrong," the stored
facts are usually the answer.

---

## What context to send, in priority order

When you're out of budget, drop things in this order (drop the bottom first):

1. **System prompt** — never drop.
2. **The original task** — never drop. This is what stops drift.
3. **Known facts (IDs, decisions)** — tiny, and expensive to lose.
4. **The latest tool result** — this is what the model is reasoning about now.
5. **Last 2–3 exchanges** — needed for continuity.
6. **Summary of older steps** — compressed already.
7. **Retrieved documents** — re-retrievable, so cheapest to lose.
8. **Older tool results in full** — drop these first.

---

## Interview questions

**1. "Your agent costs 4× what you estimated. Why?"**

Because you priced one step and multiplied. Context grows every step and you resend
all of it, so cost grows roughly with the square of the step count. Then talk about
budgets and compression.

**2. "The agent asks the user for the same information twice. What happened?"**

It got compressed out of the history. Fix: extract identifiers into structured storage
and re-inject them, rather than trusting a text summary to hold them.

**3. "How do you decide what to remember long-term?"**

Facts the user stated about themselves that will still be true in a month. Not moods,
not one-off details, nothing sensitive. Mention that users should be able to see and
delete it.

**4. "How would you test a summariser?"**

Not on prose quality — on what it keeps. Write cases where an ID, a number, or a
ruled-out option appears early and must survive compression.

**5. "Agent memory across users — what's the risk?"**

Cross-user leakage. Key by user, test that user B never sees user A's facts, and treat
it as a security test rather than a correctness test.

---

## What to remember

- Models remember nothing. Memory is you choosing what to resend.
- Cost grows with the *square* of the steps, because you resend everything each time.
- Set a token budget and split it. Adding to one part must take from another.
- Keep facts in a structured object, not just in prose — prose gets compressed away.
- Test summarisers on what they lose, not how they read.
- Retrieve fresh each step; don't pile up old retrievals.
- Memory crosses user boundaries. Key it, test it, let users delete it.

---

**Next:** [multi-agent.md](multi-agent.md) — when several agents help, and when they
just multiply your problems.
