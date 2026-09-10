# PII and Personal Data

Handling personal data in AI systems. This is partly a security topic and partly a legal
one, and the legal part means **the goal is not "no PII ever" — it's knowing where it is,
why it's there, and being able to control it.**

> A note on scope: this is engineering guidance, not legal advice. Data protection rules
> (GDPR, CCPA, HIPAA and others) vary by jurisdiction and by data type. The patterns here
> help you build systems that can comply; what compliance *requires* is a question for your
> legal and privacy teams. Where this file says "ask legal," it means it.

---

## Why AI systems are different

Traditional systems have personal data in known places — a users table, a specific column.
AI systems spread it around:

- Prompts contain whatever the user typed, which is anything
- Retrieved documents may contain other people's data
- Conversation history accumulates it over turns
- Logs and traces capture all of the above
- Fine-tuning datasets can absorb it into weights
- Vector embeddings are derived from it — and are themselves personal data

That last point surprises people. **An embedding of someone's personal data is a lossy but
real representation of it**, and in most frameworks it's subject to the same rules as the
source. You can't treat your vector index as exempt.

---

## Know what you're handling

Different categories carry very different obligations. You can't apply one policy to all of
it.

| Category | Examples | Rough sensitivity |
|---|---|---|
| Identifiers | name, email, phone, account ID | Standard PII |
| Financial | card numbers, account balances | High, often regulated |
| Health | conditions, medications, records | Very high (HIPAA-class) |
| Government IDs | SSN, passport, national ID | Very high |
| Biometric | face, voice, fingerprint | Very high |
| Special categories | race, religion, sexuality, politics, union | Special protection under GDPR |
| Location | real-time position, home address | High |

The engineering consequence: **tag data by category so you can apply category-specific
handling.** A system that treats an email address and a health record identically is either
over-restricting the first or under-protecting the second.

---

## Minimize before you protect

The cheapest way to secure data is not to have it. Before building controls, ask what you
actually need.

```python
def build_context(user_query, customer):
    # Don't dump the whole customer record into the prompt.
    # Include only the fields this task needs.
    return {
        "query": user_query,
        "plan": customer.plan,            # needed to answer plan questions
        "region": customer.region,        # needed for regional policy
        # NOT: full name, email, payment details, address, history
    }
```

Every field you don't put in the prompt is a field that can't leak through the response, the
cache, or the logs. Minimization is a security control, not just a compliance nicety.

---

## Redaction, and its limits

You can strip PII from text before it goes to the model or into logs.

```python
PATTERNS = {
    "email":       re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone":       re.compile(r"\b\+?\d[\d\s\-().]{7,}\d\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

def redact(text):
    for name, pattern in PATTERNS.items():
        text = pattern.sub(f"[{name.upper()}]", text)
    return text
```

**Where redaction genuinely helps:** logs, traces, training data, and any place the data
doesn't need to be to do its job.

**Where it's harder than it looks:**

- **Names and addresses don't match patterns.** Regexes catch structured data (cards, SSNs,
  emails); they miss "call me at my office, ask for Jim." Named-entity models help but
  aren't perfect.
- **Redaction can break the task.** If a support agent needs the order number to look up the
  order, redacting it makes the system useless. Redact in logs, keep it in the working
  context.
- **Reversible redaction is a liability.** If you tokenize "email → TOKEN_123" so you can
  restore it, that mapping is itself sensitive data with the same protection requirements.

Redaction is a useful layer, not a complete solution. Don't let "we redact PII" stand in
for a data-handling strategy.

---

## The logging problem

The place PII most often ends up where it shouldn't: your observability system.

Your logs contain prompts and responses. Those contain everything users typed. And logs
usually have **broader access** than your production database — more engineers can read
them, they're retained longer, they're replicated to more places.

```python
def log_request(ctx, prompt, response):
    # Structured attributes: safe, no payload
    metrics.record({
        "trace_id": ctx.trace_id,
        "input_tokens": count(prompt),
        "model": ctx.model,
        "pii_categories_present": detect_categories(prompt),   # tag, don't store
    })

    # Payloads: redacted, short retention, access-controlled, audited
    if should_store_payload(ctx):
        payload_store.put(
            ctx.trace_id,
            redact(prompt),
            ttl_days=retention_for(ctx),      # set by legal
            access="restricted",
        )
```

Four rules:

- **Redact on write, not on read** — the raw data should never land in storage.
- **Short retention, set by legal** — not by an engineer picking a round number.
- **Access controlled and audited** — reading a stored prompt is a privileged action.
- **Some product lines can't store prompts at all** — build a debugging strategy that
  doesn't depend on them, rather than discovering the constraint after the fact.

---

## Data residency

For some data, *where it lives and is processed* is legally constrained. Not just "encrypt
it" — "it must not leave this jurisdiction."

Consequences for an AI system:

- The model call must go to a regional endpoint that processes and doesn't store outside the
  region — and **contractually**, verified, not assumed.
- Embeddings are derived data. If the source can't leave the region, the vectors can't
  either. Your index must be regional.
- Logs and traces are derived data too. Regional storage, or scrubbing.
- Check what "regional endpoint" actually means — an endpoint that routes to compute
  elsewhere satisfies latency but not residency.

The distinction that trips teams up: "must not be trained on" is a contract term most
providers give you. "Must not leave our jurisdiction" is an architecture requirement.

See [../12-senior-scenarios/scaling.md](../12-senior-scenarios/scaling.md).

---

## Deletion and the right to be forgotten

A user asks you to delete their data. In an AI system, where is it?

```python
def delete_user_data(user_id):
    """Every place the data landed. Miss one and the deletion is incomplete."""
    delete_from_database(user_id)
    delete_conversation_history(user_id)
    delete_vector_embeddings(user_id)        # derived, still personal
    delete_cached_responses(user_id)         # keyed by user/scope
    purge_from_logs(user_id)                 # or rely on short retention
    delete_from_pending_training_sets(user_id)
    # Fine-tuned model: cannot un-train. This is why you don't fine-tune
    # a shared model on deletable personal data.
    return DeletionReceipt(user_id, deleted_locations=[...])
```

Two hard cases:

**Fine-tuned models can't forget.** Data baked into weights can't be removed without
retraining. This is a strong reason not to fine-tune on personal data that users can
request deletion of — you're making a promise you can't keep. See
[../02-model-selection/fine-tuning-vs-rag.md](../02-model-selection/fine-tuning-vs-rag.md).

**You need a map of where data lands.** If you can't enumerate every place a user's data
went, you can't delete it, and "we probably got most of it" is not a compliant answer.
Design the data-flow so this list is knowable.

---

## The memory feature problem

If your product "remembers" things about users across conversations, you've built a personal
data store, with everything that implies:

- Users should be able to **see** what you've stored about them
- Users should be able to **delete** it
- It should **not cross user boundaries** — one user's stored fact appearing for another is
  a breach
- **Don't store sensitive categories** just because a user mentioned them — health,
  finances, protected characteristics need a deliberate decision, not automatic capture

```python
def test_memory_does_not_cross_users():
    memory.store(user="alice", fact="works at Acme")
    assert "Acme" not in memory.load(user="bob")
```

The temptation with memory features is to capture everything useful. The discipline is to
capture only what's needed, let users control it, and keep sensitive categories out unless
there's a clear reason and consent.

---

## Interview questions

**1. "How do you handle PII in an LLM system?"**

Minimize first — only put in the prompt what the task needs. Redact in logs and training
data. Tag by category so obligations match sensitivity. Then note that embeddings are
derived personal data and vector indexes aren't exempt.

**2. "Why is redaction not enough?"**

It catches structured data but misses names and free-form references; it can break the task
if you redact what the system needs; and reversible redaction creates a sensitive mapping.
It's a layer, not a strategy.

**3. "A user requests deletion. Where's their data?"**

Database, conversation history, embeddings, cache, logs, pending training sets — and a
fine-tuned model that can't forget. The real answer is that you need a map of where data
lands, and that you shouldn't fine-tune shared models on deletable data.

**4. "What does data residency require beyond encryption?"**

That the data and everything derived from it — embeddings, logs, model processing — stays in
the jurisdiction. Regional model endpoints verified contractually, regional vector index,
regional or scrubbed logs. "Must not leave" is architecture, not just encryption.

**5. "Your logs contain user prompts. What's the risk?"**

Logs usually have broader access and longer retention than production data, so they become
the weakest link for PII. Redact on write, short retention set by legal, audited access.

**6. "You're building a memory feature. What are the obligations?"**

It's a personal data store: users can see and delete their data, it must not cross user
boundaries, and sensitive categories shouldn't be captured automatically. Test the
cross-user isolation as a security test.

---

## What to remember

- The goal isn't zero PII — it's knowing where it is and being able to control it.
- This is legal as much as engineering. Where it says "ask legal," ask legal.
- Minimize first: fields not in the prompt can't leak.
- Embeddings are derived personal data. Vector indexes aren't exempt.
- Tag data by category; obligations differ enormously across them.
- Redaction helps in logs and training data; it's a layer, not a solution.
- Logs are the usual weak link — broader access, longer retention.
- Residency covers derived data too: embeddings, logs, processing location.
- Deletion requires a map of every place data lands — and fine-tuned models can't forget.
- A memory feature is a personal data store. Treat it as one.

---

**Next:** [authorization.md](authorization.md) — who can see and do what.
