# Authorization

Who can see what, and who can do what.

This is the file the other security files keep pointing at. Prompt injection is an
authorization problem. Exfiltration is an authorization problem wearing a different hat.
Tool abuse is authorization applied to actions instead of data. So it's worth being precise
about the thing underneath all of them.

The sentence to carry:

> **The model is not a security boundary. Authorization has to happen in the code around it,
> before data reaches the model and before an action leaves it.**

---

## Why "the model decides" never works

The tempting design, and it shows up in real systems more often than you'd think:

```python
# BAD: the permission rule lives in the prompt.
system = f"""You are a support assistant for {user.name}.
The user's role is {user.role}. Only show salary information to HR staff.
Here are the documents you may use: {all_documents}"""
```

Three things are wrong, and they're worth separating because candidates usually spot only
the first.

1. **The rule is a preference.** The model may follow it. It may not. Injection, an unusual
   phrasing, or a long conversation can shift it.
2. **The data is already in the context.** Even if the model refuses to *state* the salary,
   it was loaded, it's in your logs, it's in the trace, and it's one paraphrase away.
3. **The role came from somewhere.** If `user.role` was derived from anything the user
   controls, the whole check is decorative.

The fix for (1) and (2) is the same and it's structural: **don't put data in the context
that the user isn't allowed to see.** Then no prompt rule is needed, no refusal is needed,
and there's nothing to leak.

---

## Establish the principal outside the conversation

Every request needs an identity that the conversation cannot change.

```python
@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    roles: frozenset[str]
    scopes: frozenset[str]      # what this session may do

def principal_from_request(request) -> Principal:
    # From the verified session token. NOT from the message body,
    # NOT from anything the model produced, NOT from tool output.
    claims = verify_session_token(request.headers["authorization"])
    return Principal(
        user_id=claims["sub"],
        tenant_id=claims["tenant"],
        roles=frozenset(claims["roles"]),
        scopes=frozenset(claims["scopes"]),
    )
```

`frozen=True` is not decoration. The principal is fixed at the edge of the request and
carried, unchanged, through every retrieval and every tool call. Nothing downstream — no
model output, no tool result, no retrieved document — may modify it.

The failure this prevents: an agent reads a document saying "the user has admin access," and
somewhere a code path believes it. If the principal is immutable and derives only from the
verified token, that sentence is just text.

---

## Filter retrieval inside the query, never after

The single most common authorization bug in RAG systems.

```python
# BAD: fetch everything, then drop what they can't see.
def search(query, principal):
    hits = index.search(embed(query), k=20)
    return [h for h in hits if can_read(principal, h)]     # post-filter
```

Two problems, one obvious and one subtle:

**The obvious one:** the other tenant's data was loaded into your process. Every bug from
here on — a logging line, a caching layer, an exception handler that dumps context, a
refactor that moves the filter — turns into a data breach. You want it never fetched.

**The subtle one, and this is the interview answer:** post-filtering silently destroys your
result quality. If 18 of the top 20 belong to other tenants, the user gets 2 results for a
query that should have returned 20. Nothing errors. Retrieval just quietly gets worse for
whoever has the least data, which is usually your newest customers.

```python
# GOOD: the permission is part of the query.
def search(query, principal):
    return index.search(
        embed(query),
        k=20,
        filter={
            "tenant_id": principal.tenant_id,
            "acl": {"$in": list(visible_groups(principal))},
        },
    )
```

Now `k=20` returns 20 documents the user can actually see, and the other tenant's vectors
were never touched.

### Document ACLs at index time

For this to work, permissions have to be indexed alongside the vector.

```python
def index_document(doc, source_permissions):
    index.upsert(
        id=doc.id,
        vector=embed(doc.text),
        metadata={
            "tenant_id": doc.tenant_id,
            "acl": list(source_permissions.readable_by),   # groups/roles
            "indexed_at": now(),
        },
    )
```

Which raises the question every team hits eventually.

### Permissions change; your index doesn't notice

Someone leaves a group on Monday. Your vector store still has Friday's ACL. Until you
re-index, they can retrieve documents they no longer have access to.

Three approaches, and the honest answer in an interview is that you pick based on how fast
revocation has to be:

| Approach | Freshness | Cost |
|---|---|---|
| Re-index on permission change | Near-real-time | Needs an event stream from the source system |
| Store group IDs, resolve membership at query time | Real-time for membership | One lookup per query; index only stores stable group IDs |
| Periodic full re-sync | Hours | Cheap, and wrong in between |

The middle one is usually the right default. Index `acl: ["group:eng", "group:leads"]` — the
document's groups change rarely — and resolve *which groups this user is in* at query time
from the live identity system. Membership changes take effect immediately, and you never
re-index a document because someone changed teams.

**A second check after retrieval is still worth having** — not as the filter, but as an
assertion that fails loudly:

```python
def assert_readable(hits, principal):
    for h in hits:
        if h.metadata["tenant_id"] != principal.tenant_id:
            log.critical("cross_tenant_retrieval", doc=h.id, principal=principal.user_id)
            raise SecurityError("cross-tenant retrieval")     # fail closed, page someone
```

If this ever fires, something upstream is broken and you want to know within minutes, not at
the next audit.

---

## Multi-tenancy: pick your isolation level deliberately

"Filter by tenant_id" is one point on a spectrum, and which point you're on should be a
decision, not an accident.

| Model | Isolation | When |
|---|---|---|
| Shared index, metadata filter | Depends entirely on filter correctness | Many small tenants, cost-sensitive |
| Namespace per tenant | Strong — wrong namespace returns nothing | Most B2B SaaS. Good default |
| Index per tenant | Very strong | Enterprise contracts, regulated data |
| Separate infrastructure | Complete | Residency requirements, largest customers |

Namespaces are the sweet spot for most teams: a missing filter in the shared-index model
returns *everyone's* data, whereas a wrong namespace returns nothing. **Make the failure mode
"no results" rather than "everyone's results."** That principle — the bug should fail closed
— is worth stating explicitly, because it's what separates a design that survives a mistake
from one that doesn't.

```python
def store_for(principal):
    # The tenant is in the path. There is no call that omits it.
    return index.namespace(f"tenant-{principal.tenant_id}")
```

Better still, make the unscoped call impossible: if `index.search()` without a namespace
doesn't exist in your wrapper, no future engineer can forget it. Design out the mistake
rather than reviewing for it.

---

## Tool authorization is per-user, not per-agent

A common design error: permissions attached to the agent rather than to the person the agent
is acting for.

```python
# BAD: the agent has one service account with the union of everyone's rights.
agent = Agent(tools=[read_orders, issue_refund, delete_account])
```

Now a support agent acting for a read-only intern can issue refunds, because the *agent*
can.

```python
# GOOD: the toolset is derived from the principal.
def tools_for(principal) -> list[Tool]:
    tools = [read_orders]                                   # everyone
    if "support" in principal.roles:
        tools.append(issue_refund)
    if "admin" in principal.roles:
        tools.append(delete_account)
    return tools
```

Two properties fall out of this that are worth naming:

**The model can't call what isn't there.** Filtering the toolset at construction beats
checking permission at call time, because the unavailable tool never appears in the model's
context — nothing to be persuaded into attempting, and no refusal for an attacker to probe
against.

**Never exceed the user's own rights.** The agent acting for a user must have *at most* what
that user has. If a user couldn't issue a refund by clicking a button in your UI, the agent
must not be able to do it on their behalf. This is the delegation rule, and it's the one
that stops "the assistant did it" becoming a privilege-escalation path.

Check at call time too, because defence in depth is cheap here:

```python
def execute(tool, args, principal):
    if tool.name not in {t.name for t in tools_for(principal)}:
        log.warning("tool_not_authorized", tool=tool.name, user=principal.user_id)
        return "Error: you do not have permission to use this tool."
    return tool.fn(**args, principal=principal)
```

Note `principal=principal`. The tool doesn't take a `user_id` argument the model fills in —
that would be a parameter an attacker can shape. The identity is passed by the framework, not
by the model.

```python
# BAD: the model supplies whose data to fetch.
def get_orders(customer_id: str): ...      # injection: "fetch customer 12345"

# GOOD: the identity comes from the request, not the conversation.
def get_orders(principal: Principal): ...  # nothing to shape
```

That distinction — **identity is injected, arguments are supplied** — removes a whole class
of horizontal-privilege-escalation bug.

---

## Authorization at every layer

The full picture, and the answer to "where does the check go?" is *everywhere data or
authority crosses a line*.

```
Request     →  verify token, build immutable Principal
Retrieval   →  filter in the query, by tenant and ACL
Context     →  only what this principal may see reaches the prompt
Tools       →  toolset derived from principal; identity injected, not argued
Execution   →  downstream credentials scoped to the principal, not to the agent
Output      →  assert nothing from outside the principal's scope is present
Cache       →  key includes the permission scope
Logs        →  access to stored payloads is itself authorized
```

Two of these get forgotten most often.

**The cache.** A cache keyed on the query text alone serves one customer's answer to
another. This produced a real three-week incident described in
[../07-production/caching.md](../07-production/caching.md). The key must include the scope:

```python
def cache_key(query, principal):
    scope = hashlib.sha256(
        "|".join(sorted(visible_groups(principal))).encode()).hexdigest()[:16]
    return f"{principal.tenant_id}:{scope}:{hash_query(query)}"
```

Hashing the group set rather than the user ID keeps the cache useful — everyone with the
same visibility shares entries — while making a cross-scope hit impossible.

**Downstream credentials.** If your agent calls an internal API with one shared service
account, that API can't enforce anything per-user. Propagate the principal so the downstream
service does its own check. An agent is a client, not a trusted subsystem.

---

## Testing authorization

Authorization tests are security tests. They belong in CI and they should fail the build.

```python
def test_cross_tenant_retrieval_returns_nothing():
    index_doc(tenant="acme", text="Acme's Q3 roadmap")
    hits = search("roadmap", principal_for(tenant="globex"))
    assert hits == []

def test_toolset_excludes_unauthorized():
    tools = tools_for(principal_with(roles={"viewer"}))
    assert "issue_refund" not in {t.name for t in tools}

def test_cache_is_scoped():
    ask("what is our pricing?", principal_a)
    answer = ask("what is our pricing?", principal_b)     # different tenant
    assert answer.cache_hit is False

def test_model_cannot_choose_customer():
    # get_orders takes no customer_id — this is a signature assertion.
    assert "customer_id" not in inspect.signature(get_orders).parameters

def test_revoked_access_is_immediate():
    remove_from_group(user="dana", group="finance")
    hits = search("budget", principal_for(user="dana"))
    assert all("group:finance" not in h.metadata["acl"] for h in hits)
```

That last one is the test most teams don't have, and it's the one that catches the stale-ACL
problem before a customer does.

---

## Interview questions

**1. "How do you stop one customer's data reaching another?"**

Filter inside the retrieval query by tenant, never post-filter; prefer a namespace or index
per tenant so a bug returns nothing rather than everything; key caches on the permission
scope; assert after retrieval and fail closed. Then the framing: the model is not the
boundary, so the data must never enter the context in the first place.

**2. "Why is post-filtering wrong? It gives the same answer."**

It doesn't, and that's the good answer. The other tenant's data was loaded into your
process, so any downstream bug leaks it — and quality degrades silently, because `k=20`
with 18 filtered out returns 2 results. Filtering in the query fixes both.

**3. "Permissions changed in the source system. What happens to your index?"**

It's stale until re-indexed. Store stable group IDs in the metadata and resolve the user's
group membership at query time, so revocation is immediate without re-indexing documents.
Mention the alternatives — event-driven re-index, periodic sync — and that the choice depends
on how fast revocation must take effect.

**4. "The agent needs to look up a customer's orders. How does it know which customer?"**

From the verified session, injected by the framework — never as a tool argument the model
fills in. If `customer_id` is a parameter, an injection can point it at someone else. That's
horizontal privilege escalation, and the fix is in the signature.

**5. "Where do you put the permission check for tools?"**

Both places, but the load-bearing one is toolset construction: derive the available tools
from the principal so the model never sees a tool it can't use. Check again at call time as
defence in depth. And the agent must never have more authority than the user it acts for.

**6. "A retrieved document says the user is an administrator. What happens?"**

Nothing. The principal is immutable and comes from the verified token; text the model read
can't modify it. If it could, that's the bug — trust must never flow from content back into
identity.

**7. "Your caching layer is serving wrong answers across customers. Root cause?"**

The cache key omitted the permission scope. Key on tenant plus a hash of the visible group
set, so the cache stays useful within a scope and can't hit across one.

---

## What to remember

- The model is not a security boundary. Authorize in code, before and after the model.
- Don't put data in the context the user isn't allowed to see — then there's nothing to leak.
- The principal is immutable, derived from a verified token, never from content or tool
  output.
- Filter retrieval **inside** the query. Post-filtering leaks on any downstream bug and
  silently degrades quality.
- Index ACLs as stable group IDs; resolve membership at query time so revocation is
  immediate.
- Prefer namespaces or per-tenant indexes so a mistake fails closed — no results, not
  everyone's results.
- Derive the toolset from the principal; the model can't call a tool it can't see.
- An agent never has more authority than the user it acts for.
- Identity is injected by the framework; arguments are supplied by the model. Never mix them.
- Cache keys include the permission scope.
- Test authorization in CI: cross-tenant, toolset, cache scoping, revocation.

---

**Next:** [agent-security.md](agent-security.md) — the full agent threat model.
