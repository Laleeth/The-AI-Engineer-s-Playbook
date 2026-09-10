# Tool Calling

Tool calling is how a model runs your code. You describe what a function does, the
model asks you to call it, you call it, and you send back the result.

The mechanism is simple. The hard part is making the model pick the right tool with
the right arguments — and that turns out to be mostly a **writing** problem, not a
coding problem.

---

## How it actually works

There's no magic here. The model can't run anything. It just outputs text saying "I'd
like to call this function with these arguments." Your code does the rest.

The loop:

```
1. You send: the user's message + a list of tools
2. Model replies: "call get_weather with {city: 'Berlin'}"
3. YOUR CODE runs get_weather("Berlin")  ← the model does nothing here
4. You send back: the result
5. Model replies with an answer, or asks for another tool
```

Step 3 is the important one. **The model never touches your systems.** It asks, you
decide, you run it. That's also where all your safety controls live — see
[permissions.md](permissions.md).

---

## Describing a tool

You give the model a name, a description, and the shape of the arguments (this is
called a JSON Schema — it's just a description of what fields exist and what type they
are).

```python
{
    "name": "get_order_status",
    "description": "Look up the current status of a customer order by order ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order ID, format ORD-12345",
            }
        },
        "required": ["order_id"],
    },
}
```

The model reads this the same way a new colleague would. It's documentation. If a new
hire couldn't tell when to use this tool, the model can't either.

---

## Writing good tool descriptions

This is where most tool-calling problems come from, and it's the least technical part
of the job.

### Say when to use it, not just what it does

Bad:

```python
tool = {
    "description": "Gets user data.",
}
```

Good:

```python
tool = {
    "description": (
        "Get a customer's account details: name, email, plan, signup date. "
        "Use when you need to know who the customer is or what plan they're on. "
        "Does NOT include order history — use get_orders for that."
    ),
}
```

The "does NOT" line matters more than you'd expect. Saying what a tool *isn't* for
stops the model reaching for it in the wrong situation.

### Make similar tools clearly different

If you have these three:

```
search_docs
search_articles
search_help
```

...the model will pick randomly, because you would too. Either merge them into one
tool with a `source` parameter, or make the descriptions sharply different:

```
search_product_docs   — technical documentation for developers
search_help_center    — how-to guides for end users
search_policies       — refund, privacy, and terms of service text
```

### Describe the arguments too

```python
properties = {
    "order_id": {
        "type": "string",
        "description": "Order ID like ORD-12345. If the customer gives you a "
                       "number without the prefix, add ORD- to the front.",
    },
}
```

That second sentence prevents a whole class of failed calls.

### Use enums when the options are fixed

```python
properties = {
    "status": {
        "type": "string",
        "enum": ["pending", "shipped", "delivered", "cancelled"],
    },
}
```

Now the model can't invent `"in_transit"`. Anywhere you have a fixed list of valid
values, use an enum. It's free reliability.

---

## How many tools is too many?

More tools make the model worse at choosing. There's no exact number, but a rough
guide:

- **Under 10 tools:** usually fine.
- **10–20:** you'll start seeing wrong choices, especially between similar tools.
- **Over 20:** noticeably worse, and you're also burning a lot of tokens on tool
  descriptions in every single request.

Two ways to handle a big tool set:

**1. Filter before you send.** Pick the relevant tools for this request. You can do
this with simple rules (this is a billing question → send the billing tools) or by
searching your tool descriptions.

```python
def pick_tools(user_message, all_tools, limit=8):
    # simplest version: keyword rules
    # better version: embed the message, embed tool descriptions, take top matches
    scored = [(relevance(user_message, t), t) for t in all_tools]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in scored[:limit]]
```

**2. Group them.** Instead of 30 tools, expose 5 tools that each take an `action`
parameter. Fewer choices at the top level, and the model picks the detail inside.

Trade-off: grouping makes the model's job easier but makes your validation harder,
because one tool now does several things.

---

## Handling the call

Never trust the arguments. The model is guessing, and sometimes it guesses wrong.

```python
def run_tool(name, arguments, tools):
    tool = tools.get(name)

    # 1. Does this tool exist?
    if tool is None:
        # Tell the model what DOES exist so it can fix itself
        return f"Error: no tool named '{name}'. Available: {', '.join(tools)}"

    # 2. Are the arguments valid?
    error = validate(arguments, tool.parameters)
    if error:
        return f"Error: {error}"

    # 3. Is the model allowed to do this?
    if not allowed(tool, current_user):
        return "Error: you do not have permission to use this tool."

    # 4. Actually run it — and never let it crash the loop
    try:
        result = tool.fn(**arguments)
    except FileNotFoundError as e:
        # Terminal: retrying with the same arguments won't help. Say so.
        return f"Not found: {e}. Do not retry with the same arguments."
    except TimeoutError as e:
        # Transient: might work next time
        return f"Timed out: {e}. This may be temporary."
    except Exception as e:
        log.exception("tool %s failed", name)   # full details for engineers
        return f"Error calling {name}: {type(e).__name__}"   # short version for the model

    return shorten(result, limit=8_000)
```

Five things worth noticing in that code:

**Errors go back to the model as text, not as exceptions.** A failed tool shouldn't
kill the run. The model can often recover — try different arguments, try a different
tool, or tell the user it couldn't find something.

**Terminal vs. transient errors are described differently.** This matters a lot. If a
404 comes back as just "Error", the model will try again. And again. That's how you
get the infinite loop in [agent-failure-modes.md](agent-failure-modes.md). Tell it
plainly: *this does not exist, do not retry.*

**Full errors are logged, short errors are returned.** A stack trace wastes tokens and
can leak internal details (file paths, hostnames, sometimes credentials in a
connection string) into a conversation a user might see.

**Output is capped.** A tool returning 40 MB of logs will blow your context window and
cost a fortune. Cut it, and *say* you cut it:

```python
def shorten(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[cut: {len(text)-limit} more characters. Narrow your query.]"
```

Cutting silently is worse than not cutting. The model will answer confidently based on
the first 8,000 characters and won't know it's missing anything.

**Permission is checked at call time**, not left to the prompt. More on this in
[permissions.md](permissions.md).

---

## Things that go wrong

### The model invents a tool

It asks for `get_customer_email` when you only have `get_customer`. Don't crash —
return the list of real tools. The model will usually correct itself on the next step.

### The arguments don't parse

Tool arguments come back as a JSON string. Sometimes it's malformed. Handle it:

```python
try:
    args = json.loads(raw_arguments)
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
except (json.JSONDecodeError, ValueError):
    return "Error: your arguments were not valid JSON. Please try again."
```

Again — a message back to the model, not a crash.

### It calls the same tool over and over

Usually because a failure came back looking soft. "Error: could not find runbook
A-8814" reads to the model like "maybe try again."

Track what's already been called:

```python
def fingerprint(name, args):
    return hashlib.sha256(
        json.dumps({"n": name, "a": args}, sort_keys=True).encode()
    ).hexdigest()[:16]

# If this exact call happened before, say so plainly
if fp in already_called:
    return (f"You already called {name} with these exact arguments and got: "
            f"{already_called[fp][:200]}. Do not repeat this. Try something else.")
```

Telling the model it's repeating itself works much better than raising the step limit.

### It calls tools in a silly order

It drafts a reply before looking anything up. Fix in the description or the system
prompt:

```
"Before drafting a reply, always look up the customer and check the
knowledge base first."
```

If the order really matters and must never vary, that's a sign this should be a
workflow, not an agent — see [agent-vs-workflow.md](agent-vs-workflow.md).

### It makes up argument values

It calls `get_order_status(order_id="ORD-00000")` because the user never gave an order
ID. Prevent it:

- Say so in the description: *"Only call this if the user gave you an actual order ID.
  Do not guess."*
- Give it a way out: an `ask_user` tool it can call when information is missing.
- Validate: if the ID doesn't exist, the error tells it to ask instead of guessing.

---

## Doing several tool calls at once

Models can ask for multiple tools in one turn. You can run them together — but only if
they're safe to run together.

```python
async def run_all(calls, tools):
    # Read-only tools can run at the same time
    reads = [c for c in calls if tools[c.name].effect == "read_only"]
    writes = [c for c in calls if tools[c.name].effect != "read_only"]

    results = await asyncio.gather(*[run_tool_async(c, tools) for c in reads])

    # Anything that changes something runs one at a time, in order
    for c in writes:
        results.append(await run_tool_async(c, tools))

    return results
```

**Never run write operations in parallel.** Two "send email" calls at the same time
can produce two emails, and the ordering between "cancel order" and "refund order"
matters. Reads are safe; writes are not.

---

## Structured output vs. tool calling

These look similar and get confused. The difference:

**Structured output** = "give me your answer in this shape." One call, you get JSON
back, you're done.

**Tool calling** = "you may ask me to run things." Possibly many rounds.

If you just want the model's answer formatted as JSON, use structured output. It's
simpler, cheaper, and more reliable. Tool calling is for when the model actually needs
information it doesn't have, or needs to *do* something.

A surprising number of "agents" are really just structured output with extra steps.

---

## Testing tools

Three things worth testing, and they're all cheap:

**1. Does it pick the right tool?**

Build a small set of examples: user message → which tool should be called.

```python
CASES = [
    ("where is my order ORD-123", "get_order_status"),
    ("how do I reset my password", "search_help_center"),
    ("what plan am I on", "get_customer"),
    ("cancel my subscription", "escalate"),      # should NOT self-serve
]
```

Run it whenever you change a tool description or add a tool. **Adding a tool changes
how the model picks between all the others** — it's not additive, and this test is how
you find out.

**2. Do the arguments come out right?**

Same idea, but check the arguments too. Especially formats: does it add the `ORD-`
prefix? Does it use the enum values?

**3. Does it handle failures well?**

Make a tool return a 404, a timeout, an empty result, a huge result. Check the agent
does something sensible rather than looping or giving up.

---

## Interview questions

**1. "How do you handle a tool that fails?"**

Return the error to the model as text, not an exception. Distinguish terminal from
transient, and tell the model explicitly when retrying won't help. Log the full error,
send back a short one.

**2. "You have 50 tools. What do you do?"**

Filter to the relevant ones per request, or group them. Explain that selection accuracy
drops with tool count and that all 50 descriptions cost tokens on every single request.

**3. "The model keeps calling a tool with made-up IDs. Fix it."**

Say "do not guess" in the description, give it an `ask_user` tool as an escape route,
validate arguments and return a helpful error, and check whether the ID was actually
in the conversation before the call.

**4. "Can you run tool calls in parallel?"**

Reads yes, writes no. Explain the ordering and duplicate-side-effect risks.

**5. "How do you test tool selection?"**

A small labeled set of message → expected tool, run on every tool or description
change. Mention that adding a tool changes selection across all of them.

---

## What to remember

- The model never runs anything. It asks; your code decides and runs.
- Tool descriptions are documentation. Write them for a new colleague.
- Say what a tool is **not** for. It stops wrong choices.
- Use enums wherever the values are fixed.
- Errors go back as text, and terminal errors must say "don't retry."
- Always cap tool output, and say when you cut it.
- Fewer tools = better choices.
- Parallel reads are fine. Parallel writes are not.

---

**Next:** [planning.md](planning.md) — how agents decide what to do, and when planning
helps versus hurts.
