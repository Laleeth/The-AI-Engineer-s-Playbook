# Cost vs. Quality

Cost questions are the fastest way to find out whether someone has actually
operated an AI system. Candidates who haven't will pick a model. Candidates who
have will ask what the traffic looks like, notice that "quality" is not a scalar,
and realize that the interesting decisions are almost never "which model" — they're
about *architecture that lets different traffic get different treatment*.

Three habits worth building before you read on:

- **Always compute the unit economics.** Cost per request, per user, per resolved
  ticket. A monthly number is not a decision input; a per-unit number compared to
  the per-unit value is.
- **Refuse the false dichotomy.** Most cost/quality questions are posed as a choice
  between models. The strong answer almost always changes the *shape* of the
  system — routing, caching, cascading, batching — rather than picking a point on
  someone else's menu.
- **Know where the money actually goes.** In most LLM applications, input tokens
  dominate cost and output tokens dominate latency. Optimizing the wrong one is the
  most common wasted quarter in this field.

**Contents**

1. [The Model Menu (and Why It's the Wrong Question)](#1-the-model-menu-and-why-its-the-wrong-question)
2. [The Free Tier That Ate the Margin](#2-the-free-tier-that-ate-the-margin)
3. [Cutting 60% Without Telling Users](#3-cutting-60-without-telling-users)
4. [Fine-Tune, Prompt, or Route?](#4-fine-tune-prompt-or-route)
5. [When Quality Is Worth Any Price](#5-when-quality-is-worth-any-price)

---

## 1. The Model Menu (and Why It's the Wrong Question)

### Situation

You run the AI platform for a customer-support product used by mid-market SaaS
companies. Your assistant drafts replies to customer emails; a human agent reviews
and sends.

You have evaluated three models on your production-derived eval set:

| Model | Quality (graded win-rate vs. human baseline) | Cost/1M in | Cost/1M out | p95 latency |
|---|---|---|---|---|
| A (frontier) | 92% | $3.00 | $15.00 | 4.1s |
| B (mid-tier) | 89% | $0.60 | $2.40 | 1.9s |
| C (small) | 85% | $0.15 | $0.60 | 0.8s |

At current volume: Model A ≈ **$40k/month**, Model B ≈ **$8k/month**, Model C ≈
**$2k/month**.

Your CFO has set a **$15k/month** ceiling for this line item.

### Your Task

Choose the model strategy. "Model B, it fits the budget" is a failing answer;
explain what you'd do instead and why.

### Constraints

- **1M requests/day** (≈ 30M/month).
- **2-second p95 latency** requirement — agents work in a queue and wait for the
  draft.
- Traffic analysis: **95% of emails are routine** (password resets, billing
  questions, "where's my invoice", status inquiries). **5% are complex** — angry
  escalations, multi-issue threads, technical troubleshooting, anything mentioning
  cancellation or legal.
- Product data shows that **agents edit drafts heavily on the complex 5%**, and that
  heavy editing correlates strongly with agent dissatisfaction and, at the account
  level, with churn.
- Average request: ~2,400 input tokens (email thread + retrieved account context),
  ~350 output tokens.
- Budget: $15k/month, hard.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Step 1 — Check the arithmetic before you trust the table.**

Per request: 2,400 input + 350 output tokens. At 30M requests/month that's
72B input tokens and 10.5B output tokens.

- Model A: 72,000 × $3.00 + 10,500 × $15.00 = $216,000 + $157,500 = **$373,500**.
- The stated "$40k/month" is off by an order of magnitude.

**This discrepancy is the first thing to raise.** Either the volume, the token
counts, or the quoted costs are wrong — and in a real interview, catching it is
worth more than any architecture you propose afterwards. (Assume for the rest of the
scenario that the interviewer clarifies: the quoted monthly figures assume a *much*
lower volume tier, or heavy prefix caching, or that only a fraction of requests use
the model. Ask which.)

The general lesson: **quoted model costs are meaningless without your token
profile.** A model that's 5× cheaper per token can be more expensive per task if it
needs more context or produces longer outputs.

**Step 2 — Notice that "quality" is not one number.**

92% vs. 89% vs. 85% is an average over a traffic mix. What matters is quality *per
segment*:

- On the routine 95%, the gap between models is probably tiny — these are templated,
  low-ambiguity tasks. Model C might be at 93% of A's quality here.
- On the complex 5%, the gap is probably large, and it's exactly where errors are
  expensive (churn correlation).

**Ask for the segmented eval scores.** If the interviewer says you don't have them,
that's your first deliverable — you cannot make this decision on a blended average.

**Step 3 — Convert quality into money.**

The business consequence isn't "quality points." It's:
- agent editing time (a labor cost you can measure and price),
- agent satisfaction → attrition,
- account churn on badly-handled escalations.

Once you can say "a 4-point quality drop on complex tickets costs us X in editing
time plus Y in churn risk," the $15k ceiling becomes something you can argue with
evidence rather than accept.

**Step 4 — Change the shape of the system.**

95/5 traffic split with a large quality gap on the 5% is the textbook case for
routing. But routing is only one of several levers, and the right answer stacks
them:

1. Prefix caching (biggest input-token lever, zero quality risk)
2. Context reduction (do you really need 2,400 input tokens?)
3. Routing / cascading
4. Response caching for genuinely repeated situations
5. Batching for anything non-interactive

Note that levers 1 and 2 attack the *input token* cost, which is where 60%+ of the
spend is. Most candidates jump straight to lever 3.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**The strategy, in order of implementation:**

**1. Attack input tokens first (no quality risk, largest single win).**

2,400 input tokens per request is a design choice, not a fact. Decompose it:

```
system prompt + style guide + policy       ~800 tokens   (identical every request)
tool/format schema                          ~200 tokens   (identical every request)
retrieved account context                   ~700 tokens   (varies)
email thread                                ~700 tokens   (varies)
```

- **~1,000 tokens are identical on every request.** Prefix caching eliminates most
  of their cost and cuts prefill latency. Order the prompt stable-first to make this
  work. This alone is a ~40% input-cost reduction.
- **The retrieved account context is probably over-fetched.** Are all 700 tokens
  used? Retrieve more candidates, rerank, include fewer. Typical result: 700 → 300
  tokens with *equal or better* quality, because you removed distractors.
- **Thread truncation:** for a 14-message thread, do you need all of it, or the last
  3 messages plus a running summary? Summaries can be computed once and cached per
  thread rather than re-derived per request.

Realistic combined effect: input tokens per request from 2,400 → ~900 effective.
That's a >60% cost reduction *before touching the model choice*, and it improves
latency.

**2. Route, don't choose.**

```
incoming email
    │
    ├─ hard rules (never route down):
    │     mentions cancellation / legal / refund > threshold / VIP account
    │     sentiment strongly negative
    │     thread length > 6 messages
    │           └──► Model A
    │
    └─ Model C (small) drafts the reply
           │
           ├─ escalation check:
           │     · confidence/abstention flag in structured output
           │     · retrieval score below threshold
           │     · draft fails policy validation or contains a hedge phrase
           │     · thread contains an unresolved question the draft doesn't address
           │
           ├─ pass  ──► return draft (≈ 85–88% of traffic)
           └─ fail  ──► Model B or A ──► return draft
```

Why this shape:

- The hard rules capture most of the complex 5% *before* spending anything, using
  signals that are cheap and reliable (account tier, keywords, thread length,
  sentiment). Do not use a statistical router for a query mentioning legal action.
- The cascade catches what the rules miss, routing on the *draft* rather than the
  input — far more reliable for this task, because you can validate a draft
  (does it answer the question? does it cite a policy? does it contain
  placeholders?) in ways you cannot validate a question.
- The escalation check here is unusually strong compared to most cascades, because
  drafting is a task with checkable output structure. Exploit that.

**3. Exploit the human in the loop.**

This product has a reviewer on every request. That changes the economics
fundamentally, and most candidates miss it:

- **The cost of an imperfect draft is agent editing time, not a customer-visible
  failure.** This makes cheap models much more viable than in a fully-automated
  product.
- **You have free labeled data.** Every agent edit is a signal. Log the diff between
  draft and sent reply. Edit distance becomes your online quality metric — cheap,
  continuous, and far more representative than any offline eval. Within weeks you
  can measure per-model, per-segment quality from production directly.
- This data also gives you the training set for a learned router, and eventually
  for fine-tuning.

**4. Batching for the non-interactive slice.**

If any portion of the workload is not agent-waiting (overnight queue drain,
pre-drafting replies for the morning queue), run it through a batch API tier where
available — typically ~50% cheaper — with relaxed latency.

**Expected outcome:** input-token reduction (~60%) × routing (~80% of traffic on the
cheap model) compounds. The blended cost lands comfortably under the ceiling with
quality *higher* on the complex traffic than a flat Model B choice would deliver,
because the 5% that matters now gets Model A instead of B.

**5. Reframe the ceiling.**

Present to the CFO: cost per drafted reply, and cost per reply *sent without heavy
editing*. If Model A on the complex 5% prevents even a small amount of churn, its
marginal cost is trivially justified. The right conversation is not "stay under
$15k"; it's "here is the cost curve and here is the value curve, where do you want
to sit?" A senior engineer who can produce that curve changes the decision.

</details>

### Trade-offs

**Single mid-tier model (Model B everywhere).**
*Advantages:* simplest possible system; one model to evaluate, prompt, monitor, and
upgrade; predictable cost; no routing bugs; no cascade latency; a new engineer
understands it in an hour.
*Disadvantages:* pays mid-tier prices for the 95% of traffic that doesn't need it,
and delivers mid-tier quality on the 5% where quality is worth the most. Optimizes
the average and neglects both tails.
*Choose when:* volume is low enough that the savings don't justify the complexity,
the team is small, or the traffic genuinely is homogeneous. **This is a much more
defensible answer than candidates think** — at 100k requests/month rather than 30M,
it's clearly correct.

**Cascade (small → escalate).**
*Advantages:* best cost/quality frontier; escalation rate doubles as a live quality
signal; degrades gracefully.
*Disadvantages:* escalated requests pay twice and are slower — the p95 for the
hardest 15% becomes small-call + large-call, which may exceed the 2-second budget
exactly where users are least patient. Requires an escalation check that must be
maintained and re-validated with every prompt change.
*Choose when:* output is checkable and the traffic mix is skewed — as here.

**Input-side routing (classifier).**
*Advantages:* one model call, lowest latency, cheapest.
*Disadvantages:* predicts difficulty from the email alone, which is often not
predictable; drifts; a wrong route on an escalation is expensive.
*Choose when:* segments are obvious from metadata (account tier, channel, language)
— which is exactly why the *hard rules* in the design above are input-side and the
statistical part is output-side. Use each where it's strong.

**Fine-tune a small model on your data.**
*Advantages:* potentially frontier-adjacent quality on your specific task at small-
model prices; the human-review loop gives you an ideal training set.
*Disadvantages:* training and maintenance cost; a new artifact to version, evaluate,
and re-train as products change; you now own model drift. Rarely the *first* move.
*Choose when:* the task is narrow, stable, high-volume, and you have thousands of
high-quality examples — which, notably, this product will have in six months. Put it
on the roadmap, not in the first quarter.

### What Could Go Wrong?

- **The 95/5 split is measured wrong, or moves.** If "complex" grows to 20% (a big
  customer with a messy product onboards), your economics change overnight. Monitor
  the route distribution as a first-class metric and alert on drift.
- **Cascade latency busts the SLA.** Small model (0.8s) + escalation check + Model A
  (4.1s) ≈ 5s for escalated traffic, against a 2s p95 requirement. If escalation
  exceeds ~15% of traffic this breaks the SLA. Mitigations: speculative parallel
  execution for high-risk segments (pay double, save latency), or accept a
  segment-specific SLA negotiated with product.
- **Agents lose trust in the cheap model.** If agents notice drafts are worse for
  some tickets and can't tell which, they start editing everything heavily and the
  product's value evaporates — cost savings with zero labor savings is a net loss.
  Consider surfacing a confidence indicator so agents know when to read carefully.
- **Prefix cache invalidated by a style-guide edit.** The 800-token style guide is
  exactly the kind of thing a content team edits weekly. Every edit costs you the
  cache. Structure and govern it (see the latency incident in
  `production-incidents.md`).
- **Optimizing for the metric you can measure.** Edit distance is a great proxy but
  agents may accept a mediocre draft rather than rewrite it, especially under queue
  pressure. Low edit distance can mean "good draft" or "tired agent." Validate the
  proxy against sampled quality grading.
- **The savings are real and nobody notices the quality drop for six weeks**, because
  cost is on a dashboard and quality isn't. This is the default outcome unless you
  build the quality dashboard first.

### Metrics

*Unit economics*
- Cost per draft; cost per **sent** reply; cost per resolved ticket.
- Blended cost per request by route; marginal cost of the escalation path.
- Input vs. output token spend split (they need different optimizations).

*Quality*
- Segmented win-rate by traffic class (routine vs. complex) — never a blended number.
- **Agent edit distance** distribution, per route; share of drafts sent unedited.
- Escalation rate and escalation precision.
- Policy-violation rate in drafts (a hard failure, not a quality gradient).

*Business*
- Agent handling time per ticket (the actual labor saving being sold).
- Agent satisfaction / attrition.
- Account churn correlated with escalation-handling quality.
- Tickets per agent per hour — the number that justifies the whole product.

### Follow-up Questions

1. Your escalation rate is 14% and cost is at $11k/month. Product asks you to
   improve quality on the routine 95% instead of saving more money. Where do you
   spend the remaining $4k, and how do you decide?
2. Model B's price drops 60% overnight. Does your architecture change? What if
   Model C's price drops instead?
3. A customer on your top tier demands "always use the best model." How do you price
   that, and what does it do to your architecture?
4. Six months in, you have 4M agent-edited examples. Make the case for and against
   fine-tuning, with the numbers you'd need to decide.
5. The CFO asks you to cut another 40%. What goes, in order, and what do you tell
   them it costs?
6. How would this design change if there were **no human reviewer** — drafts sent
   automatically?

### What Separates Senior From Staff?

**Senior:** designs the routing and caching architecture, hits the budget, measures
quality per segment, and defends the design with numbers.

**Staff:**

- Catches the arithmetic error in the premise and reframes the whole discussion.
- Notices the human-in-the-loop changes the economics and turns agent edits into the
  company's quality measurement infrastructure — a capability that outlives this
  decision and serves every future model change.
- Reframes the CFO conversation from "cost ceiling" to "cost curve vs. value curve,"
  and brings the churn correlation to that conversation, changing the constraint
  rather than optimizing within it.
- Sequences the fine-tuning bet correctly: not now, but sets up the data collection
  today so it's available in two quarters. Staff engineers create options.
- Writes down the conditions under which this architecture should be dismantled
  (e.g. "if the routine/complex split moves past 80/20, collapse to a single mid
  model and revisit"), so the next engineer doesn't inherit an unexamined design.

### Interviewer Notes

*What you're evaluating:*

- **Does the candidate compute anything?** The arithmetic discrepancy is deliberate.
  Candidates who accept the monthly numbers without checking are telling you they
  don't validate inputs.
- **Do they ask for segmented quality?** A blended 92 vs. 89 is not decision-grade
  data and a strong candidate says so within the first two minutes.
- **Do they attack input tokens before model choice?** Very few do, and it's the
  biggest lever. This is a strong differentiator between people who've run these
  systems and people who've read about them.
- **Do they notice the human reviewer?** It's stated plainly in the setup and it
  changes everything — both the risk tolerance and the availability of labels.
  Candidates who ignore it are not reading the business.
- **Do they compute the cascade's latency consequence?** Many propose cascading and
  never check it against the stated 2-second requirement.

*Red flags:* picking a model and defending it; treating quality as a scalar;
proposing fine-tuning as the first move; no mention of measurement.

*Green flags:* asks for the token profile; asks whether escalations correlate with
account value; proposes to ship the cheapest safe lever first; states explicitly
what they'd measure before and after.

---

## 2. The Free Tier That Ate the Margin

### Situation

A B2C productivity app added an AI assistant nine months ago. Pricing:

- **Free tier:** 20 AI actions/month. 2.1M free users, ~340k monthly active on AI.
- **Pro tier:** $12/month, "unlimited" AI. 84,000 subscribers.
- **Team tier:** $25/seat/month. 11,000 seats.

AI infrastructure spend is **$430k/month** and growing 11% MoM. Total subscription
revenue is $1.29M/month. Gross margin on the AI feature has gone from comfortable
to alarming, and the board has noticed.

Breakdown of spend:

```
free tier        $147k/month  (34%)   — 0 revenue
pro tier         $221k/month  (51%)
team tier         $62k/month  (15%)
```

Within Pro, usage is extremely skewed: the **top 3% of Pro users generate 61% of Pro
tier AI spend.** The heaviest 200 users each cost more than $190/month against $12
of revenue.

### Your Task

Fix the unit economics without gutting the product. You own the engineering side;
pricing changes require product and finance buy-in, which you can request but not
decide.

### Constraints

- Churn is highly sensitive to feature removal; a previous downgrade of the free
  tier caused a measurable spike.
- The free tier is the primary acquisition channel; free→Pro conversion is 4.1%,
  and users who use the AI feature convert at **7.8%** vs. 2.2% for those who don't.
- Team tier customers have contractual "unlimited" language.
- Engineering: you plus two.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**First: stop treating this as one problem.** There are three distinct economic
situations here and they need different answers:

- **Free tier ($147k, no revenue):** this is a *marketing spend*, not a cost overrun.
  The question is not "how do we cut it" but "is a customer-acquisition cost of
  $147k/month producing enough conversions?" Do the math: 340k AI-active free users
  × 7.8% conversion = ~26,500 conversions attributable... except conversion is
  measured over a period, not monthly, and correlation isn't causation. But
  directionally: if the AI feature is lifting conversion from 2.2% to 7.8%, $147k/mo
  buys a substantial number of $12/mo subscribers. **The free tier might be
  underspending, not overspending.**
- **Pro tier ($221k):** this is a classic unlimited-plan abuse curve. 3% of users
  driving 61% of cost is not "power users," it's a pricing design failure meeting an
  automation-friendly product. Investigate what those 200 users are actually doing
  before assuming anything.
- **Team tier ($62k against $275k revenue):** healthy. Leave it alone.

**Second: characterize the heavy users before you throttle them.** The distinction
matters enormously:
- Legitimate power users (a writer running 400 drafts/month) — these are your
  advocates; throttling them is expensive in ways the spreadsheet won't show.
- Automated/scripted use — someone driving your API through the UI, or reselling.
- A product bug — a retry loop, an auto-refresh, a feature that fires the model on
  every keystroke. **Check this first; it's free money if true.**

**Third: separate cost-per-action reduction from action-count reduction.** They're
independent levers:
- Reducing cost per action (caching, routing, context reduction) is invisible to
  users and should be maxed out before anything user-visible.
- Reducing action count (rate limits, pricing) is visible and costs churn.

Do all of the first before any of the second. A 55% cost-per-action reduction turns
$430k into $195k with zero product change, which may end the conversation entirely.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Phase 0 (week 1): find the bug.** Pull the top 200 users' event traces. In
practice, a distribution this skewed is usually 40% legitimate power use, 30%
automation, and 30% *a product defect*. Common defects: the assistant re-runs on
document autosave; a failed request retries client-side without backoff; a "regenerate"
button double-fires; a background feature (auto-summarize on open) runs on every
document view. Fixing one of these can remove 20% of spend in a week with zero
product impact. **Always look here first.**

**Phase 1 (weeks 2–8): reduce cost per action. No user-visible change.**

Apply the standard stack, ordered by (savings × safety):

1. **Prefix caching** on the shared system prompt and tool schemas.
2. **Response caching** for genuinely repeated, impersonal actions (the app almost
   certainly has "summarize", "fix grammar", "translate" actions where identical
   input recurs — especially in the free tier where users experiment with the same
   sample text).
3. **Routing:** the vast majority of "fix grammar" and "shorten this" actions do not
   need a frontier model. Route by *action type* — this app has an enormous
   advantage over a chat product: the user explicitly tells you what they want by
   clicking a button. Use it. Action-type routing is deterministic, auditable, and
   requires no classifier.
4. **Context discipline:** does "summarize this paragraph" send the whole document?
   Very often yes, because it was easier. Scope context to the action.
5. **Batch tier** for anything asynchronous (background indexing, scheduled digests).

Realistic outcome: 45–65% reduction in cost per action. $430k → ~$180k.

**Phase 2 (weeks 6–12): tiered quality by tier — deliberately.**

This is a legitimate product decision, not a trick, *if* it's honest:

- Free tier: small model, aggressive caching, lower context limits. Quality is good
  enough to demonstrate value — which is the free tier's only job.
- Pro: mid-tier by default, frontier on the actions that need it.
- Team: frontier, priority routing, higher limits.

The rule: **make the tiers differ in ways users can perceive as value, not as
degradation.** "Pro gets longer documents and faster responses" is a feature. "Free
gets a worse model" is churn if discovered and framed that way.

**Phase 3 (needs product/finance): fix the pricing design.**

Present, don't decide. Bring:

- The cost-per-user distribution (a histogram, not an average — the average hides
  everything).
- The margin curve: at what usage level does a $12 Pro user become unprofitable?
  (After Phase 1, probably around 800 actions/month rather than 90.)
- Options with modeled churn/revenue impact:
  - **Fair-use policy** with a soft ceiling: unlimited until a high threshold that
    99.7% of users never hit, then throttled speed (not blocked). Preserves the
    "unlimited" promise in spirit and in most contracts.
  - **A usage-based Pro+ tier** for genuine power users, priced above cost.
  - **Action-weighted quotas:** a "summarize" costs 1 credit, an "agentic research
    task" costs 20. Makes cost legible to users and aligns their behavior with
    yours. This is usually the best long-run answer.
- The recommendation, with your reasoning, and the explicit statement that the top
  200 users are 200 people — a personal outreach, not a policy change, may be the
  cheapest fix.

**What not to do:** don't unilaterally throttle Pro users. It's a pricing decision
with revenue and legal implications, and engineering making it quietly is how
companies end up with a class of angry users and a surprised CFO.

**On the free tier:** bring the acquisition math to the table and argue that the
$147k is CAC, not COGS, and should be evaluated against conversion — possibly
increased, not cut, if the marginal conversion economics are good. This reframe is
the highest-value thing an engineer can contribute to this conversation.

</details>

### Trade-offs

**Cut cost per action only (no product change).**
*Advantages:* invisible to users, no churn, no pricing negotiation, fastest.
*Disadvantages:* has a floor; growth at 11% MoM will re-consume the savings in about
seven months. Buys time, doesn't fix the structure.

**Rate limits / quotas.**
*Advantages:* directly caps the tail; predictable spend; simple to implement and
explain.
*Disadvantages:* hits your most engaged users — the ones who evangelize the product;
"unlimited" reversals damage trust disproportionately; support burden.

**Tiered model quality.**
*Advantages:* aligns cost with revenue; creates a real upgrade incentive; scales
naturally.
*Disadvantages:* a quality difference users can detect but can't attribute feels
like a broken product, not a tier; complicates evaluation (you now maintain quality
bars for three configurations); risky if the free tier's quality drops below the
"this is impressive" threshold that drives conversion.

**Usage-based pricing.**
*Advantages:* the only structurally sound answer — cost and revenue scale together;
aligns incentives; makes heavy users profitable instead of dangerous.
*Disadvantages:* consumer users hate metered pricing; conversion drops when pricing
is uncertain; a large migration and communication project; may not fit the
competitive landscape.

### What Could Go Wrong?

- **Throttling your evangelists.** The 200 heaviest users likely include your most
  vocal advocates and several people who make the product's demos. Segment by
  influence before applying policy.
- **The free tier quality drop kills conversion.** You save $60k/month and lose more
  than that in subscriptions, with a two-month reporting lag that hides the
  causation. Any free-tier change must be A/B tested on conversion, not just cost.
- **Caching leaks across users** in a consumer product with personal documents. Same
  discipline as any multi-tenant cache; the free tier's shared sample content is
  exactly where a careless key design gets exposed.
- **Contractual exposure on Team tier "unlimited."** Ask legal before touching it.
- **Growth outruns your savings.** At 11% MoM, a one-time 55% cut is consumed in
  ~7 months. Anything that isn't structural is a delay, and you should say so
  explicitly when you present it, or you'll be asked to do this again in Q3 with the
  easy levers already spent.
- **Metrics stay in aggregate.** If you don't build per-user cost attribution, you
  cannot do any of this. Build that in week 1.

### Metrics

Cost per active user per tier (and the *distribution*, not the mean); cost per AI
action by action type; gross margin per tier; free→Pro conversion segmented by AI
usage (the number that determines whether free-tier spend is good or bad); action
volume per user percentile; cache hit rate by action type; churn by usage decile
(to see whether heavy users are also loyal); and the projected month at which
current growth re-crosses the budget.

### Follow-up Questions

1. Your top user costs $340/month and is a well-known creator with 200k followers
   who mentions your product regularly. What do you do?
2. Free tier cost is cut 60% but conversion drops from 7.8% to 6.1%. Was that a good
   trade? What data do you need to answer?
3. Marketing wants to *increase* free-tier AI limits to drive acquisition. Model the
   decision. What would make you support it?
4. Growth continues at 11% MoM. Sketch the 18-month cost trajectory under your plan
   and say when you'd need to act again.
5. A competitor launches an unlimited AI plan at $9/month. What's your response, and
   is it an engineering response at all?

### What Separates Senior From Staff?

**Senior:** finds the bug, executes the cost-per-action reduction expertly, builds
per-user cost attribution, presents options to product.

**Staff:** reframes the free tier as CAC rather than COGS and changes what the board
is even asking. They recognize that a 55% one-time cut against 11% MoM growth is a
seven-month reprieve and say so up front rather than declaring victory. They bring
the cost-per-user *distribution* to the pricing conversation — the artifact that
lets finance make a good decision — and they propose action-weighted credits as the
structural fix, knowing it's a year-long cross-functional project they'll need to
sponsor. They also handle the top-200 problem as 200 relationships rather than a
policy.

### Interviewer Notes

Watch for whether the candidate immediately separates the three tiers into three
different economic problems — it's the key structural insight and it should come
fast. Then watch for the free-tier reframe (CAC not COGS); it's rare and it's the
strongest signal in this scenario. Candidates who propose rate-limiting Pro users in
the first two minutes are optimizing a spreadsheet without a product model.
Also probe: do they check for a product bug before assuming user behavior? The
skewed distribution has a mundane explanation more often than not.

---

## 3. Cutting 60% Without Telling Users

### Situation

You have been asked to reduce AI infrastructure cost by 60% in one quarter for a
document-intelligence product. The explicit constraint from leadership: **"users
must not notice."**

Current architecture per document processed:

```
1. classify document type          small model,   ~800 in /  ~20 out
2. extract entities (per page)     frontier,    ~2,200 in / ~600 out   × N pages
3. validate extraction             frontier,    ~2,800 in / ~400 out   × N pages
4. summarize document              frontier,   ~12,000 in / ~800 out
5. generate compliance flags       frontier,    ~9,000 in / ~500 out
```

Average document: 22 pages. 90,000 documents/month.
Current cost: **$268k/month**.

### Your Task

Find 60%. Show your work. State explicitly what quality risk each change carries and
how you'd verify it.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Build the cost model first.** Per document (22 pages), using rough frontier pricing
of $3/1M in and $15/1M out:

| Step | Input tokens | Output tokens | Input $ | Output $ | Total $ |
|---|---|---|---|---|---|
| 1. classify | 800 | 20 | ~0.0001 | ~0.0000 | ~$0.001 |
| 2. extract × 22 | 48,400 | 13,200 | $0.145 | $0.198 | **$0.343** |
| 3. validate × 22 | 61,600 | 8,800 | $0.185 | $0.132 | **$0.317** |
| 4. summarize | 12,000 | 800 | $0.036 | $0.012 | $0.048 |
| 5. flags | 9,000 | 500 | $0.027 | $0.008 | $0.035 |
| **Total** | **131,800** | **23,320** | **$0.393** | **$0.350** | **~$0.74** |

× 90,000 documents ≈ **$67k/month**... which doesn't match $268k. So either volume,
page count, or pricing differs — **ask**. (Perhaps there are retries, or multiple
passes, or the frontier model is more expensive than assumed.) Again: *build the
model, notice when it disagrees with the reported number, and chase the discrepancy.*
The gap between your model and the invoice is usually where the waste is hiding.

**Then rank by share of spend.** Steps 2 and 3 are ~89% of cost. Any optimization of
steps 1, 4, or 5 is rounding error. This is the most common mistake in cost work:
optimizing the thing that's easy to optimize rather than the thing that's expensive.

**Then ask the design question about step 3.** "Validate extraction" is a frontier
model call costing nearly as much as the extraction itself. Ask: what does it catch?
If it catches schema errors, a schema validator does it for free. If it catches
hallucinated entities, a string-match against the source page does most of it for
free. **A frontier model used as a validator is almost always a design smell** —
you're paying generation prices for a checking task, and a checking task usually has
a deterministic implementation.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Ranked by savings, with the risk each carries:**

**1. Replace step 3 (validation) with deterministic checks + selective model review.
Saves ~40% of total. Risk: low, verifiable.**

Decompose what validation actually catches by sampling 500 validation runs and
classifying the errors it finds:
- Schema/type errors → JSON Schema validation. Free. Catches these perfectly.
- Entities not present in source text → exact/fuzzy string match against the page.
  Free, deterministic, catches the dominant hallucination class.
- Semantic errors (right string, wrong field; misattributed party) → genuinely needs
  a model, but only for the ~10–15% of extractions that are ambiguous.

New design: run free checks on 100%; route only failures and low-confidence
extractions to a model validator, and use a **mid-tier** model for it. Effective cost
of step 3 drops by 85%+.

*Verification:* run old and new validation side by side on 5,000 documents; measure
agreement and specifically the false-negative rate (errors the new path misses).
Ship only if false negatives are within an agreed bound.

**2. Batch pages in step 2 instead of one call per page. Saves ~25% of step 2.
Risk: medium, needs evaluation.**

22 separate calls each re-send the ~1,400-token instruction/schema prefix — that's
~30,800 tokens of pure repetition per document. Options:
- **Prefix caching** — zero risk, immediate, cuts most of that repetition.
- **Batch 4–6 pages per call** — cuts per-call overhead further and lets the model
  see cross-page context (tables spanning pages, entities defined on page 1 and
  referenced on page 14), which may *improve* accuracy. But it increases output
  length per call and risks truncation, and errors now affect a batch rather than a
  page.

Do prefix caching first (free), then evaluate batching with a real accuracy
comparison. Batch size is an empirical question, not a guess.

**3. Route step 2 by document type. Saves ~30% of step 2. Risk: medium.**

Step 1 already classifies the document. Use it. Structured forms (invoices,
standardized filings) extract reliably with a small model; free-form contracts and
handwritten annotations need the frontier model. Measure per-type accuracy for each
model, set the routing table from data, and re-measure quarterly.

*Verification:* per-document-type accuracy gates. Never route a type down without an
accuracy measurement for that type specifically.

**4. Skip work that isn't needed. Saves 5–15%. Risk: low.**

- Are all 22 pages substantive? Blank pages, signature pages, and exhibit dividers
  need no extraction. A cheap page classifier (or even a heuristic on text density)
  removes them.
- Are steps 4 and 5 always used? Instrument whether users ever open the summary or
  the compliance flags. Products accumulate features nobody reads. Compute
  on-demand rather than eagerly for anything with low read rates — this can be the
  easiest large saving in the entire system and it requires no model work at all.

**5. Deduplicate. Saves whatever the duplicate rate is. Risk: none.**

Document-intelligence pipelines reprocess the same documents constantly (re-uploads,
re-runs, versioned copies). Content-hash every page and cache extraction results
keyed on `(page_hash, extraction_prompt_version, model_version)`. In many real
pipelines this alone is 10–20%.

**Total:** ~40% (validation) + ~15% (batching/prefix on extraction) + ~10% (routing)
+ ~8% (skipping) + dedupe. Comfortably past 60%, with the largest single win being a
*design correction* rather than a model downgrade.

**The verification plan is the deliverable, not the savings.** Every change ships
behind:
- a per-step accuracy gate on a labeled set of ≥2,000 documents stratified by type,
- a canary at 5% traffic with automated comparison against the control,
- a rollback path per change (each change independently revertible — do not ship
  them as one release),
- ongoing production sampling with human grading at a fixed rate.

Because "users must not notice" is the constraint, the measurement is the work.
</details>

### Trade-offs

**Fewer model calls (batching, merging steps).**
*Advantages:* attacks per-call overhead, which is often the dominant waste; can
improve quality via shared context.
*Disadvantages:* larger blast radius per failure; truncation risk; harder to attribute
errors to a step; less parallelism, so latency may rise.

**Cheaper models per step.**
*Advantages:* simple, predictable, easily reverted.
*Disadvantages:* quality risk is direct and needs per-segment measurement; creates a
multi-model system to evaluate and maintain.

**Replacing model calls with deterministic code.**
*Advantages:* free, fast, exactly reproducible, testable. **Usually the best
available option and consistently under-used.**
*Disadvantages:* only works where the task is actually deterministic; brittle to
format changes; requires someone to understand the domain well enough to write the
rules.

**Doing less work (lazy computation, skipping pages).**
*Advantages:* free savings; often improves latency too.
*Disadvantages:* requires knowing what's actually used, which requires
instrumentation nobody built; risk of a user needing the thing you made lazy.

### What Could Go Wrong?

- **"Users must not notice" is unfalsifiable without measurement,** and the failure
  mode is silent: accuracy degrades 3%, no complaints arrive for two months, then a
  customer audit finds systematic extraction errors in their contracts. Sampling and
  human grading are not optional here.
- **Compounding degradation.** Five changes each costing 1% accuracy compound to
  ~5%, and each was individually "within tolerance." Measure the *stack*, not just
  each change.
- **The deterministic validator has a blind spot** that the model validator covered,
  and it's exactly the class of error that matters legally. Sample the disagreements,
  don't just count them.
- **Dedupe cache returns stale results** after a prompt or model version change.
  Version the key or you'll serve last quarter's extraction logic indefinitely.
- **Batching breaks page-level attribution.** If the product shows "extracted from
  page 7," batched extraction must still emit page provenance, or you break a
  user-visible feature while "not changing anything users notice."
- **You hit 60% and growth resumes.** Same as before: name the durability of the
  savings when you present them.

### Metrics

Cost per document and per page, by document type; per-step cost share (to know where
the next 10% is); extraction accuracy (precision and recall per entity type) against
a labeled set; validation false-negative rate; deduplication hit rate; share of
documents where a downstream user actually opened the summary/flags; canary
comparison deltas per change; and a standing production sample graded by humans
weekly — the only thing that actually tests "users must not notice."

### Follow-up Questions

1. Your deterministic validator has a 2% false-negative rate against the model
   validator. Is that acceptable? What would you need to know to answer?
2. You hit 58%, not 60%. Where does the last 2% come from, and would you take it?
3. Six weeks after shipping, a customer finds a systematic extraction error. Walk
   through the investigation. How do you know which change caused it?
4. Leadership is pleased and asks for another 40% next quarter. What's your answer?
5. How would you have designed this pipeline originally to avoid the step-3 problem?

### What Separates Senior From Staff?

**Senior:** builds the cost model, finds the savings, ships them safely with per-step
gates.

**Staff:** identifies that using a frontier model as a validator was an *architectural
error*, not just an expensive choice — and asks why it was made, because the same
reasoning is probably in three other pipelines. They question whether steps 4 and 5
should exist at all by looking at product usage data, which is a product question an
engineer is well-placed to raise. They also refuse the framing "users must not
notice" as unmeasurable and replace it with a specific, agreed quality bar
("extraction F1 must not drop more than 0.5 points on any document type, verified
weekly on a 500-document human-graded sample") — converting an unfalsifiable
constraint into an engineering contract. That reframe is the staff-level move.

### Interviewer Notes

Give the candidate the table and see if they build the per-step cost model
unprompted. The 89%-of-spend concentration in steps 2 and 3 should be identified in
the first few minutes; candidates who start optimizing step 4 are not prioritizing.
The single best signal is whether they question step 3's existence — "why is a
frontier model validating another frontier model's output?" is the question that
unlocks the largest saving, and it comes from design instinct rather than
optimization technique. Also watch whether they insist on a measurable definition of
"users must not notice"; accepting it as stated is a senior-level miss.

---

## 4. Fine-Tune, Prompt, or Route?

### Situation

A logistics company classifies inbound customer emails into 40 categories, extracts
5 structured fields, and routes to the right team. Volume: **8M emails/month**.

Current: a frontier model with a 6,000-token prompt (category definitions,
examples, edge-case rules accumulated over 18 months). Accuracy: **94.2%**. Cost:
**$142k/month**. p95 latency: 3.4s (fine — this is asynchronous).

You have **2.4M labeled examples** from 18 months of production plus human
corrections.

Three proposals are on the table:

- **A:** Fine-tune a small open model on the labeled data, self-host.
- **B:** Prompt-optimize and use a mid-tier model with a compressed prompt.
- **C:** Route — cheap model for the easy 80%, frontier for the rest.

### Your Task

Recommend one, or a sequence. Justify with the specific properties of this task.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

This is one of the few scenarios where **fine-tuning is genuinely the right answer**,
and knowing why requires knowing what fine-tuning is good at.

The task has every property that favors fine-tuning:

- **Narrow and well-defined.** 40 fixed categories, 5 fixed fields. Not open-ended
  generation.
- **Enormous labeled dataset** (2.4M) that came free from production.
- **High, stable volume** — 8M/month amortizes training cost immediately.
- **Latency-insensitive** — you can self-host without a latency risk.
- **The prompt is 6,000 tokens of accumulated rules**, which is the clearest possible
  signal that the task knowledge should live in weights rather than in context. You
  are paying 6,000 input tokens × 8M requests = 48B tokens/month to re-explain the
  task on every single email.

That last point is the key insight: **a long, rule-laden prompt on a high-volume
narrow task is a fine-tuning signal.** The prompt is doing the job of training data,
inefficiently, 8 million times a month.

Counter-considerations to hold:
- Categories change (new product lines, reorganizations) → retraining cadence.
- You now own a model artifact, a training pipeline, and serving infrastructure.
- The 5.8% error tail may be genuinely hard and not improved by fine-tuning.
- Fine-tuned models are worse at handling inputs unlike their training distribution
  — and email distributions shift.

So the answer is likely "fine-tune, *and* keep a frontier fallback," which is a
routing architecture anyway. The proposals are not mutually exclusive, and saying so
is the strong answer.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Recommendation: B first, then A with a C-shaped fallback. Not A alone.**

**Sequence and reasoning:**

**Step 1 (weeks 1–3): prompt compression + mid-tier model. Cheap, fast, informative.**

6,000 tokens of accumulated rules almost certainly contains substantial dead weight:
rules for cases that no longer occur, redundant examples, verbose category
definitions. Compress systematically — measure per-rule contribution by ablation on
a held-out set. Typical outcome: 6,000 → 2,000 tokens with equal accuracy.

Then test a mid-tier model with the compressed prompt. Cost drops sharply. If
accuracy holds at ~93–94%, you've captured most of the value in three weeks with no
new infrastructure.

**Crucially, this step tells you what fine-tuning must beat.** Establishing the
cheap baseline before the expensive project is basic discipline and most teams skip
it, then attribute the fine-tune's gains to fine-tuning when half came from prompt
cleanup.

**Step 2 (weeks 4–12): fine-tune a small model.**

- Dataset: sample from the 2.4M, stratified by category (the tail categories will be
  rare — a 40-class problem always has a long tail, and naive sampling gives you a
  model that's excellent at the top 8 categories and useless at the bottom 15).
- Hold out a temporally-recent test set, not a random split. Random splits leak: near-
  duplicate emails from the same customer thread end up on both sides and inflate
  accuracy. **Split by time and by customer.**
- Include the human corrections as a distinct, higher-weight signal — they're your
  hardest examples.
- Target: match or beat the 94.2% baseline at a fraction of the cost. Expect a
  well-tuned small model to reach or exceed frontier accuracy on a narrow
  classification task with 2.4M examples. This is the regime where fine-tuning wins
  decisively.

**Step 3: keep the frontier model in the architecture as an escape hatch.**

```
email ──► fine-tuned small model (self-hosted)
              │
              ├─ high confidence (calibrated) ──► route
              │
              └─ low confidence / unknown-pattern / new category
                        └──► frontier model with the full prompt
                                    └──► route + log as a training example
```

Why the fallback is non-negotiable:
- New categories and novel email patterns will appear before the next retrain.
- A fine-tuned classifier's confidence is calibratable (unlike a chat model's
  self-reported confidence), so the routing signal here is *good* — use it.
- Every fallback invocation is a labeled hard example for the next training run.
  The architecture improves itself.

**Step 4: operationalize the model lifecycle.** This is what you're really signing
up for and it should be in the proposal:
- Monthly retraining on fresh data, automated.
- A frozen benchmark plus a rolling recent benchmark; block deploys on either
  regressing.
- Drift monitoring: input distribution, per-category volume, confidence
  distribution, fallback rate.
- A documented rollback to the previous model version, and to the frontier-only path
  as a break-glass.

**Expected economics:** self-hosted small model at 8M requests/month with modest
GPU footprint is typically 10–30× cheaper per request than a frontier API. $142k →
plausibly $8–15k including infrastructure and the frontier fallback on ~10% of
traffic. But note that a portion of that gain came from Step 1 and would have been
available without the fine-tune — report it honestly.

</details>

### Trade-offs

**Fine-tuning.**
*Advantages:* task knowledge in weights, so no per-request prompt tax; best
accuracy-per-dollar on narrow high-volume tasks; latency drops; calibratable
confidence; no per-token vendor pricing exposure.
*Disadvantages:* you own a training pipeline, a model registry, and serving
infrastructure forever; retraining cadence is a permanent operational commitment;
brittle to distribution shift; slow to respond to a new category (hours-to-days vs.
a prompt edit in minutes); requires ML engineering skills the team may not have.
*Choose when:* narrow task, abundant labels, high stable volume, latency-tolerant —
i.e. exactly here.

**Prompt optimization + cheaper model.**
*Advantages:* days not months; no new infrastructure; instantly reversible; changes
to categories are a text edit; keeps vendor flexibility.
*Disadvantages:* has a ceiling; still pays the per-request prompt tax; accuracy on
the hardest classes may not hold; prompt entropy re-accumulates over time.
*Choose when:* volume is moderate, requirements change often, or the team has no ML
capacity. **Always do this first regardless**, because it establishes the baseline.

**Routing only.**
*Advantages:* keeps frontier quality where needed; no training; incremental.
*Disadvantages:* still pays frontier prices on the routed-up fraction and mid prices
elsewhere; the 6,000-token prompt tax remains on every request; doesn't exploit the
2.4M labels at all — which, in this scenario, is leaving the single largest asset
unused.
*Choose when:* labels are scarce or the task is open-ended.

### What Could Go Wrong?

- **Random train/test split inflates accuracy.** Email threads produce near-duplicates.
  A model that memorizes them scores 97% offline and 91% in production. Split by time
  *and* customer.
- **Label noise.** 2.4M production labels include the routing decisions of tired
  humans. If 4% are wrong, your ceiling is 96%. Audit a sample before training and
  know your label quality — it bounds everything.
- **Category drift.** The business adds "sustainability inquiries" as category 41 and
  your model has never seen it. Without the fallback and a fast retrain path, these
  silently misroute for a month.
- **Confidence miscalibration after retraining.** Your routing threshold was tuned on
  the old model. Recalibrate on every release, or the fallback rate swings wildly.
- **Self-hosting cost surprises.** GPU instances are billed for uptime, not usage. At
  8M/month with diurnal traffic, low-utilization hours can erase the savings. Model
  the *utilization*, not the peak throughput.
- **Nobody owns the model after you.** Fine-tuning creates a permanent maintenance
  obligation. If the team can't sustain monthly retraining, the model rots and
  accuracy quietly declines. Be honest about this before you propose it.

### Metrics

Accuracy overall and **per category** (the tail categories are where fine-tuning
fails and where the business impact often is); field-level extraction precision/recall;
confidence calibration (reliability diagram, ECE); fallback rate and fallback
accuracy; misroute cost (what does a wrong routing decision cost in handling time?);
cost per email including amortized training and idle GPU time; time-to-support a new
category; and model-version-tagged accuracy over time to detect rot.

### Follow-up Questions

1. Your fine-tuned model hits 94.0% — slightly *below* the 94.2% baseline — at 8% of
   the cost. Ship it?
2. The business adds 12 new categories next quarter. Walk through your response, and
   say how long each option takes.
3. How do you detect that your fine-tuned model has degraded, given that production
   labels come from the humans it's supposed to be helping?
4. Would your answer change at 200k emails/month instead of 8M? At what volume does
   fine-tuning stop making sense?
5. Your fine-tuned model is 96% accurate overall but 61% on the three rarest
   categories, which happen to be the highest-value ones. What do you do?

### What Separates Senior From Staff?

**Senior:** picks fine-tuning for the right reasons, designs the fallback, sets up
the retraining pipeline and the eval gates.

**Staff:** insists on the cheap baseline first so the fine-tune's contribution is
measurable rather than assumed. They surface the *ongoing* cost of owning a model —
headcount, retraining cadence, on-call — and get explicit organizational commitment
before starting, because a fine-tuned model with no owner is worse than an API call.
They notice the label-quality ceiling and audit it before anyone spends a month
training against noise. And they ask the question nobody asked: *is 94.2% even the
right target?* What does a misroute cost, and would 98% on the three high-value
categories be worth more than 96% overall? That reframing — optimizing the business
objective rather than the reported metric — is the staff move.

### Interviewer Notes

This scenario rewards candidates who know *when* fine-tuning is correct, which is
rarer than knowing how. The signals: do they identify the 6,000-token prompt as the
tell? Do they insist on establishing a cheap baseline first? Do they raise train/test
leakage from email threads unprompted? Do they treat the fallback as mandatory rather
than optional? And do they name the operational commitment honestly, or sell
fine-tuning as free savings? A candidate who says "fine-tune" in the first thirty
seconds without asking about label quality or category stability is pattern-matching.

---

## 5. When Quality Is Worth Any Price

### Situation

A medical-triage support tool used by nurse practitioners. It reviews a patient's
intake notes and prior records and suggests possible conditions to consider, flags
red-flag symptoms requiring immediate escalation, and drafts a documentation summary.

A clinician reviews everything; the tool never acts autonomously.

Current: frontier model, extensive retrieval over clinical guidelines, multi-step
verification. Cost: **$61 per encounter**. Volume: 14,000 encounters/month =
**$854k/month**.

The CFO points out that a competitor's tool costs $4 per encounter. Your CEO asks
you to "get to $20."

Clinical safety has veto power over any change.

### Your Task

Respond to the request.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

The temptation is to either capitulate ("here's how to get to $20") or refuse
("safety, no"). Both are weak. The strong response reframes:

**What is a missed red flag worth?** If the tool fails to flag a symptom requiring
immediate escalation and a patient is harmed, the cost is measured in patient
outcomes, malpractice exposure, regulatory action, and the end of the product. The
relevant comparison is not $61 vs. $4 — it's $61 vs. the expected cost of the errors
that a cheaper system would produce.

Then: **not all of the $61 carries the same risk.**

- Red-flag detection: safety-critical. Do not degrade. Arguably should be
  *more* expensive — run it twice with different models and take the union.
- Differential suggestions: clinically important but reviewed by a clinician who
  applies judgment. Some quality latitude exists.
- Documentation summary: administratively useful, clinically low-risk, and probably
  the largest token consumer (long records, long output).

Decompose by risk tier and optimize only where risk is low. This is the same
decomposition logic as every other scenario in this file — the difference is that
here, the risk tiers are *asymmetric to the point of being non-negotiable*, and
recognizing that is the answer.

Also worth investigating: **why is the competitor $4?** Possibilities: they do less;
they're subsidizing; they're using a cheaper model and their error rate is higher and
nobody has measured it; they process shorter records; the number is marketing. A $4
tool and a $61 tool that do the same thing with the same error rate is implausible.
Ask what their false-negative rate on red flags is. Usually nobody knows, and that's
the point.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**The response has three parts: reframe, decompose, deliver.**

**1. Reframe the comparison.**

Bring to the CEO:
- The current false-negative rate on red-flag detection, with a confidence interval,
  measured against clinician-adjudicated ground truth.
- The estimated rate for a cheap-model configuration, measured — not assumed — on the
  same set.
- The expected cost of a missed red flag: incident cost, liability exposure,
  regulatory consequence, contract termination risk with health-system customers.
- The observation that the competitor's price tells you nothing without their error
  rate, and a concrete suggestion: ask a customer who evaluated both for their
  comparison data, or run their tool against your labeled set if procurement allows.

The point is not to win the argument. It's to make the decision explicit and
attributable: *this is a clinical risk decision, not an engineering cost decision,
and it needs the clinical safety officer and probably the board.*

**2. Decompose by risk tier and optimize the low-risk portion aggressively.**

Estimated split of the $61:

```
red-flag detection          ~$14   safety-critical      DO NOT DEGRADE
differential suggestions    ~$19   clinically reviewed  optimize carefully
documentation summary       ~$21   low clinical risk    optimize aggressively
retrieval + orchestration    ~$7   no clinical risk     optimize freely
```

- **Documentation summary (~$21):** long records, long outputs. Chunk and summarize
  hierarchically with a mid-tier model, cache per-record summaries across encounters
  (a patient's history doesn't change between visits — this is a big win in a
  longitudinal product), and generate on demand rather than eagerly. Realistic: $21
  → $5.
- **Retrieval and orchestration ($7):** prefix caching, reranking to reduce context,
  guideline snippets cached and versioned. $7 → $2.
- **Differential suggestions ($19):** careful. A cascade is possible — mid-tier model
  generates, frontier model reviews — but the review step must be validated to catch
  what the direct frontier path catches. Only ship with clinical sign-off on a
  measured non-inferiority result. Maybe $19 → $12, maybe not at all.
- **Red-flag detection ($14):** leave it, or increase it. Consider ensembling two
  models and taking the union of flags, accepting more false positives (a clinician
  dismisses a false positive in seconds; a false negative can be catastrophic). This
  might cost *more*, and that should be proposed proactively — it's the single
  highest-value change available and it goes the opposite direction from the ask.

**Realistic landing: $61 → ~$33**, with red-flag safety held constant or improved.
Not $20. Say so clearly, with the reasoning, rather than hitting $20 by degrading
something and hoping.

**3. Change the pricing conversation instead.**

$33/encounter against what value? If the tool saves 25 minutes of clinician
documentation time per encounter, the labor value alone dwarfs $61. The problem may
be that the product is *priced* wrong, not that it costs too much. An engineer who
brings the value-per-encounter analysis to this conversation is far more useful than
one who delivers a 60% cut.

**What you refuse, explicitly and in writing:** any change to red-flag detection
without a measured non-inferiority study reviewed by clinical safety. Put it in the
design doc. If you're overruled, that's a documented organizational decision, not a
quiet engineering one.

</details>

### Trade-offs

**Hold quality, cut only the safe parts.** Slower savings, defensible, keeps the
product's core promise. The right default in a safety-critical domain.

**Cut aggressively and monitor.** Faster savings and a monitoring plan sounds
responsible — but in a domain where the failure mode is a rare, severe, *silent*
error, monitoring is inadequate. You'd need enormous sample sizes to detect a
change in a low-base-rate event, and by then patients have been affected. Name this
statistical reality explicitly; it's the strongest argument against the "just measure
it" position.

**Tier the product.** A cheaper tier with clear scope limits ("documentation only, no
clinical suggestions") is honest and viable. A cheaper tier that does the same job
worse is not.

### What Could Go Wrong?

- **Silent degradation in a low-base-rate failure mode.** If missed red flags occur
  in 0.3% of encounters, detecting a change from 0.3% to 0.5% requires thousands of
  adjudicated encounters. You will not detect it in production before it matters.
- **Automation bias.** Clinicians trust the tool more over time. Degrading quality
  while trust is high is more dangerous than degrading it early, and this
  relationship is invisible in your metrics.
- **The cached patient summary goes stale** across encounters and misses a new
  diagnosis. Cache keys must include record version; treat any record change as an
  invalidation.
- **The cheaper differential path shifts the *distribution* of suggestions** rather
  than the accuracy — subtly biasing toward common conditions and away from rare
  ones, which is exactly backwards for a triage tool. Measure per-condition recall,
  especially on rare high-severity conditions, not aggregate accuracy.
- **Regulatory classification changes** if the tool's behavior changes materially.
  Ask before you ship.

### Metrics

Red-flag sensitivity (recall) with confidence intervals — the primary safety metric,
and specificity as the secondary; per-condition recall for rare high-severity
conditions; clinician override rate and override *direction*; documentation time
saved per encounter; cost per encounter by component; and an adjudication pipeline
where a clinical panel reviews a fixed sample continuously — the only credible source
of ground truth in this domain.

### Follow-up Questions

1. You're overruled and told to hit $20. What do you do, and what do you put in
   writing?
2. Design the non-inferiority study for the differential-suggestions change. How many
   encounters, adjudicated by whom, and what's your acceptance criterion?
3. The competitor at $4/encounter wins a major customer. Does that change your
   position? Should it?
4. How do you detect automation bias in your clinician users, and what would you do
   about it?
5. A new model is released that is cheaper *and* scores better on your benchmarks.
   What's your adoption process in this domain, and how long does it take?

### What Separates Senior From Staff?

**Senior:** decomposes by risk tier, finds the safe savings, holds the line on red
flags, documents the decision.

**Staff:** changes what's being decided. They bring the value analysis that reframes
the product as underpriced rather than overbuilt; they make the risk decision
explicit and route it to the people who own it (clinical safety, legal, the board)
rather than absorbing it as an engineering trade-off; they propose *increasing* spend
on red-flag detection because that's where the expected value is, even though nobody
asked; and they establish the adjudication pipeline as permanent infrastructure so
every future model change in this product is decidable. They also state the
statistical argument — that you cannot monitor your way to safety on a rare failure
mode — clearly enough that non-technical leadership understands it.

### Interviewer Notes

This scenario tests judgment about *when not to optimize*, which is harder to teach
than optimization. Strong candidates decompose by risk tier immediately and are
willing to say "no" with reasoning attached. The best candidates propose spending
*more* on red-flag detection, unprompted — it demonstrates they're optimizing
expected value rather than the stated objective. Watch for candidates who
capitulate to the $20 target and start cutting; also watch for candidates who refuse
without producing any savings, which is equally unhelpful. The right posture is:
here's what I can safely deliver, here's what I won't do and why, here's the decision
that isn't mine to make, and here's the conversation I think we should be having
instead.

---

## Cost Reasoning Cheat Sheet

Worth internalizing before any cost interview.

**The cost identity.** For an API-based system:

```
monthly cost = requests × (input_tokens × price_in + output_tokens × price_out)
```

Every optimization attacks one of four terms. Know which one you're attacking:

| Lever | Attacks | Typical savings | Quality risk |
|---|---|---|---|
| Prefix / KV caching | `input_tokens` (effective) | 20–50% | None |
| Reranking → fewer chunks | `input_tokens` | 30–60% | Often *improves* quality |
| Context scoping / truncation | `input_tokens` | 10–40% | Low–medium |
| Response caching | `requests` | 10–40% | Medium (correctness risk) |
| Deduplication | `requests` | 5–25% | None |
| Routing / cascading | `price` (blended) | 30–70% | Medium |
| Cheaper model outright | `price` | 50–90% | High |
| Batch API tier | `price` | ~50% | None (latency cost) |
| Fine-tuning | `price` + `input_tokens` | 70–95% | Medium, plus ops burden |
| Doing less work | `requests` | Varies | Low, if measured |
| Deterministic code instead of a model call | `requests` | 100% of that call | Low, and testable |

**Rules of thumb:**

- Input tokens usually dominate **cost**; output tokens usually dominate **latency**.
  Check before optimizing.
- The cheapest call is the one you don't make. Look for unnecessary calls before
  cheaper calls.
- A model used as a validator, a classifier, or a formatter is often replaceable with
  code. Check.
- Never compare model prices without your own token profile.
- Quality is never a scalar. Segment it or you will optimize the average into a tail
  disaster.
- One-time savings against compounding growth is a delay, not a fix. Say so when you
  present it.
- Build cost-per-unit-of-work attribution **before** you need it. Every scenario in
  this file is unanswerable without it.
