# Staff-Level Scenarios

Staff and principal interviews are not harder versions of senior interviews. They
test a different thing.

A senior engineer is evaluated on whether they can solve a hard problem well. A staff
engineer is evaluated on whether they can decide **which problem the organization
should be solving**, get people who don't report to them to agree, and make a decision
that still looks right in two years. The technical depth is assumed; the failure mode
at this level is not "wrong architecture," it's "correct architecture that nobody
adopted," or "solved the stated problem while the real one got worse."

What these scenarios test:

- **Problem selection.** Given ten things that are wrong, which three matter?
- **Influence without authority.** You cannot mandate. What do you do instead?
- **Sequencing under irreversibility.** Which decisions can wait, and which foreclose
  options?
- **Organizational design.** Team boundaries, ownership, and incentives are
  engineering tools.
- **Honesty about cost**, including the cost of your own proposals.
- **Time horizon.** What does this look like in three years, and who maintains it?

A recurring test while reading: **what would you say no to, and to whom?** Staff-level
answers always contain a refusal.

**Contents**

1. [Eight Teams, Eight RAG Systems](#1-eight-teams-eight-rag-systems)
2. [The AI Strategy Nobody Asked You For](#2-the-ai-strategy-nobody-asked-you-for)
3. [Model Governance After the Incident](#3-model-governance-after-the-incident)
4. [The Migration That Will Take Two Years](#4-the-migration-that-will-take-two-years)
5. [Ambiguous Mandate, Hostile Stakeholders](#5-ambiguous-mandate-hostile-stakeholders)
6. [Prioritizing When Everything Is Urgent](#6-prioritizing-when-everything-is-urgent)

---

## 1. Eight Teams, Eight RAG Systems

### Situation

You are the newly-hired Staff AI Engineer at a 1,200-person B2B software company. Over
two years, eight product teams independently built RAG systems.

```
team          embedding model    vector store         eval        cost/mo
──────────────────────────────────────────────────────────────────────────
Search        hosted, 1536d      managed vendor A     yes, good    $84k
Support       hosted, 1536d      managed vendor A     minimal      $61k
Docs          open, 768d         pgvector             none         $12k
Analytics     hosted, 3072d      managed vendor B     none        $103k
Onboarding    open, 384d         in-memory, rebuilt   none          $7k
                                 on every deploy
Compliance    hosted, 1536d      managed vendor A     yes, weak    $44k
Sales         hosted, 3072d      managed vendor B     none         $71k
Field Eng     open, 768d         pgvector             none          $9k
──────────────────────────────────────────────────────────────────────────
                                                      total     $391k/mo
```

Also true:

- Four teams built chunking pipelines independently. Three have subtly different bugs
  in PDF table handling.
- Two teams index **the same corpus** (product documentation) separately.
- No team has a data-classification story. Two index content containing customer names
  and contract terms.
- Security flagged one system in an audit. Legal doesn't know about the other seven.
- Six of eight have no evaluation, so nobody can say whether any of it works.
- Three vector-database vendors, no negotiated contracts.
- Three teams have separately told you they'd love to stop maintaining retrieval.

You report to the VP of Engineering. **No direct reports.** You can request headcount
but not assume it.

### Your Task

Decide what to do, in what order, and how — given that you cannot mandate anything.

### Constraints

- Every team has a committed product roadmap. Nobody has slack.
- The VP supports you in principle but will not force adoption.
- The Analytics team ($103k/month, no eval) is the highest-revenue product and its
  lead is skeptical of platform work.
- You will be judged at your 6-month review on something visible.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Resist the obvious answer.** "Build a shared RAG platform and migrate all eight
teams" is what everyone expects and it is a two-year project with a high failure rate.
Platform consolidations fail predictably: the platform fits the first two use cases and
not the rest, migration competes with product roadmaps, and adoption is voluntary in
practice regardless of what leadership says.

**Rank the problems by harm, not by how much they offend your sense of order.**

| Problem | Real harm | Urgency |
|---|---|---|
| Unclassified customer data in indexes | Breach / legal exposure | **Immediate** |
| Six teams with no evaluation | Shipping unmeasured quality | High |
| Duplicated chunking with PDF bugs | Silent quality bugs | High |
| Duplicate indexing of the same corpus | ~$40k/mo waste | Medium |
| Three vendors, no contracts | Overpaying, no leverage | Medium |
| In-memory index rebuilt on deploy | An outage waiting | Medium |
| Different embedding models | Mostly aesthetic | **Low** |
| Different vector stores | Mostly aesthetic | **Low** |

**This ranking is the answer.** Heterogeneity itself is not the problem. Eight teams
using different embedding models costs the company almost nothing directly; it offends
a tidiness instinct. What costs the company is unclassified data, unmeasured quality,
duplicated buggy work, and unnegotiated spend.

A staff engineer who spends a year standardizing embedding models has optimized the
wrong variable while a breach was pending. Say this out loud in the interview — the
willingness to *not* consolidate is the signal.

**Then: where is your actual leverage?** With no reports and no mandate, leverage comes
from three places:

1. **Risk that leadership must act on.** Unclassified customer data in eight
   unreviewed indexes is a legal problem, and legal problems create mandates that
   platform proposals cannot.
2. **Things teams already want.** Three teams have said they want out of the retrieval
   business. That's a pull you can serve rather than a push you have to make.
3. **Things that are nearly free to adopt.** Anything requiring a migration will be
   scheduled behind product work, forever.

**Finally, separate three categories** — this distinction is the core of platform work:

- **Centralize** (one implementation, non-optional): data classification and permission
  filtering, cost attribution, vendor contracts.
- **Share** (a good library, opt-in): chunking, evaluation harness, retrieval patterns.
- **Leave alone** (explicitly, and say so): embedding model, vector store, prompts,
  ranking strategy.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Month 1: audit, and convert risk into a mandate.**

Do the thing nobody has done: produce a single document that inventories all eight
systems along the dimensions that matter — what corpus, what classification of data,
who can query it, what permission filtering exists, what evaluation exists, what it
costs.

Take the data-classification findings to security and legal directly. Two systems
indexing customer names and contract terms with no classification and no permission
model is a finding, not a project proposal. **This is the highest-leverage thing you
can do in month one**, because it converts a discretionary platform initiative into a
compliance requirement with executive attention and a deadline.

Be careful with the politics: you are new, and going to legal about other teams' systems
can read as an attack. Do it *with* the teams — share the audit with each team lead
first, frame it as "here's what I found across all of us, including what I'd want fixed
if it were mine," and go to security together. You want the finding to arrive as
engineering diligence, not as an accusation.

**Months 1–3: ship the two things that are free to adopt.**

1. **A retrieval-permission library.** One implementation of "given a user and a query,
   which documents are they allowed to see" with classification-aware filtering and an
   audit log. Adoption cost: a few days per team, and it satisfies the security finding
   they now have to address anyway. This is centralized *because it must be*, and teams
   adopt it because the alternative is writing it themselves under a deadline.

2. **A chunking library.** Four teams built this, three have PDF bugs. Nobody wants to
   own chunking. Publish a good one — well-tested against real documents, handling
   tables, headers, and the awkward cases — and teams will adopt it because it's
   strictly better than what they have and requires no migration of stored data (they
   re-chunk on next reindex, on their own schedule).

Both are libraries, not services. No network hop, no availability dependency, no
platform SPOF. Adoption cost near zero. **This is what "influence without authority"
looks like in practice: make the right thing the easy thing.**

**Months 2–5: evaluation, by pull not push.**

Six teams have no evaluation. You cannot make them build eval suites. What you can do:

- Publish a reference eval harness that's genuinely good.
- Build a results registry (a table and a dashboard) showing every AI feature's quality
  score and when it was last measured. Teams with no evaluation appear blank.
- **Embed with one team at a time** — starting with a team that wants help — and build
  their first eval set *with* them, from their production traffic. Four to six weeks
  each. This is unglamorous and it is the work.

Visibility does what a mandate can't. Once three teams have scores on a dashboard the
VP looks at, the blank rows become uncomfortable without you having to say anything.

**Months 3–6: the money and the pull.**

3. **Consolidate the duplicate corpus.** Two teams index product documentation
   separately. Offer to run it: one indexing pipeline, one index, both teams query it.
   Saves ~$40k/month and removes a class of "the two systems disagree" bugs. This is a
   concrete, attributable win for your six-month review.

4. **Negotiate vendor contracts.** $391k/month across three vendors with no contracts
   is leaving 20–40% on the table. This requires no team to change anything — it's pure
   savings, it makes you useful to finance, and it takes weeks of your time rather than
   theirs. Consolidating to two vendors (not one) preserves leverage.

5. **Serve the three teams who want out.** They asked. Offer a managed retrieval
   service for teams that don't want to own it — but build it *for them specifically*,
   with their requirements, rather than as a general platform. If it works for three
   teams, others will ask. If nobody else asks, you've still helped three teams and
   spent no capital.

**What you explicitly do not do, and say so publicly:**

- Do not standardize embedding models. The cost of migration (re-embedding every
  corpus) vastly exceeds the benefit, and the benefit is mostly tidiness.
- Do not force consolidation onto one vector store.
- Do not build a general RAG framework in year one.

Writing down what the platform will *not* do is what makes teams stop being defensive,
and it's the single most useful sentence you can put in a platform charter.

**The Analytics team specifically.** $103k/month, no evaluation, highest revenue,
skeptical lead. Do not start there. Start with a team that wants help, produce a
visible result, and let Analytics come to you — or approach them with something they
want (cost reduction on their $103k, which is the largest line item and something their
VP notices) rather than something you want (standardization).

**At six months, the visible result is:** a security finding closed, ~$40k/month saved
on the duplicate corpus, 20–40% off the vendor bill, three teams on shared permission
and chunking libraries, and four teams with evaluation where there was one. None of
that required a mandate.

</details>

### Trade-offs

**Big-bang platform consolidation.**
*Advantages:* if it works, one system, one cost model, one security posture, one thing
to improve. Real long-term leverage.
*Disadvantages:* 18–24 months, high failure rate, requires sustained executive
enforcement, competes with eight product roadmaps, and the platform inevitably fits the
first two adopters and fights the rest.
*Choose when:* you have a real mandate, real headcount, and the teams' use cases are
genuinely similar.

**Federated with shared libraries (chosen).**
*Advantages:* near-zero adoption cost; teams keep autonomy and velocity; you can start
immediately with no headcount; failure is cheap.
*Disadvantages:* heterogeneity persists; libraries drift as teams fork them; no
central control over cost or quality; you're always negotiating rather than deciding.
*Choose when:* no mandate, small platform footprint, high team autonomy — i.e. here.

**Do nothing structural; just fix the security problem.**
*Advantages:* addresses the only genuinely urgent issue; no political capital spent.
*Disadvantages:* the $391k keeps growing, quality stays unmeasured, and in a year
there are twelve systems instead of eight.
*Choose when:* you have three months, not two years — worth naming as the minimum
viable answer.

### What Could Go Wrong?

- **The audit reads as an attack** and eight team leads become adversaries in month
  one. Mitigation: share findings with each team first; go to security jointly; frame
  it as shared risk.
- **Libraries get forked.** A team needs a change, forks, and never merges back. Within
  a year there are five versions of the chunking library. Mitigation: fast response to
  contribution requests, and treat the teams as customers whose PRs get reviewed same
  day.
- **You become the RAG on-call for eight teams** without owning any of them.
  Responsibility without authority is the classic staff trap. Be explicit about support
  boundaries in writing.
- **The evaluation dashboard becomes a shaming device**, teams game it, and the numbers
  stop meaning anything. Introduce scores private-to-team first; add a held-out set.
- **Vendor consolidation to one vendor** removes your negotiating leverage entirely.
  Keep two.
- **The three teams who wanted out change their minds** when they see the migration
  cost. Scope the first one small enough to finish.
- **Your six-month review has nothing visible** because everything you did was
  preventive. Pick at least one legible, attributable win early — the duplicate corpus
  consolidation is designed for this.

### Metrics

*Risk:* AI systems with a documented data classification (target 8/8); systems with
permission filtering; open security findings.
*Quality:* features with a live eval score and a named owner; features with a blocking
CI gate; regressions caught pre-release.
*Cost:* total AI spend and spend per team (attributable, which it currently isn't);
duplicate infrastructure eliminated; effective vendor rate vs. list.
*Adoption:* teams using the permission library and chunking library; contributions from
other teams (a healthy shared library receives PRs).
*Anti-metric:* if any team's delivery velocity drops after adopting your work, the work
is failing regardless of its elegance.

### Follow-up Questions

1. Six months in, five teams have adopted your libraries and three haven't — including
   Analytics. What do you do, and what do you *not* do?
2. The VP now offers you a mandate: all teams must migrate to a central platform in 12
   months. Do you take it?
3. A ninth team is starting a new AI product next month. What do you ask of them, and
   what do you give them?
4. Two teams' libraries have diverged and a security fix must land in both. How did you
   let this happen, and how do you fix it?
5. You get two engineers. Where do they go?
6. How do you know, at month 12, whether you've succeeded? Name three things.

### What Separates Senior From Staff?

A senior engineer given this situation builds an excellent shared RAG platform. It is
technically better than anything the eight teams have, and in 18 months three teams use
it.

A staff engineer:

- **Ranks the problems by harm and publicly declines to fix the ones that don't
  matter.** The refusal to standardize embedding models — and the ability to explain
  why — is more informative than any architecture.
- **Converts a discretionary initiative into a non-discretionary one** by routing the
  data-classification finding through security and legal, which is the only mandate
  available.
- **Chooses libraries over services** because adoption cost, not technical quality,
  determines outcomes when you have no authority.
- **Spends most of the effort on evaluation adoption** — the thing with no visible
  artifact — because six teams shipping unmeasured quality is the largest hidden risk.
- **Serves demand rather than creating it**: starts with the teams that asked.
- **Writes down what the platform will not do**, which is what makes autonomous teams
  stop defending themselves.

### Interviewer Notes

*What to listen for:*

- Does the candidate immediately propose consolidation? That's the trap. The best
  answers explicitly rank problems and decline the low-harm ones.
- Do they identify the data-classification issue as the urgent one? It's the only item
  with unbounded downside and it's easy to skim past among the cost figures.
- Do they distinguish "centralize" from "share" from "leave alone"? This is the single
  most useful distinction in platform work.
- Do they choose libraries over services, and can they say why? (Adoption cost,
  availability coupling, no mandate.)
- Do they handle the Analytics team as a customer to be won rather than an obstacle?
- Do they have a plan for their own visible result at six months? Staff engineers
  operate inside a career too, and pretending otherwise is naive.

*Red flags:* proposing a 12-month platform build with no adoption strategy; treating
heterogeneity as the primary problem; going to legal about other teams without
involving them; no plan for what they won't do.

*Green flags:* asks about team autonomy culture; asks who owns the budget; asks what
happened to the previous person in this role; proposes to start with the team that
wants help; names the risk of becoming on-call for eight systems.

---

## 2. The AI Strategy Nobody Asked You For

### Situation

You are a Staff AI Engineer at a 400-person company that sells scheduling software to
healthcare clinics. AI has arrived through the side door:

- The CEO saw a competitor's demo and told the board the company will "be AI-first
  within a year."
- Three product teams have shipped AI features reactively. Two of them are barely used
  (feature usage under 4% of active accounts).
- One AI feature — appointment note summarization — has 61% weekly usage and customers
  cite it in renewals.
- The company has no AI roadmap, no shared infrastructure, no evaluation, and no
  position on customer data usage.
- Sales is promising AI capabilities in deals that don't exist.
- A large customer's security team has asked, in writing, whether patient data is sent
  to third-party model providers. Nobody has answered yet.

Your VP asks you to "write the AI strategy." There is no template, no deadline, and no
defined audience.

### Your Task

Write it. Decide what it should contain and what decisions it should force.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**A strategy document that lists technologies is not a strategy.** "We will use RAG,
agents, and fine-tuning" is a shopping list. A strategy makes *choices* — it says what
the company will do, what it will not do, and why, in a way that lets people make
decisions without asking you.

**Start from evidence, not ambition.** You have one strong data point most companies
lack: one AI feature has 61% usage and shows up in renewals; two have under 4%. That
asymmetry is the most valuable input available and it should shape everything.

Ask: *what is different about the note summarization feature?* Almost certainly:
- It automates something clinicians already do, hate doing, and do many times daily.
- Its output is reviewed by a human before it matters.
- It saves measurable time.
- It fits an existing workflow rather than creating a new one.

That's a **thesis**: in this product, AI wins when it removes repetitive documentation
work inside an existing clinical workflow with a human check. Everything else has been
speculative.

A strategy built on that thesis is defensible, testable, and immediately useful for
prioritization. A strategy built on "be AI-first" is not.

**Second: the unanswered security question is the most urgent item in the document.**
A healthcare customer asking in writing whether patient data goes to third parties is
not a strategy question, it's a live risk with a clock on it. Any strategy that doesn't
lead with the data position is unserious in this industry.

**Third: the CEO's "AI-first within a year" needs handling, not obedience.** Your job is
not to contradict it publicly, and not to pretend it's a plan. Translate it into
something executable and measurable, and let the evidence set the pace. Give the CEO a
version of their goal that can actually be achieved and defended to the board.

**Fourth: a strategy nobody can act on is a memo.** The test for every section: *does
this let someone make a decision without asking me?*
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**The document has six sections. It is 6–8 pages, not 40.**

**1. The data position (first, because it's blocking).**

State plainly:
- What data goes to which providers today, per feature.
- What contractual protections exist (zero-retention, no-training, BAA where required
  for protected health information).
- What the company commits to going forward.

Then propose the standard: **no protected health information leaves the company's
control without an executed agreement and a named approver; features that need PHI use
either a provider under a signed BAA or in-tenancy inference.** Route it through legal
and compliance for sign-off, and answer the customer this week.

This section is what makes the whole document credible in a healthcare company. It also
converts an ambient risk into a written policy that unblocks every future feature —
which is the highest-value output of the entire exercise.

**2. The thesis: where AI creates value in this product.**

Present the usage evidence. State the thesis: *AI in this product succeeds when it
removes repetitive documentation and communication work inside an existing clinical
workflow, with a human review step.* Name the counter-evidence honestly (two features
under 4%, and why).

Then derive a **prioritization filter** anyone can apply:

> A proposed AI feature should score well on: (a) does it replace work a clinician
> does repeatedly today? (b) is there a natural human review point? (c) does it fit an
> existing workflow rather than requiring a new one? (d) can we measure the time saved?
> Features scoring poorly on three or more get built only with an explicit exception.

This is the part that lets other people make decisions without you, which is the point
of writing a strategy at all.

**3. What we will build in the next four quarters.**

Three to five bets, sequenced, each with a hypothesis and a kill criterion:

- **Deepen the winner.** Note summarization has 61% usage; extend it (structured
  extraction into the chart, coding assistance, follow-up drafting). The highest-return
  AI work available is usually improving the AI feature people already use.
- **One or two new features that pass the filter**, with explicit success criteria and
  a date at which they're killed if unmet.
- **Sunset or fix the two low-usage features.** Say this out loud. A strategy that only
  adds is a wish list.

**4. What we will not do.**

- No general-purpose chatbot. (It fails the filter: no existing workflow, no review
  point, unmeasurable.)
- No autonomous clinical decision-making. (Regulatory and safety.)
- No fine-tuning on customer data in year one. (Data position, and we lack evaluation
  capability to know if it helps.)
- No self-hosted inference in year one. (Team size.)

The "will not" section is the most-read and most-useful part of any strategy document.
It's also where you spend your credibility, so each item needs a reason a
non-engineer accepts.

**5. Capability investments.**

The unglamorous foundations, framed by what they unblock:

- **Evaluation** — without it, no AI feature can be improved or safely changed. First
  investment, and the prerequisite for everything else in the document.
- **A shared provider gateway** — cost attribution, quota isolation, one place to
  enforce the data policy. This is how section 1 gets *enforced* rather than merely
  written.
- **AI observability** — so failures can be diagnosed.
- **A per-feature quality bar and review process** before anything ships to clinicians.

Each with a cost and an owner. If you're asking for headcount, this is where it goes,
and the ask is tied to a named risk rather than to ambition.

**6. How we'll know it's working.**

- Feature-level: usage, retention lift, time saved per clinician per day, quality score.
- Company-level: AI features in renewal conversations; AI-attributed churn prevention;
  AI cost as a percentage of gross margin.
- Capability-level: features with evaluation; incidents; time-to-ship a new AI feature.

**Handling the CEO's "AI-first."** Translate rather than contradict:

> "AI-first, for us, means every clinical documentation workflow has an AI-assisted
> path by end of year, measured by clinician time saved. Here is the sequence and what
> it requires."

That's a version of their goal that is achievable, measurable, and defensible to the
board — and it quietly replaces an unmeasurable slogan with a plan. Get the CEO's
agreement on that definition early; it's the single most valuable conversation in this
scenario.

**Handling sales.** Sales promising nonexistent features is a real problem with a real
fix: a published capability sheet of what exists, what's coming with dates, and what is
explicitly not planned — plus a fast path for sales to ask before promising. Make it
easy to do the right thing.

</details>

### Trade-offs

**Evidence-driven, narrow strategy (chosen).** Defensible, focused, high hit-rate. But
it constrains ambition, may look unambitious to a board that wants AI headlines, and
risks missing a genuinely new opportunity that doesn't fit the existing-workflow filter.

**Broad "AI everywhere" strategy.** Matches executive enthusiasm, funds more
exploration, higher variance and possibly higher ceiling. But it spreads a small team
across many bets, most of which will look like the two features at 4% usage, and it
creates a large surface of unevaluated AI touching patient data.

**Infrastructure-first.** Build evaluation, gateway, and observability before new
features. Sound engineering, but a year of no visible AI progress in a company whose
CEO promised the board otherwise is politically fatal. The right answer interleaves:
build the capability that unblocks the next feature, alongside the feature.

### What Could Go Wrong?

- **The CEO reads the "will not do" section as insubordination.** Socialize before
  publishing; get the CEO's input on the definition of AI-first *before* it's in
  writing.
- **The document is written and never used.** A strategy that doesn't change a
  prioritization meeting is decoration. Attach it to the actual planning process — the
  filter should appear in the feature-proposal template.
- **The thesis is wrong.** One successful feature is a small sample. State it as a
  hypothesis with a review date, not a law.
- **The data policy blocks a deal** and gets overridden under commercial pressure.
  Anticipate this: define an exception process with a named approver, so exceptions are
  decisions rather than erosions.
- **Sunsetting the two low-usage features** angers the teams that built them. Frame it
  as portfolio management with data, and let those teams work on the winner.
- **Sales keeps promising.** The capability sheet only works if it's maintained and if
  sales leadership backs it. That's a relationship, not a document.

### Metrics

For the strategy itself: is the filter used in prioritization meetings? Do feature
proposals cite it? Has the "will not do" list held? Is the data policy referenced in
security questionnaires (this is the fastest way to see whether it's real)?

For the portfolio: AI feature usage and retention lift; clinician time saved; AI spend
as a share of gross margin; features with evaluation coverage; time-to-ship for a new
AI feature.

### Follow-up Questions

1. The CEO reads your draft and says it's "not ambitious enough." What do you do?
2. Six months in, the note-summarization deepening works but the two new bets are at 6%
   usage. What does the strategy say you should do, and do you follow it?
3. A competitor ships an autonomous scheduling agent and wins a deal. Does your "will
   not do" list change?
4. How would you write this differently if the company had zero AI features rather than
   three?
5. Sales closes a $2M deal contingent on a feature your filter rejects. What happens?
6. Who is the audience for this document, and how does that change what you write?

### What Separates Senior From Staff?

A senior engineer writes a technically excellent document about architecture,
infrastructure, and best practices. It is correct and nobody's decisions change.

A staff engineer writes a document that **forces choices**: a thesis grounded in the
one piece of real evidence the company has, a filter other people can apply without
them, an explicit "will not do" list, and a data position that unblocks every future
conversation with a healthcare customer. They handle the CEO's slogan by translating it
into something measurable rather than obeying or contradicting it. They treat the sales
problem as a systems problem with a fix. And they name what gets sunset, because a
strategy that only adds is a wish list.

### Interviewer Notes

The usage asymmetry (61% vs. 4%) is the most important input in the scenario and it's
buried among other facts. Candidates who build their strategy on it are reasoning from
evidence; candidates who produce a technology roadmap are producing a shopping list.
The second signal is whether they lead with the unanswered security question — in
healthcare, an unanswered PHI question outranks everything. Third: does the document
contain a "will not do" section? Almost no candidate volunteers one, and it's the
clearest marker of strategic thinking. Fourth: how do they handle the CEO? Both
obedience and contradiction are wrong answers; translation is the staff move.

---

## 3. Model Governance After the Incident

### Situation

Your company's AI-powered loan pre-qualification tool produced systematically different
outcomes for applicants from certain postal codes. A journalist noticed. There is now:

- A regulator inquiry.
- A board-level demand for "AI governance."
- An engineering organization that has shipped 14 AI features in 18 months with no
  review process.
- A legal team proposing a review board that must approve every model change.
- An engineering organization that regards that proposal as an existential threat to
  velocity.

You are the Staff AI Engineer asked to design the governance framework. You have three
weeks before a board presentation.

### Your Task

Design something that satisfies the regulator, protects the company, and doesn't stop
engineering.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**First, understand what actually failed** — not to assign blame, but because a
governance framework that doesn't address the actual failure is theatre, and regulators
recognize theatre.

Likely mechanism: the model wasn't given postal code as a feature (nobody is that
careless), but postal code correlates with features that were used — income proxies,
employment history, prior banking relationships, or something in a retrieved document.
The disparity emerged from proxy variables, which means:

- It would not have been caught by "don't use protected attributes" as a rule.
- It **would** have been caught by outcome testing across demographic groups.

That distinction determines the entire framework: **governance must test outputs, not
just review inputs.** A review board reading design documents would have approved this
system.

**Second, characterize the two failure modes of governance:**

- *Too heavy:* every change needs board approval, velocity collapses, teams route
  around the process, and governance becomes a document nobody reads. Common outcome.
- *Too light:* governance exists on paper, nothing changes, the next incident happens
  and now you also have a documented process you didn't follow — which is worse than
  having none.

The design goal is **proportionality**: the weight of review should scale with the risk
of the system. A loan decision and an internal meeting summarizer should not go through
the same process, and any framework that treats them identically will be either
useless or intolerable.

**Third: most governance should be automated.** Human review boards don't scale and are
inconsistent. What scales is: a mandatory test suite that includes fairness testing, run
in CI, blocking deploys. Humans review the *risk classification* and the *exceptions*,
not every change.

**Fourth: the regulator wants evidence, not intentions.** They will ask "show me that
you tested for this, what you found, and what you did." That's a data and
record-keeping requirement, and it's the requirement most engineering organizations are
least prepared for.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**A three-tier, risk-proportionate framework.**

**Tier the systems first.** Classify every AI system by consequence, not by technology:

```
Tier 1 — Consequential decisions about people
  (credit, employment, housing, insurance, access to services)
  → full governance

Tier 2 — Customer-facing, influences decisions but human-mediated
  (recommendations, drafted communications, prioritization)
  → moderate governance

Tier 3 — Internal productivity, no external consequence
  (meeting summaries, code assistance, internal search)
  → light governance
```

Of 14 features, probably 1–2 are Tier 1, 4–5 Tier 2, and the rest Tier 3. **Getting
this classification right is what makes the framework survivable** — it concentrates
the expensive process where the risk is and leaves most of engineering alone.

**What each tier requires:**

*Tier 1:*
- Documented purpose, data sources, and features (including proxies considered).
- **Mandatory disparate-impact testing across protected classes and their proxies**,
  run automatically on every model, prompt, or data change, blocking deploy on
  regression beyond a defined threshold.
- Explainability: the ability to state, for any individual decision, the primary
  factors — and to reproduce that decision from stored inputs and versions.
- Human review of the *test results* before launch and quarterly thereafter, by a
  standing committee with legal, compliance, and engineering representation.
- Retained records: every decision with inputs, model version, and outputs, for the
  regulatory retention period.
- An appeal path for affected individuals.

*Tier 2:*
- Documented purpose and data sources.
- Mandatory evaluation suite with quality gates in CI.
- Fairness testing where the output could plausibly differ across groups.
- Self-certification against a checklist, with audit sampling by the committee.

*Tier 3:*
- Registration in an inventory (so nothing is invisible).
- Basic evaluation.
- No pre-approval required.

**The engineering interface — this is the part that determines whether it works:**

Governance is a **library and a CI gate**, not a meeting:

```
every AI system declares, in code:
    tier, purpose, data_sources, owner, eval_suite, protected_attributes_tested

CI enforces:
    tier 1/2 → eval suite must exist, must pass, fairness tests must pass
    tier 1   → decision logging must be enabled, explainability endpoint must exist
    all      → registered in the inventory, has an owner
```

Teams experience governance as a build failure with a clear message, not as a
scheduling problem. This is the crucial design decision and it's what makes the
difference between adoption and circumvention.

**Fairness testing, concretely** (the thing that would have caught this incident):

- Define outcome metrics: approval rate, average offered terms, false-negative rate.
- Compute across protected groups *and* known proxies (postal code, name-derived
  signals where legally permissible to test, device type, referral source).
- Test for disparate impact using an agreed statistical standard, with thresholds set
  by legal.
- Run on a held-out set that includes demographic labels — which means you must have or
  construct such a dataset, carefully, with legal guidance. **This is usually the
  hardest practical step and it should be started in week one.**
- Report results as an artifact retained with the deploy.

**The review board — narrow it deliberately.** Legal's proposal (approve every model
change) would review hundreds of changes a year and become a rubber stamp within two
months. Counter-propose:

- The board approves **tier classifications**, **new Tier 1 systems**, **threshold
  definitions**, and **exceptions**. It reviews quarterly reports.
- It does not approve individual changes to systems whose automated gates pass.
- It has a standing engineering seat, so it is not a purely legal body reviewing
  technical artifacts it can't interpret.

Present this to legal as *stronger* than their proposal, because it tests every change
automatically rather than reviewing a sample of documents — and that framing is usually
persuasive, because it's true.

**For the board presentation, three things:**
1. What happened, mechanically and honestly, including that input review would not have
   caught it.
2. The framework, with the tiering and the automated enforcement.
3. What it costs (headcount, timeline, velocity impact) and what remains unaddressed.
   Overclaiming to a board after an incident is how the second incident becomes
   unsurvivable.

**Remediation, separately and in parallel:** identify affected applicants, define
redress, and fix the model. Governance is about the future; the regulator is also
asking about the past.

</details>

### Trade-offs

**Heavy centralized review.** Maximum defensibility, clear accountability, satisfying
to a board. But velocity collapses, teams route around it, the board becomes a
bottleneck and then a rubber stamp, and you get the appearance of governance with none
of the substance.

**Automated gates with narrow human review (chosen).** Scales, tests every change,
gives engineers fast feedback, produces retained evidence. But it requires real
engineering investment, the tests must be well-designed (a bad fairness test is worse
than none because it manufactures false assurance), and it can miss novel harms that no
test anticipates.

**Guidelines without enforcement.** Cheap, no velocity impact. Does nothing. After an
incident, documented-but-unfollowed guidelines are an aggravating factor rather than a
defense.

### What Could Go Wrong?

- **The fairness tests are wrong.** A poorly-specified test produces confident
  assurance that the system is fair when it isn't. Have the methodology reviewed
  externally.
- **You can't build the labeled demographic dataset** legally or practically, so the
  central control is unimplementable. Start this in week one; it's the long pole and
  discovering it in week three is fatal to the timeline.
- **Tier classification is gamed.** Teams classify down to avoid the process. Mitigate:
  classification is proposed by the team and approved by the board, with audit.
- **Tier 3 systems drift into Tier 1 consequences** — an internal tool starts
  influencing customer outcomes. Require re-classification on material change, and audit
  periodically.
- **Velocity impact is real and unacknowledged**, engineering leadership loses trust,
  and the framework becomes an adversarial process. Measure and publish the velocity
  impact honestly.
- **The framework covers models but not prompts, retrieval, or data changes** — all of
  which change behavior. Governance must cover any change that alters outputs, which is
  a broader definition than most frameworks use.

### Metrics

Systems registered in the inventory (target 100%); tier distribution; Tier 1/2 systems
with passing fairness tests and current results; deploy-blocking events and their
resolution time; velocity impact (time-to-deploy before vs. after, by tier); audit
sampling results for Tier 2 self-certifications; time-to-produce evidence for a
regulator request (a direct measure of whether record-keeping works); and — the outcome
metric — measured disparity across groups on Tier 1 systems, trended.

### Follow-up Questions

1. A team classifies their system Tier 3 and you believe it's Tier 1. How is that
   resolved?
2. Your fairness test blocks a deploy that the team believes is a false positive and a
   customer commitment depends on it. What's the process?
3. The regulator asks for evidence about a decision made 14 months ago. Can you produce
   it? What would you need to have built?
4. How do you govern a third-party model whose behavior changes without notice?
5. Legal wants the review board to approve every prompt change. Make the counter-case
   in one paragraph.
6. Eighteen months later, no fairness test has ever blocked a deploy. Is that good news?

### What Separates Senior From Staff?

A senior engineer builds excellent fairness testing infrastructure.

A staff engineer recognizes that **the governance design is primarily an organizational
design problem**: it must satisfy a regulator, be acceptable to legal, be tolerable to
engineering, and actually prevent the failure. They achieve that by making the process
proportionate to risk and mostly automated, so that the expensive human process applies
to two systems rather than fourteen. They negotiate legal's proposal down by offering
something demonstrably stronger rather than by resisting. They tell the board what the
framework costs and what it doesn't cover, because credibility after an incident is the
scarcest resource. And they identify the demographic-labeled dataset as the critical
path in week one rather than week three.

### Interviewer Notes

The key technical insight is that input review would not have caught this — the
disparity came through proxies, so governance must test outputs. Candidates who propose
"don't use protected attributes" as the control have not understood the incident. The
key organizational insight is proportionality: a framework applied uniformly to 14
systems will fail. Watch for whether the candidate engages with legal's proposal
constructively (counter-proposing something stronger) or adversarially. Also probe the
demographic dataset question — it's the practical blocker in every real fairness
program and candidates who've done this work raise it unprompted.

---

## 4. The Migration That Will Take Two Years

### Situation

Your company's core AI product is built on an architecture chosen three years ago that
is now clearly wrong:

- A monolithic Python service that does retrieval, generation, orchestration, and
  serving, with 180,000 lines of code and no clear internal boundaries.
- Retrieval logic is coupled to a vector store the company is contractually leaving in
  14 months (the vendor is being acquired and the product is being sunset).
- Prompts are embedded in code, so any prompt change is a full deploy — and deploys
  take 40 minutes.
- Test coverage of AI behavior is near zero; the eval suite runs manually.
- The service handles 40M requests/day and is the company's primary revenue driver.
- Eleven engineers work in this codebase. Four of them wrote most of it and have strong
  opinions.

You've been asked to lead the modernization. You estimate two years. Leadership has
budgeted "a couple of quarters."

### Your Task

Plan it, staff it, and manage the expectation gap.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**First: is a two-year migration the right answer at all?** Large rewrites of
revenue-critical systems have a poor track record. Before committing, ask what's
actually forcing the work:

- The vector store sunset in 14 months is **a hard deadline with a real consequence.**
  That's not modernization, it's survival, and it has a date.
- Everything else — coupling, prompt deploys, test coverage — is pain, not deadline.

That reframe changes the plan entirely. **The vendor migration is the project. The
modernization is what you do around it**, using it as the forcing function to
establish boundaries you'd otherwise never get funded to build.

**Second: never propose a big-bang rewrite of a system generating primary revenue.**
The strangler pattern applies: extract seams incrementally, run old and new in
parallel, migrate traffic gradually, delete the old path only when the new one is
proven. Slower on paper, dramatically higher success rate, and it delivers value
continuously rather than at the end.

**Third: the expectation gap is a real deliverable.** "Leadership budgeted two quarters,
I need two years" is not resolved by asserting the estimate harder. It's resolved by
decomposing into increments with independent value, so the conversation becomes "what
do we get each quarter" rather than "will you be done in two." Most two-year projects
die because they have nothing to show at month six.

**Fourth: the four original authors are the biggest risk and the biggest asset.** A
modernization that treats them as obstacles will fail — they have context nobody else
has, and they can block progress passively for a year. They need to be architects of
the change, not subjects of it.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Reframe: this is a vendor migration with a hard deadline, and a modernization
program that rides on it.**

**Phase 0 (weeks 1–6): make change safe.**

You cannot refactor a system whose behavior you can't verify. Before touching
architecture:

1. **Build the automated eval suite and put it in CI.** Sampled from production traffic,
   segmented, with confidence intervals. This is the prerequisite for every subsequent
   step — without it, every refactor is a leap of faith and the four original authors
   are right to resist.
2. **Characterization tests** around the current behavior at the seams you intend to
   cut. You're not testing correctness; you're pinning current behavior so you can
   detect change.
3. **Instrumentation:** per-stage latency, per-request cost, error taxonomy. You'll need
   to prove the new path is equivalent.

Six weeks of no visible progress, and it is non-negotiable. Sell it as risk reduction
for the vendor migration, which is true.

**Phase 1 (months 2–6): extract retrieval behind an interface. Ship the vendor
migration.**

The vendor deadline forces retrieval extraction, so do it properly:

- Define a retrieval interface (query, filters, permissions → ranked results). Keep it
  narrow.
- Implement it twice: over the old vendor, and over the replacement.
- Route via config; dual-read and compare on live traffic for weeks; migrate
  progressively by tenant or traffic percentage.
- Delete the old implementation only when the new one has been at 100% for a month.

**Outcome at month 6:** the contractual deadline is met with 8 months to spare, and the
codebase now has one real internal boundary. That's a legible win for leadership and it
buys you the credibility for phases 2 and 3.

**Phase 2 (months 5–10): extract prompts from code.**

- Move prompts to versioned configuration with a review process, canary deploys, and an
  eval gate. Ship in hours, not 40-minute deploys.
- This is high-visibility to product and the fastest quality-of-life improvement for
  everyone. It also removes a class of incidents (see the prompt-change incident in
  `production-incidents.md`).

**Phase 3 (months 8–18): extract serving and orchestration, incrementally.**

- Separate the request-serving path from the orchestration logic, one workflow at a
  time, highest-traffic first.
- Each extraction is independently shippable and independently revertible.
- The monolith shrinks; nothing is rewritten wholesale.

**Phase 4 (ongoing): delete.**

Explicitly budget time for removing the old code paths. Migrations that never delete
leave you with two systems and twice the maintenance — the most common way a
modernization makes things worse.

**Managing the expectation gap:**

Don't argue about "two years." Present:

- **The hard deadline** (14 months, vendor sunset) with the plan to hit it in 6.
- **A quarterly value sequence:** Q1 safety net + retrieval interface; Q2 vendor
  migration complete; Q3 prompt velocity (ship prompt changes in an hour); Q4 first
  orchestration extraction, deploy time down.
- **What "done" means:** not "the rewrite is finished" but a set of measurable
  properties (deploy time, change failure rate, time-to-ship a prompt change, coupling
  metrics).
- **What happens if it's cut short**, honestly: after Q2 you have the vendor migration
  and one boundary. After Q4 you have a materially better system. Stopping after Q1 is
  the only genuinely bad outcome.

That framing converts "two years" from a request for patience into a portfolio of
quarterly deliverables that can be evaluated on their own merits.

**The four original authors:**

- Make them the designers of the interfaces. They know where the bodies are buried, and
  interface design is the highest-judgment work in the project.
- Frame the work as extending their system's life, not replacing their mistakes. The
  architecture was reasonable three years ago at a tenth of the scale; say that, and
  mean it.
- Give them the vendor migration, which is the most technically interesting piece.
- Watch for the failure mode where they block progress through review latency or
  litigating decided questions. Escalate early and privately if it happens.

**Staffing:** 11 engineers cannot all work on migration and none on product. A realistic
split is ~4 on migration and ~7 on product, with rotation so knowledge spreads and the
migration doesn't become one person's project. Rotation also prevents the migration team
becoming isolated from the reality of shipping features on the old system.

</details>

### Trade-offs

**Big-bang rewrite.** Clean result, no long period of dual maintenance, satisfying.
But high failure rate on revenue-critical systems, no value until the end, a feature
freeze nobody will accept, and the new system inherits the old one's undocumented
behaviors as a series of production incidents.

**Strangler / incremental (chosen).** Continuous value, low risk, reversible at each
step, doesn't block product work. But it takes longer, requires maintaining two paths
during transition, and demands discipline to actually delete the old code.

**Do only the vendor migration; defer modernization.** Minimal, meets the deadline,
lowest cost. But you spend the forcing function without getting the boundaries, and the
next migration is equally painful. Worth naming as the fallback if funding is cut.

### What Could Go Wrong?

- **The eval suite isn't good enough** and you migrate a regression you can't detect.
  The entire plan rests on Phase 0; if the eval suite is weak, everything downstream is
  unsafe.
- **The vendor sunset date moves earlier.** Have a contingency; talk to the vendor
  early.
- **Dual-running costs double** during migration and nobody budgeted it. Say it up
  front.
- **The old paths are never deleted** because deletion has no visible value. Schedule it
  as explicit work with an owner.
- **Product pressure eats the migration team** one engineer at a time until it stalls.
  Protect the allocation formally and report on it.
- **Leadership cuts the project after Q2** because the vendor deadline is met and the
  rest looks optional. Sequence so each quarter's deliverable stands alone, and be
  honest that this outcome is survivable.
- **A key author leaves mid-migration.** Rotation and documentation are the mitigation,
  and they must start in Phase 0.

### Metrics

*Progress:* percentage of traffic on the new retrieval path; number of extracted
boundaries; lines of code in the monolith (a crude but honest trend); prompt changes
deployed without a code deploy.
*Safety:* eval suite coverage and pass rate; change failure rate; incidents attributable
to migration work (should be near zero — if it isn't, slow down).
*Value:* deploy time; time-to-ship a prompt change; time-to-ship a new AI feature;
on-call load.
*Cost:* dual-running infrastructure cost; engineer-months spent vs. plan.

### Follow-up Questions

1. Month 8: the vendor migration is done, leadership wants the team back on product.
   What's your case for continuing?
2. Your eval suite passes but a customer reports a behavior change after the retrieval
   migration. How do you investigate, and what does it say about your suite?
3. One of the four original authors is blocking design decisions through review latency.
   How do you handle it?
4. The vendor announces the sunset is in 6 months, not 14. Replan.
5. How do you keep 7 engineers shipping product features in a codebase that's being
   actively restructured?
6. What's the smallest version of this plan that still leaves the company better off?

### What Separates Senior From Staff?

A senior engineer plans a competent incremental migration and executes it well.

A staff engineer **reframes the project around the only hard constraint** — the vendor
deadline — and uses it as the forcing function that funds the architectural work
nobody would otherwise approve. They refuse to start refactoring before the safety net
exists, and they defend those six "unproductive" weeks. They convert an unwinnable
"two years vs. two quarters" argument into a quarterly value sequence, which is a
communication design as much as a technical one. They handle the four original authors
as the project's most important stakeholders rather than as resistance. And they say
plainly what happens if the project is cut short at each stage, because a plan that only
works if fully funded is not a plan.

### Interviewer Notes

The reframe — vendor deadline is the project, modernization rides on it — is the
strongest available signal and it comes from noticing that only one item in the
situation has a date attached. Second signal: does the candidate insist on the eval
suite and characterization tests *before* refactoring? Candidates who start extracting
services in month one are describing a project that will fail. Third: how do they handle
the estimate gap? Arguing for two years is a junior response; decomposing into
independently valuable quarters is the staff response. Fourth: the four original
authors. Candidates who don't mention them are missing the largest organizational risk
in the scenario.

---

## 5. Ambiguous Mandate, Hostile Stakeholders

### Situation

You've been hired as a Staff AI Engineer into a company where your role was created
after an argument.

The context:

- The **Data Science** team (14 people, reports to the Chief Data Officer) believes AI
  belongs to them. They have ML expertise, own the feature store and the model registry,
  and have shipped classical ML models for years. They have limited LLM experience.
- The **Platform Engineering** team (9 people, reports to the VP of Engineering)
  believes AI infrastructure is infrastructure and belongs to them. They own Kubernetes,
  CI/CD, and observability. They have no ML experience.
- **Three product teams** have shipped LLM features by ignoring both groups.
- Your role was created by the CTO to "bring these together." Your reporting line is to
  the CTO. Neither the CDO nor the VP of Engineering was consulted before you were
  hired.
- Your first week includes two separate meetings where each group explains why the
  other shouldn't own AI.

There is no written charter for your role.

### Your Task

Establish yourself and produce something useful within six months.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**The technical problem is not the problem.** This is an organizational conflict with a
technical surface, and treating it as a technical problem — proposing an architecture in
week two — will make it worse, because whatever you propose will be read as a
territorial claim by whichever group it favors.

**Understand the actual dispute.** It's usually not really about who's right; it's about
headcount, scope, career paths, and which executive gets to say AI reports to them.
Diagnosing this correctly determines everything you do next.

**Your position is genuinely weak and it's important to see that clearly:** hired over
both leaders' heads, no team, no budget, no charter, and a mandate ("bring these
together") that is a wish rather than an instruction. Anything you do that looks like
taking territory will unite them against you.

**The available move is to be useful to both, on something neither owns.** Find work
that:
- Neither group is doing.
- Both groups benefit from.
- Product teams urgently need.
- Doesn't require you to win the ownership argument.

Almost always this is **evaluation** and **the shared operational layer** (cost
attribution, quota, provider abstraction, observability for LLM calls). Data Science
doesn't do it because it's engineering; Platform doesn't do it because it requires ML
judgment; product teams need it and have been improvising.

**Second: the three product teams are your actual constituency.** They ignored both
groups because both were unhelpful. If you help them, you build a base of support that
doesn't depend on either executive, and you generate evidence about what the company
actually needs.

**Third: get a charter, but earn it first.** Asking the CTO for a written charter in
week one produces a document neither the CDO nor the VP accepts. Asking in month four,
with evidence and three product teams' support, produces one that sticks.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**Weeks 1–4: listen, and build an accurate map.**

- Meet everyone: both leaders, their senior engineers, all three product teams, plus
  finance (who sees the AI bill), security, and a few PMs.
- Ask about problems, not ownership. "What's hard about shipping an AI feature here?"
  produces useful answers; "who should own AI?" produces positions.
- Produce a written **inventory**: every AI system, what it does, who owns it, what it
  costs, what its quality is (mostly: unknown), and what's blocking each team.

This document is your first deliverable and it is deliberately neutral: it makes no
ownership claims and it is useful to everyone. It also establishes you as someone who
finds out rather than someone who arrives with answers.

**Weeks 4–10: solve a real problem for the product teams.**

Pick the most acute shared pain. Likely candidates:
- No cost attribution (finance is unhappy, teams can't optimize).
- No evaluation (teams ship blind and know it).
- Provider rate limits causing incidents across teams.

Ship something small and genuinely useful in six weeks — a gateway library with cost
attribution and quota isolation, or a reference eval harness plus help building the
first suite. **Do it with the product teams, and credit them.**

Two effects: product teams now want you to succeed, and you have evidence rather than
opinions about what the company needs.

**Months 3–4: make both groups' expertise necessary rather than redundant.**

This is the key move. The ownership argument dissolves if the work visibly requires
both:

- **Data Science owns evaluation methodology.** They have the statistical expertise
  nobody else has — sampling, significance, bias measurement, experiment design. Frame
  evaluation as their domain and ask for their help. It's true, it's flattering, and it
  gives them a real role in LLM work that plays to their strengths rather than asking
  them to become prompt engineers.
- **Platform owns the serving, deployment, and observability layer.** Also true, also
  plays to strengths, and doesn't require ML knowledge.
- **You own the seam**: the LLM-specific practices, the shared libraries, the standards,
  and the relationships with product teams.

Propose this explicitly, to both leaders, as a division of labor rather than a division
of territory — and propose it in a document where each of them gets something they
wanted. Get their edits before it goes to the CTO. **The goal is a proposal that both
can claim as partly theirs.**

**Months 4–6: get the charter written, with their names on it.**

Now go to the CTO — ideally with the CDO and VP of Engineering — with a proposed
operating model backed by four months of evidence and three product teams' experience.
A charter co-signed by both leaders is durable; one imposed by the CTO is not.

**Things to avoid, deliberately:**

- Do not propose an org change in your first three months.
- Do not let either group's engineers be assigned to you before there's a charter; a
  team without a mandate becomes a target.
- Do not become the person who runs everyone's AI infrastructure by default because
  nobody else will. Responsibility without authority is the staff engineer's most
  common career failure and it is very easy to fall into here.
- Do not win the argument. If one group "loses," you inherit a hostile stakeholder with
  a long memory.

**If it doesn't work:** be honest with yourself at month six. If both leaders are still
hostile, the CTO won't back a charter, and you have no base, the role may be
unworkable as constructed. Knowing when a role is structurally impossible is a legitimate
staff-level skill, and the interview answer that acknowledges this is stronger than one
that assumes success.

</details>

### Trade-offs

**Claim ownership early and decisively.** Fast clarity, and if the CTO backs you it
works. But it makes an enemy of at least one executive, and probably both, and you have
no team to execute with. High variance, usually bad.

**Stay neutral and be useful (chosen).** Builds a base, accumulates evidence, keeps both
groups as potential allies, and produces value regardless of how the ownership question
resolves. But it's slow, you have no formal authority for months, and if you never get
a charter you're a well-liked individual contributor with a staff title.

**Ask the CTO to resolve it immediately.** Clean in principle. But an imposed resolution
in month one, without evidence, will be relitigated continuously and will poison both
relationships. It also signals you can't operate in ambiguity, which is most of what the
job is.

### What Could Go Wrong?

- **Both groups unite against you** as the outsider. Mitigation: never take territory;
  always give credit; make both groups more necessary, not less.
- **You become everyone's helper and own nothing.** Six months of being useful with no
  charter is a career problem. Set an internal deadline for the charter conversation.
- **A product team's AI feature has an incident** and you're blamed because you "own AI"
  informally without any control. Be explicit in writing about what you are and are not
  responsible for.
- **The CTO leaves.** Your only sponsor. Build relationships with the CDO and VP of
  Engineering as insurance from week one.
- **The evaluation work gets captured** by Data Science and turned into a heavyweight
  process that product teams route around. Stay involved in how it lands, not just who
  owns it.
- **You solve the wrong problem** because you listened to the loudest stakeholder rather
  than measuring. The inventory protects against this.

### Metrics

*Early:* the inventory exists and is used; product teams request your help unprompted (a
direct measure of whether you're useful); cost attribution coverage.
*Mid:* AI features with evaluation; incidents caused by shared infrastructure problems
(rate limits, quota contention); time-to-ship a new AI feature.
*Organizational:* a written charter co-signed by both leaders; joint work between Data
Science and Platform on AI initiatives; whether the ownership argument still comes up in
meetings.

### Follow-up Questions

1. Month 3: the CDO proposes to the CTO that the AI function should report to them and
   that your role should be folded into Data Science. What do you do?
2. A product team asks you to take over operating their AI feature. Do you?
3. The VP of Engineering assigns two of their engineers to "help you" without discussing
   it. What's your response?
4. What would make you conclude this role is unworkable, and what would you do then?
5. How would your approach differ if you'd been hired by the CDO instead of the CTO?
6. Six months in, you have product-team support and no charter. What's your move?

### What Separates Senior From Staff?

A senior engineer in this situation does excellent technical work and waits for the
organization to sort itself out.

A staff engineer recognizes the ownership conflict as the primary problem, declines to
resolve it by winning, and instead **restructures the work so that both groups'
expertise is necessary** — which makes the conflict irrelevant rather than resolved.
They build a constituency among product teams that doesn't depend on either executive.
They sequence the charter conversation for month four rather than week one, because a
charter backed by evidence and co-signed by rivals is durable and one imposed from above
is not. They are explicit in writing about what they are not responsible for. And they
know what "unworkable" looks like and are willing to name it.

### Interviewer Notes

This scenario has no technical content by design, and candidates who steer it toward
architecture are answering a different question. The signals: do they recognize the
conflict is about scope and headcount rather than correctness? Do they refuse to claim
territory early? Do they identify the product teams as the available constituency? Do
they find work neither group owns? The strongest answer includes making both groups
*more* necessary — that's the move that dissolves rather than wins the argument. Also
listen for whether they protect themselves from responsibility-without-authority; it's
the most common way staff engineers fail in real organizations and awareness of it is a
maturity signal.

---

## 6. Prioritizing When Everything Is Urgent

### Situation

You are the Staff AI Engineer on a team of six. Over one week, the following arrive:

1. **Security:** an LLM-powered internal tool can be prompt-injected to retrieve
   documents the user shouldn't see. Proof of concept attached. No evidence of
   exploitation.
2. **Finance:** AI spend is 40% over budget and trending worse. They want a plan by
   month-end.
3. **The largest customer** (18% of ARR) says the AI feature "hallucinates" and has
   escalated to their CEO, who has escalated to yours.
4. **Product:** a competitor shipped an agentic feature; product wants an equivalent in
   the next quarter. The CEO is asking about it in every all-hands.
5. **Your own backlog:** the eval suite is 9 months stale and no longer reflects
   production traffic. Three of the last five incidents would have been caught by a
   current one.
6. **An engineer on your team** has told you privately they're considering leaving
   because they've spent six months on infrastructure and want to work on modeling.
7. **A partner integration** goes live in three weeks and its AI component has never
   been load-tested.

Your team of six is currently fully allocated to (4) — the agentic feature.

### Your Task

Prioritize. Say what you do, what you delay, and what you decline.

### How Would You Approach It?

<details>
<summary>💡 Reveal the reasoning path</summary>

**Classify by consequence and reversibility, not by who is shouting.**

| Item | Worst case | Reversible? | Clock |
|---|---|---|---|
| 1. Prompt injection / data exposure | Data breach, legal, trust | **No** | Now |
| 3. Largest customer escalation | 18% of ARR churns | Partly | Days |
| 7. Untested partner launch | Public failure at launch | Partly | 3 weeks |
| 2. Budget overrun | Financial pain, credibility | Yes | Month-end |
| 5. Stale eval suite | Recurring incidents | Yes | Chronic |
| 6. Engineer attrition risk | Lose a person, 6-month gap | **Hard to reverse** | Weeks |
| 4. Competitive feature | Lost deals, CEO pressure | Yes | Quarter |

**The ranking almost writes itself once you frame it this way**, and the interesting
part is what it says about item 4: the thing your entire team is working on, and the
thing the CEO asks about weekly, is the *lowest-consequence, most-reversible* item on
the list. That's the uncomfortable finding, and stating it is the point of this
scenario.

**Second: several of these are the same problem.** Items 3 and 5 are connected — the
customer complaint is evidence that quality is unmeasured, which is what the stale eval
suite means. Fixing 5 is how you actually fix 3 rather than patching it. Item 2 and
item 4 also interact: the agentic feature will increase spend, so shipping it without
addressing cost makes 2 worse.

**Third: not everything needs your team.** Item 2 (budget) needs analysis and a plan,
not six engineers. Item 6 needs a conversation, not a sprint. Item 7 needs a day of load
testing, not a project. Sizing correctly is most of prioritization; treating every
incoming item as a project is how teams become paralyzed.

**Fourth: item 6 is easy to deprioritize and shouldn't be.** Losing an engineer from a
team of six is a 17% capacity cut plus months of hiring and ramp. It's also the item
that will be invisible until it's irreversible. Staff engineers who ignore team health
because it isn't technical are making a resourcing error.
</details>

<details>
<summary>✅ Reveal a strong answer</summary>

**This week:**

**Item 1 (prompt injection) — immediate, 2 engineers.** A demonstrated ability to
retrieve unauthorized documents is a security vulnerability with unbounded downside and
no undo. Response: assess exploitability today, restrict the tool's document access
scope or disable the affected path within 24 hours, notify security and legal, check
logs for prior exploitation. The durable fix is capability restriction — permission
filtering enforced at the retrieval layer, not in the prompt — and it takes longer, but
containment happens now.

**Item 3 (customer escalation) — 1 engineer plus you, immediately.** But do the right
thing rather than the reflexive thing: **get specifics before promising anything.** Ask
the customer for concrete examples. Then diagnose properly — is it retrieval failure
(the right document wasn't found), generation failure (the model ignored the context),
or expectation mismatch (the product promises more than it does)? These have completely
different fixes and the customer's word "hallucinates" tells you nothing about which.

Communicate within 24 hours with what you know, what you're doing, and when they'll hear
next. Escalations become churn through silence more often than through the underlying
problem.

**Item 6 (the engineer) — a conversation this week, costs an hour.** Find out what they
actually want, and whether there's a version of it available. If the next two quarters
are infrastructure, say so honestly rather than making a vague promise. Losing them is
worse than losing a quarter of the agentic feature, and this is the cheapest item on the
list to address.

**Item 7 (partner launch) — schedule now, 1 engineer for 2 days in week 2.** It's small
and it has a hard date. Do it early; a load test that finds a problem in week 1 is
fixable, one in week 3 is a launch delay.

**Next two weeks:**

**Item 5 (eval suite) — 2 engineers, ongoing.** This is the item that everything else
depends on. The customer escalation cannot be resolved credibly without it (how do you
prove you fixed it?), the agentic feature cannot be shipped safely without it, and three
of five recent incidents would have been caught. **Rebuilding the eval suite from
current production traffic is the highest-leverage work available**, and it is exactly
the kind of work that never gets prioritized because it has no external advocate.

Make the case explicitly to leadership using the incident data: three of the last five
incidents, with their cost, would have been prevented.

**Item 2 (budget) — you, a few days of analysis.** Produce the cost decomposition and a
plan with options and their quality/velocity consequences. This is analysis work, not
team work. Deliver by month-end as asked. Note in it that the agentic feature will
increase spend and quantify by how much — leadership should decide that trade
knowingly.

**Item 4 (agentic feature) — descoped and delayed, deliberately.**

This is the hard conversation and the one the scenario is really testing. Go to the CEO
and product leadership with:

- Why the other items outrank it (consequence and reversibility, with the security item
  first).
- What the feature actually needs to ship safely: evaluation for agent behavior, cost
  controls for an unbounded-loop workload, and security review for tool access — none of
  which exist.
- A revised plan: a narrower version of the feature, one quarter later, built on
  foundations that exist by then.
- The honest statement that shipping an agentic feature in a system with a known
  injection vulnerability, no current eval suite, and a 40% budget overrun is how you
  get a much worse week than this one.

**What you decline outright:** shipping the full agentic feature this quarter. Say it
clearly rather than accepting the deadline and missing it, which is the failure mode
that destroys trust.

**How to present the whole thing:** a single page showing all seven items, their
consequence, reversibility, and your allocation. Give leadership the ability to override
your ranking with full information — that's their right — but make the trade explicit.
**A prioritization decision made visibly, with reasoning, is defensible even when it's
wrong; one made silently is not.**

</details>

### Trade-offs

**Follow the loudest signal (CEO's competitive feature).** Politically safe short-term,
keeps leadership happy. But it ships an agentic feature onto a foundation with a known
security hole, no evaluation, and a budget overrun — an incident-generating machine, and
the resulting incident is far more damaging than the delay would have been.

**Pure risk-ranking (chosen).** Correct on consequences. But it requires pushing back on
the CEO in your first year, and if the competitive threat is real, being right about
security while losing the market is not obviously a win. This is why the answer includes
a *revised* plan for item 4 rather than a refusal.

**Try to do everything.** Six engineers across seven initiatives means nothing finishes,
everything is late, and quality suffers everywhere. The most common failure and the most
tempting.

### What Could Go Wrong?

- **You're overruled on item 4** and told to ship anyway. Then: document the risks in
  writing, get the security fix done regardless (it's non-negotiable), descope
  aggressively, and ship the narrowest version. Don't sulk, and don't quietly comply
  while expecting failure.
- **The customer escalation turns out to be an expectation problem**, not a quality
  problem — meaning the fix is in sales and product messaging, not engineering. Diagnose
  before committing engineering effort.
- **The prompt injection was already exploited** and you find evidence in the logs. Now
  it's an incident with disclosure obligations, and your week changes again.
- **The engineer leaves anyway.** Sometimes they do. The conversation is still correct;
  it just doesn't always work.
- **The eval suite rebuild takes longer than two weeks** — it usually does, because the
  hard part is labeling. Scope it to a useful subset first.
- **You spend all your credibility on the item-4 conversation** and have none left for
  the next one. Pick the battles that matter and make the case with data.

### Metrics

For the prioritization itself: are the highest-consequence items resolved first? Is the
allocation visible to stakeholders? Was any item resolved by someone other than your
team (correct delegation)?

For the outcomes: time-to-containment on the security issue; customer escalation
resolved and the account retained; eval suite coverage of current production traffic;
budget trajectory; partner launch without incident; team retention; and — the honest one
— whether the delayed feature, when shipped, was better for having waited.

### Follow-up Questions

1. The CEO says the competitive feature is existential and must ship this quarter.
   Everything else waits. What do you do?
2. How do you decide the eval suite is "good enough" to stop working on it?
3. Your two engineers on the security fix say the proper fix takes eight weeks, not one.
   How do you handle the gap?
4. What if the customer escalation turns out to be about a feature working as designed?
5. Which of these seven would you have prevented, and how?
6. A month later, three more urgent items arrive. Has your process changed?

### What Separates Senior From Staff?

A senior engineer prioritizes well within the work they've been given and executes.

A staff engineer prioritizes across the *whole* set, including the items that aren't
engineering tickets (the departing engineer, the budget analysis, the customer
relationship). They classify by consequence and reversibility rather than by volume of
escalation, and they notice that the item everyone is watching is the least consequential
one. They size items correctly — a load test is two days, not a project — which is most
of what prioritization is. They have the hard conversation with the CEO with a *revised
plan* rather than a refusal. They make the trade-off visible so leadership can override
with full information. And they name what they are declining, out loud, rather than
accepting everything and delivering late.

### Interviewer Notes

This scenario is deliberately overloaded and the signal is in the sorting. Watch for:
does the candidate use an explicit framework (consequence × reversibility, or similar)
or just react? Do they put the security issue first? Do they notice that items 3 and 5
are the same underlying problem? Do they size items correctly, or treat everything as a
project? Do they address the departing engineer — most candidates skip it, and it's the
item with the worst effort-to-impact ratio if ignored? And critically: do they push back
on item 4 with a revised plan, or do they either capitulate or refuse? The presence of a
*credible alternative* is what makes pushback a staff-level behavior rather than
obstruction.

---

## Staff-Level Interview Patterns

Recurring shapes worth recognizing.

**The consolidation trap.** You're shown a mess of heterogeneous systems and expected to
propose standardization. The staff answer ranks the problems by harm, standardizes the
few things that matter, and publicly declines the rest.

**The mandate you don't have.** You're asked to change behavior across teams you don't
control. The answer is never "get leadership to mandate it." It's: find the risk that
creates its own mandate, make adoption nearly free, serve teams who already want help,
and let visibility do the work of authority.

**The unmeasurable requirement.** "Users must not notice." "Be AI-first." "Improve
quality." The staff move is to convert it into something falsifiable with a threshold, a
measurement method, and an owner — then get agreement on that definition.

**The reframe.** The stated problem is not the real one. A "model selection" problem is a
verification problem; an "agent framework" problem is a durable-workflow problem; a
"two-year migration" is a vendor deadline with modernization attached. Look for the
constraint with a date on it and the requirement nobody has questioned.

**The refusal.** Every good staff answer contains something the candidate will not do,
stated clearly, with a reason a non-engineer would accept. A plan with no "no" in it is
a wish list.

**The successor.** Staff work outlives the person doing it. Trigger conditions for
revisiting decisions, written rationale for rejected options, versioned metadata that
makes future migrations possible, and named owners. The question behind all of it:
*will the next person understand why, and be able to change it?*
