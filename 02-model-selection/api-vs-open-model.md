# API or Your Own Model?

Two options: call someone else's model over the internet, or run a model yourself on
your own hardware.

Most teams should use an API. That's not a cop-out — it's what the numbers usually say.
But there are real cases where running your own is clearly right, and this file is about
telling them apart.

> **Note on numbers.** All prices and hardware figures here are illustrative. Prices
> change constantly. What doesn't change is the *shape* of the calculation, which is
> what you should learn.

---

## The short version

| | API | Your own |
|---|---|---|
| Setup time | Minutes | Weeks |
| Cost at low volume | Cheap | Very expensive |
| Cost at high volume | Expensive | Much cheaper |
| Who's on call | Them | You |
| Model quality | Usually the best available | Usually a step behind |
| Rate limits | Yes | Only your hardware |
| Your data | Leaves your network | Stays put |
| Model changes under you | Yes, sometimes silently | Never |

---

## Do the arithmetic first

People argue about this from taste. Don't. Work it out.

### What you need to know

```
requests per month
average input tokens per request
average output tokens per request
```

Then:

```python
def monthly_api_cost(requests, in_tokens, out_tokens, price_in, price_out):
    """price_* are per 1 million tokens."""
    total_in = requests * in_tokens
    total_out = requests * out_tokens
    return (total_in / 1_000_000 * price_in) + (total_out / 1_000_000 * price_out)
```

Example: 5 million requests a month, 800 tokens in, 200 out, at $0.50/M input and
$1.50/M output:

```
input:  5,000,000 × 800   = 4,000,000,000 tokens → $2,000
output: 5,000,000 × 200   = 1,000,000,000 tokens →   $1,500
                                            total = $3,500/month
```

$3,500 a month. **Do not self-host this.** One engineer's time for a month costs more
than a year of that bill.

Now change one number — 200 million requests a month:

```
input:  160,000,000,000 tokens → $80,000
output:  40,000,000,000 tokens → $60,000
                         total = $140,000/month
```

That's $1.7M a year. Now self-hosting is worth a serious look.

**The rough dividing line is somewhere around $30–50k/month of API spend.** Below that,
the engineering and operational cost of self-hosting usually exceeds the savings.

### The cost people forget

Self-hosting isn't just GPU rental. The real number:

```python
def monthly_selfhost_cost(gpus, gpu_hourly, engineers, engineer_monthly):
    hardware = gpus * gpu_hourly * 730          # hours in a month
    people = engineers * engineer_monthly
    return hardware + people
```

Two engineers at a fully-loaded $25k/month each is **$50,000/month before you rent a
single GPU.** That's the number that kills most self-hosting business cases, and it's
the one that's missing from most self-hosting business cases.

And those engineers aren't a one-off. Serving infrastructure needs ongoing care:
capacity planning, model upgrades, on-call, debugging.

---

## What actually decides it

Cost is one input. Four others matter as much or more.

### 1. What shape is your workload?

This is the part people miss, and it changes the answer more than volume does.

**Good for self-hosting:**
- Short inputs, short outputs (classification, extraction, routing, moderation)
- Steady, predictable traffic
- One narrow task
- Latency-tolerant, or latency-critical in a way an API can't meet

**Bad for self-hosting:**
- Long inputs and long outputs (document summarization, long-form writing)
- Spiky traffic — quiet at night, 10× at lunch
- Many different tasks needing different capabilities
- You need the strongest reasoning available

Why? Because self-hosting economics live or die on **GPU utilization**. A GPU costs the
same whether it's busy or idle. If your traffic is 10× at peak and quiet overnight, you
either pay for peak capacity all day (terrible economics) or queue at peak (bad
latency).

Short outputs help because generating tokens is the slow, expensive part. A workload
that reads 200 tokens and writes 20 gets far more throughput per GPU than one that
writes 2,000.

### 2. Where does your data have to live?

Sometimes this decides it on its own, and cost doesn't get a vote.

- Health records, financial data, or anything under a regulation that says data can't
  leave your network.
- A customer contract that names where data may be processed.
- A security team that won't approve a new supplier.

If data genuinely cannot leave your environment, self-hosting (or a private deployment
of a provider's model) is the answer regardless of the spreadsheet.

Be careful to check what the requirement *actually* is. "Must not be trained on" is a
contract term most providers will give you. "Must not leave our network" is an
architecture requirement. They sound similar and cost very different amounts.

### 3. Can your team run it?

Be honest here. Self-hosting means owning:

- GPU capacity planning
- A serving stack (batching, memory management, queueing)
- Model upgrades and the testing around them
- Cold starts — loading model weights takes minutes, so you can't scale up quickly
- On-call for all of it

If nobody on the team has done this, you're not choosing between two options — you're
choosing between an API and a hiring plan.

And ask the uncomfortable question: **if the two people who know this leave, can anyone
else run it?** A system saving $1M a year that only two people understand is a risk, not
just a saving.

### 4. Does the model quality clear your bar?

Open models have got very good, especially for narrow tasks. But test before you commit.

The pattern that works:

1. Take a real sample of your traffic.
2. Run it through the API model you use now. Record the results.
3. Run it through the candidate open model.
4. Compare, **per segment** — easy cases and hard cases separately.

Often you'll find the open model matches on the easy 80% and loses on the hard 20%.
That's not a failure — that's a routing design (see
[model-routing.md](model-routing.md)).

---

## The middle options

It's not two choices. Several things sit in between.

**Private deployment from a provider.** The provider's model, running in your cloud
account or a dedicated instance. You get data residency without running a serving stack.
Costs more than the shared API, less hassle than self-hosting.

**Open model, hosted by someone else.** Several companies will serve open-weights models
for you. You get model control (they can't silently change it) without the
infrastructure. Good middle ground, and often overlooked.

**Split the traffic.** Self-host the high-volume simple path, use an API for the hard
cases. This is usually the best answer at scale — you get most of the savings with
much less risk.

```python
def route(request):
    if is_simple(request):
        return local_model(request)        # 85% of traffic, self-hosted
    return api_model(request)              # 15%, hard cases
```

---

## The one thing people underestimate

**Self-hosting means you own model quality forever.**

On an API, the provider improves the model and you get better for free. Self-hosted,
every improvement is a project you run: find a better model, evaluate it, migrate,
re-tune your prompts, re-check your thresholds.

That's fine if you've decided model quality is stable enough for your task. It's a
problem if you're in a fast-moving area and expected to keep up.

The flip side is also true and worth naming: **on an API, the provider can change the
model under you.** A silent update can change your outputs, break your parsing, or shift
your quality — and you find out from a metric, not an announcement. Self-hosting
removes that entirely, which for some products is worth real money.

---

## If you do self-host

Things to get right:

**Use a proper serving stack.** Don't write your own inference loop. Existing servers
handle continuous batching (running many requests together efficiently) and paged
attention (managing memory for many concurrent requests) far better than anything you'll
write in a month.

**Measure utilization, not throughput.** Your business case assumed a utilization
number. Check it. 45% utilization instead of 80% roughly doubles your cost per request
and can wipe out the whole saving.

**Keep an API fallback, with real traffic on it.** Two reasons: it absorbs traffic
spikes that your fixed GPU count can't, and it stays working because you use it. A
fallback you never exercise is a fallback that's broken when you need it. Sending 2–5%
of traffic there permanently is cheap insurance.

**Plan for cold starts.** Loading weights takes minutes. Autoscaling that assumes
seconds will fail you at exactly the wrong moment.

**Watch for input length spikes.** Long inputs eat memory much faster than short ones.
Cap input length at the door, not inside the model server.

---

## Migrating from API to self-hosted

If you decide to move, do it in this order. The order matters.

**1. Prove the model first, not the infrastructure.**

Can an open model hit your quality bar? Test it against a real sample before you buy a
single GPU. Many teams build the serving stack first and then discover a quality gap
they can't close.

**2. Shadow mode.**

Run the self-hosted model on real traffic without using its output. Log both, compare.
Full-scale comparison at zero risk to users. This is the standard way to de-risk a
migration and skipping it is a common mistake.

**3. Small percentage, then ramp.**

1% → 5% → 25% → 100%, with a feature flag and instant rollback at each step.

**4. Keep the API path forever.**

Not as a transition step — as permanent overflow and fallback.

---

## Interview questions

**1. "API or self-hosted?"**

The right first move is to ask for numbers: volume, input and output token counts,
traffic shape, data rules, team experience. Then do the arithmetic out loud. A candidate
who picks a side before asking is guessing.

**2. "At what point does self-hosting make sense?"**

Roughly $30–50k/month of API spend, *and* a workload shape that gives high GPU
utilization, *and* a team that can run it. All three, not just the first. Then mention
that a data-residency requirement can override the economics entirely.

**3. "Your business case assumed 80% GPU utilization. It's running at 45%. Now what?"**

Cost per request has roughly doubled and the saving may be gone. Options: consolidate
workloads onto the same GPUs, use the spare capacity for batch jobs, shrink the fleet
and send peak traffic to the API, or unwind. The important part is having modeled the
diurnal traffic curve rather than the average.

**4. "What's the biggest hidden cost of self-hosting?"**

Engineering time, ongoing — not the GPUs. Two engineers is $50k/month before hardware.
Then mention that you now own model quality forever, and the bus-factor risk.

**5. "How would you migrate?"**

Model quality first, then shadow mode on full traffic, then a percentage ramp with
rollback, keeping the API path permanently for overflow.

---

## What to remember

- Do the arithmetic before the argument. Volume, token counts, traffic shape.
- Below roughly $30–50k/month API spend, self-hosting rarely pays.
- Engineers cost more than GPUs. Two of them is $50k/month.
- Utilization decides the economics. Spiky traffic ruins it.
- Short inputs and outputs suit self-hosting; long ones don't.
- Data-residency rules can override cost entirely — check what the rule actually says.
- Self-hosting means you own model quality forever; APIs can change under you.
- Middle options exist: private deployments, hosted open models, split traffic.
- Prove the model before building the infrastructure.

---

**Next:** [model-routing.md](model-routing.md) — sending different requests to different
models.
