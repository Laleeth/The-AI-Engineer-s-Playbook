#!/usr/bin/env python3
"""Verify the repository's code, links, claims and self-description.

Four checks:

  1. Every ```python block parses as valid Python.
  2. Every internal markdown link points at a file that exists, and every
     anchor points at a heading that exists.
  3. Numeric claims made in the prose match what the code actually computes.
  4. The README and ROADMAP describe the repository that actually exists —
     section counts, file counts, word counts, scenario counts, and the list
     of sections that are still unwritten.

Check 4 exists because the first three didn't catch the failure that actually
happened: the file counts, the "still to come" list and a worked example were
all edited in one place and left stale in another. A repo that claims every
number is checked has to check the numbers it makes about itself.

Run:  python scripts/verify.py
CI runs this on every push. If it fails, the README's "verified" badge is lying.
"""

from __future__ import annotations

import ast
import math
import pathlib
import re
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Fenced blocks can be indented when nested in a list; capture the indent
# so it can be stripped before parsing.
FENCE = re.compile(r"^([ \t]*)```python\n(.*?)^[ \t]*```", re.DOTALL | re.MULTILINE)
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6} (.+)$", re.MULTILINE)


def markdown_files() -> list[pathlib.Path]:
    return sorted(
        p for p in ROOT.rglob("*.md")
        if ".git" not in p.parts
    )


def slug(heading: str) -> str:
    """GitHub's heading-anchor rule, near enough for our purposes."""
    return re.sub(r"[^a-z0-9 -]", "", heading.strip().lower()).replace(" ", "-")


# ---------------------------------------------------------------------------
# 1. Python blocks parse
# ---------------------------------------------------------------------------

def check_python_blocks() -> tuple[int, list[str]]:
    total, failures = 0, []
    for md in markdown_files():
        for i, (indent, raw) in enumerate(FENCE.findall(md.read_text()), start=1):
            block = textwrap.dedent(raw) if indent else raw
            total += 1
            try:
                ast.parse(block)
            except SyntaxError as exc:
                rel = md.relative_to(ROOT)
                failures.append(f"{rel} block #{i} line {exc.lineno}: {exc.msg}")
    return total, failures


# ---------------------------------------------------------------------------
# 2. Internal links resolve
# ---------------------------------------------------------------------------

def check_links() -> tuple[int, list[str]]:
    total, failures = 0, []
    for md in markdown_files():
        text = md.read_text()
        own_anchors = {slug(h) for h in HEADING.findall(text)}
        rel = md.relative_to(ROOT)

        for _, target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            total += 1
            path, _, frag = target.partition("#")

            if path:
                dest = (md.parent / path).resolve()
                if not dest.exists():
                    failures.append(f"{rel}: missing file -> {target}")
                    continue
                if frag and dest.suffix == ".md":
                    anchors = {slug(h) for h in HEADING.findall(dest.read_text())}
                    if frag not in anchors:
                        failures.append(f"{rel}: missing anchor -> {target}")
            elif frag and frag not in own_anchors:
                failures.append(f"{rel}: missing local anchor -> #{frag}")

    return total, failures


# ---------------------------------------------------------------------------
# 3. Numeric claims in the prose are actually true
# ---------------------------------------------------------------------------

def check_claims() -> tuple[int, list[str]]:
    """Each entry re-computes a figure quoted somewhere in the docs.

    When you change a number in the prose, change it here too — that is the
    point. A claim nobody re-computes is a claim that quietly goes wrong.
    """
    failures: list[str] = []

    def check(name: str, got: float, expected: float, tolerance: float = 0.05):
        if expected == 0:
            ok = got == 0
        else:
            ok = abs(got - expected) / abs(expected) <= tolerance
        if not ok:
            failures.append(f"{name}: computed {got:,.4g}, docs say {expected:,.4g}")

    # --- 01-llm-internals/attention.md ---
    def attention_cost_ratio(len_a, len_b):
        return (len_b / len_a) ** 2

    check("attention 1k->4k cost ratio", attention_cost_ratio(1_000, 4_000), 16)
    check("attention 4k->128k cost ratio", attention_cost_ratio(4_000, 128_000), 1024)

    def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_each=2):
        return 2 * layers * kv_heads * head_dim * bytes_each

    check("MHA cache/token (32 kv heads)", kv_bytes_per_token(32, 32, 128), 524_288)
    check("GQA cache/token (8 kv heads)", kv_bytes_per_token(32, 8, 128), 131_072)
    check("MQA cache/token (1 kv head)", kv_bytes_per_token(32, 1, 128), 16_384)
    check("GQA is 4x smaller than MHA",
          kv_bytes_per_token(32, 32, 128) / kv_bytes_per_token(32, 8, 128), 4)

    # --- 01-llm-internals/kv-cache.md ---
    check("GQA cache/token in KB", kv_bytes_per_token(32, 8, 128) / 1024, 128)
    check("cache at 4k tokens in MB",
          kv_bytes_per_token(32, 8, 128) * 4_000 / 1e6, 512, 0.03)
    check("cache at 128k tokens in GB",
          kv_bytes_per_token(32, 8, 128) * 128_000 / 1e9, 16, 0.05)
    # 80 GB GPU, 26 GB weights, 2,500-token contexts
    check("concurrent requests on an 80GB GPU",
          (80 - 26) * 1e9 / (kv_bytes_per_token(32, 8, 128) * 2_500), 165, 0.005)

    # --- 01-llm-internals/quantization.md ---
    def model_size_gb(params_billion, bits):
        return params_billion * 1e9 * bits / 8 / 1e9

    check("8B at BF16", model_size_gb(8, 16), 16)
    check("70B at BF16", model_size_gb(70, 16), 140)
    check("8B at FP8", model_size_gb(8, 8), 8)
    check("70B at INT4", model_size_gb(70, 4), 35)

    # --- 01-llm-internals/prefill-vs-decode.md ---
    check("prefill 2k tokens at 20k/s (ms)", 2_000 / 20_000 * 1000, 100)
    check("decode 500 tokens at 50/s (ms)", 500 / 50 * 1000, 10_000)
    check("decode is ~100x slower per token",
          (1 / 50) / (1 / 20_000), 400, 0.01)   # per-token ratio
    # capacity example: 500 rps, 2000 in, 500 out
    check("prefill load tokens/sec", 500 * 2_000, 1_000_000)
    check("decode load tokens/sec", 500 * 500, 250_000)
    check("gpus for prefill", 500 * 2_000 / 20_000, 50)
    check("gpus for decode", 500 * 500 / 2_000, 125)

    # --- 02-model-selection/api-vs-open-model.md ---
    def api_cost(requests, tin, tout, pin, pout):
        return requests * tin / 1e6 * pin + requests * tout / 1e6 * pout

    check("api cost 5M requests", api_cost(5_000_000, 800, 200, 0.50, 1.50), 3_500)
    check("api cost 200M requests", api_cost(200_000_000, 800, 200, 0.50, 1.50), 140_000)
    check("api cost 200M annualised", 140_000 * 12, 1_680_000)
    check("two engineers per month", 2 * 25_000, 50_000)

    # --- 02-model-selection/cost-quality-latency.md ---
    cost_in = 3000 / 1e6 * 0.50
    cost_out = 300 / 1e6 * 1.50
    check("RAG input share of cost", cost_in / (cost_in + cost_out), 0.77, 0.03)

    # --- 03-rag: storage arithmetic ---
    def storage_gb(n, dims, bytes_each=4):
        return n * dims * bytes_each / 1e9

    check("10M x 768d storage", storage_gb(10_000_000, 768), 30.7)
    check("10M x 1536d storage", storage_gb(10_000_000, 1536), 61.4)
    check("10M x 3072d storage", storage_gb(10_000_000, 3072), 122.9)
    check("50k x 768d storage MB", storage_gb(50_000, 768) * 1000, 154)
    check("50M x 1536d storage", storage_gb(50_000_000, 1536), 307)

    # --- 12-senior-scenarios/model-selection.md: the legal corpus ---
    check("400M x 3072d storage", storage_gb(400_000_000, 3072), 4_915)
    check("400M x 1536d storage", storage_gb(400_000_000, 1536), 2_458)
    check("400M x 768d storage", storage_gb(400_000_000, 768), 1_229)

    # --- 12-senior-scenarios/scaling.md: the 48M chunk index ---
    check("48M x 1024d storage", storage_gb(48_000_000, 1024), 196, 0.02)

    # --- 03-rag/reranking.md ---
    check("reranking context reduction", (20 - 5) / 20, 0.75)
    check("20 chunks monthly cost", 20 * 400 * 5_000_000 / 1e6 * 0.50, 20_000)
    check("5 chunks monthly cost", 5 * 400 * 5_000_000 / 1e6 * 0.50, 5_000)

    # --- 03-rag/contextual-retrieval.md ---
    check("context generation 500k chunks", 500_000 * 8_000 / 1e6 * 0.25, 1_000)

    # --- 03-rag/multimodal-rag.md ---
    def image_cost(n):
        return n * 1500 / 1e6 * 1.0 + n * 150 / 1e6 * 3.0

    check("image indexing 100k", image_cost(100_000), 195)
    check("image indexing 10M", image_cost(10_000_000), 19_500)

    # --- 04-agents/memory.md and agent-vs-workflow.md ---
    def context_tokens(steps, per_step=2000):
        return sum(per_step * i for i in range(1, steps + 1))

    check("agent context 10 steps", context_tokens(10), 110_000)
    check("agent context 20 steps", context_tokens(20), 420_000)
    check("20 vs 10 step ratio", context_tokens(20) / context_tokens(10), 3.8, 0.05)
    check("workflow 3 calls", 2000 + 4000 + 6000, 12_000)
    check("agent 12 steps", context_tokens(12), 156_000)
    check("agent vs workflow ratio", context_tokens(12) / 12_000, 13.0, 0.02)

    # --- 05-evaluation/regression-testing.md: the worked example ---
    # 144 both right, 20 both wrong, 22 fixed by B, 14 broken by B
    old_correct = 144 + 14
    new_correct = 144 + 22
    check("worked example: old score", old_correct / 200, 0.79, 0.01)
    check("worked example: new score", new_correct / 200, 0.83, 0.01)

    p = mcnemar(new_fixed=22, new_broke=14)
    check("worked example: p-value", p, 0.24, 0.10)
    if p <= 0.05:
        failures.append(
            f"worked example must NOT be significant, got p={p:.3f} — "
            f"the whole point of that section is that it is within noise"
        )

    # --- 05-evaluation/regression-testing.md: sample size table ---
    def examples_needed(effect, sd=0.42):
        return math.ceil(((1.96 + 0.84) * sd / effect) ** 2)

    check("sample size 15 points", examples_needed(0.15), 60, 0.12)
    check("sample size 10 points", examples_needed(0.10), 140, 0.12)
    check("sample size 5 points", examples_needed(0.05), 550, 0.12)
    check("sample size 2 points", examples_needed(0.02), 3_500, 0.12)
    check("sample size 1 point", examples_needed(0.01), 14_000, 0.12)
    check("sample size 4 points (the ~880 claim)", examples_needed(0.04), 880, 0.10)

    # --- 06-inference-serving/serving-stacks.md and gpu-economics.md ---
    def cost_per_m_output(tokens_per_second, utilization=1.0, gpu_hourly=2.00):
        return gpu_hourly / (tokens_per_second * utilization * 3600 / 1e6)

    check("output cost at 2,000 tok/s", cost_per_m_output(2_000), 0.28, 0.02)
    check("output cost at 400 tok/s", cost_per_m_output(400), 1.39, 0.01)
    check("cost at 75% utilization", cost_per_m_output(2_000, 0.75), 0.37, 0.01)
    check("cost at 45% utilization", cost_per_m_output(2_000, 0.45), 0.62, 0.01)
    check("cost at 20% utilization", cost_per_m_output(2_000, 0.20), 1.39, 0.01)
    # the "5x" claim in the section README is the ratio of those two extremes
    check("serving efficiency swing",
          cost_per_m_output(400) / cost_per_m_output(2_000), 5.0, 0.01)
    check("vs API at $1.50 (full utilization)", 1.50 / cost_per_m_output(2_000), 5.4, 0.01)
    check("vs API at $1.50 (45% utilization)",
          1.50 / cost_per_m_output(2_000, 0.45), 2.4, 0.02)

    # --- 06-inference-serving/continuous-batching.md ---
    seconds_per_step = 26 / 3_000            # 26 GB of weights at ~3 TB/s
    check("decode tokens/sec at batch 1", 1 / seconds_per_step, 115, 0.01)
    check("decode tokens/sec at batch 32", 32 / seconds_per_step, 3_692, 0.01)

    static_outputs = [30, 45, 60, 80, 120, 200, 400, 800]
    static_utilization = sum(static_outputs) / (max(static_outputs) * len(static_outputs))
    check("static batching utilization", static_utilization, 0.27, 0.02)
    check("prefill stall for a 2k prompt (s)", 2_000 / 20_000, 0.1)

    # --- 06-inference-serving/capacity-planning.md ---
    check("fleet for 500 rps", 500 * 2_000 / 20_000 + 500 * 500 / 2_000, 175)
    check("fleet monthly cost", 175 * 2.00 * 730, 255_500)

    def provisioned(steady, peak_multiple=2.5, target_utilization=0.75, zones=3):
        return steady * peak_multiple / target_utilization * zones / (zones - 1)

    check("provisioned fleet with headroom", provisioned(175), 875, 0.01)

    free_bytes = (80 - 26) * 1e9
    check("kv tokens on one GPU", free_bytes / 131_072, 411_987, 0.001)
    check("concurrent at 32k context", free_bytes / (131_072 * 32_000), 13, 0.02)

    # --- 06-inference-serving/multi-gpu.md ---
    check("70B at BF16 (GB)", model_size_gb(70, 16), 140)
    check("70B at INT4 (GB)", model_size_gb(70, 4), 35)
    kv_gb_64_seq = 64 * 2_500 * 131_072 / 1e9
    check("kv cache for 64 sequences (GB)", kv_gb_64_seq, 21.0, 0.02)
    check("70B INT4 plus its cache fits in 80GB",
          model_size_gb(70, 4) + kv_gb_64_seq, 56.0, 0.02)

    # --- 06-inference-serving/autoscaling.md ---
    def cold_start(weights_gb, read_gb_per_s, node_provision_s=0,
                   image_pull_s=45, warmup_s=20):
        return node_provision_s + image_pull_s + weights_gb / read_gb_per_s + warmup_s

    check("cold start, network filesystem (s)", cold_start(26, 1.0), 91, 0.01)
    check("cold start, local NVMe (s)", cold_start(26, 8.0), 68, 0.01)
    check("cold start, cold node (s)", cold_start(26, 1.0, 240), 331, 0.01)

    # --- 06-inference-serving/gpu-economics.md ---
    def diurnal_utilization(peak_hours, peak_multiple):
        return (peak_hours * peak_multiple + (24 - peak_hours)) / (24 * peak_multiple)

    check("diurnal utilization", diurnal_utilization(6, 2.5), 0.55, 0.02)
    blended = (100 * 2.00 + 50 * 3.50 + 50 * 0.80) / 200
    check("blended GPU rate", blended, 2.075, 0.01)
    check("blended hourly spend", 100 * 2.00 + 50 * 3.50 + 50 * 0.80, 415)

    # --- 09-data-pipelines/document-parsing.md ---
    check("parsing upgrade for 200k documents", 200_000 * 12 * 0.015, 36_000)
    check("chunks affected by a reparse", 200_000 * 20, 4_000_000)

    # --- 09-data-pipelines/incremental-indexing.md ---
    overnight_capacity = 45_000 * 10
    check("overnight reindex capacity", overnight_capacity, 450_000)
    check("change rate vs capacity", 2_000_000 / overnight_capacity, 4.4, 0.02)

    # --- 09-data-pipelines/backfills.md ---
    def backfill_days(items, per_second):
        return items / per_second / 3600 / 24

    check("900M chunks at 46/s (days)", backfill_days(900_000_000, 46), 227, 0.01)
    check("900M chunks at 162/s (days)", backfill_days(900_000_000, 162), 64, 0.02)
    # six weeks = 42 days — the rate the scenario actually requires
    check("rate needed for 900M in six weeks",
          900_000_000 / (42 * 24 * 3600), 248, 0.01)
    # the scenario's reported rates, expressed per second
    check("14M chunks/day in items/sec", 14_000_000 / 86_400, 162, 0.01)
    check("4M chunks/day in items/sec", 4_000_000 / 86_400, 46, 0.02)

    def embedding_cost(chunks, tokens_per_chunk=400, price_per_million=0.02):
        return chunks * tokens_per_chunk / 1e6 * price_per_million

    check("re-embed 50M chunks", embedding_cost(50_000_000), 400)
    check("re-embed 900M chunks", embedding_cost(900_000_000), 7_200)

    # --- 10-system-design-patterns/request-response-vs-async.md ---
    budget = [round(60 * (0.9 ** i), 1) for i in range(4)]
    check("innermost timeout of a 60s budget", budget[-1], 43.7, 0.01)

    # --- 10-system-design-patterns/human-in-the-loop.md ---
    def review_capacity(reviewers, minutes_per_item, hours_per_day=6):
        return reviewers * (hours_per_day * 60 / minutes_per_item)

    check("four reviewers at 3 min/item", review_capacity(4, 3), 480)
    check("review share of 50k daily items", review_capacity(4, 3) / 50_000, 0.0096, 0.02)

    # --- 10-system-design-patterns/batch-vs-realtime.md ---
    def batch_vs_realtime(items, tokens_each, tokens_per_gpu_second=2_000):
        gpu_seconds = items * tokens_each / tokens_per_gpu_second
        return (gpu_seconds / 3600 * 2.00 / 0.45,      # real-time
                gpu_seconds / 3600 * 0.80 / 0.90)      # batch

    realtime_cost, batch_cost = batch_vs_realtime(10_000_000, 300)
    check("real-time cost for 10M items", realtime_cost, 1_852, 0.01)
    check("batch cost for 10M items", batch_cost, 370, 0.01)
    check("batch is ~5x cheaper", realtime_cost / batch_cost, 5.0, 0.02)

    def staleness(interval_hours, change_rate_per_day):
        return min(1.0, change_rate_per_day * interval_hours / 24)

    check("staleness at 2%/day, nightly", staleness(24, 0.02), 0.02)
    check("staleness at 50%/day, nightly", staleness(24, 0.50), 0.50)

    # --- 10-system-design-patterns/multi-region.md ---
    check("network share of a 3.35s response", 150 / (150 + 200 + 3_000), 0.045, 0.02)

    # --- 05-evaluation: Wilson interval on the quoted example ---
    lo, hi = wilson([1.0] * 162 + [0.0] * 38)     # 0.81 on n=200
    check("wilson lower bound", lo, 0.75, 0.02)
    check("wilson upper bound", hi, 0.86, 0.02)
    for values, label in [([1.0] * 20, "all correct"), ([0.0] * 20, "all wrong")]:
        lo, hi = wilson(values)
        if not (0.0 <= lo <= hi <= 1.0):
            failures.append(f"wilson interval escaped [0,1] for {label}: {lo}, {hi}")

    return 0, failures


# ---------------------------------------------------------------------------
# 4. The README describes the repo that exists
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve",
    13: "Thirteen", 14: "Fourteen", 15: "Fifteen",
}


def section_dirs() -> list[pathlib.Path]:
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())


def content_files(section: pathlib.Path) -> list[pathlib.Path]:
    """Files a reader would count: everything but the section's own README."""
    return sorted(p for p in section.glob("*.md") if p.name != "README.md")


def word_count(section: pathlib.Path) -> int:
    """Whitespace-separated tokens across every file in the section.

    Including the section's own README, and including code blocks — the figure
    is meant as a reading-length signal, not a prose count. Use this definition
    when updating the table; `wc -w` disagrees slightly on multibyte punctuation.
    """
    return sum(len(p.read_text(encoding="utf-8").split()) for p in section.glob("*.md"))


def check_self_description() -> list[str]:
    failures: list[str] = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")

    sections = section_dirs()
    actual = {
        s.name[:2]: (len(content_files(s)), word_count(s))
        for s in sections
    }

    # --- the section table in the README ---
    rows = re.findall(
        r"\*\*\[(\d\d) · [^\]]+\]\((\S+?)/\)\*\*"
        r".*?"
        r'align="right" valign="top">(\d+)</td>'
        r'<td align="right" valign="top">(\d+)k</td>',
        readme,
        re.DOTALL,
    )
    if len(rows) != len(sections):
        failures.append(
            f"README section table has {len(rows)} rows, repo has {len(sections)} sections"
        )

    for number, directory, files_claimed, words_claimed in rows:
        if number not in actual:
            failures.append(f"README lists section {number}, which is not in the repo")
            continue
        if not (ROOT / directory).is_dir():
            failures.append(f"README section {number} points at missing dir {directory}/")
        files_real, words_real = actual[number]
        if int(files_claimed) != files_real:
            failures.append(
                f"README says section {number} has {files_claimed} files, it has {files_real}"
            )
        if round(words_real / 1000) != int(words_claimed):
            failures.append(
                f"README says section {number} is {words_claimed}k words, "
                f"it is {words_real:,} ({round(words_real / 1000)}k)"
            )

    # --- the totals row ---
    total = re.search(
        r"<b>Total</b></td><td align=\"right\"><b>(\d+)</b></td>"
        r"<td align=\"right\"><b>(\d+)k</b></td>",
        readme,
    )
    files_real = sum(f for f, _ in actual.values())
    words_real = sum(w for _, w in actual.values())
    if not total:
        failures.append("README has no totals row to check")
    else:
        if int(total.group(1)) != files_real:
            failures.append(
                f"README totals say {total.group(1)} files, repo has {files_real}"
            )
        if round(words_real / 1000) != int(total.group(2)):
            failures.append(
                f"README totals say {total.group(2)}k words, repo has "
                f"{words_real:,} ({round(words_real / 1000)}k)"
            )

    # --- which sections are missing ---
    # The numbering runs 00..N with deliberate gaps, so the gaps *are* the
    # unwritten sections. Nothing here is hardcoded: add a directory and the
    # counts below move on their own.
    highest = max(int(n) for n in actual)
    missing = {f"{n:02d}" for n in range(highest + 1)} - set(actual)

    # --- the sections badge, and the prose that repeats it ---
    # Two shapes: "13 written" once nothing is outstanding, and
    # "11 written, 2 to go" while gaps remain.
    badge = re.search(r"sections-(\d+)%20written(?:%2C%20(\d+)%20to%20go)?", readme)
    if not badge:
        failures.append(
            "README has no readable sections badge (expected "
            "'sections-<n>%20written' or 'sections-<n>%20written%2C%20<m>%20to%20go')"
        )
    else:
        if int(badge.group(1)) != len(sections):
            failures.append(
                f"sections badge says {badge.group(1)} written, repo has {len(sections)}"
            )
        badge_remaining = int(badge.group(2)) if badge.group(2) else 0
        if badge_remaining != len(missing):
            failures.append(
                f"sections badge says {badge_remaining} to go, "
                f"the numbering has {len(missing)} gaps: {sorted(missing)}"
            )

    written_word = NUMBER_WORDS[len(sections)]
    for name, text in (("README.md", readme), ("ROADMAP.md", roadmap)):
        if not re.search(rf"{written_word} sections are written", text, re.I):
            failures.append(
                f'{name} does not say "{written_word} sections are written" — '
                f"{len(sections)} are"
            )
        if missing:
            missing_word = NUMBER_WORDS[len(missing)].lower()
            if not re.search(rf"{missing_word} are not", text, re.I):
                failures.append(
                    f'{name} does not say "{missing_word} are not" — '
                    f"{len(missing)} sections are unwritten"
                )
        elif not re.search(r"none are outstanding", text, re.I):
            failures.append(
                f'{name} does not say "none are outstanding" — every section is written'
            )

    readme_missing = set(re.findall(r"\| \*\*(\d\d)\*\* \|", readme))
    if readme_missing != missing:
        failures.append(
            f"README's missing-sections table lists {sorted(readme_missing) or 'nothing'}, "
            f"the gaps are {sorted(missing)}"
        )

    roadmap_planned = set(re.findall(r"\| (\d\d) \| [^|]+ \| — \| 🔜 Planned \|", roadmap))
    if roadmap_planned != missing:
        failures.append(
            f"ROADMAP marks {sorted(roadmap_planned) or 'nothing'} as planned, "
            f"the gaps are {sorted(missing)}"
        )

    roadmap_done = re.findall(r"\| (\d\d) \| \[[^\]]+\]\([^)]+\) \| (\d+) \| ✅ Done \|", roadmap)
    for number, files_claimed in roadmap_done:
        if number not in actual:
            failures.append(f"ROADMAP marks section {number} done, it does not exist")
        elif int(files_claimed) != actual[number][0]:
            failures.append(
                f"ROADMAP says section {number} has {files_claimed} files, "
                f"it has {actual[number][0]}"
            )
    if len(roadmap_done) != len(sections):
        failures.append(
            f"ROADMAP marks {len(roadmap_done)} sections done, repo has {len(sections)}"
        )
    if len(roadmap_done) + len(roadmap_planned) != highest + 1:
        failures.append(
            f"ROADMAP's status table has {len(roadmap_done) + len(roadmap_planned)} rows "
            f"for sections 00–{highest:02d} ({highest + 1} numbers)"
        )

    # --- scenario and problem counts ---
    scenarios = {}
    for md in content_files(ROOT / "12-senior-scenarios"):
        scenarios[md.name] = len(
            re.findall(r"^### Situation$", md.read_text(encoding="utf-8"), re.M)
        )
    problems = {}
    for md in content_files(ROOT / "11-coding-rounds"):
        problems[md.name] = len(
            re.findall(r"^## (?:Problem )?\d+", md.read_text(encoding="utf-8"), re.M)
        )

    headline = re.search(r"\*\*(\d+) scenarios and coding problems\*\*", readme)
    real_total = sum(scenarios.values()) + sum(problems.values())
    if not headline:
        failures.append("README no longer states a scenario count")
    elif int(headline.group(1)) != real_total:
        failures.append(
            f"README claims {headline.group(1)} scenarios and coding problems, "
            f"there are {real_total} ({sum(scenarios.values())} scenarios + "
            f"{sum(problems.values())} coding problems)"
        )

    for folder, counts in (("12-senior-scenarios", scenarios), ("11-coding-rounds", problems)):
        index = (ROOT / folder / "README.md").read_text(encoding="utf-8")
        for name, claimed in re.findall(
            r"\|\s*\[([\w.-]+\.md)\]\([^)]+\)\s*\|[^|\n]*\|\s*(\d+)", index
        ):
            if name in counts and int(claimed) != counts[name]:
                failures.append(
                    f"{folder}/README.md says {name} has {claimed}, it has {counts[name]}"
                )

    # --- the python-block count claimed inside a section ---
    coding = (ROOT / "11-coding-rounds" / "README.md").read_text(encoding="utf-8")
    # normalised so the claim can be line-wrapped without breaking the check
    claim = re.search(
        r"All (\d+) Python blocks in this section parse",
        re.sub(r"\s+", " ", coding),
    )
    real_blocks = sum(
        len(FENCE.findall(p.read_text(encoding="utf-8")))
        for p in (ROOT / "11-coding-rounds").glob("*.md")
    )
    if not claim:
        failures.append("11-coding-rounds/README.md no longer states a block count")
    elif int(claim.group(1)) != real_blocks:
        failures.append(
            f"11-coding-rounds/README.md claims {claim.group(1)} python blocks, "
            f"there are {real_blocks}"
        )

    # --- the paired-comparison example, wherever it is quoted ---
    failures.extend(check_worked_example())

    return failures


BREAKDOWN = re.compile(
    r"both correct\s+(\d+)\s*\n"
    r"both wrong\s+(\d+)\s*\n"
    r"B fixed it\s+(\d+)[^\n]*\n"
    r"B broke it\s+(\d+)",
)


def check_worked_example() -> list[str]:
    """The 0.79-vs-0.83 example is quoted in more than one file.

    Every copy is re-derived from its own numbers here, because the failure this
    catches already happened once: a breakdown was corrected in one file and
    left stale in another, where it summed to 0.81 and 0.85 under prose that
    said 0.79 and 0.83.
    """
    failures: list[str] = []
    seen = 0

    for md in markdown_files():
        text = md.read_text(encoding="utf-8")
        for both_right, both_wrong, fixed, broke in BREAKDOWN.findall(text):
            seen += 1
            rel = md.relative_to(ROOT)
            both_right, both_wrong = int(both_right), int(both_wrong)
            fixed, broke = int(fixed), int(broke)
            n = both_right + both_wrong + fixed + broke

            stated = [float(x) for x in re.findall(r"\b0\.\d\d\b", text)]
            old = (both_right + broke) / n
            new = (both_right + fixed) / n

            if n != 200:
                failures.append(f"{rel}: breakdown sums to {n}, the prose says 200 cases")
            for label, value in (("A", old), ("B", new)):
                if not any(abs(value - s) < 0.005 for s in stated):
                    failures.append(
                        f"{rel}: breakdown gives prompt {label} = {value:.2f}, "
                        f"which appears nowhere in the prose (found {sorted(set(stated))})"
                    )

            p = mcnemar(new_fixed=fixed, new_broke=broke)
            if p <= 0.05:
                failures.append(
                    f"{rel}: breakdown gives p={p:.3f} — the example only works "
                    f"if the difference is NOT significant"
                )
            quoted_p = re.search(r"p ≈ (\d\.\d+)", text)
            if quoted_p and abs(p - float(quoted_p.group(1))) > 0.05:
                failures.append(
                    f"{rel}: breakdown gives p={p:.2f}, prose says p ≈ {quoted_p.group(1)}"
                )

    if seen == 0:
        failures.append("the paired-comparison worked example has disappeared from the docs")
    return failures


def mcnemar(new_fixed: int, new_broke: int) -> float:
    """Two-sided McNemar's test. Only discordant pairs carry information."""
    n = new_fixed + new_broke
    if n == 0:
        return 1.0
    if n < 25:
        k = min(new_fixed, new_broke)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)
    chi2 = (abs(new_fixed - new_broke) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2))


def wilson(values, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct near 0 and 1, where eval sets live."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    p = sum(values) / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------

def main() -> int:
    print("Verifying The AI Engineer's Playbook\n")
    ok = True

    n_blocks, block_failures = check_python_blocks()
    if block_failures:
        ok = False
        print(f"FAIL  python blocks ({len(block_failures)} of {n_blocks})")
        for f in block_failures:
            print(f"        {f}")
    else:
        print(f"PASS  {n_blocks} python blocks parse")

    n_links, link_failures = check_links()
    if link_failures:
        ok = False
        print(f"FAIL  internal links ({len(link_failures)} of {n_links})")
        for f in link_failures:
            print(f"        {f}")
    else:
        print(f"PASS  {n_links} internal links resolve")

    _, claim_failures = check_claims()
    if claim_failures:
        ok = False
        print(f"FAIL  numeric claims ({len(claim_failures)})")
        for f in claim_failures:
            print(f"        {f}")
    else:
        print("PASS  numeric claims match the prose")

    self_failures = check_self_description()
    if self_failures:
        ok = False
        print(f"FAIL  repo self-description ({len(self_failures)})")
        for f in self_failures:
            print(f"        {f}")
    else:
        print("PASS  the README describes the repo that exists")

    print()
    if ok:
        print("All checks passed.")
        return 0
    print("Verification failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
