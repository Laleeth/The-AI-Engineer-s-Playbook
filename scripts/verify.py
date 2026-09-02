#!/usr/bin/env python3
"""Verify the repository's code and links.

Three checks:

  1. Every ```python block parses as valid Python.
  2. Every internal markdown link points at a file that exists, and every
     anchor points at a heading that exists.
  3. Numeric claims made in the prose match what the code actually computes.

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

    # --- 05-evaluation: Wilson interval on the quoted example ---
    lo, hi = wilson([1.0] * 162 + [0.0] * 38)     # 0.81 on n=200
    check("wilson lower bound", lo, 0.75, 0.02)
    check("wilson upper bound", hi, 0.86, 0.02)
    for values, label in [([1.0] * 20, "all correct"), ([0.0] * 20, "all wrong")]:
        lo, hi = wilson(values)
        if not (0.0 <= lo <= hi <= 1.0):
            failures.append(f"wilson interval escaped [0,1] for {label}: {lo}, {hi}")

    return 0, failures


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

    print()
    if ok:
        print("All checks passed.")
        return 0
    print("Verification failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
