#!/usr/bin/env python3
"""Turn a rollout directory + the official grading report into the headline number.

The pass/fail signal comes entirely from the official SWE-bench harness. This
script only joins it against our own per-instance records so that every failure
lands in exactly one bucket -- which is the part that actually tells you what to
fix next.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pct(k: int, n: int) -> float:
    return (k / n * 100) if n else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Never report a subset score without it."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def bucket(rec: dict, resolved: set[str]) -> str:
    """One instance, one bucket -- checked most-specific first."""
    iid = rec["instance_id"]
    if iid in resolved:
        return "resolved"
    status = rec.get("status")
    if status == "agent_timeout":
        return "agent_timeout"
    if status in ("container_error", "harness_error"):
        return "infra_error"
    agent = rec.get("agent") or {}
    # Checked *before* no_patch: exhausting the budget usually ends with no diff,
    # and the whole point of this bucket is to tell "raise --max-steps" apart
    # from "the model could not do it".
    if any("step budget exhausted" in e for e in agent.get("errors") or []):
        return "step_budget_exhausted"
    if status == "no_patch":
        return "no_patch"
    if agent.get("status") == "error":
        return "agent_error"
    return "tests_failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="rollout dir from run_agent.py")
    ap.add_argument("--grading", type=Path, required=True, help="official harness report JSON")
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((args.run / "instances").glob("*.json"))]
    grading = json.loads(args.grading.read_text())
    resolved = set(grading.get("resolved_ids", []))

    n = len(records)
    k = sum(1 for r in records if r["instance_id"] in resolved)
    lo, hi = wilson(k, n)

    buckets: dict[str, int] = {}
    for rec in records:
        b = bucket(rec, resolved)
        buckets[b] = buckets.get(b, 0) + 1

    agents = [r.get("agent") or {} for r in records]
    tok_in = sum(a.get("input_tokens", 0) for a in agents)
    tok_out = sum(a.get("output_tokens", 0) for a in agents)
    auto_approved = sum(a.get("auto_approved", 0) for a in agents)
    wall = sum(r.get("wall_s", 0) for r in records)

    subset_note = ""
    if cfg.get("subset"):
        subset_note = f" (random subset, seed={cfg['seed']})"

    print()
    print("=" * 66)
    print(f"  {cfg['dataset']}  n={n}{subset_note}")
    print(f"  agent: codewright @ {cfg.get('codewright_commit','?')[:9]}   model: {cfg['model']}")
    print(f"  max_steps={cfg['max_steps']}  timeout={cfg['timeout_s']}s  pass@1  {cfg.get('started_utc','')}")
    print("=" * 66)
    if n == 0:
        print("  no instances recorded -- nothing to report")
        return

    print(f"  RESOLVE RATE   {k}/{n} = {pct(k, n):.1f}%   95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%]")
    print("-" * 66)
    print("  failure attribution")
    for name, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<26} {count:>4}  ({pct(count, n):4.1f}%)")
    print("-" * 66)
    print(f"  tokens   in={tok_in:,}  out={tok_out:,}   avg/instance={(tok_in + tok_out) // max(n, 1):,}")
    print(f"  wall     {wall / 3600:.1f} h total,  {wall / max(n, 1) / 60:.1f} min/instance avg")
    print(f"  full-auto approvals granted: {auto_approved}")
    print("=" * 66)
    print()
    print("  Report it as:")
    print(f"    codewright + {cfg['model']} on {cfg['dataset']}"
          f"{subset_note}: {pct(k, n):.1f}% resolved (pass@1, n={n}, 95% CI "
          f"[{lo * 100:.1f}, {hi * 100:.1f}])")
    print()

    (args.run / "report.json").write_text(json.dumps({
        "config": cfg, "n": n, "resolved": k, "resolve_rate": k / n if n else 0,
        "ci95": [lo, hi], "buckets": buckets,
        "input_tokens": tok_in, "output_tokens": tok_out, "wall_s": wall,
    }, indent=2))


if __name__ == "__main__":
    main()
