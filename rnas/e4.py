"""E4: official LIBERO test results (50 official init states x 40 tasks per candidate)."""

import json
import math
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    cands = json.load(open("results/final_candidates.json"))["candidates"]
    lat = {}
    for line in open("outputs/latency/a100_bench.jsonl"):
        r = json.loads(line)
        lat[r["arch"]] = r
    lines = ["| candidate | arch | spatial | object | goal | long | **avg** [95% CI] | n eps | ms/call | ms/step |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cands:
        succ = {s: [] for s in SUITES}
        for f in Path(f"outputs/test/{c['name']}").glob("rollouts_*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                assert r["mode"] == "test" and r["arch"] == c["arch"]
                succ[r["suite"]] += r["success"]
        allv = sum(succ.values(), [])
        n, k = len(allv), sum(allv)
        lo, hi = wilson(k, n)
        L = lat.get(c["arch"], {})
        per = " | ".join(f"{sum(v) / len(v):.3f}" if v else "-" for v in succ.values())
        lines.append(f"| {c['name']} | {c['arch']} | {per} | **{k / max(n, 1):.3f}** [{lo:.3f}, {hi:.3f}] | {n} | "
                     f"{L.get('call_ms', float('nan')):.0f} | {L.get('ms_per_step', float('nan')):.1f} |")
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    Path("outputs/analysis/e4.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
