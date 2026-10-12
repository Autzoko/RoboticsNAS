"""S3: does short standalone training (step10k) preserve the closed-loop ranking of full training (30k)?

Paired on S1 search-val cells (suite, task, episode). Also applies the selection rule of the two-stage method
("cheapest net statistically tied with the best") to each ranking source and reports what it would pick.
"""

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr


def cells(f):
    out = {}
    for line in open(f):
        r = json.loads(line)
        for e, s in zip(r["episodes"], r["success"]):
            out.setdefault(r["arch"], {})[(r["suite"], r["task"], e)] = float(s)
    return {k: v for k, v in out.items() if len(v) >= 400}


def load(pattern):
    d = {}
    for f in glob.glob(pattern):
        d.update(cells(f))
    return d


def pick(M, keys, cost, rng, nboot=2000):
    """Cheapest arch whose paired SR deficit vs the empirical best is not significant (one-sided 95% bootstrap)."""
    sr = M.mean(1)
    b = int(np.argmax(sr))
    idx = rng.integers(0, M.shape[1], (nboot, M.shape[1]))
    tied = [i for i in range(len(keys)) if np.quantile((M[b][idx] - M[i][idx]).mean(1), 0.05) <= 0]
    return min(tied, key=lambda i: cost[keys[i]]), tied


def main():
    full, short = load("outputs/s1/*.jsonl"), load("outputs/s3/*.jsonl")
    sn = load("outputs/s2/V0.jsonl")
    cost = {}
    for line in open("results/cost/a100_v2.jsonl"):
        r = json.loads(line)
        cost[r["arch"]] = r
    keys = sorted(set(full) & set(short))
    cs = sorted(set.intersection(*[set(full[k]) & set(short[k]) for k in keys]))
    F = np.array([[full[k][c] for c in cs] for k in keys])
    S = np.array([[short[k][c] for c in cs] for k in keys])
    f, s = F.mean(1), S.mean(1)
    rng = np.random.default_rng(0)
    bt = []
    for _ in range(2000):
        i = rng.integers(0, len(cs), len(cs))
        bt.append(kendalltau(S[:, i].mean(1), F[:, i].mean(1)).statistic)
    tau = kendalltau(s, f).statistic
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    top3_full = set(np.argsort(-f)[:3]); top5_short = set(np.argsort(-s)[:5])
    gate = tau >= 0.40 and top3_full <= top5_short
    ms = {k: cost[k]["ms_per_step"] for k in keys}
    lines = [f"# S3: SR@10k vs SR@30k (standalone, s2-h10, search-val, {len(cs)} paired episodes, n={len(keys)})", "",
             f"Kendall {tau:.3f} [95% CI {lo:.3f}, {hi:.3f}] | Spearman {spearmanr(s, f).statistic:.3f} | "
             f"30k top-3 within 10k top-5: {top3_full <= top5_short} | mean SR 10k {s.mean():.3f} vs 30k {f.mean():.3f}",
             f"Reference: test-retest Kendall of the 30k ranking ~0.56; supernet V0..V4 0.08 to -0.03.",
             f"**Pre-registered gate (tau >= 0.40 and top-3 in top-5): {'PASS' if gate else 'FAIL'}**", "",
             "| arch | SR@10k | SR@30k | supernet V0 | ms/step | deploy MB |", "|---|---|---|---|---|---|"]
    for i in np.argsort(-f):
        k = keys[i]
        v0 = f"{np.mean(list(sn[k].values())):.3f}" if k in sn else "-"
        lines.append(f"| {k} | {s[i]:.3f} | {f[i]:.3f} | {v0} | {ms[k]:.1f} | {cost[k]['deploy_MB']:.0f} |")
    lines += ["", "Selection rule 'cheapest (ms/step) arch tied with the best' applied to each ranking source; "
              "the pick is scored by its 30k SR:", "",
              "| ranking source | pick | tied set size | pick SR@30k | regret vs 30k best | ms/step | deploy MB |",
              "|---|---|---|---|---|---|---|"]
    srcs = [("30k standalone (oracle, 3x cost)", F), ("10k standalone (two-stage)", S)]
    if all(k in sn for k in keys):
        srcs.append(("supernet V0 (weight sharing)", np.array([[sn[k].get(c, np.nan) for c in cs] for k in keys])))
    for name, M in srcs:
        if np.isnan(M).any():
            continue
        p, tied = pick(M, keys, ms, np.random.default_rng(1))
        k = keys[p]
        lines.append(f"| {name} | {k} | {len(tied)} | {f[p]:.3f} | {f.max() - f[p]:.3f} | {ms[k]:.1f} | "
                     f"{cost[k]['deploy_MB']:.0f} |")
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    Path("outputs/analysis/s3_rank.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
