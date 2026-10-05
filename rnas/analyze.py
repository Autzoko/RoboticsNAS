"""RQ1/RQ2 analysis: rank agreement between proxies and closed-loop success on the E2 benchmark.

Ground truth for proxy evaluation = SR on held-out episode half (EVAL_EPS) so that closed-loop low-fidelity
proxies (computed on SEARCH_EPS) and offline proxies are compared against the same independent target.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .space import Arch
from .search import load_table

LOWER_IS_BETTER = {"fm_loss", "act_l1", "act_l1_exec", "grip_err", "kd_offline", "kd_onpolicy", "self_cons",
                   "call_ms", "ms_per_step", "expert_params_M"}


def boot_tau(x, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        vals.append(kendalltau(x[s], y[s]).statistic)
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="outputs/bench_<tag>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b = Path(args.bench)
    keys, S, cells = load_table(sorted(glob.glob(str(b / "rollouts_g*.jsonl"))))
    ep = np.array([c[2] for c in cells])
    eps = sorted(set(ep))
    half = len(eps) // 2
    s_eps, e_eps = eps[:half], eps[half:]
    sr_all = S.mean(1)
    sr_search = S[:, np.isin(ep, s_eps)].mean(1)
    sr_eval = S[:, np.isin(ep, e_eps)].mean(1)
    df = pd.DataFrame({"arch": keys, "sr": sr_all, "sr_search": sr_search, "sr_eval": sr_eval})
    for suite in sorted({c[0] for c in cells}):
        m = np.array([c[0] == suite for c in cells])
        df[f"sr_{suite}"] = S[:, m].mean(1)
    # low-fidelity closed-loop proxies: first k search episodes per task
    for k in (1, 2):
        m = np.isin(ep, s_eps[:k])
        df[f"cl_{k}ep"] = S[:, m].mean(1)
    for f in ("proxies.jsonl", "latency.jsonl"):
        p = b / f
        if p.exists():
            q = pd.read_json(p, lines=True).drop_duplicates("arch", keep="last")
            df = df.merge(q.drop(columns=[c for c in ("gpu",) if c in q]), on="arch", how="left")
    for a in ("n_vlm", "n_exp", "bridge", "ffn", "vtok", "steps", "horizon"):
        df[a] = [getattr(Arch.from_key(k), a) for k in df.arch]

    # split-half reliability (noise ceiling for any proxy predicting sr_eval)
    r_half = spearmanr(sr_search, sr_eval).statistic
    tau_half = kendalltau(sr_search, sr_eval).statistic
    lines = [f"# E2 proxy analysis ({b.name})", "",
             f"archs with complete grid: {len(keys)}; cells/arch: {S.shape[1]} (search {np.isin(ep, s_eps).sum()}, eval {np.isin(ep, e_eps).sum()})",
             f"SR: mean {sr_all.mean():.3f}, min {sr_all.min():.3f}, max {sr_all.max():.3f}",
             f"split-half reliability (search vs eval SR): Spearman {r_half:.3f}, Kendall {tau_half:.3f}", "",
             "| signal | Kendall tau vs SR_eval [95% CI] | Spearman | top-10 overlap |", "|---|---|---|---|"]
    y = df.sr_eval.to_numpy()
    top10 = set(np.argsort(-y)[:10])
    sigs = [c for c in df.columns if c in LOWER_IS_BETTER or c.startswith("cl_") or c == "sr_search"]
    res = []
    for c in sigs:
        x = df[c].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 5:
            continue
        sgn = -1 if c in LOWER_IS_BETTER else 1
        t = kendalltau(sgn * x[ok], y[ok]).statistic
        lo, hi = boot_tau(sgn * x[ok], y[ok])
        r = spearmanr(sgn * x[ok], y[ok]).statistic
        ov = len(set(np.argsort(-(sgn * np.where(ok, x, -np.inf * sgn)))[:10]) & top10)
        res.append((c, t, lo, hi, r, ov))
    for c, t, lo, hi, r, ov in sorted(res, key=lambda z: -z[1]):
        lines.append(f"| {c} | {t:.3f} [{lo:.3f}, {hi:.3f}] | {r:.3f} | {ov}/10 |")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / "bench_table.csv", index=False)
    (Path(args.out) / "proxy_correlations.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
