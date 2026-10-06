"""E5: weight-sharing validity. Standalone-trained vs supernet-inherited closed-loop SR (same search-val seeds)."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .search import load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="outputs/bench_sn")
    ap.add_argument("--out", default="results/e5")
    a = ap.parse_args()
    keys, S, _ = load_table(sorted(glob.glob(f"{a.bench}/rollouts_g*.jsonl")))
    sn = dict(zip(keys, S.mean(1)))
    e5 = json.load(open("results/e5_archs.json"))["archs"]
    files = {k: f"outputs/e5/{k}.jsonl" for k in e5}
    files[e5[0]] = "outputs/curve_fixed/step30k_s1.jsonl"  # default arch standalone (30k)
    rows = []
    for k in e5:
        _, s2, _ = load_table([files[k]])
        k2, s2, _ = load_table([files[k]])
        sa = dict(zip(k2, s2.mean(1)))[k]
        rows.append((k, sn.get(k, np.nan), sa))
    x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    lines = ["| arch | supernet-inherited SR | standalone SR | gap |", "|---|---|---|---|"]
    for k, s, t in rows:
        lines.append(f"| {k} | {s:.3f} | {t:.3f} | {t - s:+.3f} |")
    lines.append(f"\nKendall {kendalltau(x, y).statistic:.3f}, Spearman {spearmanr(x, y).statistic:.3f} (n={len(rows)}); "
                 f"mean gap {np.mean(y - x):+.3f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "e5.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
