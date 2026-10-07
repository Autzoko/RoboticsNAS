"""Gate 1: supernet-inherited vs standalone closed-loop SR on the S1 keys (s2-h10, search-val, same seeds)."""

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .space import Arch


def sr_file(f):
    by = {}
    for line in open(f):
        r = json.loads(line)
        by.setdefault(r["arch"], []).extend(r["success"])
    return {k: float(np.mean(v)) for k, v in by.items() if len(v) >= 400}


def main():
    sa = {}
    for f in glob.glob("outputs/s1/*.jsonl"):
        sa.update(sr_file(f))
    lines = ["| variant | n | Kendall | Spearman | mean gap (standalone - supernet) | gap e4/e8 | gap e12/e16 |",
             "|---|---|---|---|---|---|---|"]
    detail = {}
    for f in sorted(glob.glob("outputs/s2/V*.jsonl")):
        v = Path(f).stem
        sn = sr_file(f)
        ks = sorted(set(sn) & set(sa))
        if len(ks) < 4:
            continue
        x = np.array([sn[k] for k in ks]); y = np.array([sa[k] for k in ks])
        shallow = np.array([Arch.from_key(k).n_exp <= 8 for k in ks])
        g = y - x
        lines.append(f"| {v} | {len(ks)} | {kendalltau(x, y).statistic:.3f} | {spearmanr(x, y).statistic:.3f} | "
                     f"{g.mean():+.3f} | {g[shallow].mean():+.3f} | {g[~shallow].mean():+.3f} |")
        detail[v] = {k: {"supernet": sn[k], "standalone": sa[k]} for k in ks}
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    Path("outputs/analysis/s2_rank.md").write_text("\n".join(lines) + "\n")
    json.dump(detail, open("outputs/analysis/s2_rank_detail.json", "w"), indent=1)
    print("\n".join(lines))
    for v, d in detail.items():
        print(f"\n{v}:")
        for k, r in sorted(d.items(), key=lambda z: -z[1]["standalone"]):
            print(f"  {k:36s} standalone {r['standalone']:.3f}  supernet {r['supernet']:.3f}")


if __name__ == "__main__":
    main()
