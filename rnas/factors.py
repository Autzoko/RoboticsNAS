"""RQ4: design-axis effects on closed-loop SR (supernet-inherited, E2 table): marginal means + drop-one R^2."""

import sys

import numpy as np
import pandas as pd

AXES = ["n_vlm", "n_exp", "bridge", "ffn", "vtok", "steps", "horizon"]


def r2(d, axes):
    X = pd.get_dummies(d[axes].astype(str), drop_first=True).astype(float)
    X.insert(0, "c", 1.0)
    y = d["sr"].to_numpy()
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    res = y - X.to_numpy() @ beta
    return 1 - res.var() / y.var()


d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/e2/bench_table.csv")
full = r2(d, AXES)
lines = [f"Main-effects OLS on SR (n={len(d)}): R^2 = {full:.3f}", "",
         "| axis | marginal mean SR by level (n) | drop-one delta R^2 |", "|---|---|---|"]
for a in AXES:
    g = d.groupby(a)["sr"].agg(["mean", "count"])
    lv = ", ".join(f"{i}: {r['mean']:.3f} ({int(r['count'])})" for i, r in g.iterrows())
    lines.append(f"| {a} | {lv} | {full - r2(d, [x for x in AXES if x != a]):.3f} |")
print("\n".join(lines))
open("results/e2/factors.md", "w").write("\n".join(lines) + "\n")
