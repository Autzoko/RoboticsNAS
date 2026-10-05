"""RQ3: compare search strategies by replaying them on the E2 benchmark table.

For each repetition: random paired episode order; methods spend at most `budget` episodes on the SEARCH half;
the chosen arch is scored on the held-out EVAL half. Reported: mean held-out SR of the chosen arch and regret
w.r.t. the best arch (by held-out SR) satisfying the same latency cap.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import search as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--budgets", default="400,800,1600,3200")
    ap.add_argument("--proxies", default="fm_loss,act_l1_exec,kd_onpolicy")
    args = ap.parse_args()
    b = Path(args.bench)
    keys, T, cells = S.load_table(sorted(glob.glob(str(b / "rollouts_g*.jsonl"))))
    ep = sorted({c[2] for c in cells})
    s_eps, e_eps = ep[: len(ep) // 2], ep[len(ep) // 2 :]
    side = pd.DataFrame({"arch": keys})
    for f in ("proxies.jsonl", "latency.jsonl"):
        if (b / f).exists():
            q = pd.read_json(b / f, lines=True).drop_duplicates("arch", keep="last")
            side = side.merge(q, on="arch", how="left")
    lat = side["ms_per_step"].to_numpy() if "ms_per_step" in side else np.zeros(len(keys))
    caps = {"none": np.inf, "lat<=median": float(np.median(lat))}
    proxies = [p for p in args.proxies.split(",") if p in side]

    rows = []
    for cap_name, cap in caps.items():
        cand = [i for i in range(len(keys)) if lat[i] <= cap]
        for budget in [int(x) for x in args.budgets.split(",")]:
            res = {}
            for r in range(args.reps):
                R = S.Replay(keys, T, cells, s_eps, e_eps, np.random.default_rng(r))
                best = max(R.true[i] for i in cand)
                runs = {"random_full": S.random_full(R, cand, budget)}
                for n0 in (8, 16, 32):
                    runs[f"sh_n{n0}"] = S.sh(R, cand, budget, n0)
                for p in proxies:
                    pv = side[p].to_numpy(dtype=float)
                    pv = np.where(np.isfinite(pv), pv, np.inf)
                    runs[f"proxy_top1[{p}]"] = S.proxy_top1(R, cand, pv)
                    for n0 in (8, 16):
                        runs[f"proxy_sh[{p}]_n{n0}"] = S.proxy_sh(R, cand, budget, n0, pv)
                for m, (i, used) in runs.items():
                    res.setdefault(m, []).append((R.true[i], best - R.true[i], used))
            for m, v in res.items():
                v = np.array(v)
                rows.append({"cap": cap_name, "budget": budget, "method": m, "sr_eval": v[:, 0].mean(),
                             "regret": v[:, 1].mean(), "regret_se": v[:, 1].std() / np.sqrt(len(v)),
                             "episodes_used": v[:, 2].mean()})
    df = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / "replay.csv", index=False)
    with open(Path(args.out) / "replay.md", "w") as f:
        for (cap, budget), g in df.groupby(["cap", "budget"]):
            f.write(f"\n### cap={cap}, budget={budget} episodes\n\n")
            f.write(g.sort_values("regret")[["method", "sr_eval", "regret", "regret_se", "episodes_used"]]
                    .to_markdown(index=False, floatfmt=".3f") + "\n")
    print(open(Path(args.out) / "replay.md").read())


if __name__ == "__main__":
    main()
