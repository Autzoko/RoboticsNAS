"""RQ3: compare search strategies by replaying them on the E2 table (closed-loop SR per (arch, task, episode)).

For each repetition: random paired episode order; methods spend at most `budget` episodes on the SEARCH half;
the chosen arch is scored on the held-out EVAL half. Reported: mean held-out SR of the chosen arch and regret w.r.t.
the best arch (by held-out SR) satisfying the same cost budget.

Cost budgets come from the cost table (`rnas/latency.py`): per-control-step compute (ms_per_step), per-call latency
(call_ms) and deploy memory (deploy_MB), each as a quantile cap over the benchmark archs.
Proxies: any column of proxies.jsonl / zerocost.jsonl; direction from HIGHER_IS_BETTER.
Reference divergences for bound_race: column `kd_onpolicy` (reference = anchor) and any `D_ref[<arch key>]` columns.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import search as S
from .space import Arch

HIGHER_IS_BETTER = {"zc_gradnorm", "zc_snip", "zc_naswot"}
ANCHOR = "v16-e16-stretch-f1-t64-s10-h10"  # anchor-visited states were recorded with the anchor at h10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost", default=None, help="cost jsonl (default: <bench>/latency.jsonl)")
    ap.add_argument("--caps", default="none,ms_per_step:0.5,deploy_MB:0.5,ms_per_step:0.25")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--budgets", default="400,800,1600")
    ap.add_argument("--proxies", default="fm_loss,act_l1_exec,kd_onpolicy,zc_gradnorm,zc_snip,zc_naswot")
    args = ap.parse_args()
    b = Path(args.bench)
    keys, T, cells = S.load_table(sorted(glob.glob(str(b / "rollouts_g*.jsonl"))))
    ep = sorted({c[2] for c in cells})
    s_eps, e_eps = ep[: len(ep) // 2], ep[len(ep) // 2 :]
    side = pd.DataFrame({"arch": keys})
    files = [b / "proxies.jsonl", b / "zerocost.jsonl", Path(args.cost) if args.cost else b / "latency.jsonl"]
    for f in files:
        if f.exists():
            q = pd.read_json(f, lines=True).drop_duplicates("arch", keep="last")
            q = q[[c for c in q.columns if c == "arch" or c not in side.columns]]
            side = side.merge(q, on="arch", how="left")
    X = S.arch_features(keys)
    hor = np.array([Arch.from_key(k).horizon for k in keys], dtype=float)
    # reference divergence vectors for bound_race
    D = {}
    if "kd_onpolicy" in side:
        r0 = keys.index(ANCHOR) if ANCHOR in keys else int(np.nanargmin(side["kd_onpolicy"].to_numpy()))
        D[r0] = np.nan_to_num(side["kd_onpolicy"].to_numpy(dtype=float), nan=np.inf)  # D_h: per-step mean discrepancy
    for c in side.columns:
        if c.startswith("D_ref[") and c[6:-1] in keys:
            D[keys.index(c[6:-1])] = np.nan_to_num(side[c].to_numpy(dtype=float), nan=np.inf)

    proxies = {}
    for p in args.proxies.split(","):
        if p in side:
            v = side[p].to_numpy(dtype=float)
            v = -v if p in HIGHER_IS_BETTER else v
            proxies[p] = np.where(np.isfinite(v), v, np.inf)

    caps = {}
    for spec in args.caps.split(","):
        if spec == "none":
            caps["none"] = np.ones(len(keys), bool)
        else:
            k, qv = spec.split(":")
            if k in side:
                v = side[k].to_numpy(dtype=float)
                caps[spec] = v <= np.nanquantile(v, float(qv))

    rows = []
    for cap_name, ok in caps.items():
        cand = [i for i in range(len(keys)) if ok[i]]
        if len(cand) < 4:
            continue
        for budget in [int(x) for x in args.budgets.split(",")]:
            res = {}
            for r in range(args.reps):
                R = S.Replay(keys, T, cells, s_eps, e_eps, np.random.default_rng(r))
                best = max(R.true[i] for i in cand)
                runs = {"random_full": S.random_full(R, cand, budget),
                        "sh_n16": S.sh(R, cand, budget, 16),
                        "predictor_ridge": S.predictor_search(R, cand, budget, X)}
                for p, pv in proxies.items():
                    runs[f"proxy_top1[{p}]"] = S.proxy_top1(R, cand, pv)
                    runs[f"proxy_sh[{p}]_n8"] = S.proxy_sh(R, cand, budget, 8, pv)
                if D:
                    Dc = {k: v for k, v in D.items()}
                    runs["bound_race(M1+M3)"] = S.bound_race(R, cand, budget, Dc, hor)
                for m, (i, used) in runs.items():
                    res.setdefault(m, []).append((R.true[i], best - R.true[i], used))
            for m, v in res.items():
                v = np.array(v)
                rows.append({"cap": cap_name, "n_cand": len(cand), "budget": budget, "method": m,
                             "sr_eval": v[:, 0].mean(), "regret": v[:, 1].mean(),
                             "regret_se": v[:, 1].std() / np.sqrt(len(v)), "episodes_used": v[:, 2].mean()})
    df = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / "replay.csv", index=False)
    with open(Path(args.out) / "replay.md", "w") as f:
        for (cap, budget), g in df.groupby(["cap", "budget"]):
            f.write(f"\n### cap={cap} (n={g.n_cand.iloc[0]}), budget={budget} episodes\n\n")
            f.write(g.sort_values("regret")[["method", "sr_eval", "regret", "regret_se", "episodes_used"]]
                    .to_markdown(index=False, floatfmt=".3f") + "\n")
    print(open(Path(args.out) / "replay.md").read())


if __name__ == "__main__":
    main()
