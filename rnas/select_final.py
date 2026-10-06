"""Freeze E4 final candidates (before any official-test access). One deterministic run per method on the
SEARCH half of the E2 table (seed 0, fixed budget); candidates are then fine-tuned 5k steps from the supernet
and evaluated on the official LIBERO init states."""

import glob
import json
import subprocess

import numpy as np
import pandas as pd

from . import search as S

BENCH, BUDGET = "outputs/bench_sn", 1600


def main():
    keys, T, cells = S.load_table(sorted(glob.glob(f"{BENCH}/rollouts_g*.jsonl")))
    ep = sorted({c[2] for c in cells})
    side = pd.DataFrame({"arch": keys})
    for f in ("proxies.jsonl", "latency.jsonl"):
        side = side.merge(pd.read_json(f"{BENCH}/{f}", lines=True).drop_duplicates("arch", keep="last"),
                          on="arch", how="left")
    lat = side["ms_per_step"].to_numpy()
    cap = float(np.median(lat))
    # all episodes of the benchmark are "search" data now: selection uses the full search-val table
    R = S.Replay(keys, T, cells, ep, [], np.random.default_rng(0))
    cands = []
    for cap_name, c in (("lat<=median", cap), ("none", np.inf)):
        pool = [i for i in range(len(keys)) if lat[i] <= c]
        kd = side["kd_onpolicy"].to_numpy()
        fm = side["fm_loss"].to_numpy()
        picks = {
            "ours_proxySH_kd": S.proxy_sh(R, pool, BUDGET, 8, kd)[0],
            "nas_valloss_top1": S.proxy_top1(R, pool, fm)[0],
            "random_search": S.random_full(R, pool, BUDGET)[0],
        }
        for m, i in picks.items():
            cands.append({"name": f"{m}__{cap_name}", "arch": keys[i], "init": "supernet+ft5k",
                          "search_val_sr": float(T[i].mean()), "ms_per_step": float(lat[i])})
    # schedule-only tuning of the published architecture (best knobs for the default net on search-val)
    from .space import DEFAULT, Arch
    dn = [i for i, k in enumerate(keys) if Arch.from_key(k).net == DEFAULT.net]
    best_sched = keys[max(dn, key=lambda i: T[i].mean())]
    cands += [
        {"name": "smolvla_default_tuned_schedule", "arch": best_sched, "init": "standalone30k"},
        {"name": "smolvla_default_published", "arch": "v16-e16-stretch-f1-t64-s10-h50", "init": "standalone30k"},
        {"name": "smolvla_default_ft5k", "arch": "v16-e16-stretch-f1-t64-s10-h50", "init": "supernet+ft5k"},
    ]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"budget": BUDGET, "latency_cap_ms_per_step": cap, "frozen_at_commit": commit, "candidates": cands}
    json.dump(out, open("outputs/final_candidates.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
