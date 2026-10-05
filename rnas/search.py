"""Search strategies under a simulation-episode budget, replayed on the closed-loop benchmark table.

Each benchmarked arch has per-episode success on a fixed grid of (task, episode-seed) cells shared by all
archs (common random numbers). Episodes are split into a SEARCH half (seen by search methods) and a
held-out EVAL half used only to score the chosen arch (avoids winner's-curse bias).

Methods (all return one chosen arch under an optional latency cap):
  proxy_top1(p)       zero rollouts; pick best by proxy p
  random_full(m)      m random archs, each evaluated on all search episodes (budget-limited)
  sh(n0)              successive halving from n0 random archs, paired episodes, doubling per round
  proxy_sh(p, n0)     successive halving from the n0 best archs by proxy p   (ours: closed-loop-aware)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_table(rollout_jsonl: list[str]) -> tuple[list[str], np.ndarray, list[tuple]]:
    """-> arch keys, success tensor S[arch, cell] (cell = (suite, task, episode)), cell ids."""
    rows = []
    for p in rollout_jsonl:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            for e, s in zip(r["episodes"], r["success"]):
                rows.append((r["arch"], r["suite"], r["task"], e, float(s)))
    df = pd.DataFrame(rows, columns=["arch", "suite", "task", "ep", "succ"])
    df = df.groupby(["arch", "suite", "task", "ep"], as_index=False)["succ"].mean()
    piv = df.pivot_table(index="arch", columns=["suite", "task", "ep"], values="succ")
    piv = piv.dropna(axis=0)  # archs with complete grids only
    return list(piv.index), piv.to_numpy(), list(piv.columns)


class Replay:
    def __init__(self, keys, S, cells, search_eps, eval_eps, rng):
        self.keys = keys
        ep = np.array([c[2] for c in cells])
        self.Ss = S[:, np.isin(ep, search_eps)]
        self.Se = S[:, np.isin(ep, eval_eps)]
        self.true = self.Se.mean(1)
        self.rng = rng
        # column order for paired evaluation: random permutation of search cells, fixed per replay
        self.order = rng.permutation(self.Ss.shape[1])

    def sr(self, i, n):
        """Observed SR of arch i on the first n cells of the shared (paired) order."""
        return self.Ss[i, self.order[:n]].mean()


def proxy_top1(R, cand, proxy):
    return min(cand, key=lambda i: proxy[i]), 0


def random_full(R, cand, budget):
    n_cells = R.Ss.shape[1]
    m = max(1, min(len(cand), budget // n_cells))
    pick = R.rng.choice(cand, size=m, replace=False)
    return max(pick, key=lambda i: R.sr(i, n_cells)), m * n_cells


def successive_halving(R, pool, budget):
    """Paired SH: every survivor sees the same cells; cells per arch double each round."""
    pool = list(pool)
    n_cells = R.Ss.shape[1]
    rounds = max(1, math.ceil(math.log2(len(pool))))
    per = max(1, budget // (len(pool) * rounds))  # cells per arch in round 0
    used, n = 0, 0
    while len(pool) > 1 and used < budget:
        n = min(n_cells, max(n + 1, per))
        used += len(pool) * n
        pool.sort(key=lambda i: -R.sr(i, n))
        pool = pool[: max(1, len(pool) // 2)]
        per *= 2
    return pool[0], used


def sh(R, cand, budget, n0):
    pool = R.rng.choice(cand, size=min(n0, len(cand)), replace=False)
    return successive_halving(R, pool, budget)


def proxy_sh(R, cand, budget, n0, proxy):
    pool = sorted(cand, key=lambda i: proxy[i])[:n0]
    return successive_halving(R, pool, budget)
