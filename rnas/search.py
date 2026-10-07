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


# ----------------------------------------------------------------------------- baselines / M1-M3
def arch_features(keys: list[str]) -> np.ndarray:
    """One-hot encoding of all axes (+ bias) for predictor baselines."""
    from .space import BRIDGE, FFN, HORIZON, N_EXP, N_VLM, STEPS, VTOK, Arch

    axes = [("n_vlm", N_VLM), ("n_exp", N_EXP), ("bridge", BRIDGE), ("ffn", FFN), ("vtok", VTOK),
            ("steps", STEPS), ("horizon", HORIZON)]
    X = []
    for k in keys:
        a = Arch.from_key(k)
        row = [1.0]
        for name, vals in axes:
            row += [1.0 if getattr(a, name) == v else 0.0 for v in vals]
        X.append(row)
    return np.array(X)


def predictor_search(R, cand, budget, X, n_init=10, cells_per=40, ridge=1.0):
    """Sequential model-based search (BO-like): low-fidelity rollouts (cells_per cells) on n_init random archs,
    ridge regression on one-hot features, then repeatedly evaluate the predicted-best unevaluated arch."""
    n_cells = R.Ss.shape[1]
    cells_per = min(cells_per, n_cells)
    seen, y, used = [], [], 0
    pool = list(R.rng.permutation(cand))
    for i in pool[:n_init]:
        if used + cells_per > budget:
            break
        seen.append(i); y.append(R.sr(i, cells_per)); used += cells_per
    while used + cells_per <= budget and len(seen) < len(cand):
        A = X[seen]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ np.array(y))
        rest = [i for i in cand if i not in seen]
        nxt = max(rest, key=lambda i: X[i] @ w)
        seen.append(nxt); y.append(R.sr(nxt, cells_per)); used += cells_per
    best = seen[int(np.argmax(y))] if seen else cand[0]
    return best, used


def bound_race(R, cand, budget, D: dict, h, n0=8, ref_cells=200, q=0.9, delta=0.01):
    """M1+M3: bootstrapped-reference bound racing.
    D: {ref_idx: divergence vector over all archs, D_h(a; ref)} (lower = closer to ref); h: executed horizon per arch.
    LB(a) = J(r) - L * D(a; r) / h(a)   (T absorbed into L);  UB(a) = J(r) + L * D(a; r) / h(a).
    Round: evaluate incumbent r (ref_cells); prune candidates whose UB cannot beat the best SR so far by delta (after L
    is calibrated); shortlist the rest ordered by LB; paired SH;
    calibrate L (q-quantile of |J_a - J_r| / (D/h)) from raced pairs; if the SH winner beats r on the same paired cells,
    it becomes the new reference (when its D vector is available)."""
    refs = [r for r in D if r in cand] or list(D)
    r = refs[0]
    used = ref_cells
    Jr = R.sr(r, ref_cells)
    L = None
    raced = set([r])
    best, best_sr = r, Jr
    while used < budget:
        d = D[r] / np.maximum(h, 1)
        pool = [i for i in cand if i not in raced]
        if L is not None:
            pool = [i for i in pool if Jr + L * d[i] >= best_sr + delta]
        if not pool:
            break
        pool = sorted(pool, key=lambda i: d[i])[:n0]  # highest LB first (LB is monotone in d for fixed r)
        win, u = successive_halving(R, pool + [r], max(1, budget - used))
        used += u
        n_eval = min(R.Ss.shape[1], ref_cells)
        ratios = [abs(R.sr(i, n_eval) - Jr) / d[i] for i in pool if d[i] > 0]
        if ratios:
            L = float(np.quantile(ratios, q))
        raced |= set(pool)
        w_sr = R.sr(win, n_eval)
        if w_sr > best_sr:
            best, best_sr = win, w_sr
            if win in D:  # bootstrap the reference
                r, Jr = win, w_sr
        if u == 0:
            break
    return best, used
