"""S4: does teacher-imitation training at 10k steps rank archs like plain 30k training? (pre-registered: results/s4_plan.json)

Paired on S1 search-val cells. Compares distill@10k and plain@10k (S3) rankings against plain@30k (S1) on the 8 S4 nets,
and applies the 'cheapest tied with the best' selection rule (s3_rank.pick) to each ranking source.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .s3_rank import load, pick


def main():
    plan = json.load(open("results/s4_plan.json"))
    keys = plan["archs"]
    full, short, dist = load("outputs/s1/*.jsonl"), load("outputs/s3/*.jsonl"), load("outputs/s4/eval/*.jsonl")
    keys = [k for k in keys if k in full and k in short and k in dist]
    cs = sorted(set.intersection(*[set(full[k]) & set(short[k]) & set(dist[k]) for k in keys]))
    F, S, D = (np.array([[src[k][c] for c in cs] for k in keys]) for src in (full, short, dist))
    f, s, d = F.mean(1), S.mean(1), D.mean(1)
    cost = {}
    for line in open("results/cost/a100_v2.jsonl"):
        r = json.loads(line)
        cost[r["arch"]] = r
    ms = {k: cost[k]["ms_per_step"] for k in keys}
    rng = np.random.default_rng(0)
    bt = {"distill": [], "plain": [], "diff": []}
    for _ in range(2000):
        i = rng.integers(0, len(cs), len(cs))
        fi = F[:, i].mean(1)
        a, b = kendalltau(D[:, i].mean(1), fi).statistic, kendalltau(S[:, i].mean(1), fi).statistic
        bt["distill"].append(a); bt["plain"].append(b); bt["diff"].append(a - b)
    ci = {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in bt.items()}
    td, tp = kendalltau(d, f).statistic, kendalltau(s, f).statistic
    gate = td >= 0.40 and td >= tp + 0.2
    se = np.sqrt(f * (1 - f) / len(cs)).mean()
    lines = [f"# S4: teacher-imitation @10k vs plain @30k (n={len(keys)}, {len(cs)} paired search-val episodes, s2-h10)", "",
             "| ranking source | Kendall vs plain@30k [95% CI] | Spearman | mean SR | SR range | SR std across nets |",
             "|---|---|---|---|---|---|",
             f"| distill@10k (S4) | {td:.3f} [{ci['distill'][0]:.3f}, {ci['distill'][1]:.3f}] | "
             f"{spearmanr(d, f).statistic:.3f} | {d.mean():.3f} | {d.min():.3f}-{d.max():.3f} | {d.std():.3f} |",
             f"| plain@10k (S3) | {tp:.3f} [{ci['plain'][0]:.3f}, {ci['plain'][1]:.3f}] | "
             f"{spearmanr(s, f).statistic:.3f} | {s.mean():.3f} | {s.min():.3f}-{s.max():.3f} | {s.std():.3f} |",
             f"| plain@30k (S1, target) | 1 | 1 | {f.mean():.3f} | {f.min():.3f}-{f.max():.3f} | {f.std():.3f} |", "",
             f"Kendall difference distill - plain@10k: {td - tp:+.3f} [95% CI {ci['diff'][0]:+.3f}, {ci['diff'][1]:+.3f}]. "
             f"Per-net binomial SE at 400 eps ~{se:.3f}.",
             f"**Pre-registered gate (tau >= 0.40 and >= plain@10k + 0.2): {'PASS' if gate else 'FAIL'}**", "",
             "| arch | distill@10k | plain@10k | plain@30k | ms/step |", "|---|---|---|---|---|"]
    for i in np.argsort(-f):
        lines.append(f"| {keys[i]} | {d[i]:.3f} | {s[i]:.3f} | {f[i]:.3f} | {ms[keys[i]]:.1f} |")
    lines += ["", "Selection rule 'cheapest (ms/step) tied with the best', scored by plain@30k SR:", "",
              "| ranking source | pick | tied set | pick SR@30k | regret | ms/step |", "|---|---|---|---|---|---|"]
    for name, M in (("plain@30k (oracle)", F), ("distill@10k", D), ("plain@10k", S)):
        p, tied = pick(M, keys, ms, np.random.default_rng(1))
        lines.append(f"| {name} | {keys[p]} | {len(tied)} | {f[p]:.3f} | {f.max() - f[p]:.3f} | {ms[keys[p]]:.1f} |")
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    Path("outputs/analysis/s4_rank.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
