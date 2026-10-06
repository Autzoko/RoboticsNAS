# RoboticsNAS — Results summary (draft, 2026-10-06)

Base: SmolVLA (`lerobot/smolvla_base`, no LIBERO pretraining) on LIBERO (4 suites, 40 tasks). Elastic weight-sharing
search space over VLM depth, expert depth/width, VLM->expert bridge, visual tokens, denoising steps, executed horizon
(2,448 configs). All selection on *search-val* (LIBERO with freshly sampled layouts, dedicated seeds); official init
states touched once, for candidates frozen at commit `186b175`. Details/jobs: `EXPERIMENT_LOG.md`.

## Claim 1 — Offline validation loss does not rank VLA architectures by closed-loop success (RQ1)

E2: 111 architectures x 400 closed-loop episodes (44,400 episodes) with shared supernet weights.
Ground truth = SR on a held-out half of episodes (split-half ceiling Kendall tau = 0.840).

| signal | rollouts / arch | Kendall tau [95% CI] | top-10 overlap |
|---|---|---|---|
| closed loop, 2 eps/task | 80 | 0.819 [0.769, 0.860] | 6/10 |
| closed loop, 1 ep/task | 40 | 0.775 [0.726, 0.818] | 6/10 |
| agreement with anchor policy, offline states | 0 | 0.756 [0.697, 0.811] | 8/10 |
| agreement with anchor policy, anchor-visited states | 0 (+anchor once) | 0.738 [0.670, 0.796] | 8/10 |
| executed-action L1 vs demos (sees steps & horizon) | 0 | 0.667 [0.605, 0.726] | 4/10 |
| full-chunk action L1 vs demos | 0 | 0.538 [0.445, 0.621] | 0/10 |
| **flow-matching validation loss (standard NAS signal)** | 0 | **0.165 [0.005, 0.330]** | **0/10** |
| latency | 0 | ~0 | 0/10 |

Same phenomenon across training of one fixed architecture (standalone default SmolVLA, search-val 400 eps):

| ckpt | offline val flow loss | SR (h50) | SR (h10) |
|---|---|---|---|
| 10k | 0.515 | 0.477 | 0.700 |
| 20k | 0.738 | 0.625 | 0.772 |
| 30k | 0.961 | 0.713 | 0.850 |

Val loss is minimal at 6k (0.495) and would select the worst checkpoint.

Why: (i) flow-matching loss is invariant to inference-schedule axes (denoising steps, executed horizon) that move SR
by >20 points; (ii) it scores per-step velocity regression on expert states, not chunk-level action quality where it
matters. Signals that *execute the inference schedule* (denoised executed actions, agreement with a strong policy)
recover most of the ranking.

## Claim 2 — Closed-loop-aware search finds near-optimal architectures with few rollouts (RQ3)

Replay on the E2 table (300 repetitions, chosen arch scored on held-out episodes). Regret vs best eligible arch,
latency cap <= median (12 ms per control step):

| method | 400 eps | 800 | 1600 |
|---|---|---|---|
| **proxy shortlist (anchor agreement) + paired successive halving** | **0.024** | **0.009** | **0.001** |
| successive halving, random pool | 0.084 | 0.070 | 0.058 |
| successive halving, val-loss shortlist | 0.112 | 0.097 | 0.078 |
| random search | 0.178 | 0.123 | 0.086 |
| val-loss top-1 (standard NAS, 0 rollouts) | 0.210 | 0.210 | 0.210 |

## Claim 3 — Inference schedule is a first-class architecture axis (RQ4)

- Executing the full 50-step chunk (SmolVLA default) is the worst horizon in every setting; h10-h25 best.
- 2 Euler denoising steps match or beat 10 (supernet: 0.611 vs 0.516 marginal; standalone default: 0.848 vs 0.828 at
  h25) -> 5x fewer expert passes.
- 16 visual tokens/camera ~ 64 (0.564 vs 0.579 marginal): 4x fewer visual tokens nearly free.
- Bridging: uniform `stretch` mapping slightly better than reading only the top VLM layers.

## Claim 4 (limitation) — Weight sharing distorts the deployment-relevant ranking of expert depth (E5/E5b)

| arch | supernet-inherited | +5k fine-tune | standalone 30k |
|---|---|---|---|
| v16-e16 (default) h50 | 0.693 | 0.723 | 0.713 |
| v8-e12 h50 | 0.458 | 0.748 | 0.690 |
| v12-e4 h5 | 0.450 | 0.680 | 0.853 |
| v16-e8-top h5 | 0.665 | 0.752 | 0.840 |
| v12-e8-top h5 | 0.762 | 0.800 | 0.835 |
| v24-e4-top h25 | 0.085 | 0.352 | 0.640 |
| v24-e16-top h10 | 0.695 | 0.767 | 0.853 |

Supernet vs standalone Kendall 0.255 (n=8), mean gap +0.21, largest for 4-layer experts. Standalone-trained small
networks are competitive with the default once the schedule is short. The closed-loop proxies rank *supernet*
subnets well (Claim 1/2), but the supernet itself under-rates shallow experts; 5k-step fine-tuning closes about half the gap.
Search therefore favours the anchor network with a cheaper schedule.

## Final test (official LIBERO init states, 2000 episodes per policy) — E4

See `results/e4.md` (filled when all runs complete).
