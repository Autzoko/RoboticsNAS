# RoboticsNAS — Closed-Loop-Aware Architecture Search for VLA Models

Working title: **"Offline Proxies Mislead VLA Architecture Search: Closed-Loop Weight-Sharing NAS for Flow-Matching VLAs"**

## 1. Motivation and positioning

VLA architectures (VLM depth, which VLM features condition the action head, action-expert size,
visual-token budget, denoising steps, executed action horizon) are chosen by hand and trial-and-error.
Recent controlled studies ("What Makes an Efficient VLA?", arXiv 2609.13984) sweep a few axes manually;
hardware-aware NAS for robot policies (DC-QFA, arXiv 2604.10170) searches *deployment* configurations
(architecture x bit-width) per device and selects subnets with offline/latency criteria.
"Anatomy of a Closed-Loop Collapse" (arXiv 2609.23048) shows one compressed VLA that passes offline checks
but collapses in closed loop.

Gap we target: **no NAS work for VLAs measures whether the search signal ranks architectures the way
closed-loop task success does, and none treats closed-loop rollouts as a first-class, budgeted fidelity
inside the search.** Standard NAS relies on validation loss; for flow-matching VLAs this signal is
(i) blind to inference-time knobs (denoising steps, executed horizon) and (ii) measured on expert
(off-policy) states, not on the states the policy actually visits.

## 2. Research questions

- **RQ1 (proxy validity).** Across a weight-sharing VLA search space, how well do standard NAS signals
  (flow-matching validation loss, denoised action error, zero-cost proxies, FLOPs/params) rank architectures
  by closed-loop success rate (Kendall tau / Spearman rho / top-k overlap)?
- **RQ2 (closed-loop-aware proxies).** Do cheap signals computed on *policy-visited* states
  (on-policy agreement with the anchor policy, chunk-boundary consistency) or low-fidelity rollouts
  (1-2 episodes/task with common random numbers) rank architectures better per unit compute?
- **RQ3 (search).** Under an equal simulation budget, does closed-loop multi-fidelity search
  (proxy-filtered candidates + paired successive-halving racing on identical init seeds) find better
  success-latency Pareto fronts than random search, offline-loss-guided evolution, and the hand-designed
  SmolVLA configuration?
- **RQ4 (design rules).** What do the searched/benchmarked architectures say about VLM depth, VLM->expert
  bridging, expert depth/width, visual tokens, denoising steps and execution horizon (and their interactions)?

Contributions: (C1) an elastic weight-sharing search space for a flow-matching VLA covering architecture
*and* inference-schedule axes; (C2) a closed-loop VLA NAS benchmark table (closed-loop SR for ~100
architectures + offline/on-policy proxies) and the proxy-validity study; (C3) a rollout-efficient
closed-loop search procedure; (C4) design findings.

## 3. Base model and data (leakage-free protocol)

- **Initialization:** `lerobot/smolvla_base` (expert pretrained on SO-100 community datasets; *no LIBERO*),
  VLM `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (internet data). Revisions pinned in `configs/pins.json`.
  **Forbidden:** any LIBERO-finetuned checkpoint (e.g. `lerobot/smolvla_libero`, `HuggingFaceVLA/smolvla_libero`,
  `YuankaiLuo/SimVLA-LIBERO`, pi0/pi05 LIBERO checkpoints).
- **Data:** `lerobot/libero` (4 suites, 40 tasks, 1693 demos, local copy `/scratch/ll5582/LIBERO/lerobot_libero`).
  Per task, a fixed set of demos is held out as **offline-val** (used only for offline proxies);
  the rest is **train**. State/action normalization statistics are computed on **train only**.
- **Search-time closed loop ("search-val"):** LIBERO simulator with `init_states=False`; initial object
  layouts are sampled by the LIBERO/robosuite placement samplers from dedicated reset seeds
  (`SEARCH_SEED_BASE = 1_000_000`). The official fixed init-state files are never loaded during
  search / proxy studies / model selection (enforced in code).
- **Final test:** official LIBERO init states (50 per task, 40 tasks = 2000 episodes per policy), run only
  for architectures frozen in `results/final_candidates.json` (committed before testing).
  All test runs are logged in `docs/EXPERIMENT_LOG.md`.
- VLM (incl. vision encoder) frozen as in SmolVLA; trained: action expert, state/action/time projections.

## 4. Search space (SmolVLA family)

SmolVLA: frozen SmolVLM2 (first 16 of 32 text layers), action expert with one layer per VLM layer,
width 0.75x, alternating self-attention (even) / cross-attention (odd) layers; expert layer j reads
the KV of VLM layer j. 64 visual tokens per camera; 10 Euler denoising steps; chunk 50.

| axis | symbol | choices | default (SmolVLA) |
|---|---|---|---|
| VLM layers computed | `n_vlm` | 8, 12, 16, 20, 24 | 16 |
| expert layers (first-k of 16 pretrained) | `n_exp` | 4, 8, 12, 16 | 16 |
| VLM->expert bridge | `bridge` | `stretch` (b(j)=floor((j+1)n_vlm/n_exp)-1), `top` (b(j)=n_vlm-n_exp+j) | stretch(=aligned) |
| expert FFN width (sorted neurons) | `ffn` | 0.5, 0.75, 1.0 | 1.0 |
| visual tokens / camera | `vtok` | 16 (2x2 avg-pool), 64 | 64 |
| denoising steps (inference) | `steps` | 2, 4, 10 | 10 |
| executed action horizon (inference) | `horizon` | 5, 10, 25, 50 | 50 |

`top` requires n_exp <= n_vlm. Total ~2.6k valid configurations. Latency/FLOPs measured per config.

## 5. Supernet training

Sandwich rule per step: anchor (= SmolVLA default arch), smallest arch, 2 uniformly random archs.
All subnets share one prefix pass (frozen VLM, with gradient flowing only to the state projection).
Loss = flow-matching MSE to ground truth for every subnet + in-place distillation of non-anchor
subnets to the anchor's predicted velocity (same x_t, t; teacher detached). FFN neurons sorted by
importance before training. Inference knobs (steps, horizon) are evaluation-only.

## 6. Experiments

- **E0 correctness:** elastic implementation at the default arch reproduces LeRobot SmolVLA loss/actions
  numerically (one check, no other smoke tests).
- **E1 supernet + standalone baseline:** train supernet; train the default SmolVLA arch standalone with the
  same data/steps (reference).
- **E2 benchmark (RQ1/RQ2):** ~100 architectures (stratified random + anchor + extremes), each evaluated on
  search-val (10 episodes/task x 40 tasks, common seeds) + all proxies + latency.
- **E3 search (RQ3):** same total episode budget for random / offline-proxy evolution / ours; tabular replay
  on E2 for many repetitions + one live search run per method.
- **E4 final test:** top Pareto archs from each method + default SmolVLA (standalone and supernet-inherited)
  fine-tuned briefly from supernet weights, then official LIBERO test (2000 episodes each). Latency on A100.
- **E5 weight-sharing validity:** ~8 archs trained standalone; rank agreement with supernet-inherited SR.

## 7. Compute plan (Jubail, <= 12 GPUs/user)

Supernet ~1 GPU-day (A100). Standalone baseline ~0.5 GPU-day. E2 ~100 archs x 400 episodes ≈ 40-60 GPU-h
(parallel jobs). E4 ~10 policies x 2000 episodes. Total target: ~1 week wall-clock.
