# RoboticsNAS v2 — From an engineering gain to a NAS method (plan, 2026-10-06)

## 0. Where v1 stands (honest assessment)

- The final gain (+14.7 pts official LIBERO) comes from the inference schedule of the *unchanged* SmolVLA network;
  a plain schedule sweep gets the same (0.810 vs 0.821, CIs overlap). It is an engineering result for one model.
- The NAS pipeline could not return a smaller network because (a) weight sharing under-rates shallow action experts
  (supernet vs standalone Kendall 0.26, n=8; e4 subnets 0.45 vs 0.85 standalone) and (b) the best proxy measures
  agreement with the anchor, so it is anchored to the default network.
- Reusable, defensible findings: offline flow-matching loss does not rank VLA architectures (tau 0.165); closed-loop
  evidence can be obtained cheaply (proxy + paired racing); inference-schedule axes belong in the space.

v2 goal: a NAS method that (i) has a principled search signal, (ii) has a supernet whose ranking matches standalone
training, (iii) actually returns smaller/faster architectures that hold up on the official test, and (iv) transfers
to other VLA families and a second benchmark.

## 1. Method contributions to develop

### M1. A closed-loop policy-gap proxy with a bound (replaces "agreement with anchor")
Use the performance-difference / simulation lemma for chunked policies: for a candidate policy pi_a and a reference
pi_r executed with horizon h over task length T,
  |J(pi_a) - J(pi_r)| <= C * T * E_{s ~ d^{pi_r}} [ D(pi_a(.|s), pi_r(.|s)) ] (+ a term in the chunk boundary
  mismatch that grows with h).
Consequences we can test, not just assert:
- Flow-matching loss is not a bound on D (it is a per-time-step velocity error averaged over noise levels and is
  invariant to steps/horizon), so it can be arbitrarily uninformative -> explains tau 0.165.
- The bound is *one-sided around the reference*: it explains the anchor bias of v1. Fix: **bootstrapped reference** —
  during racing, the reference is replaced by the current empirical best subnet (new rollouts only for it), so the
  proxy is evaluated on the state distribution of the best-so-far policy instead of the default network.
- Horizon term: weight divergence by executed horizon and measure it on visited states at chunk boundaries.
Deliverable: proxy = estimated bound, compared against v1 KD proxy, zero-cost proxies, and closed-loop low fidelity.

### M2. Depth-decoupled supernet training for flow-matching action experts (fixes the ranking)
Hypothesis: shallow subnets are under-trained because all depths share one readout (final norm + action_out_proj)
and the first k layers must act both as intermediate and final features; the sandwich rule plus KD to the anchor
favours deep subnets. Planned variants (each one supernet run, ~8 h on one A100):
- V0 v1 baseline (sandwich, KD to anchor).
- V1 depth-specific readouts: per-depth RMSNorm + small output adapter (<1% params), shared trunk.
- V2 depth-balanced sampling (uniform over depth, not over configs) + per-depth KD teacher (deepest of same depth
  family) instead of the anchor.
- V3 gradient-conflict control: measure cosine between depth-subnet gradients; project conflicting components
  (PCGrad-style) on shared layers.
Primary metric: rank agreement between supernet-inherited SR and standalone SR (Kendall), target >= 0.6.
Diagnostic (cheap, no rollouts): gradient-conflict per depth pair over training; correlate with the E5 gap.

### M3. Racing with bound-based pruning (search procedure with a budget guarantee)
Paired successive elimination on common random numbers (same seeds for all candidates), with confidence intervals
from paired differences; candidates whose M1 lower bound is below the current best upper bound are pruned without
rollouts. Report the episode budget needed to reach epsilon-regret and compare with v1 SH, random, BO on proxies.

## 2. Experiments (ordered; each step gates the next)

| step | what | compute | gate / decision |
|---|---|---|---|
| S0 | Re-analyse v1 data: compute M1 proxy variants (horizon-weighted, chunk-boundary) from saved visited states; gradient-conflict diagnostic on v1 supernet | ~3 GPU-h | M1 proxy tau >= KD proxy; conflict correlates with gap |
| S1 | Enlarge standalone ground truth for the weight-sharing study: 8 -> 16 pre-registered archs (stratified by depth/width/tokens), 30k steps each, search-val 400 eps | ~60 GPU-h | needed for a credible rank test (n=16) |
| S2 | Supernet variants V0-V3 (SmolVLA, same data/steps), evaluate the 16 S1 archs with inherited weights | 4 x 8 h + ~25 GPU-h eval | pick the variant with best Kendall vs standalone |
| S3 | Full search with best supernet + M1 + M3 under 2 latency caps; compare with v1 search, random, val-loss NAS, zero-cost proxies (SynFlow, grad-norm, NASWOT), a GP predictor on proxies | ~40 GPU-h | does search now return smaller nets with standalone-level SR? |
| S4 | Second VLA family: pi0 (PaliGemma 3B + flow expert; VLM frozen, expert-only supernet) on LIBERO with the same space (expert depth/width, VLM layers, steps, horizon) | ~4 days A100 | transfer of M1/M2/M3 conclusions |
| S5 | Second benchmark: SmolVLA on MetaWorld MT50 (LeRobot env, cheap sim) and LIBERO-Plus perturbation splits for robustness of searched archs | ~2 days | not LIBERO-specific |
| S6 | Final tests: frozen candidates per (model, benchmark), 3 training seeds for the top-2 and the baseline, paired bootstrap tests; latency on A100 + one edge device (Jetson Orin if available) | ~3 days | headline table |

Optional (if S4 is too slow): X-VLA (0.9B, LeRobot) instead of pi0 as the second family; it has a different
action head, which also tests generality beyond flow-matching experts.

## 3. Paper framing after v2

Title direction: "Closed-loop-aware weight-sharing NAS for vision-language-action models".
1. Analysis: offline losses fail; a bound-based closed-loop proxy and why (M1).
2. Method: depth-decoupled supernet (M2) + bound-pruned paired racing (M3).
3. Results: on SmolVLA and pi0, LIBERO and MetaWorld, searched architectures that are smaller/faster at equal or
   better official-test SR than the published configs and than schedule-only tuning; supernet ranking validated
   against standalone training (n=16).
v1 results become the motivation section and the ablation baseline (V0).

## 4. Protocol (unchanged, plus)
- Same leakage rules: no benchmark-finetuned checkpoints (pi0 must start from `lerobot/pi0_base`-style weights
  without LIBERO data; verify its pretraining mixture), train-only stats, search on fresh layouts, official init
  states only for frozen candidates.
- Pre-register S1 architectures and S6 candidates in `results/` before running them.
- Exclude flaky simulation nodes (cn012/014/015/016/270); rollout code already has timeouts + resume.

## 5. Risks
- M2 may not close the gap -> fall back to two-stage search (supernet shortlist -> short standalone fine-tune ->
  race) and report weight-sharing failure as a negative result with its diagnostic.
- pi0 cost -> X-VLA or a reduced pi0 (frozen VLM, fewer expert axes).
- Fair-share/queue limits (~12 GPUs, priority drops after heavy use) -> S1 and S2 run in parallel; S4 last.
