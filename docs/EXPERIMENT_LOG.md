# Experiment log

Chronological record of jobs, settings, results and decisions. Times are Jubail local (GST, UTC+4) unless noted.
Every final-test (official LIBERO init states) run must be listed in the "Test-set access" table.

## Test-set access

| date | policy / arch | ckpt | job | note |
|---|---|---|---|---|
| 2026-10-06 07:06 | smolvla_default_published / v16-e16-stretch-f1-t64-s10-h50 | fixed_default/final.pt | 18671894-905 | E4 final, frozen @186b175 |
| 2026-10-06 07:06 | smolvla_default_tuned_schedule / v16-e16-stretch-f1-t64-s2-h10 | fixed_default/final.pt | " | " |
| 2026-10-06 07:06 | ours_proxySH_kd_cap / v16-e16-stretch-f1-t64-s2-h10 | ft5k/default | " | " |
| 2026-10-06 07:06 | random_search_cap_ft5k / v16-e16-stretch-f1-t64-s10-h50 | ft5k/default | " | " |
| 2026-10-06 07:06 | nas_valloss_top1_cap / v16-e8-stretch-f1-t16-s4-h50 | ft5k/v16-e8-stretch-f1-t16-s4-h50 | " (after ft) | " |
| 2026-10-06 07:06 | standalone_race_best_small / v12-e4-stretch-f1-t16-s4-h5 | standalone/v12-e4 | " | " |

## Log

### 2026-10-05 — setup
- Repo initialised (local + GitHub `Autzoko/RoboticsNAS`, HPC checkout at `/scratch/ll5582/RoboticsNAS`).
- LeRobot pinned at `3f2c29e` (`third_party/lerobot`), env `env/` built by job 18656149 (`scripts/setup_env.sbatch`).
- Dataset: local `lerobot/libero` snapshot at `/scratch/ll5582/LIBERO/lerobot_libero` (HF revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`).
- Env fixes: gcc module needed for `egl_probe` build; `opencv-python` replaced by headless (no libGL on nodes);
  project-local LIBERO config (`configs/libero/config.yaml`, `LIBERO_CONFIG_PATH`) because `~/.libero` points to a
  non-existent path; LIBERO assets downloaded from HF Hub into `cache/libero/assets` (+ package symlink).
- Pins (`configs/pins.json`): smolvla_base `d9f33c94`, SmolVLM2-500M-Video-Instruct `7b375e1b`.
- Split (`configs/split.json`): 3 demos/task held out as offline-val (120 eps), 1573 train eps; stats on train only.

### 2026-10-05 — E0 correctness (jobs 18656389, 18656437)
- smolvla_base VLM layers 0-15 are bit-identical to SmolVLM2 (max diff 0) -> VLM was frozen in pretraining;
  appending SmolVLM2 layers 16-23 for the `n_vlm>16` axis is consistent.
- Elastic forward at default arch vs LeRobot SmolVLA (fp32): train-loss max abs diff 5.7e-6 (mean 0.377);
  `sample_actions` max abs diff 0. FFN neuron sorting function-preserving (5e-7).
- LIBERO (EGL, 256x256, hard reset) runs on A100 nodes. Common random numbers verified: identical reset seeds give
  identical sim-state hashes across archs; different episodes give different layouts.
- Untrained (smolvla_base, no LIBERO training) SR = 0/2 on spatial task 0, as expected. 280-step batch: 15-31 s.

### 2026-10-05 — E1 training launch
- First launch (jobs 18656510 supernet, 18656511 fixed) aborted: transformers builds the action expert in bf16
  (config dtype), so AdamW updated bf16 master weights; the supernet's flat grad all-reduce crashed on it.
  Fix (commit "Keep trainable params in fp32"): all trainable params cast to fp32, bf16 autocast for compute.
  Both runs restarted from scratch for a fair supernet-vs-standalone comparison.
- Throughput reference (old run, fixed default arch, H100 NVL, batch 64): 2.5 it/s, GPU-bound (98% util).
- Latency (A100 80GB PCIe, batch 1, eager bf16): ~85-220 ms per policy call across archs (job 18656514).
- Relaunch: supernet 18656828 (2 GPUs, 30k steps, global batch 64, sandwich: anchor+smallest+2 random, KD w=1),
  fixed default 18656829 (1 GPU, 30k steps, batch 64).
- 14:18: supernet 18656828 (2xH100, manual flat all-reduce) crashed at step ~1550 with a CUDA illegal memory
  access on rank 1 (NCCL watchdog). Loss curve until then was healthy (step 1500: fm anchor 0.71, smallest 0.76).
  Not debugged further; relaunched single-GPU (18657517, batch 64, ckpt every 1000 steps, auto-resume).
- Single-GPU supernet 18657517: 1.09 it/s on A100 (ETA ~22:20). 
- E5 (weight-sharing validity) launched early to use idle time: 7 archs pre-registered in `results/e5_archs.json`
  (random among benchmark nets, seed 1, chosen before any closed-loop result) trained standalone 30k steps
  (jobs 18657910-18657917); default arch standalone = `outputs/fixed_default`.

### 2026-10-05 — Pipeline validation (job 18658750)
Standalone default arch, step 10k (`outputs/fixed_default/step10k.pt`), search-val, 3 eps/task (ep-offset 200,
disjoint from benchmark seeds), 40 tasks:

| arch | spatial | object | goal | long | all |
|---|---|---|---|---|---|
| v16-e16-stretch-f1-t64-s10-h50 | 0.50 | 0.50 | 0.63 | 0.23 | 0.467 |
| v16-e16-stretch-f1-t64-s10-h10 | 0.63 | 0.93 | 0.83 | 0.43 | 0.708 |

- Same weights, only the executed horizon changes: +24 pts. Offline flow loss is identical for both by construction ->
  first concrete evidence for the inference-schedule axes / offline-proxy blindness.
- Offline-val flow loss of this run rises after 6k steps (0.495@6k -> 0.515@10k) while train loss falls; snapshots
  kept at 10k/20k/30k to relate offline loss and closed-loop SR over training.
- Cost: ~18 s wall per (arch, task) with 3 envs.

### 2026-10-05 17:40 — Standalone default arch finished (30k steps)
Offline-val flow loss over training (fixed noise/t, 8x32 val samples):
2k 0.543 | 4k 0.505 | 6k **0.495** | 8k 0.507 | 10k 0.515 | 12k 0.550 | 16k 0.625 | 20k 0.738 | 24k 0.863 | 28k 0.944 | 30k 0.961.
Offline loss would select the 6k checkpoint. Closed-loop search-val evals launched (jobs 18660168-72):
10k/20k ckpts x {h50,h10}; 30k ckpt x full {steps 2,4,10} x {horizon 5,10,25,50} grid, 10 eps/task (bench seeds).
Supernet at 11.5k steps: val fm anchor 0.495, smallest 0.494 @10k.

### 2026-10-05 18:41 — Status / handoff
- Partial curve evals (spatial + part of object): SR rises 10k->20k (h50 0.47->0.60, h10 0.69->0.82) while offline-val
  flow loss rises 0.515->0.738. Final numbers in `outputs/curve_fixed/*.jsonl` (jobs 18660168-72).
- Running: supernet 18657517 (step ~16k/30k, ETA ~22:20), E5 standalone 18657911-17 (~3.5 h each); 18657910 done.
- NEXT (in order):
  1. Supernet done -> anchor visited-state recording: `rnas.rollout --ckpt outputs/supernet_v1/final.pt
     --archs v16-e16-stretch-f1-t64-s10-h10 --n-eps 5 --ep-offset 100 --record-dir outputs/bench_sn/visited`.
  2. E2: `bash scripts/launch_bench.sh outputs/supernet_v1/final.pt sn 10` (111 archs, 10 eps/task).
  3. Proxies: `rnas.proxies --ckpt ... --archs results/bench_archs.json --record-dir outputs/bench_sn/visited
     --out outputs/bench_sn/proxies.jsonl`; copy `outputs/latency/a100_bench.jsonl` -> `outputs/bench_sn/latency.jsonl`.
  4. `rnas.analyze --bench outputs/bench_sn --out results/e2` ; search replay (rnas/search.py).
  5. E5: eval each standalone ckpt with its arch key (search-val, 10 eps) vs supernet-inherited SR.
  6. Freeze `results/final_candidates.json`, short fixed fine-tunes from supernet, then `scripts/launch_test.sh`.

### 2026-10-05 20:40 — RESULT: offline loss vs closed-loop SR over training (standalone default arch)
Search-val, 10 eps x 40 tasks = 400 episodes per cell, benchmark seeds (jobs 18660168-72).

| ckpt | offline-val fm loss | SR h50 | SR h10 |
|---|---|---|---|
| 10k | 0.515 | 0.477 | 0.700 |
| 20k | 0.738 | 0.625 | 0.772 |
| 30k | 0.961 | 0.713 | 0.850 |

Inference-schedule grid at 30k (rows: denoising steps, cols: executed horizon):

| steps \ h | 5 | 10 | 25 | 50 |
|---|---|---|---|---|
| 2  | 0.815 | 0.818 | 0.848 | 0.685 |
| 4  | 0.813* | 0.853* | 0.829* | 0.715 |
| 10 | 0.815 | 0.850 | 0.828 | 0.713 |
(* 38-39/40 tasks at time of logging)

Findings: (1) offline val loss is anti-correlated with closed-loop SR across checkpoints (would pick the worst ckpt);
(2) horizon 50 (SmolVLA default execution) is clearly worst; (3) 2 Euler steps ~= 10 steps -> 5x fewer expert passes.

### 2026-10-05 22:30 — Supernet finished (18657517, 30k steps, 1 GPU, ~7.6 h)
Offline-val fm loss: anchor 0.495@10k -> 0.546@30k (mild rise; standalone rose to 0.961);
smallest subnet (v8-e4-f0.5-t16) 0.494@10k -> **0.467@30k** (< anchor: offline loss prefers the smallest net).
Auto-launched (`scripts/post_supernet.sh`): record_anchor 18666150 (anchor h10, 5 eps/task, ep-offset 100,
visited states) -> proxies (afterok); E2 benchmark bench_sn g0-g9 (18666152-62), 111 archs x 400 eps.
E5 evals auto-launched as standalone runs finish (`scripts/e5_watch.sh`).
- 23:30: all E2/E5/proxy jobs pending on priority (fair-share drop after today's ~60 GPU-h; no idle GPUs on the
  cluster). Time limits lowered to 8 h for backfill. Scheduler estimates first starts ~10:00 on 10-06.
- `rnas/replay.py` (RQ3 search replay) + `rnas/analyze.py` verified end-to-end on a synthetic table on Jubail.

### 2026-10-06 00:30 — E5 partial + anchor recording
- record_anchor 18666150 done (34 min): supernet anchor @h10, 5 eps/task, ep-offset 100: SR 0.795 (standalone
  default @h10 on bench seeds: 0.850). 724 MB visited states in `outputs/bench_sn/visited`.
- E5 standalone (30k steps each), search-val 400 eps:
  v12-e4-stretch-f1-t16-s4-h5 0.853 | v16-e8-top-f1-t16-s4-h5 0.840 | v12-e8-top-f0.75-t64-s2-h5 0.835 |
  v16-e12-top-f0.5-t16-s2-h25 0.828 | v8-e12-stretch-f0.75-t64-s2-h50 0.690 | default h50/h10 0.713/0.850.
  -> much smaller networks match the default once the execution horizon is short; h50 dominates failures.
- E2 bench g0-g4 running since 00:06-00:25 (backfill); g5-g9 est. 03:00-08:30; proxies est. 11:00.
- 01:30: bench g6 (18666158, cn015) hung after 3 rows (env workers idle, GPU 0%) -> added reset/step timeouts
  (600/300 s) with worker recreation + up to 3 retries per (arch, task) to `rnas/rollout.py`; g6 cancelled and
  resubmitted as 18668036 (resumes; finished rows are skipped). Other groups at ~2.2 h/group pace.
- 02:20: g9 on cn015 very slow (up to 249 s per arch-task vs ~20 s) -> cancelled, resubmitted excluding cn015 (g9r); g6r also excludes cn015.

### 2026-10-06 03:05 — E2 interim (17 complete archs, supernet weights)
- Split-half reliability (200 vs 200 eps): Spearman 0.900 / Kendall 0.787 -> 400-ep grid ranks reliably.
- cl_1ep (1 ep/task = 40 eps) Kendall 0.752 vs held-out SR; cl_2ep 0.742.
- SR range 0.21-0.82; top: v16-e16-f1-t64-s2-h5 0.818 (2 denoise steps), default s10-h10 0.800, s10-h50 0.693.
- **Weight-sharing gap suspicion:** e4 subnets 0.21-0.49 under supernet weights vs standalone
  v12-e4-stretch-f1-t16-s4-h5 0.853 (E5). Shallow experts look under-trained by the sandwich rule -> E5 rank check and
  post-search fine-tuning are essential; report honestly.

### 2026-10-06 04:05 — E2 interim with proxies (67 complete archs)
Split-half ceiling Kendall 0.814. Kendall tau vs held-out SR [95% CI], top-10 overlap:
sr_search 0.814 | cl_2ep 0.792 [.713,.858] 6/10 | kd_offline 0.755 [.679,.824] 8/10 | kd_onpolicy 0.754 8/10 |
cl_1ep 0.745 6/10 | act_l1_exec 0.646 4/10 | act_l1 0.491 | self_cons 0.430 | grip_err 0.154 0/10 |
**fm_loss 0.108 [-.096,.330] 2/10** | latency ~0 | expert params (bigger=better) 0.498.
-> standard NAS val loss fails; anchor-agreement (KD) proxies ~ 2-episode closed loop at zero rollouts;
   on-policy states give no gain over offline states for KD (honest negative).
Proxies job 18666151 done (111 archs); all 7 E5 evals done.

### 2026-10-06 05:00 — E5 (partial) weight-sharing gap; E5b launched
supernet-inherited vs standalone-30k SR (same seeds): default h50 0.693/0.713 | v16-e12-top 0.748/0.828 |
v8-e12 0.458/0.690 | v12-e4 0.450/0.853 | v24-e4-top 0.085/0.640 | v24-e16-top 0.695/0.853 (2 pending).
Gap grows as the expert gets shallower: supernet strongly under-rates e4 subnets; standalone SR is flat (0.64-0.85).
E5b: fine-tune each E5 arch from supernet 5k steps (lr 5e-5, 1/6 of standalone cost) + eval
(jobs 18671599-614, dependent evals) to test whether cheap fine-tuning restores standalone SR/ranking.

### 2026-10-06 06:20 — E2 FINAL (111 archs x 400 eps = 44,400 episodes), E5 FINAL
Files: `results/e2/proxy_correlations.md`, `results/e2/bench_table.csv`, `results/e2/factors.md`, `results/e5/e5.md`.
- Split-half ceiling Kendall 0.840. fm_loss **0.165 [0.005,0.330], top-10 0/10**; kd_offline 0.756 (8/10);
  kd_onpolicy 0.738 (8/10); cl_1ep 0.775; cl_2ep 0.819; act_l1_exec 0.667; latency/params uninformative or inverse.
- RQ4 (supernet weights): n_exp dominates (dR2 .40, likely weight-sharing artifact), steps 2>4>10, vtok 16~64,
  stretch > top, horizon 10-25 best.
- E5: supernet-inherited vs standalone Kendall 0.255 (n=8), mean gap +0.212 -> weight sharing distorts the
  deployment-relevant ranking (shallow experts most). Motivates fine-tune-after-search (E5b running).
- Hangs: g5/g6/g9 stalls on cn014/cn015 -> restarted with env timeouts (rows resumed, no data lost).

### 2026-10-06 06:45 — E3 search replay FINAL (`results/e3/replay.md`, 300 reps, held-out-half scoring)
Regret (held-out SR of best eligible arch minus chosen), latency cap <= median ms/step:
| method | B=400 | 800 | 1600 | 3200 |
|---|---|---|---|---|
| proxy_sh[kd_onpolicy] n8 | 0.024 | 0.009 | 0.001 | 0.000 |
| proxy_sh[act_l1_exec] n8 | 0.035 | 0.013 | 0.001 | 0.000 |
| sh n16 (no proxy) | 0.084 | 0.070 | 0.058 | 0.057 |
| proxy_sh[fm_loss] n8 | 0.112 | 0.097 | 0.078 | 0.075 |
| random_full | 0.178 | 0.123 | 0.086 | 0.052 |
| proxy_top1[fm_loss] (0 eps) | 0.210 | - | - | - |
No cap, B=400: proxy_sh[kd] 0.053-0.058, proxy_sh[act_l1_exec] 0.051, sh 0.086-0.098, fm-guided 0.126-0.130,
random 0.181; proxy_top1[kd_onpolicy] 0.045 (single deterministic pick).
- 06:18: ev_ft v16-e12 (18671601) died on cn270: env worker EOFError at first reset x3 (EGL on condo H100 NVL node). Resubmitted excluding cn014/cn015/cn270. Exclude these nodes for all sim jobs.

### 2026-10-06 06:25 — Draft final selection (NOT frozen; `outputs/final_candidates.json` on HPC)
Deterministic picks (budget 1600, full search-val table, seed 0), cap = median 12.05 ms/step:
ours proxySH[kd] -> v16-e16-stretch-f1-t64-s2-h10 (SR .845) | val-loss top1 -> v16-e8-stretch-f1-t16-s4-h50 (.628) |
random -> default s10-h50 (.693). No cap: ours -> v16-e16-s10-h5 (.793), val-loss -> v16-e8-f0.75-t16-s4-h5 (.748),
random -> v16-e16-s2-h25 (.808). Default-net schedule-only tuning -> s2-h10.
Issue: supernet bias (E5) + KD-to-anchor bias -> search returns the anchor network with a cheaper schedule; cannot
surface small nets that are competitive when trained properly (standalone v12-e4 .853). Freeze deferred until E5b
(fine-tune 5k from supernet) shows whether cheap fine-tuning restores standalone-level SR -> then a
shortlist -> ft5k -> closed-loop race stage.

### 2026-10-06 07:10 — E5b result; FINAL CANDIDATES FROZEN (`results/final_candidates.json`)
ft5k (5k steps from supernet) vs supernet vs standalone-30k SR: default h50 .723/.693/.713 | v8-e12 .748/.458/.690 |
v12-e4 .680/.450/.853 | v16-e8-top .752/.665/.840 | v12-e8-top .800/.762/.835 | v24-e4-top .352/.085/.640 |
v24-e16-top .767/.695/.853 (v16-e12-top pending). ft5k closes ~half the weight-sharing gap; shallow experts still
lag standalone; rank agreement weak (tau ~.29, n=7). Key limitation of supernet VLA-NAS -> reported as such.
Frozen 6 test candidates (see file) before any test access. nas_valloss net (v16-e8-stretch-f1-t16) needs ft5k first.
- 09:20: 3 test halves stalled (>35 min no rows; cn016 x2, cn012) -> cancelled, resubmitted excluding cn012/cn016 (18672285-87), resuming finished rows. Same frozen candidates/ckpts.

### 2026-10-06 10:50 — E4 FINAL (official LIBERO test, 2000 eps each; `results/e4.md`)
published 0.674 [.653,.694] | tuned schedule (standalone) 0.810 | **ours 0.821 [.803,.837]** | random 0.662 |
val-loss NAS 0.701 | standalone small v12-e4-t16-s4-h5 0.823 (85 ms/call). Ours: +14.7 pts, 3.1x lower call latency.
E5b final: ft5k v16-e12-top-f0.5-t16-s2-h25 = 0.835 (standalone 0.828, supernet 0.748).
Test-set accessed only for the 6 frozen candidates (+3 resumed halves after node stalls).

## v2 (plan: `docs/PLAN_v2.md`; user decisions 2026-10-06: second family = pi0; no extra seed tests of v1)

### 2026-10-06 11:30 — S1 + S0 launched
- S1: 16 standalone nets pre-registered (`results/s1_archs.json`): 8 E5 nets + 8 new (2 per expert depth, seed 2),
  all evaluated at fixed schedule s2-h10 on search-val (400 eps). Jobs: 8 evals of existing ckpts + 8 trainings
  (30k steps) with dependent evals (18674351-74). Time limits tightened (train 6 h, eval 2 h) for backfill.
- S0: (a) gradient conflict between depth subnets on shared expert layers 0-3 and readout (18674389);
  (b) readout-only probe: fine-tune only expert final norm + action_out_proj for 5k steps from the supernet for
  v12-e4-stretch-f1-t16-s4-h5 and v24-e4-top-f0.75-t64-s10-h25, eval at the same keys as E5/E5b (18674390-93).
  If readout-only ~ full ft5k -> shared-readout interference is the main cause -> V1 (per-depth readouts) first.
- S4 prep (pi0): `lerobot/pi0_base` = PI base model; HF card and openpi do not list the pretraining mixture. The pi0
  paper describes it as PI cross-embodiment data + an OXE subset (OXE has no LIBERO). Plan: use pi0_base, document
  this, and add a leakage-free control initialised from PaliGemma only (no robot pretraining) for the main comparison.
- 11:40: per user request (run step by step, don't flood the queue): cancelled all pending S1 jobs (26). Kept S0
  (5 jobs) and the running S1 eval of v16-e12 at s2-h10; v16-e16 s2-h10 eval already finished. Remaining S1
  (6 existing-net evals, 8 trainings + evals) will be resubmitted in small stages after S0 is analysed.
  Working rule from now on: at most ~4 GPU jobs in the queue at a time.
- 11:53: per user, keep only ONE RoboticsNAS job in the queue: cancelled the 5 S0 jobs; only s1ev v16-e12 (18674352, running) remains. Next jobs submitted one at a time (S0 grad-conflict first).
- 12:23: S1 standalone @s2-h10 (search-val 400 eps): v16-e16 (default) 0.843, v16-e12-top-f0.5-t16 0.875. Submitted s0_gradconf (only job).
- 12:54: S0 gradient conflict (v1 supernet; 16 batches; arch v16-e{4,8,12,16}-stretch-f1-t64; `outputs/s0/grad_conflict.json`):
  cosine on shared expert layers 0-3: e4-e8 .28, e4-e12 .21, **e4-e16 .17**, e8-e12 .28, e8-e16 .27, e12-e16 .46;
  readout: .52-.69. Shared-layer grad norm e4 .27 > e8 .19 > e12 .17 > e16 .13.
  -> conflict is mainly in the shared trunk (early layers must be "final" for e4 and "intermediate" for e16), not
  the readout; e4 remains far from its optimum. Suggests per-depth adaptation of shared layers / conflict-aware
  updates (V3) over per-depth readout only (V1). Readout probe still run to confirm. Submitted s0_ro v12-e4 (only job).
- 13:46: readout probe 18677115 failed at start (param-name filter matched nothing -> empty optimizer; no GPU time used). Fixed filter + assert; resubmitted as 18677795 (only job).
- 14:36: readout probe v12-e4 trained (0.024M params, 5k steps): offline val fm 0.478 (full ft5k 0.472). Submitted its closed-loop eval (only job).
- 15:06: **S0 readout probe v12-e4-stretch-f1-t16-s4-h5: SR 0.415** (supernet 0.450, full ft5k 0.680, standalone 0.853).
  Training only the readout (0.024M params) does not help -> the shared readout is NOT the cause of the weight-sharing
  gap; consistent with the gradient-conflict result (conflict in the shared trunk). Decision: skip the second readout
  probe (v24-e4); M2 will target trunk sharing (per-depth adaptation of shared layers / conflict-aware updates), not
  per-depth readouts. Next: S1 existing-net evals one by one; submitted v8-e12 @s2-h10 (only job).
