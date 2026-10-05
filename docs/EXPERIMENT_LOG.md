# Experiment log

Chronological record of jobs, settings, results and decisions. Times are Jubail local (GST, UTC+4) unless noted.
Every final-test (official LIBERO init states) run must be listed in the "Test-set access" table.

## Test-set access

| date | policy / arch | ckpt | job | note |
|---|---|---|---|---|

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
