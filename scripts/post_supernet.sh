#!/bin/bash
# After supernet training: visited-state recording -> proxies (dependent), E2 benchmark, latency copy.
set -e
cd /scratch/ll5582/RoboticsNAS
CK=outputs/supernet_v1/final.pt
B=outputs/bench_sn
mkdir -p $B
cp outputs/latency/a100_bench.jsonl $B/latency.jsonl
J=$(sbatch --parsable -J record_anchor --time=1-00:00:00 scripts/gpu.sbatch rnas.rollout --ckpt $CK \
    --archs v16-e16-stretch-f1-t64-s10-h10 --mode search --n-eps 5 --ep-offset 100 \
    --record-dir $B/visited --out $B/anchor_record_rollouts.jsonl)
sbatch -J proxies --dependency=afterok:$J --time=1-00:00:00 scripts/gpu.sbatch rnas.proxies --ckpt $CK \
    --archs results/bench_archs.json --record-dir $B/visited --out $B/proxies.jsonl
bash scripts/launch_bench.sh $CK sn 10
