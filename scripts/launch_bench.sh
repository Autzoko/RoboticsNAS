#!/bin/bash
# E2: closed-loop search-val benchmark of results/bench_archs.json with a given checkpoint.
# Usage: bash scripts/launch_bench.sh <ckpt> <tag> <n_groups> [n_eps]
set -e
CKPT=$1; TAG=$2; G=$3; NEPS=${4:-10}
cd /scratch/ll5582/RoboticsNAS
mkdir -p outputs/bench_$TAG/groups
python3 - "$G" "$TAG" <<'PY'
import json, sys
g, tag = int(sys.argv[1]), sys.argv[2]
keys = json.load(open("results/bench_archs.json"))
for i in range(g):
    json.dump(keys[i::g], open(f"outputs/bench_{tag}/groups/g{i}.json", "w"))
PY
for i in $(seq 0 $((G-1))); do
  sbatch -J bench_${TAG}_g$i --time=1-00:00:00 scripts/gpu.sbatch rnas.rollout --ckpt $CKPT \
    --archs outputs/bench_$TAG/groups/g$i.json --mode search --n-eps $NEPS \
    --out outputs/bench_$TAG/rollouts_g$i.jsonl
done
