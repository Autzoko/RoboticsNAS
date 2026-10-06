#!/bin/bash
# E4 final test on official LIBERO init states. Only for candidates frozen in results/final_candidates.json.
# Usage: bash scripts/launch_test.sh <name> <ckpt> <arch_key>
set -e
NAME=$1; CKPT=$2; ARCH=$3
cd /scratch/ll5582/RoboticsNAS
python3 - "$NAME" "$ARCH" <<'PY'
import json, sys
c = json.load(open("results/final_candidates.json"))
assert any(x["name"] == sys.argv[1] and x["arch"] == sys.argv[2] for x in c["candidates"]), "not a frozen candidate"
PY
mkdir -p outputs/test/$NAME
for OFF in 0 25; do
  sbatch -J test_${NAME}_$OFF ${DEP:+--dependency=$DEP} --gres=gpu:1 --constraint="a100|h100|h200" --time=10:00:00 --export=ALL,RNAS_ALLOW_TEST=1 scripts/gpu.sbatch rnas.rollout \
    --ckpt $CKPT --archs $ARCH --mode test --n-eps 25 --ep-offset $OFF --out outputs/test/$NAME/rollouts_$OFF.jsonl
done
echo "$(date -Is) $NAME $ARCH $CKPT" >> outputs/test/ACCESS_LOG.txt
