#!/bin/bash
# One-job-at-a-time driver for S1. Prints SRs; if no RoboticsNAS job is queued, submits exactly one next job:
# existing-net evals (A100) -> new-net training (any GPU) -> its eval (A100) -> next new net ...
cd /scratch/ll5582/RoboticsNAS
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/s1/*.jsonl")):
    r = [json.loads(l) for l in open(f)]
    n = sum(len(x["success"]) for x in r)
    if n: print(f"{f.split('/')[-1][:-6]:40s} tasks={len(r):2d} SR={sum(sum(x['success']) for x in r)/n:.3f}")
PY
if squeue -h -u ll5582 -o "%j" | grep -qE "^(s0|s1|bench|test|ft_|ev_)"; then echo "RNAS job in queue -> nothing submitted"; exit 0; fi
NEXT=$(python3 - <<'PY'
import json, os
d = json.load(open("results/s1_archs.json")); e5 = json.load(open("results/e5_archs.json"))["archs"]
def done(k): return os.path.exists(f"outputs/s1/{k}.jsonl") and sum(1 for _ in open(f"outputs/s1/{k}.jsonl")) >= 40
for k5, k in zip(e5, d["existing_e5_nets"]):
    if not done(k):
        ck = "outputs/fixed_default/final.pt" if k5 == e5[0] else f"outputs/standalone/{k5}/final.pt"
        print("eval", k, ck); raise SystemExit
for k in d["new_nets"]:
    ck = f"outputs/standalone/{k}/final.pt"
    if not os.path.exists(ck): print("train", k, ck); raise SystemExit
    if not done(k): print("eval", k, ck); raise SystemExit
print("done - -")
PY
)
read M K CK <<< "$NEXT"
case $M in
  eval)  sbatch -J s1ev_$K --gres=gpu:a100:1 --time=02:00:00 scripts/gpu.sbatch rnas.rollout --ckpt $CK --archs $K \
           --mode search --n-eps 10 --out outputs/s1/$K.jsonl ;;
  train) sbatch -J s1_$K --gres=gpu:1 --constraint="a100|h100|h200" --cpus-per-task=16 --mem=110G --time=06:00:00 \
           scripts/gpu.sbatch rnas.train --mode fixed --arch $K --out outputs/standalone/$K --steps 30000 --batch 64 --workers 12 ;;
  done)  echo "S1 complete" ;;
esac
echo "next: $M $K"
