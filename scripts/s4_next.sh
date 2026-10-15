#!/bin/bash
# S4 driver (teacher imitation). Keeps <= 2 S4 jobs queued: teacher recording -> trainings -> evals.
cd /scratch/ll5582/RoboticsNAS
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/s4/eval/*.jsonl")):
    r = [json.loads(l) for l in open(f)]
    n = sum(len(x["success"]) for x in r)
    if n: print(f"{f.split('/')[-1][:-6]:40s} tasks={len(r):2d} SR={sum(sum(x['success']) for x in r)/n:.3f}")
PY
Q=$(squeue -h -u ll5582 -o "%j" | grep -E "^s4" || true)
echo "queued: $(echo $Q)"
TODO=$(python3 - "$Q" <<'PY'
import json, os, sys
q = set(sys.argv[1].split())
p = json.load(open("results/s4_plan.json"))
rec_done = len([f for f in os.listdir("outputs/s4/teacher_visited") if f.endswith(".npz")]) >= 40 if os.path.isdir("outputs/s4/teacher_visited") else False
out = []
if not rec_done:
    if "s4rec" not in q: out.append("rec - -")
else:
    for k in p["archs"]:
        ck = f"outputs/s4/{k}/step10k.pt"; ev = f"outputs/s4/eval/{k}.jsonl"
        if not os.path.exists(ck):
            if f"s4tr_{k}" not in q: out.append(f"train {k} {ck}")
        elif not (os.path.exists(ev) and sum(1 for _ in open(ev)) >= 40):
            if f"s4ev_{k}" not in q: out.append(f"eval {k} {ck}")
for x in sorted(out, key=lambda x: x.startswith("train")): print(x)  # evals first
PY
)
SLOTS=$((2 - $(echo "$Q" | grep -c . )))
echo "$TODO" | while read M K CK && [ -n "$M" ] && [ $SLOTS -gt 0 ]; do
  case $M in
    rec)   sbatch -J s4rec --time=03:00:00 scripts/gpu.sbatch rnas.rollout --ckpt outputs/fixed_default/final.pt \
             --archs v16-e16-stretch-f1-t64-s2-h10 --mode search --n-eps 20 --ep-offset 500 \
             --record-dir outputs/s4/teacher_visited --out outputs/s4/teacher_rollouts.jsonl ;;
    train) sbatch -J s4tr_$K --gres=gpu:1 --constraint="80g|h100|h200" --cpus-per-task=16 --mem=160G --time=08:00:00 \
             scripts/gpu.sbatch rnas.train --mode fixed --arch $K --out outputs/s4/$K --steps 30000 --stop-at 10000 \
             --batch 64 --workers 12 --teacher outputs/fixed_default/final.pt --visited outputs/s4/teacher_visited ;;
    eval)  sbatch -J s4ev_$K --gres=gpu:a100:1 --time=02:00:00 scripts/gpu.sbatch rnas.rollout --ckpt $CK --archs $K \
             --mode search --n-eps 10 --out outputs/s4/eval/$K.jsonl ;;
  esac
  SLOTS=$((SLOTS - 1))
  [ "$M" = rec ] && break
done
