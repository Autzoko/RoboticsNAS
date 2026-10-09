#!/bin/bash
# One-job S2 driver (second concurrent slot). Sequence: train V1 -> eval V0 -> eval V1 -> train V2 -> eval V2 ->
# train V3 -> eval V3. Training on any GPU; evals (simulation) on A100 only. Evals use the 16 S1 keys (s2-h10).
cd /scratch/ll5582/RoboticsNAS
mkdir -p outputs/s2
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/s2/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    by = {}
    for r in rows:
        by.setdefault(r["arch"], []).extend(r["success"])
    done = sum(1 for v in by.values() if len(v) >= 400)
    print(f"{f}: {len(rows)} rows, archs complete {done}/16")
PY
if squeue -h -u ll5582 -o "%j" | grep -qE "^s2_"; then echo "S2 job in queue -> nothing submitted"; exit 0; fi
TRAIN=(  "V1:--depth-gain"  "V2:--sampler depth --kd same_depth"  "V3:--pcgrad" )
done_eval() { [ -f outputs/s2/$1.jsonl ] && [ $(wc -l < outputs/s2/$1.jsonl) -ge 640 ]; }
ckpt() { [ $1 = V0 ] && echo outputs/supernet_v1/final.pt || echo outputs/supernet_m2_$1/final.pt; }
submit_train() { V=$1; shift
  sbatch -J s2_train_$V --gres=gpu:1 --constraint="a100|h100|h200" --cpus-per-task=16 --mem=110G --time=10:00:00 \
    scripts/gpu.sbatch rnas.train --mode supernet --out outputs/supernet_m2_$V --steps 30000 --batch 64 --workers 12 \
    --save-every 1000 "$@"; echo "next: train $V"; }
submit_eval() { V=$1
  sbatch -J s2_eval_$V --gres=gpu:a100:1 --time=10:00:00 scripts/gpu.sbatch rnas.rollout --ckpt $(ckpt $V) \
    --archs results/s1_all16.json --mode search --n-eps 10 --out outputs/s2/$V.jsonl; echo "next: eval $V"; }
# cost table (latency + memory) for all archs, once, before the next S2 step (user request 2026-10-07)
if [ ! -s outputs/cost/a100_v2.jsonl ]; then mkdir -p outputs/cost
  sbatch -J s2_cost --gres=gpu:a100:1 --time=01:00:00 scripts/gpu.sbatch rnas.latency --archs results/cost_archs.json \
    --out outputs/cost/a100_v2.jsonl; echo "next: cost table"; exit 0; fi
# order
[ -f $(ckpt V1) ] || { submit_train V1 --depth-gain; exit 0; }
done_eval V0 || { submit_eval V0; exit 0; }
done_eval V1 || { submit_eval V1; exit 0; }
# V4 (bridge-conditioned K/V adapters) — prioritised after V1 failed and the gap analysis implicated the bridge
[ -f $(ckpt V4) ] || { submit_train V4 --kv-adapter; exit 0; }
done_eval V4 || { submit_eval V4; exit 0; }
# M1 groundwork on the v1 supernet (E2 table): zero-cost baselines, then bootstrapped references
B=outputs/bench_sn
if [ ! -s $B/zerocost.jsonl ]; then
  sbatch -J s2_zerocost --gres=gpu:1 --constraint="a100|h100|h200" --time=02:00:00 scripts/gpu.sbatch rnas.zerocost \
    --ckpt outputs/supernet_v1/final.pt --archs results/bench_archs.json --out $B/zerocost.jsonl; echo "next: zerocost"; exit 0; fi
for REF in $(python3 -c "import json;print(' '.join(json.load(open('results/m1_refs.json'))['refs']))"); do
  if [ ! -s $B/refs/$REF/rollouts.jsonl ] || [ $(wc -l < $B/refs/$REF/rollouts.jsonl) -lt 40 ]; then
    sbatch -J s2_refrec_$REF --gres=gpu:a100:1 --time=04:00:00 scripts/gpu.sbatch rnas.rollout --ckpt outputs/supernet_v1/final.pt \
      --archs $REF --mode search --n-eps 5 --ep-offset 100 --record-dir $B/refs/$REF/visited --out $B/refs/$REF/rollouts.jsonl
    echo "next: record ref $REF"; exit 0; fi
  if ! grep -q "D_ref\[$REF\]" $B/proxies.jsonl 2>/dev/null; then
    sbatch -J s2_refD_$REF --gres=gpu:1 --constraint="a100|h100|h200" --time=04:00:00 scripts/gpu.sbatch rnas.proxies \
      --ckpt outputs/supernet_v1/final.pt --archs results/bench_archs.json --ref-arch $REF --only-onpolicy \
      --record-dir $B/refs/$REF/visited --out $B/proxies.jsonl
    echo "next: D_ref $REF"; exit 0; fi
done
if [ ! -f $(ckpt V3) ]; then  # PCGrad keeps 4 subnet graphs + 4 flat grads: OOM on A100-40GB -> 80GB+ only
  sbatch -J s2_train_V3 --gres=gpu:1 --constraint="80g|h100|h200" --cpus-per-task=16 --mem=110G --time=12:00:00 \
    scripts/gpu.sbatch rnas.train --mode supernet --out outputs/supernet_m2_V3 --steps 30000 --batch 64 --workers 12 \
    --save-every 1000 --pcgrad; echo "next: train V3"; exit 0; fi
done_eval V3 || { submit_eval V3; exit 0; }
[ -f $(ckpt V2) ] || { submit_train V2 --sampler depth --kd same_depth; exit 0; }
done_eval V2 || { submit_eval V2; exit 0; }
echo "S2 complete"
