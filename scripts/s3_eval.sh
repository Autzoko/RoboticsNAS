#!/bin/bash
# S3: evaluate step10k standalone snapshots @s2-h10 on search-val (paired with S1). Usage: s3_eval.sh <shard> <nshards>
set -eo pipefail
python - "$1" "$2" <<'PY' | while read K CK; do
import json, sys
c = json.load(open("results/s3_plan.json"))["ckpts"]; s, n = int(sys.argv[1]), int(sys.argv[2])
for i, (k, ck) in enumerate(c.items()):
    if i % n == s: print(k, ck)
PY
  out=outputs/s3/$K.jsonl
  if [ -f $out ] && [ $(wc -l < $out) -ge 40 ]; then echo "skip $K"; continue; fi
  echo "== $K $CK $(date -Is)"
  python -m rnas.rollout --ckpt $CK --archs $K --mode search --n-eps 10 --out $out
done
