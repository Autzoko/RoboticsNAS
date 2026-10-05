#!/bin/bash
# E0: split/stats, numerical equivalence, env + seed (common-random-number) check. Run via gpu.sbatch-like env.
set -eo pipefail
python -m rnas.data --out-dir configs
python -m rnas.check_equiv
python -m rnas.rollout --ckpt none --archs v16-e16-stretch-f1-t64-s10-h50,v8-e4-stretch-f0.5-t16-s2-h5 \
  --suites libero_spatial --tasks 0 --n-eps 2 --out outputs/e0/rollout_check.jsonl
python - <<'PY'
import json
rows=[json.loads(l) for l in open("outputs/e0/rollout_check.jsonl")]
for r in rows: print(r["arch"], r["init_hash"], r["success"], r["length"], round(r["wall_s"]), r["policy_calls"])
assert rows[0]["init_hash"]==rows[1]["init_hash"], "CRN violated: same seeds gave different init states"
assert rows[0]["init_hash"][0]!=rows[0]["init_hash"][1], "different seeds gave identical init states"
print("CRN check OK")
PY
