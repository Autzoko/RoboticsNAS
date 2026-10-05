#!/bin/bash
# launch search-val eval for each finished standalone run (once)
cd /scratch/ll5582/RoboticsNAS
for d in outputs/standalone/*; do
  k=$(basename $d)
  if [ -f $d/final.pt ] && [ ! -f outputs/e5/$k.launched ]; then
    sbatch -J e5_$k --time=1-00:00:00 scripts/gpu.sbatch rnas.rollout --ckpt $d/final.pt --archs $k --mode search --n-eps 10 --out outputs/e5/$k.jsonl && touch outputs/e5/$k.launched
  fi
done
ls outputs/e5/*.launched 2>/dev/null | wc -l
