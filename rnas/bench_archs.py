"""E2 benchmark architecture set (fixed before any closed-loop result is seen).

- anchor (SmolVLA default)
- inference-knob factorial on the anchor network: 3 steps x 4 horizons
- 100 architectures sampled uniformly from the full valid space (seed 0), de-duplicated
"""

import json
import random

from .space import DEFAULT, HORIZON, STEPS, Arch, all_archs

space = all_archs()
keys = [DEFAULT.key()]
keys += [Arch(*DEFAULT.net, steps=s, horizon=h).key() for s in STEPS for h in HORIZON]
rng = random.Random(0)
for a in rng.sample(space, 140):
    if len(keys) >= 112:
        break
    if a.key() not in keys:
        keys.append(a.key())
keys = list(dict.fromkeys(keys))
print(len(space), "valid archs;", len(keys), "benchmark archs")
json.dump(keys, open("results/bench_archs.json", "w"), indent=1)
