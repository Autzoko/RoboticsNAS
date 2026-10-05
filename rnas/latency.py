"""Per-architecture inference latency (batch 1, bf16, CUDA-synchronized), measured with supernet weights."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .build import build_elastic, make_processors, setup_env_vars
from .space import Arch


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--stats", default="configs/train_stats.json")
    args = ap.parse_args()
    setup_env_vars()
    keys = json.loads(Path(args.archs).read_text()) if args.archs.endswith(".json") else args.archs.split(",")
    model = build_elastic("cuda", sort_neurons=False).eval()
    pre, _ = make_processors(model, args.stats)
    b = pre({
        "observation.images.image": torch.rand(1, 3, 256, 256),
        "observation.images.image2": torch.rand(1, 3, 256, 256),
        "observation.state": torch.zeros(1, 8),
        "task": ["put the black bowl on the plate"],
    })
    gpu = torch.cuda.get_device_name()
    with open(args.out, "a") as f:
        for k in keys:
            a = Arch.from_key(k)
            ts = []
            for i in range(args.reps + 5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.sample_actions(b, a)
                torch.cuda.synchronize()
                if i >= 5:
                    ts.append(time.perf_counter() - t0)
            ts.sort()
            ms = 1000 * ts[len(ts) // 2]
            exp_params = sum(
                l.self_attn.q_proj.weight.numel() + l.self_attn.k_proj.weight.numel()
                + l.self_attn.v_proj.weight.numel() + l.self_attn.o_proj.weight.numel()
                + 3 * l.mlp.gate_proj.weight.numel() * a.ffn
                for l in list(model.exp_layers)[: a.n_exp]
            )
            r = {"arch": k, "gpu": gpu, "call_ms": ms, "ms_per_step": ms / a.horizon,
                 "expert_params_M": exp_params / 1e6}
            f.write(json.dumps(r) + "\n")
            print(r, flush=True)


if __name__ == "__main__":
    main()
