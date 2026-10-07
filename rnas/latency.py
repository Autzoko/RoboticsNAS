"""Per-architecture inference cost (batch 1, bf16, CUDA-synchronized), measured with supernet weights.

Reported per arch:
  call_ms        median latency of one policy call (prefix + `steps` denoising passes)
  ms_per_step    call_ms / executed horizon  (compute per control step)
  params_M       parameters actually used by the subnet (vision+connector, embeddings, VLM layers it reads,
                 sliced expert, projections); expert_params_M / vlm_params_M breakdown
  weights_MB     bf16 size of those parameters (deployment weight memory)
  act_peak_MB    measured peak extra GPU memory during one call above the resident model (activations, KV cache)
  deploy_MB      weights_MB + act_peak_MB  (memory to deploy this subnet alone)
The elastic model keeps all candidates' weights resident, so raw peak memory would overstate a subnet; hence the
analytic weight count + measured activation peak.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .build import build_elastic, make_processors, setup_env_vars
from .space import Arch

BYTES = 2  # bf16 deployment


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


def active_params(model, a: Arch) -> dict:
    vlm = model.vwe.get_vlm_model()
    vision = _n(vlm.vision_model) + _n(vlm.connector)
    embed = vlm.text_model.get_input_embeddings().weight.numel()
    vlm_layers = sum(_n(l) for l in list(model.vlm_layers)[: a.vlm_layers_needed()])
    exp = 0
    for l in list(model.exp_layers)[: a.n_exp]:
        at = l.self_attn
        exp += sum(p.numel() for p in at.parameters())
        exp += sum(p.numel() for p in l.input_layernorm.parameters()) + sum(
            p.numel() for p in l.post_attention_layernorm.parameters())
        exp += int(round(3 * l.mlp.gate_proj.weight.numel() * a.ffn))
    proj = sum(_n(m) for m in (model.m.state_proj, model.m.action_in_proj, model.m.action_out_proj,
                               model.m.action_time_mlp_in, model.m.action_time_mlp_out))
    proj += _n(model.vwe.lm_expert.norm)
    return {"vision_params_M": vision / 1e6, "vlm_params_M": (embed + vlm_layers) / 1e6,
            "expert_params_M": (exp + proj) / 1e6, "params_M": (vision + embed + vlm_layers + exp + proj) / 1e6}


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
            # activation peak for one call, above the resident model
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.sample_actions(b, a)
            torch.cuda.synchronize()
            act_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
            pc = active_params(model, a)
            w_mb = pc["params_M"] * 1e6 * BYTES / 2**20
            r = {"arch": k, "gpu": gpu, "call_ms": ms, "ms_per_step": ms / a.horizon, **pc,
                 "weights_MB": w_mb, "act_peak_MB": act_mb, "deploy_MB": w_mb + act_mb}
            f.write(json.dumps(r) + "\n")
            print(r, flush=True)


if __name__ == "__main__":
    main()
