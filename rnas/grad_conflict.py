"""S0 diagnostic: gradient conflict between depth subnets on shared expert layers of a supernet checkpoint."""

import argparse
import json

import torch

from .build import build_elastic, load_trainable, make_processors, setup_env_vars
from .data import make_dataset
from .space import Arch
from .train import flow_losses

DEPTHS = (4, 8, 12, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup_env_vars()
    model = build_elastic("cuda", sort_neurons=False)
    load_trainable(model, a.ckpt)
    model.eval()
    pre, _ = make_processors(model, "configs/train_stats.json")
    split = json.load(open("configs/split.json"))
    dl = torch.utils.data.DataLoader(make_dataset(split["train"], chunk_size=model.chunk), batch_size=32,
                                     shuffle=True, num_workers=8, generator=torch.Generator().manual_seed(0))
    groups = {
        "shared_layers0-3": [p for n, p in model.named_parameters() if p.requires_grad and any(
            f"lm_expert.layers.{i}." in n for i in range(4))],
        "readout": [p for n, p in model.named_parameters() if p.requires_grad and (
            "lm_expert.norm." in n or "action_out_proj" in n)],
    }
    archs = {d: Arch(16, d, "stretch", 1.0, 64) for d in DEPTHS}
    acc = {g: {d: [] for d in DEPTHS} for g in groups}
    for bi, batch in enumerate(dl):
        if bi == a.batches:
            break
        batch = pre(batch)
        g0 = torch.Generator(device="cuda").manual_seed(bi)
        act = model.policy.prepare_action(batch)
        nz = torch.randn(act.shape, device="cuda", generator=g0)
        tt = torch.rand(act.shape[0], device="cuda", generator=g0) * 0.999 + 0.001
        for d, arch in archs.items():
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fm = flow_losses(model, batch, [arch], 0.0, noise=nz, t=tt)[0][1]
            fm.backward()
            for g, ps in groups.items():
                acc[g][d].append(torch.cat([p.grad.flatten().float() for p in ps if p.grad is not None]).cpu())
    res = {}
    for g in groups:
        mean = {d: torch.stack(v).mean(0) for d, v in acc[g].items()}
        res[g] = {f"e{i}-e{j}": float(torch.nn.functional.cosine_similarity(mean[i], mean[j], dim=0))
                  for i in DEPTHS for j in DEPTHS if i < j}
        res[g]["norms"] = {f"e{d}": float(mean[d].norm()) for d in DEPTHS}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
