"""Zero-cost NAS proxies (baselines) per architecture, on supernet weights.

  zc_gradnorm  ||grad L_FM|| over the subnet's active trainable params          (Abdelfattah et al. 2021)
  zc_snip      sum |theta * grad_theta L_FM| over the same params              (Lee et al. 2019 / Abdelfattah 2021)
  zc_naswot    log|det K_H|, K_H = Hamming-kernel of binary codes from the sign of expert MLP gate pre-activations
               (one code per sample = per-channel sign of the token-mean pre-activation; Mellor et al. 2021)
Higher is better for all three.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .build import build_elastic, load_trainable, make_processors, setup_env_vars
from .data import make_dataset
from .space import Arch
from .train import flow_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--archs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    setup_env_vars()
    keys = json.loads(Path(a.archs).read_text()) if a.archs.endswith(".json") else a.archs.split(",")
    model = build_elastic("cuda", sort_neurons=False)
    load_trainable(model, a.ckpt)
    model.eval()
    pre, _ = make_processors(model, "configs/train_stats.json")
    split = json.load(open("configs/split.json"))
    dl = torch.utils.data.DataLoader(make_dataset(split["train"], chunk_size=model.chunk), batch_size=a.bs,
                                     shuffle=True, num_workers=8, generator=torch.Generator().manual_seed(7))
    batches = []
    for i, b in enumerate(dl):
        if i == a.batches:
            break
        batches.append(pre(b))
    expert_params = {n: p for n, p in model.named_parameters() if p.requires_grad and "lm_expert.layers." in n}

    def active(name, arch):
        li = int(name.split("lm_expert.layers.")[1].split(".")[0])
        return li < arch.n_exp

    with open(a.out, "a") as f:
        for k in keys:
            arch = Arch.from_key(k)
            gn, snip = 0.0, 0.0
            codes = []
            hooks = []
            kgate = int(round(model.exp_layers[0].mlp.gate_proj.out_features * arch.ffn))
            for j in range(arch.n_exp):
                layer = model.exp_layers[j]

                def hook(m, i, o, w=layer.mlp.gate_proj.weight):  # sliced gate pre-activation (elastic FFN)
                    with torch.no_grad():
                        pre_act = torch.nn.functional.linear(o.float(), w[:kgate].float())
                    codes.append((pre_act.mean(1) > 0).flatten(1).cpu())

                hooks.append(layer.post_attention_layernorm.register_forward_hook(hook))
            for bi, b in enumerate(batches):
                g = torch.Generator(device="cuda").manual_seed(bi)
                act = model.policy.prepare_action(b)
                nz = torch.randn(act.shape, device="cuda", generator=g)
                tt = torch.rand(act.shape[0], device="cuda", generator=g) * 0.999 + 0.001
                model.zero_grad(set_to_none=True)
                codes.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fm = flow_losses(model, b, [arch], 0.0, noise=nz, t=tt)[0][1]
                fm.backward()
                for n, p in expert_params.items():
                    if p.grad is None or not active(n, arch):
                        continue
                    gn += float(p.grad.float().pow(2).sum())
                    snip += float((p.detach().float() * p.grad.float()).abs().sum())
            # NASWOT on the last batch's codes (one forward per denoise/velocity call -> use all collected)
            C = torch.cat(codes, dim=1).float() if codes else torch.zeros(1, 1)
            K = C @ C.T + (1 - C) @ (1 - C).T
            sign, logdet = np.linalg.slogdet(K.numpy().astype(np.float64))
            for h in hooks:
                h.remove()
            r = {"arch": k, "zc_gradnorm": gn ** 0.5 / len(batches), "zc_snip": snip / len(batches),
                 "zc_naswot": float(logdet) if sign > 0 else float("-inf")}
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r, flush=True)


if __name__ == "__main__":
    main()
