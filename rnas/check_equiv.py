"""E0: the elastic forward at the default arch must reproduce LeRobot SmolVLA (fp32, same noise/time).

Also checks (a) smolvla_base VLM layers == original SmolVLM2 layers (VLM was frozen in pretraining, so
appending SmolVLM2 layers 16..23 is consistent), (b) FFN neuron sorting is function-preserving,
(c) data pipeline returns the expected keys/shapes.
"""

from __future__ import annotations

import json

import torch

from .build import build_elastic, make_processors, setup_env_vars
from .data import make_dataset
from .space import DEFAULT, Arch


def main():
    setup_env_vars()
    torch.manual_seed(0)
    model = build_elastic("cuda", sort_neurons=False).float().eval()
    pre, post = make_processors(model, "configs/train_stats.json")
    split = json.load(open("configs/split.json"))
    ds = make_dataset(split["val"][:4], chunk_size=model.chunk)
    batch = torch.utils.data.default_collate([ds[i] for i in (0, 50, 100, 150)])
    print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in batch.items()})
    batch = pre(batch)

    # (a) VLM layer identity
    from transformers import AutoModelForImageTextToText

    ref = AutoModelForImageTextToText.from_pretrained(model.policy.config.vlm_model_name, torch_dtype=torch.float32)
    d = max(
        (p1.cuda() - p2).abs().max().item()
        for p1, p2 in zip(ref.model.text_model.layers[:16].parameters(), model.vlm_layers[:16].parameters())
    )
    print(f"[a] max |smolvla_base VLM - SmolVLM2| over first 16 layers = {d:.3e}")

    pol = model.policy
    act = pol.prepare_action(batch)
    noise = torch.randn(act.shape, device="cuda")
    t = torch.rand(act.shape[0], device="cuda") * 0.9 + 0.05
    with torch.no_grad():
        ref_losses = pol.model.forward(*_inputs(pol, batch), act, noise, t)  # [B, T, 32]
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * act
        images, img_masks, lt, lm, state = model.prepare(batch)
        kvs, pad = model.prefix_kv(images, img_masks, lt, lm, state, 64, DEFAULT.vlm_layers_needed())
        v = model.velocity(kvs, pad, x_t, t, DEFAULT)
        my_losses = (v - (noise - act)) ** 2
    print(f"[E0] train-loss max abs diff = {(ref_losses - my_losses).abs().max().item():.3e} "
          f"(ref mean {ref_losses.mean().item():.4f})")

    with torch.no_grad():
        nz = torch.randn(act.shape[0], model.chunk, 32, device="cuda")
        a_ref = pol.model.sample_actions(*_inputs(pol, batch), noise=nz)[:, :, :7]
        a_my = model.sample_actions(batch, DEFAULT, noise=nz)
    print(f"[E0] sample_actions max abs diff = {(a_ref - a_my).abs().max().item():.3e}")

    # (b) neuron sorting preserves function
    from .elastic import sort_ffn_neurons

    sort_ffn_neurons(model.exp_layers)
    with torch.no_grad():
        a_sorted = model.sample_actions(batch, DEFAULT, noise=nz)
    print(f"[b] after neuron sort max abs diff = {(a_sorted - a_my).abs().max().item():.3e}")

    # every arch class runs; report output scale
    for a in [Arch(8, 4, "stretch", 0.5, 16, 2, 5), Arch(24, 8, "top", 0.75, 64, 4, 10), Arch(12, 16, "stretch", 1.0, 16)]:
        with torch.no_grad():
            out = model.sample_actions(batch, a, noise=nz)
        print(f"   {a.key()}: out shape {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")


def _inputs(pol, batch):
    images, img_masks = pol.prepare_images(batch)
    state = pol.prepare_state(batch)
    return images, img_masks, batch["observation.language.tokens"], batch[
        "observation.language.attention_mask"
    ], state


if __name__ == "__main__":
    main()
