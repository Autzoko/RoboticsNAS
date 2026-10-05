"""Elastic (weight-sharing) SmolVLA.

Re-implements the SmolVLA forward pass (LeRobot 3f2c29e, `smolvlm_with_expert.py`) as two explicit stages
so that a single frozen-VLM prefix pass can serve many sub-networks:

1. `prefix_kv`: run the frozen VLM text layers over [image tokens, language, state] and keep the
   post-RoPE K/V of every layer. Prefix tokens never attend to action tokens (block-causal mask), so these
   K/V are identical to what SmolVLA computes in its joint forward.
2. `velocity`: run the first `n_exp` action-expert layers; expert layer j reads the K/V of VLM layer
   `arch.bridge_map()[j]` (SmolVLA default: j). Even layers are joint self-attention over
   [prefix K/V ; own K/V], odd layers are cross-attention through the expert's k/v projections.

Elastic axes: VLM depth, expert depth (first-k layers), bridge map, expert FFN width (first-k sorted
neurons), visual tokens per camera (2x2 average pooling of the 8x8 token grid), denoising steps.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

from .space import MAX_VLM, Arch


def pool_tokens(x: Tensor, n_out: int) -> Tensor:
    """[B, 64, D] row-major 8x8 grid -> [B, n_out, D] by average pooling."""
    b, n, d = x.shape
    if n == n_out:
        return x
    s, so = int(math.isqrt(n)), int(math.isqrt(n_out))
    assert s * s == n and so * so == n_out and s % so == 0, (n, n_out)
    g = x.reshape(b, s, s, d).permute(0, 3, 1, 2)
    g = F.avg_pool2d(g.float(), kernel_size=s // so).to(x.dtype)
    return g.permute(0, 2, 3, 1).reshape(b, n_out, d)


def elastic_mlp(mlp: nn.Module, x: Tensor, ratio: float) -> Tensor:
    if ratio >= 1.0:
        return mlp(x)
    k = int(round(mlp.gate_proj.out_features * ratio))
    g = F.linear(x, mlp.gate_proj.weight[:k])
    u = F.linear(x, mlp.up_proj.weight[:k])
    return F.linear(mlp.act_fn(g) * u, mlp.down_proj.weight[:, :k])


@torch.no_grad()
def sort_ffn_neurons(expert_layers: nn.ModuleList) -> None:
    """Function-preserving permutation of FFN neurons by importance (OFA-style), so that first-k slices
    keep the most important neurons."""
    for layer in expert_layers:
        mlp = layer.mlp
        imp = (
            mlp.gate_proj.weight.abs().sum(1)
            + mlp.up_proj.weight.abs().sum(1)
            + mlp.down_proj.weight.abs().sum(0)
        )
        perm = torch.argsort(imp, descending=True)
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[perm])
        mlp.up_proj.weight.copy_(mlp.up_proj.weight[perm])
        mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, perm])


class ElasticSmolVLA(nn.Module):
    def __init__(self, policy, extra_vlm_layers: nn.ModuleList | None = None):
        super().__init__()
        self.policy = policy  # lerobot SmolVLAPolicy (holds config + prepare_* helpers)
        self.m = policy.model  # VLAFlowMatching
        self.vwe = self.m.vlm_with_expert
        text = self.vwe.get_vlm_model().text_model
        if extra_vlm_layers is not None and len(text.layers) < MAX_VLM:
            for layer in extra_vlm_layers[len(text.layers) : MAX_VLM]:
                text.layers.append(layer)
        for p in text.layers.parameters():
            p.requires_grad = False
        self.vlm_layers = text.layers
        self.exp_layers = self.vwe.lm_expert.layers
        self.sa_every = self.vwe.self_attn_every_n_layers
        self.head_dim = self.vwe.vlm.config.text_config.head_dim
        self.chunk = self.m.config.chunk_size

    # ------------------------------------------------------------------ prefix
    @contextlib.contextmanager
    def _pooled_images(self, vtok: int):
        orig = self.vwe.embed_image
        if vtok == 64:
            yield
            return
        self.vwe.embed_image = lambda img: pool_tokens(orig(img), vtok)
        try:
            yield
        finally:
            self.vwe.embed_image = orig

    @contextlib.contextmanager
    def image_cache(self):
        """Memoize the frozen SigLIP+connector output per image tensor within a step (vtok variants share it)."""
        orig, memo = self.vwe.embed_image, {}

        def cached(img):
            k = (img.data_ptr(), tuple(img.shape))
            if k not in memo:
                memo[k] = orig(img)
            return memo[k]

        self.vwe.embed_image = cached
        try:
            yield
        finally:
            self.vwe.embed_image = orig

    def _attend(self, mask: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.vwe.eager_attention_forward(mask, q.shape[0], self.head_dim, q, k, v)

    def prefix_kv(self, images, img_masks, lang_tokens, lang_masks, state, vtok: int, n_layers: int):
        """Returns (list of (K, V) per VLM layer [B, P, Hkv, Dh], prefix_pad_masks)."""
        with self._pooled_images(vtok):
            embs, pad, att = self.m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att_2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        h = embs
        kvs = []
        for li in range(n_layers):
            layer = self.vlm_layers[li]
            x = layer.input_layernorm(h).to(dtype=layer.self_attn.q_proj.weight.dtype)
            shp = (*x.shape[:-1], -1, self.head_dim)
            q = apply_rope(layer.self_attn.q_proj(x).view(shp), pos)
            k = apply_rope(layer.self_attn.k_proj(x).view(shp), pos)
            v = layer.self_attn.v_proj(x).view(shp)
            kvs.append((k, v))
            if li == n_layers - 1:
                break  # deeper hidden states are not needed
            att_out = self._attend(att_2d, q, k, v).to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(att_out) + h
            h = out + layer.mlp(layer.post_attention_layernorm(out))
        return kvs, pad

    # ---------------------------------------------------------------- expert
    def velocity(self, kvs, prefix_pad: Tensor, x_t: Tensor, time: Tensor, arch: Arch) -> Tensor:
        suffix_embs, suffix_pad, suffix_att = self.m.embed_suffix(x_t, time)
        b, s = suffix_pad.shape
        p = prefix_pad.shape[1]
        prefix_2d = prefix_pad[:, None, :].expand(b, s, p)
        sa_mask = torch.cat([prefix_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        pos_sa = prefix_pad.sum(-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
        pos_ca = pos_sa - pos_sa.min(dim=1, keepdim=True).values
        bmap = arch.bridge_map()
        h = suffix_embs
        for j in range(arch.n_exp):
            layer = self.exp_layers[j]
            kp, vp = kvs[bmap[j]]
            x = layer.input_layernorm(h).to(dtype=layer.self_attn.q_proj.weight.dtype)
            shp = (*x.shape[:-1], -1, self.head_dim)
            q = layer.self_attn.q_proj(x).view(shp)
            if self.sa_every > 0 and j % self.sa_every == 0:
                q = apply_rope(q, pos_sa)
                k = apply_rope(layer.self_attn.k_proj(x).view(shp), pos_sa)
                v = layer.self_attn.v_proj(x).view(shp)
                k = torch.cat([kp.to(k.dtype), k], dim=1)
                v = torch.cat([vp.to(v.dtype), v], dim=1)
                att_out = self._attend(sa_mask, q, k, v)
            else:
                q = apply_rope(q, pos_ca)
                wd = layer.self_attn.k_proj.weight.dtype
                k = layer.self_attn.k_proj(kp.to(wd).reshape(b, p, -1)).view(b, p, -1, self.head_dim)
                v = layer.self_attn.v_proj(vp.to(wd).reshape(b, p, -1)).view(b, p, -1, self.head_dim)
                att_out = self._attend(prefix_2d, q, k, v)
            att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(att_out) + h
            h = out + elastic_mlp(layer.mlp, layer.post_attention_layernorm(out), arch.ffn)
        h = self.vwe.lm_expert.norm(h)
        return self.m.action_out_proj(h[:, -self.chunk :].to(torch.float32))

    # ----------------------------------------------------------------- batch
    def prepare(self, batch: dict) -> tuple:
        pol = self.policy
        images, img_masks = pol.prepare_images(batch)
        state = pol.prepare_state(batch)
        return images, img_masks, batch["observation.language.tokens"], batch[
            "observation.language.attention_mask"
        ], state

    @torch.no_grad()
    def sample_actions(self, batch: dict, arch: Arch, noise: Tensor | None = None) -> Tensor:
        images, img_masks, lt, lm, state = self.prepare(batch)
        kvs, pad = self.prefix_kv(images, img_masks, lt, lm, state, arch.vtok, arch.vlm_layers_needed())
        bsz = state.shape[0]
        if noise is None:
            noise = self.m.sample_noise((bsz, self.chunk, self.m.config.max_action_dim), state.device)
        x = noise
        dt = -1.0 / arch.steps
        for i in range(arch.steps):
            t = torch.full((bsz,), 1.0 + i * dt, dtype=torch.float32, device=state.device)
            x = x + dt * self.velocity(kvs, pad, x, t, arch)
        n_act = self.policy.config.action_feature.shape[0]
        return x[:, :, :n_act]
