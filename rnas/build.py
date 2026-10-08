"""Construct the elastic SmolVLA from pinned, LIBERO-free checkpoints and train-only stats."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from .elastic import ElasticSmolVLA, sort_ffn_neurons
from .space import MAX_VLM

REPO = Path(__file__).resolve().parents[1]
PINS = json.loads((REPO / "configs" / "pins.json").read_text())
FORBIDDEN_SUBSTR = ("libero", "LIBERO", "pi0", "pi05")  # no benchmark-finetuned weights may be loaded


def _snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    assert not any(s in repo_id for s in FORBIDDEN_SUBSTR), f"forbidden checkpoint {repo_id}"
    return snapshot_download(repo_id=repo_id, revision=PINS[repo_id])


def libero_features() -> tuple[dict, dict]:
    inp = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    return inp, out


def build_elastic(device: str = "cuda", sort_neurons: bool = True) -> ElasticSmolVLA:
    from transformers import AutoModelForImageTextToText

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    base = _snapshot("lerobot/smolvla_base")
    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.input_features, cfg.output_features = libero_features()
    cfg.empty_cameras = 0
    cfg.load_vlm_weights = False  # VLM weights come from the smolvla_base checkpoint itself
    cfg.vlm_model_name = _snapshot(PINS["vlm_repo"])
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(base, config=cfg)

    # Extra VLM layers beyond the 16 kept by SmolVLA come from the original SmolVLM2 (frozen).
    vlm_full = AutoModelForImageTextToText.from_pretrained(cfg.vlm_model_name, torch_dtype=torch.float32)
    extra = vlm_full.model.text_model.layers
    model = ElasticSmolVLA(policy, extra_vlm_layers=extra)
    del vlm_full
    assert len(model.vlm_layers) == MAX_VLM
    if sort_neurons:
        sort_ffn_neurons(model.exp_layers)
    # transformers builds the expert in the config dtype (bf16); keep master weights of everything trainable
    # in fp32 (bf16 autocast is used for compute).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model.to(device)


def make_processors(model: ElasticSmolVLA, stats_path: str):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    from .data import load_stats_tensors

    return make_smolvla_pre_post_processors(model.policy.config, dataset_stats=load_stats_tensors(stats_path))


def trainable_state_dict(model: ElasticSmolVLA, names: set | None = None) -> dict:
    names = names or {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def load_trainable(model: ElasticSmolVLA, path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "depth_gain" in ck["model"] and model.depth_gain is None:
        model.enable_depth_gain()
    if "kv_adapt" in ck["model"] and model.kv_adapt is None:
        model.enable_kv_adapter()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, unexpected
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing_tr = {n for n in trainable & set(missing) if n not in ("depth_gain", "kv_adapt")}  # gains may be newly enabled (init 1)
    assert not missing_tr, sorted(missing_tr)[:5]
    return ck


def setup_env_vars() -> None:
    os.environ.setdefault("HF_HOME", "/scratch/ll5582/RoboticsNAS/cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
