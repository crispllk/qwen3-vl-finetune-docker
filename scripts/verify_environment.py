#!/usr/bin/env python3
"""Offline checks; a random tiny model tests code paths, not 4B model quality."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED = {
    "torch": "2.9.0+cu126",
    "torchvision": "0.24.0",
    "transformers": "4.57.6",
    "ms-swift": "4.6.0",
    "peft": "0.20.0",
    "accelerate": "1.15.0",
    "trl": "0.29.1",
    "datasets": "4.8.4",
    "qwen-vl-utils": "0.0.14",
}


def run_tiny_lora(device: str) -> dict:
    """Exercise text and synthetic image patches through the actual VL model."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(2026)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 6,
            "use_cache": False,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=2,
        video_token_id=3,
        vision_start_token_id=4,
        vision_end_token_id=5,
    )
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    base = Qwen3VLForConditionalGeneration(config)
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    ).to(device)
    model.train()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise AssertionError("Expected only LoRA parameters to be trainable")
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=1e-3)
    results = {}
    for name, token_ids in (
        ("text", [1, 10, 11, 12, 13, 14, 6]),
        ("vision", [1, 4, 2, 2, 2, 2, 5, 10, 11, 12, 6]),
    ):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.tensor([token_ids], device=device)
        labels = inputs.clone()
        labels[:, : (7 if name == "vision" else 2)] = -100
        kwargs = {}
        if name == "vision":
            # One 8x8 RGB image, duplicated temporally and partitioned into
            # sixteen 2x2 patches. Four merged image tokens enter the LLM.
            # Pixel values are synthetic and never contain user data.
            rgb = torch.linspace(-1, 1, 3 * 8 * 8, device=device).reshape(3, 8, 8)
            frames = rgb.unsqueeze(1).expand(3, 2, 8, 8)
            patches = frames.reshape(3, 2, 2, 2, 2, 2, 2, 2)
            patches = patches.permute(2, 5, 3, 6, 0, 1, 4, 7).reshape(16, 24)
            kwargs = {
                "pixel_values": patches,
                "image_grid_thw": torch.tensor([[1, 4, 4]], device=device),
            }
        # Reset position state between independent samples.
        base.model.rope_deltas = None
        before = {key: param.detach().clone() for key, param in trainable}
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            labels=labels,
            use_cache=False,
            **kwargs,
        )
        loss = output.loss
        if loss.device.type != device or not torch.isfinite(loss):
            raise AssertionError(f"{name}: wrong device or non-finite loss")
        loss.backward()
        gradients = [param.grad for _, param in trainable if param.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"{name}: missing or non-finite LoRA gradients")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise AssertionError(f"{name}: all LoRA gradients are zero")
        optimizer.step()
        if not any(not torch.equal(before[key], param.detach()) for key, param in trainable):
            raise AssertionError(f"{name}: optimizer did not change LoRA parameters")
        results[name] = {"loss": float(loss.detach()), "backward": "passed", "optimizer_step": "passed"}
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "device": device,
        "dtype": "float32",
        "attention": "sdpa",
        "model": "random tiny Qwen3VLForConditionalGeneration",
        "pretrained_weights_tested": False,
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "cases": results,
    }


def verify(require_gpu: bool) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, got {platform.python_version()}")
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("This image is validated for Linux ARM64 servers")
    versions = {name: metadata.version(name) for name in EXPECTED}
    for name, expected in EXPECTED.items():
        # ARM torchvision wheels in the cu126 index have no +cu126 suffix.
        if versions[name] != expected:
            raise RuntimeError(f"Expected {name}=={expected}; found {versions[name]}")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for module in (
        "torch", "torchvision", "transformers", "swift", "peft", "accelerate",
        "trl", "datasets", "qwen_vl_utils", "numpy", "scipy", "pandas", "pyarrow",
        "PIL", "av", "tifffile", "sentencepiece", "tiktoken", "tokenizers",
    ):
        importlib.import_module(module)
    import torch
    from torchvision.extension import _check_cuda_version
    from torchvision.ops import nms

    if torch.version.cuda != "12.6":
        raise RuntimeError(f"Expected a CUDA 12.6 build, got {torch.version.cuda}")
    torchvision_cuda = _check_cuda_version()
    if (torchvision_cuda // 1000, (torchvision_cuda % 1000) // 10) != (12, 6):
        raise RuntimeError(f"Expected CUDA 12.6 torchvision extensions, got {torchvision_cuda}")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; check --gpus all, host driver, and NVIDIA Container Toolkit")
    device = "cuda" if require_gpu else "cpu"
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    if nms(boxes, scores, 0.5).tolist() != [0]:
        raise AssertionError("torchvision NMS failed")
    tiny = run_tiny_lora(device)
    return {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "versions": versions,
        "cuda_build": torch.version.cuda,
        "torchvision_cuda_build": torchvision_cuda,
        "gpu_validation": "passed" if require_gpu else "not_run",
        "gpu_name": torch.cuda.get_device_name(0) if require_gpu else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if require_gpu else None,
        "torchvision_nms": "passed",
        "tiny_lora": tiny,
        "limitations": [
            "No pretrained 4B weights or user training data were downloaded or tested.",
            "The tiny LoRA smoke uses float32; it does not validate a full bfloat16 training run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="Fail unless real CUDA NMS and VL training checks pass")
    parser.add_argument("--report", type=Path, help="Write verification JSON")
    args = parser.parse_args()
    # Import and model construction checks must remain fully offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = verify(args.require_gpu)
    except Exception as error:
        result = {"status": "failed", "gpu_requested": args.require_gpu, "error": f"{type(error).__name__}: {error}"}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
