#!/usr/bin/env python3
"""Capture installed versions and explicit verification limits without secrets."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output-dir", type=Path, default=root / "report" / "runs")
    parser.add_argument("--verification", type=Path, help="JSON written by verify_environment.py")
    parser.add_argument("--source-pins", type=Path, default=root / "requirements" / "screenshot-pins.txt")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True).stdout
    lock_path = args.output_dir / "requirements-arm64-lock.txt"
    lock_path.write_text(freeze, encoding="utf-8")
    installed = {normalized(dist.metadata["Name"]): dist.version for dist in metadata.distributions() if dist.metadata.get("Name")}
    source = {}
    for line in args.source_pins.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            source[normalized(match[1])] = match[2]
    changes = []
    reasons = {
        "torch": "The official cu126 index has no 2.8.0 Python 3.11 ARM64 CUDA wheel; 2.9.0 is the closest available release.",
        "torchvision": "Match torch 2.9.0; the official cu126 ARM64 wheel uses version 0.24.0 without a local-version suffix.",
        "numpy": "The screenshot version requires Python >=3.12; use a compatible Python 3.11 ARM64 wheel.",
        "scipy": "The screenshot version requires Python >=3.12; use a compatible Python 3.11 ARM64 wheel.",
        "av": "The screenshot version requires Python >=3.12; use a compatible Python 3.11 ARM64 wheel.",
    }
    for name, requested in sorted(source.items()):
        actual = installed.get(name)
        if actual != requested:
            changes.append({"package": name, "screenshot": requested, "installed": actual, "reason": reasons.get(name, "Resolved dependency compatibility adjustment; see the compatibility report and installation logs.")})
    verification = json.loads(args.verification.read_text(encoding="utf-8")) if args.verification else None
    import torch

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_at_recording": torch.cuda.is_available(),
        "source_environment": {
            "lock": {"architecture": "x86_64", "python": "3.12", "device": "CPU", "original_lock_available": False},
            "referenced_cuda_environment": {"python": "3.11.17", "torch": "2.8.0+cu128"},
        },
        "intentional_environment_changes": ["CUDA 12.8 -> 12.6", "x86_64 -> aarch64", "CPU/Python 3.12 lock -> CUDA/Python 3.11 environment"],
        "version_changes_from_screenshots": changes,
        "additional_pins": {"tifffile": {"screenshot": ">=2024.1", "installed": installed.get("tifffile")}},
        "installed_packages": dict(sorted(installed.items())),
        "freeze_file": lock_path.name,
        "verification": verification,
        "gpu_validation": verification.get("gpu_validation", "not_run") if verification else "not_run",
        "limitations": [
            "Screenshots are incomplete source evidence, not the original full lock file.",
            "CUDA wheel installation and CPU smoke tests do not establish host GPU/driver compatibility.",
            "A tiny randomly initialized model check does not validate a full Qwen3-VL-4B training run.",
        ],
    }
    output = args.output_dir / "environment.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} and {lock_path}")


if __name__ == "__main__":
    main()
