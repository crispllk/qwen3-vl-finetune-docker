# ARM64 / CUDA 12.6 兼容性核查

核查日期：2026-10-28。目标是 Linux ARM64 服务器、Python 3.11、CUDA 12.6 和 Qwen3-VL-4B-Instruct 单卡 LoRA 微调。截图中的 CPU/x86_64/Python 3.12 版本表不能直接作为这一目标环境的完整锁文件。

## 必要版本调整

| 组件 | 截图版本 | ARM64 / Python 3.11 采用版本 | 原因 |
| --- | --- | --- | --- |
| Python | 3.11.17（CUDA 环境） | 3.11.17 | 官方容器标签有 ARM64 镜像 |
| torch | 2.8.0+cu126 | 2.9.0+cu126 | 官方 cu126 索引没有 2.8.0 的 CPython 3.11 ARM64 CUDA wheel；2.9.0 是最接近的较新可用版本 |
| torchvision | 0.23.0+cu126 | 0.24.0，从 cu126 索引安装 | 与 torch 2.9.0 配套；该 ARM64 wheel 文件名和包版本没有 `+cu126` 后缀 |
| numpy | 2.5.4 | 2.4.6 | 2.5.4 要求 Python >=3.12 |
| scipy | 1.18.1 | 1.17.1 | 1.18.1 要求 Python >=3.12 |
| av | 19.0.2 | 18.1.0 | 19.0.2 要求 Python >=3.12 |
| tifffile | >=2024.1 | 2026.3.3 | 补齐截图要求的 TIFF 支持，并固定到支持 Python 3.11 的版本 |

Python 限制及替代版本由 [PyPI NumPy 元数据](https://pypi.org/pypi/numpy/2.5.4/json)、[SciPy 元数据](https://pypi.org/pypi/scipy/1.18.1/json)、[AV 元数据](https://pypi.org/pypi/av/19.0.2/json) 和 [tifffile 元数据](https://pypi.org/pypi/tifffile/2026.3.3/json)核查。其余截图中的固定版本尽量保留；原始 CPU 锁文件未随截图提供，截图省略的小包版本不能声称已原样恢复，必须以目标镜像中解析并导出的完整锁文件为准。

核心训练组合 `ms-swift==4.6.0`、`transformers==4.57.6`、`peft==0.20.0`、`accelerate==1.15.0`、`trl==0.29.1`、`datasets==4.8.4`、`qwen-vl-utils==0.0.14` 的 PyPI 元数据允许 Python 3.11，并且这些固定版本满足彼此声明的直接依赖约束。元数据可解析不等于模型训练已经通过。

## PyTorch CUDA wheel 与基础镜像

官方 [torch cu126 索引](https://download.pytorch.org/whl/cu126/torch/)中确认存在：

```text
torch-2.9.0+cu126-cp311-cp311-manylinux_2_28_aarch64.whl
sha256:0ac8362cd4c8c85af5c865fb63a4580656f5f1aae39e77469a84dfb3d6c979d0
```

官方 [torchvision cu126 索引](https://download.pytorch.org/whl/cu126/torchvision/)中确认存在：

```text
torchvision-0.24.0-cp311-cp311-manylinux_2_28_aarch64.whl
sha256:5b40967d33583c00bc9dde76393e9fbba5faea1cb0fc2643afe1090f2a7cc72d
```

应从这一 CUDA 索引安装配套 wheel，不能把 ARM64 的 torchvision 固定为不存在的 `0.24.0+cu126`，也不能仅凭没有 local version 后缀判断该 wheel 是 CPU 构建。安装后检查 torch CUDA 编译版本、torchvision CUDA 扩展版本及算子导入。

已读取实际 ARM64 CUDA torch wheel 的元数据，它依赖 CUDA runtime 12.6.77、cuBLAS 12.6.4.1、cuDNN 9.10.2.21、NCCL 2.27.5、Triton 3.5.0 等。其全部 16 项精确 NVIDIA/Triton 依赖均在 PyPI 有相应 ARM64 wheel。不要以 PyPI 默认 torch wheel 的依赖表代替 cu126 wheel 的依赖表。

[Docker Hub 官方标签接口](https://hub.docker.com/v2/repositories/library/python/tags/3.11.17-slim-bookworm)确认 `python:3.11.17-slim-bookworm` 包含 `linux/arm64/v8`，无需改变 Python 补丁版本。核查时的 digest：

```text
多架构索引：sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf
ARM64 manifest：sha256:8f51ccb8569b75cf967ffe3e1638b9953042bf36142d7032cbb6f0696c9cd2c6
```

CUDA 用户态运行库随 PyTorch 依赖安装；宿主机仍需 NVIDIA 驱动以及配置完成的 NVIDIA Container Toolkit。镜像不需要内置宿主机驱动。该目标为通用 ARM64 服务器，不能据此保证适用于 Jetson/JetPack。

## 训练参数与可选加速包

按 [ms-swift 4.6 Qwen3-VL 官方示例](https://swift.readthedocs.io/en/v4.6/BestPractices/Qwen3-VL-Best-Practice.html)，LoRA 参数使用 `--tuner_type lora`、`--torch_dtype bfloat16`、`--target_modules all-linear`、`--freeze_vit true` 和 `--freeze_aligner true`。使用自定义数据集时传 `--dataset /data/train.jsonl`，需要验证集时另传 `--val_dataset /data/val.jsonl`；图片路径必须是容器内可访问的路径。

默认选择 PyTorch SDPA：

```text
--attn_impl sdpa --padding_free false --packing false
```

[ms-swift 参数文档](https://swift.readthedocs.io/en/v4.6/Instruction/Command-line-parameters.html)支持 `sdpa`；[Transformers v4.57.6 的 Qwen3-VL 源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)明确声明 `_supports_sdpa = True`。这样默认路径无需编译 FlashAttention，也无需单卡训练不使用的 DeepSpeed。BF16 仍取决于目标 GPU 能力，运行时应检查支持情况。

截图对 `cpm-kernels` 的依赖说明需要更正：[cpm-kernels 1.0.11 元数据](https://pypi.org/pypi/cpm-kernels/1.0.11/json)提供 `py3-none-any` wheel 且没有声明 Triton 依赖；此环境的 Triton 来自 torch 的依赖。ms-swift 声明依赖 `cpm-kernels`，应正常安装并进行 `pip check`，不能未核查就跳过。通用 wheel 标签只表明 Python 分发形式，不证明每个内置 CUDA kernel 都适用于所有 GPU；默认 Qwen3-VL LoRA/SDPA 路径仍需实际运行验证。

保留 Pillow、AV、tifffile；不额外引入截图未要求的 OpenCV。原始截图中的 `transformers-stream-generator==0.0.5` 只有源码发行包，需要在构建时生成 wheel。

## 驱动和验证边界

“驱动 560”不是完整兼容性证据，需记录服务器 `nvidia-smi` 输出的完整驱动版本、GPU 型号及实际可见设备。[NVIDIA CUDA 12.6 发布说明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)列出 GA 对应 Linux 驱动 560.28.03，Update 1/2 对应 560.35.03，Update 3 对应 560.35.05；CUDA minor-version compatibility 另有适用条件。因此不能仅见到 R560 就承诺所有 CUDA 功能可用，也不应把 GA 配套驱动号当作所有 cu126 wheel 的无条件保证。

构建阶段应完成依赖安装、`pip check`、版本校验、Qwen3-VL/Swift 导入及无需 GPU 的基本检查，并导出完整 `pip freeze` 和环境报告。没有 GPU 的构建器通过这些检查，仍不能宣称 GPU 训练已通过。

最终验收还需要在目标 ARM64 GPU 服务器上进行 CUDA 可见性与张量运算测试，再用少量真实图片数据完成至少一个 LoRA 前向、反向和优化步骤。环境报告应明确记录从 x86_64 到 ARM64、从 CPU 锁环境 Python 3.12 到 Python 3.11、从文档 cu128 到 cu126，以及上表额外版本调整。
