# Qwen3-VL-4B ARM64 微调 Docker

Linux ARM64 服务器的单卡 LoRA 环境：Python **3.11.17**、PyTorch **2.9.0+cu126**、torchvision **0.24.0**、ms-swift **4.6.0**、Transformers **4.57.6**。采用 SDPA，默认不安装 FlashAttention、DeepSpeed、OpenCV 或 bitsandbytes。镜像包含运行环境，不包含模型权重和训练数据。

## 版本说明

用户截图的原始版本保存在 `requirements/screenshot-pins.txt`。尽量保留固定版本；ARM64/cu126 缺少 torch 2.8 CUDA wheel，因此调整 torch/torchvision。NumPy、SciPy 和 AV 的截图版本不支持 Python 3.11，因此分别使用 2.4.6、1.17.1、18.1.0；tifffile 固定为 2026.3.3。详细原因、官方来源和驱动边界见 [兼容性核查](docs/compatibility.md)。

`requirements/runtime.in` 是目标直接依赖，`requirements/arm64.lock` 是针对 Python 3.11 / Linux ARM64 解析的全部固定依赖。PyTorch/torchvision 使用带 SHA256 的官方 CUDA wheel URL，其他包只从 PyPI 安装。截图没有提供原始完整锁文件，因此省略的传递依赖是重新解析的，不能声称恢复了未提供的锁文件。

## 拉取与验证

公开镜像文件下载见 [Releases](https://github.com/crispllk/qwen3-vl-finetune-docker/releases)。下载同一 Release 的全部镜像分片和 `SHA256SUMS` 后执行：

```bash
sha256sum -c SHA256SUMS
cat qwen3-vl-cu126-arm64.tar.gz.part-* | gzip -dc | docker load
```

GHCR 首次发布的新包默认私有，仓库公开不会自动改变包可见性。以下 `docker pull` 仅在包已设为公开或已授权登录后使用；公开 Release 下载不受此限制。加载 Release 镜像后可以直接执行后面的 `docker run`。

```bash
docker pull ghcr.io/crispllk/qwen3-vl-finetune-docker:cu126-arm64

mkdir -p reports
docker run --rm --gpus all --shm-size 8g \
  -v "$PWD/reports:/reports" \
  ghcr.io/crispllk/qwen3-vl-finetune-docker:cu126-arm64 \
  python /opt/qwen3-vl/scripts/verify_environment.py \
    --require-gpu --report /reports/gpu-verification.json
```

宿主机需为 Linux ARM64，安装 NVIDIA 驱动和 NVIDIA Container Toolkit，并已为 Docker 配置 GPU runtime。仅知道“560 驱动”不足以完成 GPU 兼容性验证，请记录完整驱动号和 GPU 型号。镜像面向通用 ARM64 服务器，未验证 Jetson/JetPack。容器提供 CUDA 用户态库，不安装宿主机驱动或完整 CUDA 编译工具链。

构建会检查 `pip check`、核心版本、CUDA 编译版本、torchvision 算子、Swift CLI，以及随机小型 Qwen3-VL 的文本和视觉 LoRA 前向/反向。构建器没有 GPU，构建成功只证明 CPU 验证通过；上面的 `--require-gpu` 验证及小规模真实数据训练必须在目标 GPU 上执行。

## 开始训练

数据使用 ms-swift JSONL 格式。例如 `data/train.jsonl`：

```json
{"messages":[{"role":"user","content":"<image>描述这张图片。"},{"role":"assistant","content":"这里填写图片对应的训练答案。"}],"images":["/data/images/example.jpg"]}
```

准备至少包含多个有效样本的真实数据，并确保图片路径在容器内存在。启动示例：

```bash
mkdir -p data output cache
docker run --rm --gpus '"device=0"' --shm-size 8g \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/output:/output" \
  -v "$PWD/cache:/cache" \
  ghcr.io/crispllk/qwen3-vl-finetune-docker:cu126-arm64 \
  train_lora.sh /data/train.jsonl --max_steps 5 --save_steps 5
```

默认模型 `Qwen/Qwen3-VL-4B-Instruct`，运行时下载到挂载缓存。已有模型可挂载 `-v /host/model:/models/qwen3-vl:ro`，并设置 `-e MODEL=/models/qwen3-vl`。正式训练可去掉 `--max_steps 5`；显存占用取决于图片分辨率、序列长度和数据，按显存调整 `--max_length` 等参数。默认 BF16，目标 GPU 必须支持；不支持时需结合 GPU 能力选择精度并重新验证。访问受限模型所需的凭据仅在运行时传入，不写入镜像。

## 锁文件和环境报告

每次成功构建会生成 `/opt/qwen3-vl/report/runs/`，包含 `environment.json`、`requirements-arm64-lock.txt`（实际 `pip freeze`）、安装报告、构建验证报告和 Swift 帮助输出。GitHub Actions 将它们作为构建 artifact 上传。也可从镜像复制：

```bash
cid=$(docker create ghcr.io/crispllk/qwen3-vl-finetune-docker:cu126-arm64)
docker cp "$cid:/opt/qwen3-vl/report/runs" ./environment-report
docker rm "$cid"
```

环境报告记录 cu128→cu126、x86_64→ARM64、原 CPU 锁环境 Python 3.12→3.11 及其他依赖调整，不把 CPU 构建结果写成 GPU 测试结果。

## 自行构建

```bash
docker buildx build --platform linux/arm64 --load -t qwen3-vl:cu126-arm64 .
```

仓库 Actions 使用原生 `ubuntu-24.04-arm` runner 构建并推送 GHCR，只有镜像内验证通过后才发布。`cu126-arm64` 是更新标签，`sha-<完整提交SHA>` 可定位具体构建；实际使用时也可固定镜像 digest。镜像为多种上游软件的组合，各软件和下载模型仍适用其各自许可证。
