ARM64 / Python 3.11.17 / CUDA 12.6 / PyTorch 2.9.0 / ms-swift 4.6.0 的 Qwen3-VL 微调环境。

下载本 Release 的全部 `qwen3-vl-cu126-arm64.tar.gz.part-*` 文件和 `SHA256SUMS`，放在同一目录，在目标 Linux ARM64 服务器执行：

```bash
sha256sum -c SHA256SUMS
cat qwen3-vl-cu126-arm64.tar.gz.part-* | gzip -dc | docker load
docker run --rm --gpus all --shm-size 8g \
  ghcr.io/crispllk/qwen3-vl-finetune-docker:cu126-arm64 \
  python /opt/qwen3-vl/scripts/verify_environment.py --require-gpu
```

这些文件是同一份 Docker 镜像压缩后的有序分片，必须全部下载。公开 Release 下载无需 GHCR 登录。镜像也推送至 GHCR，但 GitHub 的新容器包初始为私有，需单独设置包可见性才能匿名 `docker pull`；仓库公开不等于容器包自动公开。

构建通过依赖、版本、Swift CLI、torchvision CUDA 扩展以及 CPU 上的小型随机 Qwen3-VL 文本/视觉 LoRA 前向、反向和优化步骤检查。构建器没有 GPU，未验证目标 GPU、BF16 或真实 4B 模型训练。请先执行上述 GPU 检查，再根据仓库 README 用自己的真实数据试训。

`environment.json` 和 `requirements-arm64-lock.txt` 记录实际安装版本及所有必要变更。镜像不含模型权重、训练数据或凭据。
