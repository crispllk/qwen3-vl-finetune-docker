# syntax=docker/dockerfile:1.7
# Pin the official Python multi-architecture index; build is intentionally ARM64 only.
ARG PYTHON_IMAGE=python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf
FROM ${PYTHON_IMAGE}
ARG TARGETARCH
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Qwen3-VL ARM64 fine-tuning" \
      org.opencontainers.image.description="Python 3.11 / PyTorch CUDA 12.6 / ms-swift single-GPU LoRA environment" \
      org.opencontainers.image.source="https://github.com/crispllk/qwen3-vl-finetune-docker" \
      org.opencontainers.image.revision=$VCS_REF
ENV PATH=/opt/venv/bin:/opt/qwen3-vl/scripts:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/cache/huggingface MODELSCOPE_CACHE=/cache/modelscope \
    TOKENIZERS_PARALLELISM=false
COPY requirements /opt/qwen3-vl/requirements
COPY scripts /opt/qwen3-vl/scripts
COPY docs /opt/qwen3-vl/docs
# Keep installation and build-tool removal in one layer: no second CUDA venv copy.
RUN test "$TARGETARCH" = arm64 \
    && apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git ffmpeg libgomp1 libglib2.0-0 libgl1 libsndfile1 \
    && python -m venv /opt/venv \
    && mkdir -p /opt/qwen3-vl/report/runs \
    && python -m pip install --index-url https://pypi.org/simple \
      --report /opt/qwen3-vl/report/runs/install.json -r /opt/qwen3-vl/requirements/arm64.lock \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && chmod +x /opt/qwen3-vl/scripts/*.sh \
    && mkdir -p /workspace /data /models /cache /output \
    && python -m pip check \
    && python /opt/qwen3-vl/scripts/verify_environment.py \
         --report /opt/qwen3-vl/report/runs/build-verification.json \
    && python /opt/qwen3-vl/scripts/record_environment.py \
         --output-dir /opt/qwen3-vl/report/runs \
         --verification /opt/qwen3-vl/report/runs/build-verification.json \
    && swift sft --help > /opt/qwen3-vl/report/runs/swift-sft-help.txt
WORKDIR /workspace
ENTRYPOINT ["/opt/qwen3-vl/scripts/entrypoint.sh"]
CMD ["bash"]
