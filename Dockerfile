# syntax=docker/dockerfile:1.7
# Pin the official Python multi-architecture index; build is intentionally ARM64 only.
ARG PYTHON_IMAGE=python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf
FROM ${PYTHON_IMAGE} AS builder
ARG TARGETARCH
RUN test "$TARGETARCH" = arm64
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements/arm64.lock /tmp/requirements-arm64.txt
RUN mkdir -p /opt/build-report \
    && python -m pip install --index-url https://pypi.org/simple \
       --report /opt/build-report/install.json -r /tmp/requirements-arm64.txt \
    && python -m pip check

FROM ${PYTHON_IMAGE} AS runtime
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Qwen3-VL ARM64 fine-tuning" \
      org.opencontainers.image.description="Python 3.11 / PyTorch CUDA 12.6 / ms-swift single-GPU LoRA environment" \
      org.opencontainers.image.source="https://github.com/crispllk/qwen3-vl-finetune-docker" \
      org.opencontainers.image.revision=$VCS_REF
ENV PATH=/opt/venv/bin:/opt/qwen3-vl/scripts:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/cache/huggingface MODELSCOPE_CACHE=/cache/modelscope \
    TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git ffmpeg libgomp1 libglib2.0-0 libgl1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/build-report/install.json /opt/qwen3-vl/report/runs/install.json
COPY requirements /opt/qwen3-vl/requirements
COPY scripts /opt/qwen3-vl/scripts
COPY docs /opt/qwen3-vl/docs
RUN chmod +x /opt/qwen3-vl/scripts/*.sh \
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
