FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY prompts ./prompts
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests

RUN uv sync --extra cuda --extra dev

EXPOSE 30000

CMD ["uv", "run", "--extra", "cuda", "inference-lab", "serve", "--config", "configs/sglang.qwen3-0.6b.yaml"]
