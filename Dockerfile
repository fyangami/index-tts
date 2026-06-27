FROM docker.1ms.run/nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL description="IndexTTS2 OpenAI-compatible API Server"

COPY --from=ghcr.1ms.run/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV USE_MODELSCOPE=false
ENV HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --no-install-project \
    --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" \
    && uv pip install \
       --index-url "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" \
       --no-cache \
       "uvicorn[standard]" \
       fastapi \
       python-multipart \
    && rm -rf /root/.cache

COPY indextts/ indextts/
COPY server_v2.py .
COPY tools/ tools/
COPY checkpoints/ checkpoints/

RUN .venv/bin/python -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available('checkpoints')"

EXPOSE 8000

ENTRYPOINT [".venv/bin/python", "server_v2.py"]
CMD ["--model_dir", "/app/checkpoints", "--fp16"]
