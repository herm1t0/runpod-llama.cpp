# llama.cpp (CUDA, server) + RunPod Serverless worker (endpoint type: Queue).
#
# База собрана на nvidia/cuda:12.8.1-runtime-ubuntu24.04, где системный python
# помечен как externally-managed (PEP 668): `pip3 install ...` в него падает с
# "error: externally-managed-environment". Поэтому ставим пакеты в venv.
# Заодно у базы ENTRYPOINT=/app/llama-server и llama-server НЕ в PATH, а в /app.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install "huggingface_hub>=0.30" "runpod>=1.7"

ENV PATH="/opt/venv/bin:$PATH"

# --- Модель -----------------------------------------------------------------
# Репозиторий публичный, поэтому токен не нужен и в образ не попадает.
# Если репозиторий когда-нибудь закроют — качайте модель в рантайме и передавайте
# HF_TOKEN секретом/переменной окружения эндпоинта, но не ARG'ом: значение ARG
# подставляется в текст RUN-команды и остаётся в `docker history`.
ARG HF_REPO="HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
ARG HF_FILE="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"

ENV HF_REPO=${HF_REPO} \
    HF_FILE=${HF_FILE} \
    MODEL_DIR=/models \
    MODEL_PATH=/models/${HF_FILE}

# Если скачивание не укладывается в 30-минутный лимит сборки RunPod, включите
# hf_transfer: он на порядок быстрее обычного клиента.
#   && /opt/venv/bin/pip install hf_transfer   (в слое с pip, выше)
#   ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python -c "import os; from huggingface_hub import hf_hub_download; print('downloaded:', hf_hub_download(repo_id=os.environ['HF_REPO'], filename=os.environ['HF_FILE'], local_dir=os.environ['MODEL_DIR']))" \
 && test -s "$MODEL_PATH" \
 && ls -la /models

# --- Параметры llama-server (переопределяются переменными эндпоинта) ---------
ENV LLAMA_SERVER_BIN=/app/llama-server \
    LLAMA_PORT=8080 \
    N_GPU_LAYERS=99 \
    CTX_SIZE=32768 \
    PARALLEL=1 \
    LLAMA_STARTUP_TIMEOUT=900 \
    LLAMA_JOB_TIMEOUT=600 \
    LLAMA_PREWARM=0 \
    LLAMA_EXTRA_ARGS=""

WORKDIR /worker
COPY handler.py /worker/handler.py

# У базового образа ENTRYPOINT=/app/llama-server: его обязательно сбросить,
# иначе CMD допишется к нему как аргументы.
ENTRYPOINT []

# Унаследованный HEALTHCHECK бьёт в :8080/health, который до первого задания
# (при LLAMA_PREWARM=0) не поднят — для воркера это бессмысленный сигнал.
# Готовность воркера RunPod определяет сам, по исходящим пингам SDK.
HEALTHCHECK NONE
CMD ["python", "-u", "/worker/handler.py"]
