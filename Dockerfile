# syntax=docker/dockerfile:1.7

# =============================================================================
#  RunPod Serverless (Queue) worker: GGUF + llama.cpp/CUDA + HauhauCS FastMTP
# =============================================================================
#  stage 1 (builder): llama.cpp собирается из исходников с CUDA и патчем FastMTP
#  stage 2 (runtime): CUDA runtime + venv(python) + llama-server + запечённые GGUF
#
#  Сборка (нужен BuildKit -- `--mount=type=secret` ниже):
#    docker buildx build --platform linux/amd64 \
#      -t <registry>/gguf-worker:v1 --push .
#
#  Приватный HF-репозиторий -- токен секретом, ARG оседает в `docker history`:
#    docker buildx build --secret id=hf_token,src=hf_token.txt ...
# =============================================================================

# ---------- 1/2: builder -----------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

# Патч FastMTP правит src/models/qwen35.cpp и привязан к конкретному коммиту:
# LLAMA_CPP_COMMIT и FASTMTP_PATCH_URL меняются только вместе (`git apply --check`
# ниже уронит сборку, если контекст перестал совпадать).
ARG LLAMA_CPP_COMMIT=4df29be4f4c3673f428170fda944a5b19f743bb8
ARG FASTMTP_PATCH_URL=https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch

# sm_100/sm_120 (Blackwell) требуют CUDA >= 12.8. Каждая арка -- отдельный прогон
# компиляции CUDA-ядер (+5..10 мин на сборку), так что список стоит сузить под свои GPU.
ARG CUDA_ARCHITECTURES=80;86;89;90;100;120

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# --filter=blob:none: история и блобы не нужны, но checkout произвольного SHA работает
# (коммит должен быть достижим из refs репозитория -- как в инструкции к модели).
RUN git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git /src/llama.cpp \
 && git -C /src/llama.cpp checkout --detach "${LLAMA_CPP_COMMIT}" \
 && curl -fsSL -o /tmp/fastmtp.patch "${FASTMTP_PATCH_URL}" \
 && git -C /src/llama.cpp apply --check /tmp/fastmtp.patch \
 && git -C /src/llama.cpp apply /tmp/fastmtp.patch

# BUILD_SHARED_LIBS=OFF: llama-server самодостаточен, в рантайм копируется один файл,
# LD_LIBRARY_PATH не нужен.
# GGML_NATIVE=OFF обязателен: иначе host-код соберётся под CPU сборочной машины и на
# воркере упадёт с illegal instruction.
# (libcurl отключить нечем и не нужно: на этом коммите опция LLAMA_CURL удалена вместе
#  с зависимостью -- серверу curl не требуется, модель запечена в образ.)
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
 && cmake --build /src/llama.cpp/build --target llama-server -j"$(nproc)"

# ---------- 2/2: runtime -----------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

# hf_transfer не ставим: huggingface_hub 1.x уже тянет hf-xet (штатный быстрый путь Xet).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# python3-venv: в ubuntu 24.04 системный python помечен externally-managed (PEP 668),
# `pip3 install ...` в него падает с "error: externally-managed-environment".
# libgomp1 -- OpenMP-часть CPU-бэкенда ggml.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install "runpod>=1.7" requests "huggingface_hub>=0.30"
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /src/llama.cpp/build/bin/llama-server /app/llama-server

# Запускать бинарник на этапе сборки нельзя: ggml-cuda линкует CUDA::cuda_driver
# (GGML_CUDA_NO_VMM=OFF по умолчанию), поэтому libcuda.so.1 оказывается в DT_NEEDED, а в
# сборочном контейнере драйвера NVIDIA нет -- он инжектится только при старте на GPU-хосте.
# Поэтому `llama-server --version` здесь упал бы с exit 127 "cannot open shared object file".
# Вместо запуска проверяем, что не разрешена ровно одна зависимость -- драйвер.
RUN set -e; \
    test -x /app/llama-server; \
    ldd /app/llama-server > /tmp/ldd.txt 2>&1 || true; \
    cat /tmp/ldd.txt; \
    unresolved="$(grep 'not found' /tmp/ldd.txt | grep -vE 'libcuda\.so\.1|libnvidia' || true)"; \
    if [ -n "$unresolved" ]; then echo "unresolved runtime libraries:"; echo "$unresolved"; exit 1; fi

WORKDIR /worker
COPY handler.py /worker/handler.py

# ---------- GGUF: запекаются в образ на этапе сборки -------------------------
ARG HF_REPO=HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
ARG HF_FILE=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
ARG HF_DRAFT_FILE=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf

# Фиксированные имена: смена кванта задевает только ARG, но не ENV и не handler.
ENV MODEL_PATH=/models/model.gguf \
    SPEC_DRAFT_MODEL=/models/draft.gguf

# ~19 ГБ, поэтому слой последний: любое изменение Dockerfile выше перекачивало бы модель.
RUN --mount=type=secret,id=hf_token,required=false \
    set -e; \
    if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python3 -c "import os,shutil;from huggingface_hub import hf_hub_download as dl;mv=lambda n,d:shutil.move(dl(repo_id=os.environ['HF_REPO'],filename=n,local_dir='/models'),d);mv(os.environ['HF_FILE'],os.environ['MODEL_PATH']);mv(os.environ['HF_DRAFT_FILE'],os.environ['SPEC_DRAFT_MODEL']);print('downloaded OK')"; \
    rm -rf /models/.cache; \
    test -s "${MODEL_PATH}"; \
    test -s "${SPEC_DRAFT_MODEL}"; \
    ls -l /models

# ---------- Параметры llama-server (переопределяются ENV эндпоинта) ----------
# Отдельным слоем после модели: правки этих значений не инвалидируют загрузку GGUF.
ENV LLAMA_SERVER_BIN=/app/llama-server \
    LLAMA_PORT=8080 \
    N_GPU_LAYERS=all \
    CTX_SIZE=16384 \
    PARALLEL=1 \
    LLAMA_MTP=1 \
    SPEC_DRAFT_N_MAX=3 \
    SPEC_DRAFT_P_MIN=0 \
    LLAMA_EXTRA_ARGS="--flash-attn on --no-mmap --batch-size 2048 --ubatch-size 512" \
    LLAMA_STARTUP_TIMEOUT=1800 \
    LLAMA_JOB_TIMEOUT=900 \
    LLAMA_PREWARM=0

# У базового образа проверено есть ENTRYPOINT=/opt/nvidia/nvidia_entrypoint.sh --
# сбрасываем, иначе CMD допишется к нему как аргументы.
ENTRYPOINT []
CMD ["python3", "-u", "/worker/handler.py"]
