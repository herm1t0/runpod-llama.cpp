# Runpod Serverless worker (Load Balancing) для Ollama с вшитой моделью
#
# Runtime-образ = апстримный ollama/ollama + одна импортированная модель. Больше ничего:
# ни Python, ни runpod SDK, ни прокси — HTTP API самого Ollama и есть эндпоинт.

# ---------------------------------------------------------------- builder ----
# Отдельная стадия: скачиваем GGUF, импортируем его в хранилище Ollama и переносим
# в runtime ровно один чистый каталог /models — без исходного .gguf и мусора сборки.
FROM ollama/ollama:latest AS model-builder

# Точный файл модели — прямым resolve-URL, а не ссылкой вида hf.co/<repo>:<quant>.
# В hf.co-синтаксисе тэг Ollama выводит эвристикой из имени файла, а здесь имя репозитория
# (...-Aggressive-MTP-GGUF) и имя файла (...-Aggressive-Q4_K_P.gguf, без MTP) не совпадают:
# предсказать тэг нельзя. Прямой URL даёт ровно запрошенный файл и падает с 404, если путь неверен.
ARG GGUF_URL="https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"

# Имя, под которым модель видна в API: model = "qwen3.8-27b-uncensored".
ARG OLLAMA_MODEL=qwen3.8-27b-uncensored

# Необязательные дополнительные модели из реестра ollama.com, через пробел. Пусто = ничего.
ARG EXTRA_MODELS=""

ENV OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/models

# Всё в одном RUN. Это не стилистика: скачанный .gguf весит ~16 GB, и если удалять его
# отдельным слоем, веса останутся в промежуточном слое и образ вырастет почти вдвое.
RUN set -eu; \
    command -v curl >/dev/null 2>&1 || { \
        apt-get update; \
        apt-get install -y --no-install-recommends curl ca-certificates; \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    curl -fL -C - --retry 3 --retry-delay 5 -o /tmp/model.gguf "${GGUF_URL}"; \
    echo 'FROM /tmp/model.gguf' > /tmp/Modelfile; \
    ollama serve >/tmp/ollama-serve.log 2>&1 & \
    server_pid="$!"; \
    tries=0; \
    until ollama list >/dev/null 2>&1; do \
        tries=$((tries + 1)); \
        if [ "$tries" -gt 120 ]; then \
            echo "ollama serve не поднялся, лог:" >&2; \
            cat /tmp/ollama-serve.log >&2; \
            exit 1; \
        fi; \
        sleep 1; \
    done; \
    echo "=== импорт ${OLLAMA_MODEL}"; \
    ollama create "${OLLAMA_MODEL}" -f /tmp/Modelfile; \
    for model in ${EXTRA_MODELS}; do \
        echo "=== pulling ${model}"; \
        ollama pull "${model}"; \
    done; \
    echo "=== метаданные импортированной модели"; \
    ollama show "${OLLAMA_MODEL}" | head -n 40; \
    ollama list; \
    kill -TERM "${server_pid}" 2>/dev/null || true; \
    wait "${server_pid}" 2>/dev/null || true; \
    rm -f /tmp/model.gguf /tmp/Modelfile /tmp/ollama-serve.log

# ---------------------------------------------------------------- runtime ----
FROM ollama/ollama:latest

ARG OLLAMA_MODEL=qwen3.8-27b-uncensored

# OLLAMA_HOST=0.0.0.0 обязателен: Runpod проксирует запросы в контейнер снаружи.
# KEEP_ALIVE=-1 + MAX_LOADED_MODELS=1 держат модель в VRAM между запросами —
# для serverless это разница между "мгновенно" и "перезагрузка 16 GB в VRAM на каждый вызов".
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/models \
    OLLAMA_MODEL=${OLLAMA_MODEL} \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_NUM_PARALLEL=1

COPY --from=model-builder /models /models

# Падаем на сборке, если веса не доехали, а не на первом запросе в проде.
RUN set -eu; \
    test -d /models/blobs || { echo "/models/blobs отсутствует — COPY не сработал" >&2; exit 1; }; \
    test -d /models/manifests || { echo "/models/manifests отсутствует — COPY не сработал" >&2; exit 1; }; \
    du -sh /models

EXPOSE 11434

# Runpod LB-эндпоинт использует собственный health-check (GET /); это для docker run локально.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD OLLAMA_HOST=127.0.0.1:11434 ollama list >/dev/null 2>&1 || exit 1
