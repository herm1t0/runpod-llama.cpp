# Базовый образ с llama.cpp и CUDA
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Устанавливаем Python и huggingface_hub
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install huggingface_hub && \
    rm -rf /var/lib/apt/lists/*

# Аргументы сборки: репозиторий, файл модели и токен (если нужен)
ARG HF_REPO="HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
ARG HF_FILE="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
ARG HF_TOKEN="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"

# Скачиваем модель внутрь образа
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='${HF_REPO}', filename='${HF_FILE}', local_dir='/models', token='${HF_TOKEN}' if '${HF_TOKEN}' else None)"

# Переменные окружения
ENV MODEL_PATH="/models/${HF_FILE}"
ENV PORT=8080

# Открываем порт
EXPOSE 8080

# Запуск сервера
CMD ["sh", "-c", "llama-server --model $MODEL_PATH --host 0.0.0.0 --port $PORT --n-gpu-layers 99"]
