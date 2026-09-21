# RunPod Serverless (Queue) worker для GGUF-моделей

Минимальный образ: `llama.cpp` с CUDA собирается из исходников (multi-stage), в финальный
слой запекается GGUF. Задания приходят через Queue-эндпоинт и проксируются в локальный
`llama-server` (OpenAI-совместимый HTTP API на `127.0.0.1:8080`).

## Файлы

| Файл | Назначение |
|---|---|
| `Dockerfile` | builder (llama.cpp + CUDA + патч FastMTP) и runtime (CUDA runtime + venv + `llama-server` + GGUF) |
| `handler.py` | RunPod-обработчик: поднимает `llama-server`, проксирует задания |
| `.dockerignore` | держит контекст сборки крошечным (в контексте нужен только `handler.py`) |

## Сборка

BuildKit обязателен (`--mount=type=secret`), `--platform linux/amd64` — тоже: RunPod
работает только на amd64.

```bash
docker buildx build --platform linux/amd64 \
  -t <registry>/gguf-worker:v1 --push .
```

Приватный HF-репозиторий — токен только секретом, `--build-arg` оседает в `docker history`:

```bash
docker buildx build --platform linux/amd64 \
  --secret id=hf_token,src=hf_token.txt \
  -t <registry>/gguf-worker:v1 --push .
```

Основные build-arg:

| ARG | Значение по умолчанию | Зачем менять |
|---|---|---|
| `HF_REPO` / `HF_FILE` | `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` / `…-Q4_K_P.gguf` | другой квант или другая модель |
| `HF_DRAFT_FILE` | `…-FastMTP-32K.gguf` | сайдкар FastMTP (903 МБ) |
| `CUDA_ARCHITECTURES` | `80;86;89;90;100;120` | сузить под свои GPU — каждая арка это +5..10 мин сборки |
| `LLAMA_CPP_COMMIT` | `4df29be4f4c3673f428170fda944a5b19f743bb8` | только вместе с `FASTMTP_PATCH_URL` |

## Деплой

1. Запушить образ в registry, доступный из RunPod (для приватного — выдать креды в
   настройках RunPod).
2. Создать Serverless Endpoint: тип **Queue**, нужный GPU, Container Image = свой образ,
   Container Disk — с запасом (веса лежат в образе, но `llama-server` грузит их в RAM/VRAM).
3. Env эндпоинта переопределяют значения из образа (см. таблицу ниже).
4. Cold start = загрузка 19 ГБ весов с диска + прогрев CUDA-ядер. Уменьшайте простой
   через Active Workers, а не через `LLAMA_PREWARM` (см. ниже).

## Запрос

```bash
curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"Привет"}],"max_tokens":64}}'
```

Другие эндпоинты llama-server доступны через `path`:

```json
{"input": {"path": "/v1/embeddings", "input": ["текст"], "model": "local"}}
```

Весь остальной `input` уходит в llama-server без изменений, поэтому работают любые
параметры сэмплинга, `chat_template_kwargs`, `stop`, `json_schema` и т.д.

## Env

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `CTX_SIZE` | `16384` | контекст; главный потребитель VRAM после весов |
| `N_GPU_LAYERS` | `all` | `all` / `auto` / число |
| `PARALLEL` | `1` | число слотов; должно совпадать с конкурентностью эндпоинта |
| `LLAMA_MTP` | `1` | `0` — отключить FastMTP без пересборки |
| `SPEC_DRAFT_N_MAX` / `SPEC_DRAFT_P_MIN` | `3` / `0` | глубина MTP и порог принятия |
| `LLAMA_EXTRA_ARGS` | `--flash-attn on --no-mmap --batch-size 2048 --ubatch-size 512` | любые дополнительные флаги `llama-server` |
| `LLAMA_STARTUP_TIMEOUT` / `LLAMA_JOB_TIMEOUT` | `1800` / `900` | сек |
| `LLAMA_PREWARM` | `0` | `1` — грузить модель на старте воркера |

## Что учтено

- **Коммит llama.cpp зафиксирован.** Патч FastMTP меняет `src/models/qwen35.cpp` и привязан
  к конкретному SHA; шаг `git apply --check` в builder уронит сборку, если контекст
  разъедется. Апстрим на этом коммите уже содержит все нужные флаги
  (`--spec-type draft-mtp`, `--spec-draft-model`, `--spec-draft-ngl`, `-fa on|off|auto`,
  `-ngl all`) — патч добавляет только поддержку усечённого draft-словаря через `d2t`.
- **`GGML_NATIVE=OFF`.** Иначе host-код собирается под CPU сборочной машины и падает на
  воркере с `illegal instruction`.
- **Модель лежит в последнем слое**, поэтому правка любого ENV/ARG выше не вызывает
  повторную загрузку 19 ГБ.
- **venv вместо системного python** — в ubuntu 24.04 срабатывает PEP 668.
- **Два файла на диске:** `MODEL_PATH` (`/models/model.gguf`) и `SPEC_DRAFT_MODEL`
  (`/models/draft.gguf`), имена фиксированы, так что смена кванта не трогает handler.

## Чего в образе нет

- **Vision.** Проектор `mmproj-…-BF16.gguf` (931 МБ) не скачивается. Если нужен
  image/video-вход — добавьте его в `HF_FILE`-подобный ARG и `--mmproj` в `handler.py`.
- **Стриминга.** Handler возвращает ответ целиком: для Queue-эндпоинта генераторный handler
  меняет семантику `/runsync` (нужен `return_aggregate_stream`), поэтому осознанно не включён.
  Клиентам с потоковой выдачей нужен Load-Balancing-эндпоинт.
- **Проверки SHA-256 весов.** В репозитории модели есть подписанный
  `HauhauCS-RELEASE-MANIFEST.json` с хешами — при желании добавьте сверку в шаг загрузки.

## Ограничения

- Образ весит ~22–23 ГБ (19 ГБ весов + CUDA runtime + venv). Это осознанный выбор в пользу
  мгновенного старта: альтернатива — Network Volume с весами и `huggingface_hub` в рантайме.
- Формат `Q*_K_P` — кастомные кванты HauhauCS. Это обычные GGUF, специальный билд не нужен,
  но в UI могут отображаться как `?`.
- `--no-mmap` из `LLAMA_EXTRA_ARGS` держит веса в RAM контейнера (~19–20 ГБ) — если у воркера
  мало системной памяти, уберите флаг.
- VRAM: веса 17.9 + 0.9 ГБ, дальше всё решает KV-кэш, а он у этой архитектуры (48 Gated
  DeltaNet-слоёв + 16 attention) заметно меньше, чем у «обычной» 27B, но точную цифру нужно
  мерить на своём `CTX_SIZE`. `CTX_SIZE=16384` — осторожный дефолт, не измеренная величина.
  Точкой отсчёта из README модели служат `204800` контекста на RTX PRO 6000 96 ГБ.
