#!/usr/bin/env python3
"""RunPod Serverless (Queue) worker: задания проксируются в локальный llama-server.

llama-server поднимается дочерним процессом при первом задании (или в фоне при
LLAMA_PREWARM=1) и слушает только 127.0.0.1: для Queue-эндпоинта входящий порт не
нужен, воркер сам общается с RunPod исходящими запросами.

Формат задания:
    {"input": {"messages": [...], "path": "/v1/chat/completions", "timeout": 900}}

`path` (по умолчанию /v1/chat/completions) и весь остальной `input` уходят в HTTP API
llama-server как есть, а его JSON-ответ возвращается как результат задания. Так
доступны /v1/chat/completions, /v1/completions, /v1/embeddings, /tokenize, /props и т.д.

MTP-ускорение (HauhauCS FastMTP) включается автоматически, если запечённый
SPEC_DRAFT_MODEL на месте; выключается через LLAMA_MTP=0.
"""

import os
import shlex
import subprocess
import threading
import time

import requests
import runpod

LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "/app/llama-server")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.gguf")
SPEC_DRAFT_MODEL = os.environ.get("SPEC_DRAFT_MODEL", "/models/draft.gguf")
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
STARTUP_TIMEOUT = float(os.environ.get("LLAMA_STARTUP_TIMEOUT", "1800"))
JOB_TIMEOUT = float(os.environ.get("LLAMA_JOB_TIMEOUT", "900"))

_proc = None
_lock = threading.Lock()


def _log(message):
    print(message, flush=True)


def _mtp_enabled():
    if os.environ.get("LLAMA_MTP", "1") != "1":
        return False
    return bool(SPEC_DRAFT_MODEL) and os.path.isfile(SPEC_DRAFT_MODEL)


def _server_args():
    args = [
        LLAMA_SERVER_BIN,
        "--model", MODEL_PATH,
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "--n-gpu-layers", os.environ.get("N_GPU_LAYERS", "all"),
        "--ctx-size", os.environ.get("CTX_SIZE", "16384"),
        "--parallel", os.environ.get("PARALLEL", "1"),
        # Без --jinja сервер не подставит chat template из GGUF.
        "--jinja",
    ]
    if _mtp_enabled():
        # FastMTP: draft-сайдкар + llama.cpp с патчем HauhauCS (см. Dockerfile).
        args += [
            "--spec-draft-model", SPEC_DRAFT_MODEL,
            "--spec-draft-ngl", "all",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", os.environ.get("SPEC_DRAFT_N_MAX", "3"),
            "--spec-draft-p-min", os.environ.get("SPEC_DRAFT_P_MIN", "0"),
        ]
    extra = os.environ.get("LLAMA_EXTRA_ARGS", "").strip()
    if extra:
        args += shlex.split(extra)
    return args


def _healthy(timeout=2):
    """llama-server отдаёт 503 на /health, пока грузит модель."""
    try:
        return requests.get(f"{BASE_URL}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _wait_until_healthy(proc):
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with code {proc.returncode}"
            )
        if _healthy():
            _log("llama-server is ready")
            return
        time.sleep(1)
    raise TimeoutError(f"llama-server did not become healthy in {STARTUP_TIMEOUT}s")


def _ensure_server():
    """Поднять llama-server, если он не запущен, умер или перестал отвечать."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None and _healthy():
            return
        if _proc is not None:
            _log(f"llama-server (pid {_proc.pid}) is not healthy, restarting")
            _proc.kill()
            _proc.wait()
            _proc = None

        args = _server_args()
        _log("starting: " + " ".join(args))
        # stdout/stderr не перехватываем -- логи llama-server идут в stdout контейнера.
        _proc = subprocess.Popen(args)
        _wait_until_healthy(_proc)


def handler(job):
    job_input = dict(job.get("input") or {})
    path = job_input.pop("path", "/v1/chat/completions")
    timeout = float(job_input.pop("timeout", JOB_TIMEOUT))
    if not path.startswith("/"):
        path = "/" + path

    if job_input.get("stream"):
        return {"error": 'streaming is not supported here, set "stream": false'}

    # Модель грузится лениво: некорректный input не должен запускать cold start.
    _ensure_server()

    response = requests.post(
        f"{BASE_URL}{path}", json=job_input or {}, timeout=timeout
    )
    try:
        body = response.json()
    except ValueError:
        body = {"text": response.text}

    if response.status_code >= 400:
        return {"error": f"llama-server returned HTTP {response.status_code}", "detail": body}
    return body


def _prewarm():
    try:
        _ensure_server()
    except Exception as exc:  # noqa: BLE001 - прогрев не должен ломать воркер
        _log(f"prewarm failed: {exc!r}")


if __name__ == "__main__":
    if os.environ.get("LLAMA_PREWARM", "0") == "1":
        # Быстрее первое задание, но загрузка модели идёт параллельно с GPU
        # fitness-тестом SDK -- включайте, только если есть запас по VRAM.
        threading.Thread(target=_prewarm, daemon=True).start()

    runpod.serverless.start({"handler": handler})
