#!/usr/bin/env python3
"""RunPod Serverless (Queue) worker, проксирующий задания в локальный llama-server.

llama-server поднимается как дочерний процесс при первом задании (или в
фоне при LLAMA_PREWARM=1) и слушает только 127.0.0.1 — для Queue-эндпоинта
worker общается с RunPod исходящими HTTP-запросами, входящий порт не нужен.

Формат задания:
    {"input": {"messages": [...], "path": "/v1/chat/completions", "timeout": 600}}

`path` (по умолчанию /v1/chat/completions) и всё остальное содержимое `input`
без изменений уходят в HTTP API llama-server, его JSON-ответ возвращается как
результат задания.
"""

import os
import shlex
import subprocess
import threading
import time

import requests
import runpod

LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "/app/llama-server")
MODEL_PATH = os.environ["MODEL_PATH"]
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
STARTUP_TIMEOUT = float(os.environ.get("LLAMA_STARTUP_TIMEOUT", "900"))
JOB_TIMEOUT = float(os.environ.get("LLAMA_JOB_TIMEOUT", "600"))

_proc = None
_lock = threading.Lock()

def _log(message):
    print(message, flush=True)

def _server_args():
    args = [
        LLAMA_SERVER_BIN,
        "--model", MODEL_PATH,
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "--n-gpu-layers", os.environ.get("N_GPU_LAYERS", "99"),
        "--ctx-size", os.environ.get("CTX_SIZE", "32768"),
        "--parallel", os.environ.get("PARALLEL", "1"),
        "--jinja",
    ]
    extra = os.environ.get("LLAMA_EXTRA_ARGS", "").strip()
    if extra:
        args += shlex.split(extra)
    return args

def _wait_until_healthy(proc):
    """llama-server отдаёт 503 на /health, пока грузит модель."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with code {proc.returncode}"
            )
        try:
            if requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                _log("llama-server is ready")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"llama-server did not become healthy in {STARTUP_TIMEOUT}s")

def _ensure_server():
    """Поднять llama-server, если он ещё не запущен, и дождаться готовности."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _wait_until_healthy(_proc)
            return
        if _proc is not None:
            _log(f"llama-server exited with code {_proc.returncode}, restarting")
            _proc = None

        args = _server_args()
        _log("starting: " + " ".join(args))
        # stdout/stderr не перехватываем — логи llama-server идут в stdout контейнера.
        _proc = subprocess.Popen(args)
        _wait_until_healthy(_proc)

def handler(job):
    _ensure_server()

    job_input = dict(job.get("input") or {})
    path = job_input.pop("path", "/v1/chat/completions")
    timeout = float(job_input.pop("timeout", JOB_TIMEOUT))
    if not path.startswith("/"):
        path = "/" + path

    if job_input.get("stream"):
        return {"error": 'streaming is not supported here, set "stream": false'}

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
        # fitness-тестом SDK — включайте, только если есть запас по VRAM.
        threading.Thread(target=_prewarm, daemon=True).start()

    runpod.serverless.start({"handler": handler})
