from __future__ import annotations

import os
import time
from collections.abc import Callable

from google import genai
from google.genai.errors import APIError

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

DEFAULT_MODELS = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
RETRYABLE_API_CODES = {429, 500, 502, 503, 504}
RETRYABLE_HTTP_ERRORS = tuple(
    exc
    for exc in (
        getattr(httpx, "RemoteProtocolError", None) if httpx else None,
        getattr(httpx, "ConnectError", None) if httpx else None,
        getattr(httpx, "ReadTimeout", None) if httpx else None,
        getattr(httpx, "WriteError", None) if httpx else None,
        getattr(httpx, "PoolTimeout", None) if httpx else None,
    )
    if exc is not None
)


class GeminiUnavailable(RuntimeError):
    pass


def model_queue() -> list[str]:
    configured = os.getenv("GEMINI_MODEL", "").strip()
    values = [configured, *DEFAULT_MODELS]
    result: list[str] = []
    for model in values:
        if model and model not in result:
            result.append(model)
    return result


def generate_json(
    client: genai.Client,
    prompt: str,
    parse: Callable[[str], dict],
) -> dict:
    last_error: BaseException | None = None

    for model in model_queue():
        for attempt in (1, 2):
            try:
                print(f"[ai] model={model} attempt={attempt}", flush=True)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                print(f"[ai] model={model} success", flush=True)
                return parse(response.text or "")
            except APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code not in RETRYABLE_API_CODES:
                    raise
                if attempt == 1:
                    print(f"[ai] model={model} error={code}; retrying", flush=True)
                    time.sleep(2)
                else:
                    print(f"[ai] model={model} error={code}; fallback", flush=True)
            except RETRYABLE_HTTP_ERRORS as exc:
                last_error = exc
                if attempt == 1:
                    print(f"[ai] model={model} network={type(exc).__name__}; retrying", flush=True)
                    time.sleep(2)
                else:
                    print(f"[ai] model={model} network={type(exc).__name__}; fallback", flush=True)

    raise GeminiUnavailable(f"All Gemini models unavailable; last error: {last_error}") from last_error
