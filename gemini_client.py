from __future__ import annotations

import os
import time
from collections.abc import Callable

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Models confirmed available in the user's Gemini API project.
DEFAULT_MODELS = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)

# 404 is included because a model can be unavailable for a particular account.
RETRYABLE_API_CODES = {404, 429, 500, 502, 503, 504}

# A single Gemini request must never be allowed to hang until GitHub Actions
# kills the whole job. The SDK timeout is in milliseconds.
REQUEST_TIMEOUT_MS = 45_000
MAX_ATTEMPTS_PER_MODEL = 2
RETRY_DELAY_SECONDS = 2


class GeminiUnavailable(RuntimeError):
    pass


def model_queue() -> list[str]:
    configured = os.getenv("GEMINI_MODEL", "").strip()
    result: list[str] = []
    for model in (configured, *DEFAULT_MODELS):
        if model and model not in result:
            result.append(model)
    return result


def create_client(api_key: str) -> genai.Client:
    """Create a Gemini client with a hard HTTP timeout and no SDK retries."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _is_retryable_api_error(exc: APIError) -> bool:
    return getattr(exc, "code", None) in RETRYABLE_API_CODES


def generate_json(
    client: genai.Client,
    prompt: str,
    parse: Callable[[str], dict],
) -> dict:
    """Call Gemini with bounded retries and model fallback.

    Retryable service/network failures never escape until every configured
    model has been tried. Non-transient API errors fail immediately because
    retrying them would only waste the free-tier quota.
    """
    last_error: BaseException | None = None

    for model in model_queue():
        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                print(f"[ai] model={model} attempt={attempt}", flush=True)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                text = response.text or ""
                result = parse(text)
                print(f"[ai] model={model} success", flush=True)
                return result
            except APIError as exc:
                last_error = exc
                if not _is_retryable_api_error(exc):
                    raise
                code = getattr(exc, "code", "unknown")
                if attempt < MAX_ATTEMPTS_PER_MODEL:
                    print(f"[ai] model={model} error={code}; retrying", flush=True)
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"[ai] model={model} error={code}; fallback", flush=True)
            except httpx.HTTPError as exc:
                # Includes RemoteProtocolError, ConnectError, ConnectTimeout,
                # ReadTimeout, WriteError and PoolTimeout.
                last_error = exc
                if attempt < MAX_ATTEMPTS_PER_MODEL:
                    print(
                        f"[ai] model={model} network={type(exc).__name__}; retrying",
                        flush=True,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(
                        f"[ai] model={model} network={type(exc).__name__}; fallback",
                        flush=True,
                    )

    raise GeminiUnavailable(
        f"All Gemini models unavailable; last error: {last_error}"
    ) from last_error
