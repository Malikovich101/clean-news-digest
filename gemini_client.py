from __future__ import annotations

import os
import time
from collections.abc import Callable

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

# For this workload (ad filtering + duplicate detection), Flash-Lite is the
# sensible primary model: fast, inexpensive and designed for high-throughput.
# The order also reflects the free-tier quotas visible in the user's AI Studio.
DEFAULT_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
)

# 404 can mean that a model is unavailable for the current account/project.
RETRYABLE_API_CODES = {404, 429, 500, 502, 503, 504}

# One request has a hard 30-second network deadline. We do not retry the same
# model: the next model is the retry. This keeps the whole digest bounded and
# avoids burning free-tier quota on repeated calls to a failing model.
REQUEST_TIMEOUT_MS = 30_000
RETRY_DELAY_SECONDS = 1


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
    """Create a Gemini client with a hard timeout and no SDK retries."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def generate_json(
    client: genai.Client,
    prompt: str,
    parse: Callable[[str], dict],
) -> dict:
    """Call Gemini with bounded model fallback."""
    last_error: BaseException | None = None

    for index, model in enumerate(model_queue()):
        try:
            print(f"[ai] model={model} attempt=1", flush=True)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            result = parse(response.text or "")
            print(f"[ai] model={model} success", flush=True)
            return result
        except APIError as exc:
            last_error = exc
            code = getattr(exc, "code", "unknown")
            if code not in RETRYABLE_API_CODES:
                raise
            if index < len(model_queue()) - 1:
                print(f"[ai] model={model} error={code}; fallback", flush=True)
                time.sleep(RETRY_DELAY_SECONDS)
        except httpx.HTTPError as exc:
            # Includes RemoteProtocolError, ConnectError, ConnectTimeout,
            # ReadTimeout, WriteError and PoolTimeout.
            last_error = exc
            if index < len(model_queue()) - 1:
                print(
                    f"[ai] model={model} network={type(exc).__name__}; fallback",
                    flush=True,
                )
                time.sleep(RETRY_DELAY_SECONDS)

    raise GeminiUnavailable(
        f"All Gemini models unavailable; last error: {last_error}"
    ) from last_error
