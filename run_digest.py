from __future__ import annotations

import os
import time

from google.genai.errors import APIError

import digest


# Gemini can temporarily return 429 (rate limit) or 503 (high demand).
# Keep the digest logic unchanged and switch models only for the AI request.
DEFAULT_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
RETRYABLE_CODES = {429, 500, 502, 503, 504}


def models_queue() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", DEFAULT_MODELS[0]).strip()
    result: list[str] = []
    for model in (primary, *DEFAULT_MODELS):
        if model and model not in result:
            result.append(model)
    return result


def gemini_json_with_fallback(client, prompt):
    last_error: Exception | None = None

    for model in models_queue():
        for attempt in range(2):
            try:
                print(f"[ai] model={model} attempt={attempt + 1}", flush=True)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                print(f"[ai] model={model} success", flush=True)
                return digest.parse_json_object(response.text or "")
            except APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code not in RETRYABLE_CODES:
                    raise
                if attempt == 0:
                    print(f"[ai] model={model} error={code}; retrying once", flush=True)
                    time.sleep(1)
                else:
                    print(f"[ai] model={model} error={code}; switching to fallback", flush=True)
                    break

    raise RuntimeError(f"Gemini: all fallback models unavailable; last error: {last_error}") from last_error


digest.gemini_json = gemini_json_with_fallback

if __name__ == "__main__":
    digest.main()
