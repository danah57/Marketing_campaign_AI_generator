"""Anthropic Claude Haiku client for structured JSON responses."""

from __future__ import annotations

import json
import logging
import os
import re
import time

import anthropic

from utils.llm_runtime import (
    anthropic_api_key_configured,
    enable_mock_fallback_after_auth_failure,
)

logger = logging.getLogger("campaign_model.llm")

_client: anthropic.Anthropic | None = None


def _anthropic_timeout_seconds() -> float:
    return float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "90"))


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "").strip() or None,
            timeout=_anthropic_timeout_seconds(),
        )
    return _client


def _strip_markdown_json_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _retry_delay_seconds(attempt: int) -> float:
    return float(2 * (attempt + 1))


def _is_retriable_api_error(error: anthropic.APIError) -> bool:
    status_code = getattr(error, "status_code", None)
    return status_code in {429, 500, 502, 503, 529}


def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> dict:
    if not anthropic_api_key_configured():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Stages should use mock mode "
            "(set USE_MOCK_LLM=true or provide a valid API key)."
        )

    client = _get_client()
    tokens = max_tokens
    last_error: json.JSONDecodeError | None = None
    last_raw_text = ""
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.AuthenticationError as e:
            enable_mock_fallback_after_auth_failure(str(e))
            raise RuntimeError(f"Claude authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            if attempt < max_attempts - 1:
                delay = _retry_delay_seconds(attempt)
                logger.warning(
                    "Claude rate limit hit on attempt %s/%s — retrying in %.0fs",
                    attempt + 1,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"Claude rate limit exceeded after {max_attempts} attempts: {e}"
            ) from e
        except anthropic.APITimeoutError as e:
            if attempt < max_attempts - 1:
                logger.warning(
                    "Claude API timed out on attempt %s/%s — retrying",
                    attempt + 1,
                    max_attempts,
                )
                continue
            raise RuntimeError(
                f"Claude API timed out after {_anthropic_timeout_seconds()}s: {e}"
            ) from e
        except anthropic.APIError as e:
            if attempt < max_attempts - 1 and _is_retriable_api_error(e):
                delay = _retry_delay_seconds(attempt)
                logger.warning(
                    "Retriable Claude API error on attempt %s/%s (status=%s) — retrying in %.0fs",
                    attempt + 1,
                    max_attempts,
                    getattr(e, "status_code", "unknown"),
                    delay,
                )
                time.sleep(delay)
                continue
            logger.error(f"Claude API error: {e}")
            raise RuntimeError(f"Claude API call failed: {e}") from e

        last_raw_text = response.content[0].text
        clean_text = _strip_markdown_json_fences(last_raw_text)
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            last_error = e
            stop_reason = getattr(response, "stop_reason", None)
            if attempt < max_attempts - 1 and stop_reason == "max_tokens" and tokens < 8192:
                next_tokens = min(tokens * 2, 8192)
                logger.warning(
                    "Claude JSON truncated at max_tokens=%s; retrying with max_tokens=%s",
                    tokens,
                    next_tokens,
                )
                tokens = next_tokens
                continue
            break

    logger.error(f"Claude returned invalid JSON. Raw text: {last_raw_text!r}")
    raise ValueError(f"Claude returned invalid JSON: {last_error}") from last_error
