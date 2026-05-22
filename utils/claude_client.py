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


def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> dict:
    if not anthropic_api_key_configured():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Stages should use mock mode "
            "(set USE_MOCK_LLM=true or provide a valid API key)."
        )

    client = _get_client()
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.RateLimitError as e:
        logger.warning("Rate limit hit — retrying once after 2 seconds.")
        time.sleep(2)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.RateLimitError as retry_e:
            raise RuntimeError(f"Claude rate limit exceeded after retry: {retry_e}") from retry_e
    except anthropic.AuthenticationError as e:
        enable_mock_fallback_after_auth_failure(str(e))
        raise RuntimeError(f"Claude authentication failed: {e}") from e
    except anthropic.APITimeoutError as e:
        raise RuntimeError(f"Claude API timed out after {_anthropic_timeout_seconds()}s: {e}") from e
    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        raise RuntimeError(f"Claude API call failed: {e}") from e

    raw_text = response.content[0].text
    clean_text = _strip_markdown_json_fences(raw_text)
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"Claude returned invalid JSON. Raw text: {raw_text!r}")
        raise ValueError(f"Claude returned invalid JSON: {e}") from e
