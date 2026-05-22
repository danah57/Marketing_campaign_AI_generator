"""Anthropic Claude (Haiku) — JSON responses for pipeline stages."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterator

import anthropic

from utils.llm_runtime import debug_llm, enable_mock_fallback_after_auth_failure

logger = logging.getLogger("campaign_model.llm")


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def _model_id() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def _max_tokens() -> int:
    return int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))


def _anthropic_timeout_seconds() -> float:
    return float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "90"))


def call_claude(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment.")

    model = _model_id()
    max_tokens = _max_tokens()
    logger.info(
        "LLM JSON request: provider=anthropic model=%s temperature=%s max_tokens=%s system_chars=%s user_chars=%s",
        model,
        temperature,
        max_tokens,
        len(system_prompt),
        len(user_prompt),
    )
    if debug_llm():
        logger.info("DEBUG_LLM system_prompt:\n%s", system_prompt)
        logger.info("DEBUG_LLM user_prompt:\n%s", user_prompt)

    client = anthropic.Anthropic(api_key=key, timeout=_anthropic_timeout_seconds())
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APITimeoutError as e:
        raise RuntimeError(
            f"LLM API timed out after {_anthropic_timeout_seconds()}s: {e}"
        ) from e
    except anthropic.AuthenticationError as e:
        enable_mock_fallback_after_auth_failure(str(e))
        raise RuntimeError(f"Claude authentication failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {str(e)}") from e

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    preview = raw_text[:2000] + ("..." if len(raw_text) > 2000 else "")
    logger.info(
        "LLM raw response: provider=anthropic model=%s chars=%s preview=%r",
        model,
        len(raw_text),
        preview,
    )
    if debug_llm():
        logger.info("DEBUG_LLM full raw response:\n%s", raw_text)

    cleaned = _strip_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            "LLM JSON parse failed: %s | cleaned_preview=%r",
            e,
            cleaned[:1500],
            exc_info=True,
        )
        raise ValueError(f"LLM returned invalid JSON: {raw_text}") from e


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    return call_claude(system_prompt, user_prompt, temperature=temperature)


def stream_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
) -> Iterator[str]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment.")
    model = _model_id()
    max_tokens = _max_tokens()
    client = anthropic.Anthropic(api_key=key, timeout=_anthropic_timeout_seconds())
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
