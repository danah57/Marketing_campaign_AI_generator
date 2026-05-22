"""Runtime flags for LLM behavior (mock vs real, verbose debug)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger("campaign_model.llm")

# Values treated as unset / placeholder API keys (case-insensitive).
_PLACEHOLDER_API_KEYS = frozenset(
    {
        "",
        "your_key_here",
        "sk-placeholder",
        "placeholder",
        "changeme",
        "xxx",
        "none",
        "null",
    }
)

# Set when Claude returns 401 / invalid key so the rest of the pipeline uses mocks.
_force_mock_after_auth_failure: bool = False


def reset_llm_runtime_state() -> None:
    """Reset per-request flags (call at the start of each pipeline run)."""
    global _force_mock_after_auth_failure
    _force_mock_after_auth_failure = False


def enable_mock_fallback_after_auth_failure(reason: str) -> None:
    """After an Anthropic auth failure, subsequent stages use mock output."""
    global _force_mock_after_auth_failure
    if _force_mock_after_auth_failure:
        return
    _force_mock_after_auth_failure = True
    logger.warning(
        "Anthropic API key rejected (401) — using mock LLM for the rest of this request."
    )


def is_claude_auth_error(exc: BaseException) -> bool:
    """True for 401 / invalid API key errors from Claude clients."""
    if type(exc).__name__ == "AuthenticationError":
        return True
    msg = str(exc).lower()
    if "authentication_error" in msg or "invalid x-api-key" in msg:
        return True
    if "401" in msg and ("anthropic" in msg or "claude" in msg or "api call failed" in msg):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return is_claude_auth_error(cause)
    return False


def _invalid_api_key_format(key: str) -> bool:
    """Reject keys that are clearly not Anthropic (e.g. Google/OpenAI pasted by mistake)."""
    low = key.lower()
    if low.startswith("aiza"):  # Google Gemini (AIzaSy...)
        return True
    if low.startswith("sk-proj"):  # OpenAI
        return True
    if low.startswith("sk-") and not low.startswith("sk-ant"):
        return True
    return False


def anthropic_api_key_configured() -> bool:
    """True when a non-placeholder Anthropic API key is present."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key or len(key) < 16:
        return False
    lowered = key.lower()
    if lowered in _PLACEHOLDER_API_KEYS:
        return False
    if lowered.startswith("your_") or lowered.endswith("_here"):
        return False
    if _invalid_api_key_format(key):
        logger.warning(
            "ANTHROPIC_API_KEY does not look like an Anthropic key (expected sk-ant-...). "
            "Using mock LLM. Set USE_MOCK_LLM=true to silence this warning."
        )
        return False
    if not key.startswith("sk-ant-"):
        logger.warning(
            "ANTHROPIC_API_KEY format unrecognized; expected sk-ant-api.... Using mock LLM."
        )
        return False
    return True


def use_mock_llm_explicit() -> bool:
    """True when USE_MOCK_LLM is explicitly enabled."""
    v = os.getenv("USE_MOCK_LLM", "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def mock_llm_auth_fallback() -> bool:
    """True when mock mode was enabled mid-request due to Claude 401."""
    return _force_mock_after_auth_failure


def mock_llm_auto_fallback() -> bool:
    """True when mock is on because no key, or auth failed this request."""
    if mock_llm_auth_fallback():
        return True
    return not use_mock_llm_explicit() and not anthropic_api_key_configured()


def use_mock_llm() -> bool:
    """When true, stages use `_mock_output` instead of calling Claude.

    Enabled if USE_MOCK_LLM=true, ANTHROPIC_API_KEY is missing/placeholder,
    or Claude returned an authentication error earlier in this request.
    """
    if use_mock_llm_explicit():
        return True
    if _force_mock_after_auth_failure:
        return True
    return not anthropic_api_key_configured()


def debug_llm() -> bool:
    """When true, log full prompts and full raw model text (very verbose)."""
    v = os.getenv("DEBUG_LLM", "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def run_llm_stage(
    stage_label: str,
    mock_fn: Callable[[dict, dict, str], dict],
    real_fn: Callable[[dict, dict, str], dict],
    brief_dict: dict,
    context: dict,
    job_id: str,
) -> dict:
    """Run a stage with mock/real branching and silent fallback after Claude 401."""
    if use_mock_llm():
        return mock_fn(brief_dict, context, job_id)
    try:
        return real_fn(brief_dict, context, job_id)
    except (ValueError, RuntimeError) as e:
        if use_mock_llm():
            logger.warning("%s: using mock output (Claude auth unavailable).", stage_label)
            return mock_fn(brief_dict, context, job_id)
        logger.error("%s failed: %s", stage_label, e)
        raise RuntimeError(f"{stage_label} failed — AI response error: {e}") from e
