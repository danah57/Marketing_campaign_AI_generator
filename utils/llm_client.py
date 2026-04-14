"""Shared Anthropic client helper."""

import json
import os

import anthropic


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def call_claude(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {str(e)}") from e

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    cleaned_text = _strip_code_fences(raw_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {raw_text}") from e
