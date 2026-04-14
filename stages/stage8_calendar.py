"""Stage 8: Content Calendar — schedule assembly and timeline output."""

from utils.llm_client import call_claude

USE_MOCK = True


def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
    return {"total_days": 0, "start_date": "pending", "days": []}


def _real_output(brief: dict, context: dict) -> dict:
    _ = call_claude
    raise NotImplementedError("Real LLM call not yet implemented for this stage")
