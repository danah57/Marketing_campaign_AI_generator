import asyncio
import functools
import json as json_lib
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import agentpipeline as pipeline
from agentpipeline import CheckpointManager, checkpoint, _run_stage
from schemas import CampaignBrief, GenerateResponse
from stages import (
    stage2_business,
    stage3_competitors,
    stage4_audience,
    stage5_strategy,
    stage6_tactical,
    stage7b_influencer,
    stage7_evaluation,
    stage8_calendar,
)

load_dotenv()

from utils.llm_runtime import (
    anthropic_api_key_configured,
    debug_llm as _debug_llm_flag,
    mock_llm_auth_fallback,
    mock_llm_auto_fallback,
    reset_llm_runtime_state,
    use_mock_llm,
)

_log = logging.getLogger("campaign_model.llm")
_log.setLevel(logging.DEBUG if _debug_llm_flag() else logging.INFO)

PIPELINE_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "180"))
STAGE_TIMEOUT_SECONDS = int(os.getenv("STAGE_TIMEOUT_SECONDS", "120"))

app = FastAPI(title="Campaign Strategy Backend")


@app.on_event("startup")
def _startup_llm_mode() -> None:
    """Log LLM mode; auto-enable mock when ANTHROPIC_API_KEY is missing."""
    if use_mock_llm():
        if mock_llm_auto_fallback():
            _log.warning(
                "ANTHROPIC_API_KEY is not configured — all Claude stages will use mock output."
            )
        else:
            _log.info("USE_MOCK_LLM=true — running pipeline in mock mode.")
    else:
        import utils.claude_client  # noqa: F401 — load shared Anthropic client

        _log.info("Anthropic API key present — Claude stages will call the live API.")


def _ensure_safe_llm_mode() -> None:
    """No-op guard: documents that mock auto-fallback is active via use_mock_llm()."""
    if use_mock_llm() and mock_llm_auto_fallback():
        _log.debug("Request will run with mock LLM (no Anthropic API key).")


env_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
allowed_origins = ["http://localhost:3000", *env_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Pipeline failed", "detail": str(exc)},
    )


async def stream_pipeline(brief: CampaignBrief):
    """Run pipeline stage-by-stage and yield SSE progress events."""
    _ensure_safe_llm_mode()
    job_id = brief.job_id if brief.job_id else str(uuid.uuid4())
    brief_dict = brief.model_dump(mode="json")
    context: dict = {}
    cp: CheckpointManager = checkpoint

    def event(payload: dict) -> str:
        return f"data: {json_lib.dumps(payload)}\n\n"

    # Stage 1 — data collection (no LLM call)
    cp.save(1, job_id, brief_dict)
    context["stage1"] = brief_dict
    yield event({"event": "stage_complete", "stage": 1, "stage_name": "Data Collection", "progress": 8})

    # Stages 2–8 + 7b with progress percentages
    stage_plan = [
        (2, stage2_business.run, "Business Analysis", 18),
        (3, stage3_competitors.run, "Competitor Intelligence", 30),
        (4, stage4_audience.run, "Audience Profiling", 42),
        (5, stage5_strategy.run, "Campaign Strategy", 55),
        (6, stage6_tactical.run, "Content Generation", 68),
        ("7b", stage7b_influencer.run, "Influencer Matching", 78),
        (7, stage7_evaluation.run, "ML Evaluation", 88),
        (8, stage8_calendar.run, "Campaign Calendar", 96),
    ]

    try:
        loop = asyncio.get_running_loop()
        for stage_num, runner, name, progress in stage_plan:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(_run_stage, stage_num, runner, brief_dict, context, job_id),
                ),
                timeout=STAGE_TIMEOUT_SECONDS,
            )
            yield event(
                {"event": "stage_complete", "stage": str(stage_num), "stage_name": name, "progress": progress}
            )

    except asyncio.TimeoutError:
        yield event(
            {
                "event": "error",
                "message": f"Stage timed out after {STAGE_TIMEOUT_SECONDS}s",
                "progress": -1,
            }
        )
        return
    except Exception as exc:
        yield event({"event": "error", "message": str(exc), "progress": -1})
        return

    stage5_output = context.get("stage5", {})
    stage6_output = context.get("stage6", {})
    strategy = {
        "campaign_summary": stage5_output.get("campaign_summary", {}),
        "positioning_statement": stage5_output.get("positioning_statement", ""),
        "core_message": stage5_output.get("core_message", ""),
        "campaign_hooks": stage5_output.get("campaign_hooks", []),
        "content_pillars": stage5_output.get("content_pillars", []),
        "funnel": stage5_output.get("funnel", {}),
        "kpis": stage5_output.get("kpis", []),
        "budget_allocation": stage5_output.get("budget_allocation", {}),
        "tactical_plan": stage6_output,
        "influencer_strategy_note": context.get("stage7b", {}).get("influencer_strategy_note", ""),
    }
    final_result = {
        "strategy": strategy,
        "calendar": context.get("stage8", {"total_days": 0, "start_date": "pending", "days": []}),
        "influencer_matches": context.get("stage7b", {}).get("influencer_matches", []),
        "influencer_strategy_note": context.get("stage7b", {}).get("influencer_strategy_note", ""),
        "influencer_stage_skipped": context.get("stage7b", {}).get("influencer_stage_skipped", True),
    }
    yield event({"event": "complete", "progress": 100, "result": final_result})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/detailed")
def health_detailed() -> dict:
    return {
        "status": "ok",
        "claude_model": "claude-haiku-4-5",
        "anthropic_configured": anthropic_api_key_configured(),
        "mock_mode": use_mock_llm(),
        "mock_auto_fallback": mock_llm_auto_fallback(),
        "mock_auth_fallback": mock_llm_auth_fallback(),
        "stage_order": ["1", "2", "3", "4", "5", "6", "7b", "7", "8"],
        "influencer_stage": "7b",
        "endpoints": ["/health", "/health/detailed", "/generate", "/generate/stream"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(brief: CampaignBrief) -> GenerateResponse:
    _ensure_safe_llm_mode()
    _log.info(
        "POST /generate brand=%s mock=%s timeout=%ss",
        brief.brand_name,
        use_mock_llm(),
        PIPELINE_TIMEOUT_SECONDS,
    )
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(pipeline.run, brief),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Campaign generation timed out after {PIPELINE_TIMEOUT_SECONDS}s. "
                "Use /generate/stream for progress, set USE_MOCK_LLM=true for local dev, "
                "or increase PIPELINE_TIMEOUT_SECONDS."
            ),
        ) from exc
    return GenerateResponse(**result)


@app.post("/generate/stream")
async def generate_stream(brief: CampaignBrief):
    """
    SSE streaming endpoint. Yields stage progress events then the full result.

    NODE.JS FRONTEND INTEGRATION:
    EventSource does not support POST with a body.
    Use fetch() with a ReadableStream reader instead:

        const res = await fetch('/generate/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(briefPayload),
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const lines = decoder.decode(value).split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const payload = JSON.parse(line.slice(6));
                    if (payload.event === 'stage_complete') updateProgressBar(payload.progress);
                    if (payload.event === 'complete') handleResult(payload.result);
                    if (payload.event === 'error') handleError(payload.message);
                }
            }
        }
    """
    return StreamingResponse(
        stream_pipeline(brief),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
