import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from schemas import CampaignBrief
from services.campaign_logger import campaign_logger
from utils.llm_runtime import reset_llm_runtime_state, use_mock_llm
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

DEBUG_KEEP_CHECKPOINTS = True  # Set to False in production

logger = logging.getLogger("campaign_model.llm")


class CheckpointManager:
    def __init__(self, base_dir: str | Path = "checkpoints") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _stage_path(self, stage_number: int | str, job_id: str) -> Path:
        return self.base_dir / job_id / f"stage_{stage_number}.json"

    def _ensure_job_dir(self, job_id: str) -> None:
        (self.base_dir / job_id).mkdir(parents=True, exist_ok=True)

    def save(self, stage_number: int | str, job_id: str, data: dict) -> None:
        self._ensure_job_dir(job_id)
        path = self._stage_path(stage_number, job_id)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def load(self, stage_number: int | str, job_id: str) -> dict | None:
        path = self._stage_path(stage_number, job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, stage_number: int | str, job_id: str) -> bool:
        return self._stage_path(stage_number, job_id).exists()

    def clear(self, job_id: str) -> None:
        job_dir = self.base_dir / job_id
        if job_dir.exists() and job_dir.is_dir():
            shutil.rmtree(job_dir)


checkpoint = CheckpointManager()


def _run_stage(
    stage_number: int | str,
    stage_runner: Callable[[dict, dict, str], dict],
    brief_dict: dict,
    context: dict,
    job_id: str,
) -> dict:
    if checkpoint.exists(stage_number, job_id):
        cached = checkpoint.load(stage_number, job_id) or {}
        if not cached.get("stage_failed"):
            logger.info(f"Stage {stage_number}: loaded from checkpoint.")
            context[f"stage{stage_number}"] = cached
            context.update(cached)
            campaign_logger.log_stage(job_id, stage_number, cached, source="loaded from checkpoint")
            return cached
        logger.warning(f"Stage {stage_number}: previous run failed — re-running.")

    logger.info("Stage %s: starting (job %s)", stage_number, job_id)
    started = time.perf_counter()
    try:
        stage_output = stage_runner(brief_dict, context, job_id)
    except RuntimeError as e:
        error_checkpoint = {"error": str(e), "stage_failed": True, "stage": str(stage_number)}
        checkpoint.save(stage_number, job_id, error_checkpoint)
        campaign_logger.log_stage(job_id, stage_number, error_checkpoint, source="failed")
        campaign_logger.fail(job_id, stage_number, str(e))
        logger.error(f"Stage {stage_number} failed for job {job_id}: {e}")
        raise

    elapsed = time.perf_counter() - started
    logger.info("Stage %s: complete in %.1fs (job %s)", stage_number, elapsed, job_id)

    checkpoint.save(stage_number, job_id, stage_output)
    campaign_logger.log_stage(job_id, stage_number, stage_output, elapsed_seconds=elapsed)
    context[f"stage{stage_number}"] = stage_output
    context.update(stage_output)
    return stage_output


def run(brief: CampaignBrief) -> dict:
    reset_llm_runtime_state()
    job_id = brief.job_id if brief.job_id else str(uuid.uuid4())
    pipeline_started = time.perf_counter()
    logger.info(
        "Pipeline start job_id=%s brand=%s mock_llm=%s",
        job_id,
        brief.brand_name,
        use_mock_llm(),
    )
    brief_dict = brief.model_dump(mode="json")
    context: dict = {}

    campaign_logger.start(job_id, brief_dict)
    checkpoint.save(1, job_id, brief_dict)
    campaign_logger.log_stage(job_id, 1, brief_dict, source="brief saved")
    context["stage1"] = brief_dict

    _run_stage(2, stage2_business.run, brief_dict, context, job_id)
    _run_stage(3, stage3_competitors.run, brief_dict, context, job_id)
    _run_stage(4, stage4_audience.run, brief_dict, context, job_id)
    _run_stage(5, stage5_strategy.run, brief_dict, context, job_id)
    _run_stage(6, stage6_tactical.run, brief_dict, context, job_id)
    _run_stage("7b", stage7b_influencer.run, brief_dict, context, job_id)
    _run_stage(7, stage7_evaluation.run, brief_dict, context, job_id)
    _run_stage(8, stage8_calendar.run, brief_dict, context, job_id)

    stage5_output = context.get("stage5", {})
    stage6_output = context.get("stage6", {})
    stage7b_output = context.get("stage7b", {})

    strategy = stage5_strategy.build_strategy_payload(
        stage5_output,
        stage6_output,
        stage7b_output,
        brief_dict,
        context,
    )

    final_response = {
        "strategy": strategy,
        "calendar": context.get(
            "stage8",
            {
                "total_days": 0,
                "start_date": "pending",
                "days": [],
            },
        ),
        "influencer_matches": context.get("stage7b", {}).get("influencer_matches", []),
        "influencer_strategy_note": context.get("stage7b", {}).get("influencer_strategy_note", ""),
        "influencer_stage_skipped": context.get("stage7b", {}).get("influencer_stage_skipped", True),
    }

    campaign_logger.complete(job_id, final_response)

    if not DEBUG_KEEP_CHECKPOINTS:
        checkpoint.clear(job_id)

    logger.info(
        "Pipeline complete job_id=%s in %.1fs",
        job_id,
        time.perf_counter() - pipeline_started,
    )
    return final_response
