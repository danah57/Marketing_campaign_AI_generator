import json
import shutil
import uuid
from pathlib import Path
from typing import Callable

from schemas import CampaignBrief
from stages import (
    stage2_business,
    stage3_competitors,
    stage4_audience,
    stage5_strategy,
    stage6_tactical,
    stage7_evaluation,
    stage8_calendar,
)

DEBUG_KEEP_CHECKPOINTS = True  # Set to False in production


class CheckpointManager:
    def __init__(self, base_dir: str | Path = "checkpoints") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _stage_path(self, stage_number: int, job_id: str) -> Path:
        return self.base_dir / job_id / f"stage_{stage_number}.json"

    def _ensure_job_dir(self, job_id: str) -> None:
        (self.base_dir / job_id).mkdir(parents=True, exist_ok=True)

    def save(self, stage_number: int, job_id: str, data: dict) -> None:
        self._ensure_job_dir(job_id)
        path = self._stage_path(stage_number, job_id)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, stage_number: int, job_id: str) -> dict | None:
        path = self._stage_path(stage_number, job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, stage_number: int, job_id: str) -> bool:
        return self._stage_path(stage_number, job_id).exists()

    def clear(self, job_id: str) -> None:
        job_dir = self.base_dir / job_id
        if job_dir.exists() and job_dir.is_dir():
            shutil.rmtree(job_dir)


checkpoint = CheckpointManager()


def _run_stage(
    stage_number: int,
    stage_runner: Callable[[dict, dict, str], dict],
    brief_dict: dict,
    context: dict,
    job_id: str,
) -> dict:
    if checkpoint.exists(stage_number, job_id):
        stage_output = checkpoint.load(stage_number, job_id) or {}
    else:
        stage_output = stage_runner(brief_dict, context, job_id)
        checkpoint.save(stage_number, job_id, stage_output)

    context[f"stage{stage_number}"] = stage_output
    context.update(stage_output)
    return stage_output


def run(brief: CampaignBrief) -> dict:
    job_id = brief.job_id if brief.job_id else str(uuid.uuid4())
    brief_dict = brief.model_dump()
    context: dict = {}

    checkpoint.save(1, job_id, brief_dict)
    context["stage1"] = brief_dict

    _run_stage(2, stage2_business.run, brief_dict, context, job_id)
    _run_stage(3, stage3_competitors.run, brief_dict, context, job_id)
    _run_stage(4, stage4_audience.run, brief_dict, context, job_id)
    _run_stage(5, stage5_strategy.run, brief_dict, context, job_id)
    _run_stage(6, stage6_tactical.run, brief_dict, context, job_id)
    _run_stage(7, stage7_evaluation.run, brief_dict, context, job_id)
    stage8_output = _run_stage(8, stage8_calendar.run, brief_dict, context, job_id)

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
    }

    final_response = {
        "strategy": strategy,
        "calendar": stage8_output,
    }

    if not DEBUG_KEEP_CHECKPOINTS:
        checkpoint.clear(job_id)
    return final_response
