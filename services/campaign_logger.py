"""Persist each campaign generation run under campaigns/{job_id}/."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("campaign_model.campaigns")

STAGE_LABELS: dict[int | str, str] = {
    1: "Data Collection",
    2: "Business Analysis",
    3: "Competitor Intelligence",
    4: "Audience Profiling",
    5: "Campaign Strategy",
    6: "Content Generation",
    "7b": "Influencer Matching",
    7: "ML Evaluation",
    8: "Campaign Calendar",
}


def campaigns_base_dir() -> Path:
    return Path(os.getenv("CAMPAIGNS_DIR", "campaigns"))


class CampaignLogger:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or campaigns_base_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def _stage_filename(self, stage_number: int | str) -> str:
        return f"stage_{stage_number}.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_log(self, job_id: str, level: str, message: str) -> None:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "generation.log"
        line = f"{self._now_iso()} [{level}] {message}\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _write_json(self, job_id: str, filename: str, data: dict) -> None:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _update_meta(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage_completed: int | str | None = None,
        brand_name: str | None = None,
        error: str | None = None,
    ) -> None:
        job_dir = self._job_dir(job_id)
        meta_path = job_dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        meta.setdefault("job_id", job_id)
        meta["updated_at"] = self._now_iso()
        if brand_name is not None:
            meta["brand_name"] = brand_name
        if status is not None:
            meta["status"] = status
        if error is not None:
            meta["error"] = error
        if stage_completed is not None:
            completed = meta.setdefault("stages_completed", [])
            stage_key = str(stage_completed)
            if stage_key not in completed:
                completed.append(stage_key)

        self._write_json(job_id, "meta.json", meta)

    def start(self, job_id: str, brief_dict: dict) -> None:
        brand_name = brief_dict.get("brand_name", "unknown")
        started_at = self._now_iso()
        self._write_json(
            job_id,
            "meta.json",
            {
                "job_id": job_id,
                "brand_name": brand_name,
                "started_at": started_at,
                "updated_at": started_at,
                "status": "running",
                "stages_completed": [],
            },
        )
        self._write_json(job_id, "brief.json", brief_dict)
        self._append_log(
            job_id,
            "INFO",
            f"Pipeline started brand={brand_name} job_id={job_id}",
        )
        logger.info("Campaign log started: %s", self._job_dir(job_id))

    def log_stage(
        self,
        job_id: str,
        stage_number: int | str,
        data: dict,
        *,
        source: str = "generated",
        elapsed_seconds: float | None = None,
    ) -> None:
        stage_name = STAGE_LABELS.get(stage_number, f"Stage {stage_number}")
        filename = self._stage_filename(stage_number)
        self._write_json(job_id, filename, data)
        self._update_meta(job_id, stage_completed=stage_number)

        timing = f" in {elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
        self._append_log(
            job_id,
            "INFO",
            f"Stage {stage_number} ({stage_name}) {source}{timing} -> {filename}",
        )

    def complete(self, job_id: str, final_response: dict) -> None:
        self._write_json(job_id, "response.json", final_response)
        self._update_meta(job_id, status="complete")
        self._append_log(job_id, "INFO", "Pipeline complete -> response.json")

    def fail(self, job_id: str, stage_number: int | str, error: str) -> None:
        stage_name = STAGE_LABELS.get(stage_number, f"Stage {stage_number}")
        self._update_meta(job_id, status="failed", error=error)
        self._append_log(
            job_id,
            "ERROR",
            f"Stage {stage_number} ({stage_name}) failed: {error}",
        )


campaign_logger = CampaignLogger()
