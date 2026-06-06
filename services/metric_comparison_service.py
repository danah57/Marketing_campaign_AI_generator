"""Predicted vs actual metric comparison — deterministic math + optional Claude narrative."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.claude_client import call_claude
from utils.llm_runtime import use_mock_llm

logger = logging.getLogger("campaign_model.llm")

METRIC_META = {
    "reach": {"label": "Total Reach", "unit": ""},
    "engagement_rate": {"label": "Engagement Rate", "unit": "%"},
    "conversions": {"label": "Conversions", "unit": ""},
    "roi": {"label": "ROI", "unit": "x"},
    "clicks": {"label": "Clicks", "unit": ""},
}

METRICS = ["reach", "engagement_rate", "conversions", "roi", "clicks"]

_NARRATIVE_SYSTEM = (
    "You are a marketing analytics AI. Write concise performance analysis. "
    "Return ONLY valid JSON with exactly one key: ai_analysis (a string of 2-3 sentences)."
)


def _build_metric(metric: str, predicted: float, actual: float) -> dict:
    meta = METRIC_META[metric]
    achievement = round((actual / predicted * 100), 1) if predicted > 0 else 0.0
    achievement = min(achievement, 150.0)
    if achievement >= 100:
        status = "exceeded"
    elif achievement >= 75:
        status = "on_track"
    elif achievement >= 40:
        status = "below"
    else:
        status = "critical"
    return {
        "metric": metric,
        "label": meta["label"],
        "predicted": predicted,
        "actual": actual,
        "unit": meta["unit"],
        "achievement_rate": achievement,
        "status": status,
    }


def _overall_achievement(metrics: list[dict]) -> float:
    rates = [m["achievement_rate"] for m in metrics]
    return round(sum(rates) / len(rates), 1) if rates else 0.0


def _fallback_narrative(brand_level: list[dict], below: list[str], exceeded: list[str]) -> str:
    avg = _overall_achievement(brand_level)
    parts = []
    if exceeded:
        parts.append(f"{', '.join(exceeded)} exceeded predictions.")
    if below:
        parts.append(f"{', '.join(below)} are underperforming vs targets.")
    parts.append(f"Overall brand achievement stands at {avg:.1f}% of predicted targets.")
    return " ".join(parts)


def _format_metric_summary(m: dict) -> str:
    pred = m["predicted"]
    act = m["actual"]
    if m["unit"] == "%":
        return (
            f"{m['label']}: {act:.2f}/{pred:.2f}% "
            f"({m['achievement_rate']}% — {m['status']})"
        )
    if m["unit"] == "x":
        return (
            f"{m['label']}: {act:.2f}/{pred:.2f}x "
            f"({m['achievement_rate']}% — {m['status']})"
        )
    return (
        f"{m['label']}: {act:,.1f}/{pred:,.1f} "
        f"({m['achievement_rate']}% — {m['status']})"
    )


def _call_claude_narrative(data: dict, brand_level: list[dict], below: list[str], exceeded: list[str], campaign_count: int) -> str | None:
    brand_summary = ", ".join(_format_metric_summary(m) for m in brand_level)
    user_prompt = f"""
Brand: {data['brand_name']} | Industry: {data['industry']}
Predicted vs Actual: {brand_summary}
Exceeded targets: {exceeded or 'none'}
Below targets: {below or 'none'}
Campaigns analyzed: {campaign_count}

Write 2-3 sentences of concise, data-driven analysis of this brand's
performance vs predictions. Be specific about which metrics are strong
or weak.
"""
    try:
        result = call_claude(_NARRATIVE_SYSTEM, user_prompt, max_tokens=300)
        if isinstance(result, dict):
            text = result.get("ai_analysis") or result.get("analysis") or result.get("text")
            if text and str(text).strip():
                return str(text).strip()
        if isinstance(result, str) and result.strip():
            return result.strip()
    except Exception as e:
        logger.warning("Metric comparison Claude narrative failed: %s", e)
    return None


def generate_metric_comparison(data: dict) -> dict:
    """Build brand/campaign comparisons and optional AI narrative. Never raises."""
    brand_level = [
        _build_metric(m, data.get(f"predicted_{m}", 0) or 0, data.get(f"actual_{m}", 0) or 0)
        for m in METRICS
    ]

    campaign_level = []
    for c in data.get("campaigns") or []:
        if not isinstance(c, dict):
            continue
        pred = c.get("predicted") or {}
        act = c.get("actual") or {}
        metrics = [
            _build_metric(m, float(pred.get(m, 0) or 0), float(act.get(m, 0) or 0))
            for m in METRICS
        ]
        campaign_level.append(
            {
                "campaign_id": int(c.get("id", 0)),
                "campaign_name": str(c.get("name", "Unnamed Campaign")),
                "metrics": metrics,
                "overall_achievement": _overall_achievement(metrics),
                "verdict": str(c.get("ml_verdict", "UNKNOWN")),
            }
        )

    below = [m["label"] for m in brand_level if m["status"] in ("below", "critical")]
    exceeded = [m["label"] for m in brand_level if m["status"] == "exceeded"]

    ai_analysis: str
    if not use_mock_llm():
        narrative = _call_claude_narrative(
            data, brand_level, below, exceeded, len(campaign_level)
        )
        ai_analysis = narrative if narrative else _fallback_narrative(brand_level, below, exceeded)
    else:
        ai_analysis = _fallback_narrative(brand_level, below, exceeded)

    return {
        "brand_level": brand_level,
        "campaign_level": campaign_level,
        "ai_analysis": ai_analysis,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
