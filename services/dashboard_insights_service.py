"""Lightweight dashboard insights — Claude or deterministic fallback (no pipeline)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from utils.claude_client import call_claude
from utils.llm_runtime import use_mock_llm

logger = logging.getLogger("campaign_model.llm")

_SYSTEM_PROMPT = (
    "You are a marketing analytics AI. You analyze brand performance data and "
    "return ONLY valid JSON — no preamble, no markdown fences, no explanation."
)


def _compute_change(current: float, previous: float) -> float:
    if not previous or previous == 0:
        return 0.0
    return round(((current - previous) / previous) * 100, 1)


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _build_user_prompt(data: dict) -> str:
    reach_change = _compute_change(
        data.get("total_reach", 0),
        data.get("previous_period_reach") or 0,
    )
    eng_change = _compute_change(
        data.get("engagement_rate", 0),
        data.get("previous_period_engagement") or 0,
    )
    top_channel = data.get("top_channel") or "Unknown"
    budget_util = data.get("budget_utilization") or 0

    return f"""
Analyze this brand's marketing performance and return JSON insights.

Brand: {data['brand_name']}
Industry: {data['industry']}
Campaign Goal: {data['campaign_goal']}

Current Metrics:
- Active Campaigns: {data['active_campaigns']}
- Total Reach: {data['total_reach']:,.0f}
- Engagement Rate: {data['engagement_rate']:.2f}%
- Conversions: {data['total_conversions']:,.0f}
- ROI: {data['campaign_roi']:.2f}x
- Top Channel: {top_channel}
- Top Channel Engagement: {(data.get('top_channel_engagement') or 0):.2f}%
- Budget Utilization: {budget_util * 100:.0f}%

Previous Period Comparison:
- Reach Change: {reach_change:.1f}%
- Engagement Change: {eng_change:.1f}%

Return this EXACT JSON structure — fill ALL fields:
{{
  "insights": [
    {{
      "id": "ins_1",
      "type": "positive|negative|neutral|metric",
      "message": "specific insight message referencing real numbers",
      "metric": "metric name or null",
      "change": numeric_change_or_null
    }}
  ],
  "recommendations": [
    {{
      "id": "rec_1",
      "priority": "High|Medium|Low",
      "action": "specific action to take",
      "impact": "expected business impact"
    }}
  ]
}}

Rules:
- Provide exactly 4 insights (ins_1 through ins_4) and exactly 4 recommendations (rec_1 through rec_4)
- type 'positive' if metric improved or is strong
- type 'negative' if metric declined or needs attention
- type 'neutral' for general observations
- type 'metric' for pure numeric KPI statements
- Priority 'High' if ROI < 2 or engagement < 2% or budget > 90% used
- Make ALL messages specific to this brand's actual numbers
- Reference {top_channel} as the platform if engagement is highest there
"""


def _fallback_payload(data: dict) -> dict:
    reach_change = _compute_change(
        data.get("total_reach", 0),
        data.get("previous_period_reach") or 0,
    )
    eng_change = _compute_change(
        data.get("engagement_rate", 0),
        data.get("previous_period_engagement") or 0,
    )

    fallback_insights = [
        {
            "id": "ins_1",
            "type": "positive" if reach_change >= 0 else "negative",
            "message": (
                f"Reach {'increased' if reach_change >= 0 else 'decreased'} by "
                f"{abs(reach_change):.1f}% compared to last period"
            ),
            "metric": "reach",
            "change": reach_change,
        },
        {
            "id": "ins_2",
            "type": "positive" if data["engagement_rate"] >= 3 else "negative",
            "message": (
                f"Engagement rate is {data['engagement_rate']:.2f}% "
                f"{'above' if data['engagement_rate'] >= 3 else 'below'} the 3% benchmark"
            ),
            "metric": "engagement_rate",
            "change": eng_change,
        },
        {
            "id": "ins_3",
            "type": "neutral",
            "message": (
                f"You have {data['active_campaigns']} active campaign"
                f"{'s' if data['active_campaigns'] != 1 else ''} running"
            ),
            "metric": "active_campaigns",
            "change": None,
        },
        {
            "id": "ins_4",
            "type": "metric",
            "message": (
                f"{data['total_conversions']:,.0f} total conversions generated with "
                f"{data['campaign_roi']:.2f}x ROI"
            ),
            "metric": "conversions",
            "change": None,
        },
    ]

    top_ch = data.get("top_channel") or "your best channel"
    budget_util = data.get("budget_utilization") or 0.5

    fallback_recommendations = [
        {
            "id": "rec_1",
            "priority": "High" if data["engagement_rate"] < 2 else "Medium",
            "action": f"Focus content production on {top_ch} which shows highest engagement",
            "impact": "Expected 15-25% engagement increase within 4 weeks",
        },
        {
            "id": "rec_2",
            "priority": "High" if budget_util > 0.85 else "Low",
            "action": (
                f"{'Reallocate' if budget_util > 0.85 else 'Maintain'} budget allocation "
                "across campaigns"
            ),
            "impact": f"Optimize spend efficiency — current utilization at {budget_util * 100:.0f}%",
        },
        {
            "id": "rec_3",
            "priority": "Medium",
            "action": (
                f"Launch {data['campaign_goal'].lower()} campaigns targeting new audience segments"
            ),
            "impact": f"Potential {data['total_reach'] * 0.2:,.0f} additional reach",
        },
        {
            "id": "rec_4",
            "priority": "Low" if data["total_conversions"] > 100 else "Medium",
            "action": "Schedule performance review with top influencer collaborators",
            "impact": "Strengthen partnerships and improve conversion quality",
        },
    ]

    return {
        "insights": fallback_insights,
        "recommendations": fallback_recommendations,
    }


def _normalize_claude_result(raw: dict, data: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    insights = raw.get("insights")
    recommendations = raw.get("recommendations")
    if not isinstance(insights, list) or not isinstance(recommendations, list):
        return None
    if len(insights) < 1 or len(recommendations) < 1:
        return None

    fallback = _fallback_payload(data)
    while len(insights) < 4:
        insights.append(fallback["insights"][len(insights)])
    while len(recommendations) < 4:
        recommendations.append(fallback["recommendations"][len(recommendations)])

    return {
        "insights": insights[:4],
        "recommendations": recommendations[:4],
    }


def _call_claude_for_insights(data: dict) -> dict | None:
    try:
        result = call_claude(_SYSTEM_PROMPT, _build_user_prompt(data), max_tokens=1200)
        if isinstance(result, dict):
            return _normalize_claude_result(result, data)
        if isinstance(result, str):
            parsed = json.loads(_strip_json_fences(result))
            return _normalize_claude_result(parsed, data)
    except Exception as e:
        logger.warning("Dashboard insights Claude call failed: %s", e)
    return None


def generate_dashboard_insights(data: dict) -> dict:
    """Generate insights and recommendations. Never raises — always returns valid payload."""
    payload: dict | None = None

    if not use_mock_llm():
        payload = _call_claude_for_insights(data)

    if payload is None:
        payload = _fallback_payload(data)

    return {
        "insights": payload["insights"],
        "recommendations": payload["recommendations"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
