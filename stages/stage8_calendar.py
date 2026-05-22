import logging
import json
from datetime import datetime, timedelta
from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")


# ─────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────
def run(brief: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 8", _mock_output, _real_output, brief, context, job_id)


# ─────────────────────────────────────────────
# MOCK OUTPUT (Safe dev/testing fallback)
# ─────────────────────────────────────────────
def _mock_output(brief: dict, context: dict, job_id: str) -> dict:
    raw_start = brief.get("start_date")
    start_date = str(raw_start)[:10] if raw_start else datetime.now().strftime("%Y-%m-%d")

    return {
        "total_days": 3,
        "start_date": start_date,
        "days": [
            {
                "day": 1,
                "date": start_date,
                "platform": "Instagram",
                "content_type": "reel",
                "task": "Launch awareness campaign with short-form storytelling video.",
                "caption": "Discover something built for you. Tap to explore now."
            },
            {
                "day": 2,
                "date": (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                "platform": "TikTok",
                "content_type": "video",
                "task": "Run UGC-style product showcase and A/B test hooks.",
                "caption": "Real people. Real results. See the difference today."
            },
            {
                "day": 3,
                "date": (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d"),
                "platform": "Instagram",
                "content_type": "story",
                "task": "Engagement story with poll + conversion CTA.",
                "caption": "Which style fits you best? Vote now and shop your match."
            }
        ]
    }


# ─────────────────────────────────────────────
# REAL AI LOGIC
# ─────────────────────────────────────────────
def _real_output(brief: dict, context: dict, job_id: str) -> dict:

    # ── Duration ─────────────────────────────
    duration_weeks = (
        context.get("stage5", {})
        .get("campaign_summary", {})
        .get("duration_weeks")
        or brief.get("campaign_duration_weeks", 2)
    )

    total_days = int(duration_weeks) * 7

    # ── Start date ───────────────────────────
    raw_start = brief.get("start_date")
    start_date = str(raw_start)[:10] if raw_start else datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    # ── Channels ─────────────────────────────
    s6 = context.get("stage6") or {}
    posting_frequency = s6.get("posting_frequency")

    if not posting_frequency or not isinstance(posting_frequency, dict):
        platform_content = s6.get("platform_content")
        if isinstance(platform_content, dict) and platform_content:
            posting_frequency = {
                ch: len(v.get("posts", []))
                for ch, v in platform_content.items()
            }

    if not posting_frequency:
        channels = brief.get("targetAudience", {}).get(
            "platformsUsed", ["Instagram", "TikTok", "Facebook", "X", "YouTube"]
        )
    else:
        channels = list(posting_frequency.keys())

    # ── ML Context ───────────────────────────
    ml_verdict = context.get("stage7", {}).get("ml_verdict", "LAUNCH")
    ml_explanation = context.get("stage7", {}).get("written_explanation", "")

    # ── System Prompt ────────────────────────
    system_prompt = (
        "You are a senior performance marketer and media buying strategist. "
        "Generate a structured, realistic, high-performance campaign calendar. "
        "Return ONLY valid JSON. No markdown, no explanation."
    )

    max_tokens = min(4000, max(2000, int(total_days) * 120))

    # ── Improved Agency-Level Prompt ─────────
    user_prompt = f"""
Generate a REALISTIC social media campaign execution plan.

You are a senior marketing strategist at a top agency.

CAMPAIGN DATA:
Brand: {brief.get('brand_name', 'N/A')}
Product: {brief.get('product_or_service', 'N/A')}
Goal: {brief.get('campaign_goal', 'N/A')}
Platforms: {channels}
Start Date: {start_date}
ML Verdict: {ml_verdict}
Context: {ml_explanation}


━━━ PSYCHOLOGICAL PRINCIPLES TO USE ━━━
- Curiosity gaps: open loops that make viewers need to know more
- Social proof: mention community, customers, or reviews when available
- Urgency and scarcity: use naturally in Conversion and Closing phases only
- Reciprocity: give value before asking for anything
- Pattern interrupt: first frame of every video must be unexpected

STRICT RULES:
- Do NOT post every day unless necessary
- Use smart spacing (1–2 day gaps between feed posts)
-content_type MUST be EXACTLY one of:
'video', 'carousel', 'story', 'reel', 'post', 'article'
- Mix: reels, stories, ads, engagement posts
- Build narrative phases:
  Awareness → Engagement → Conversion → Closing
- Optimize for performance, NOT volume

OUTPUT JSON FORMAT:
{{
  "total_days": {total_days},
  "start_date": "{start_date}",
  "days": [
    {{
      "day": day_number(example: 1),
      "date": "{start_date}",
      "platform": "Instagram",
      "content_type": "reel",
      "task": "Highly specific execution instruction for what to do on this day",
      "caption": "Hook OR HASHTAGS"
    }}
  ]
}}
"""

    # ── LLM Call ─────────────────────────────
    try:
        raw_result = call_claude(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

        # ── Safe parsing ──────────────────────
        if isinstance(raw_result, str):
            raw_result = json.loads(raw_result)

        processed_days = raw_result.get("days", [])

    except Exception as e:
        logger.error(f"Stage 8 LLM failure: {str(e)}")
        return _mock_output(brief, context, job_id)

    # ── Validation ───────────────────────────
    if not isinstance(processed_days, list):
        logger.error("Invalid LLM output structure. Falling back.")
        return _mock_output(brief, context, job_id)

    # ── Smart Padding ────────────────────────
    fallback_platform = channels[0] if channels else "Instagram"

    if len(processed_days) < total_days:
        logger.warning(
            f"Padding missing days: got {len(processed_days)} / needed {total_days}"
        )

        for d in range(len(processed_days) + 1, total_days + 1):
            target_date = (start_dt + timedelta(days=d - 1)).strftime("%Y-%m-%d")

            processed_days.append({
                "day": d,
                "date": target_date,
                "platform": channels[d % len(channels)] if channels else fallback_platform,
                "content_type": "post",
                "task": "Run optimization audit + review campaign performance metrics.",
                "caption": "Discover our latest updates. Tap to learn more."
            })

    # ── Final Output ─────────────────────────
    return {
        "total_days": len(processed_days),
        "start_date": start_date,
        "days": processed_days
    }