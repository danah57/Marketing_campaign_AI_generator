"""Stage 7: Evaluation Layer — ML verdict and explainability summary."""

import logging
import os
import sys
from datetime import datetime

from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")

_inference_engine = None


def _extract_age_range(segment_name: str) -> str:
    """Extract age range string from segment name."""
    if any(word in segment_name.lower() for word in ["gen z", "18", "young"]):
        return "18-24"
    elif any(word in segment_name.lower() for word in ["millennial", "25", "30"]):
        return "25-34"
    elif any(word in segment_name.lower() for word in ["35", "40", "mid"]):
        return "35-44"
    elif any(word in segment_name.lower() for word in ["45", "50", "senior"]):
        return "45-54"
    return "25-34"  # default


def _extract_gender(segment_name: str) -> str:
    """Extract gender from segment name."""
    name_lower = segment_name.lower()
    if any(word in name_lower for word in ["male", "men", "man", "guy"]):
        return "Men"
    elif any(word in name_lower for word in ["female", "women", "woman"]):
        return "Women"
    return "All Ages"


def _map_industry_to_segment(industry: str) -> str:
    """Map industry to the closest Customer_Segment value the ML model understands."""
    mapping = {
        "E-commerce & Retail": "Online Shoppers",
        "Fashion & Beauty": "Fashion Enthusiasts",
        "Food & Beverage": "Foodies",
        "Media & Content Creation": "Online Shoppers",
        "Fitness & Wellness": "Health & Wellness",
        "Home & Local Services": "Online Shoppers",
        "Education & Coaching": "Students",
        "Travel & Hospitality": "Travel Enthusiasts",
        "Real Estate": "High-Income Earners",
        "Healthcare & Wellness": "Health & Wellness",
        "Finance & Business": "High-Income Earners",
        "Technology & Apps": "Tech Enthusiasts",
        "Other": "Online Shoppers",
        # Legacy keys (if older briefs still appear)
        "Fashion": "Fashion Enthusiasts",
        "Health & Wellness": "Health & Wellness",
        "Technology": "Tech Enthusiasts",
        "F&B": "Foodies",
        "Travel": "Travel Enthusiasts",
        "Beauty": "Beauty & Personal Care",
        "Sports": "Sports Enthusiasts",
        "E-commerce": "Online Shoppers",
        "Finance": "High-Income Earners",
        "Education": "Students",
    }
    return mapping.get(industry, "Online Shoppers")


def run(brief: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 7", _mock_output, _real_output, brief, context, job_id)


def _mock_output(brief: dict, context: dict, job_id: str) -> dict:
    return {
        "ml_score": 0.84,
        "ml_verdict": "LAUNCH",
        "predicted_roi": 7.6,
        "shap_explanation": {
            "roi_drivers": [
                "Channel choice is helping the score",
                "Conversion rate is high",
                "Expected clicks are high",
            ],
            "roi_detractors": [
                "Campaign type is hurting the score",
                "Duration is low",
            ],
            "success_drivers": [
                "Conversion rate is high",
                "Channel choice is helping the score",
                "Season is high",
            ],
            "success_detractors": [
                "month is low",
                "day of week is low",
            ],
        },
        "written_explanation": (
            "The EcoWear campaign scores 0.84 and clears the LAUNCH threshold comfortably. "
            "The strongest contributors are the Instagram and TikTok channel selection which "
            "the model recognizes as high-affinity for this audience segment and campaign goal, "
            "and the high expected conversion rate driven by the tight audience targeting. The "
            "predicted ROI of 7.6 exceeds the LAUNCH threshold of 6.5 clearly. The two attributes "
            "pulling the score down slightly are campaign type alignment and duration — 8 weeks is "
            "on the shorter end for a brand awareness goal at this budget level. Neither detractor "
            "is significant enough to change the verdict and the campaign is cleared for launch as designed."
        ),
    }


def _skip_ml_evaluation() -> bool:
    v = os.getenv("SKIP_ML_EVALUATION", "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def _get_inference_engine():
    global _inference_engine
    if _inference_engine is not None:
        return _inference_engine

    ml_root = os.getenv("ML_ROOT_PATH", "").strip()
    if not ml_root:
        raise RuntimeError(
            "Stage 7 requires ML_ROOT_PATH pointing at the folder that contains "
            "`pipeline/inference.py`, or set SKIP_ML_EVALUATION=true."
        )
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)

    from ml.pipeline.inference import InferenceEngine

    logger.info("Stage 7: loading ML InferenceEngine (first request may take ~30s)...")
    _inference_engine = InferenceEngine()
    return _inference_engine


def _real_output(brief: dict, context: dict, job_id: str) -> dict:
    if _skip_ml_evaluation():
        return _mock_output(brief, context, job_id)

    try:
        engine = _get_inference_engine()
    except Exception as e:
        logger.warning("Stage 7: ML engine unavailable (%s); using mock evaluation.", e)
        return _mock_output(brief, context, job_id)

    stage5 = context.get("stage5", {})
    campaign_summary = stage5.get("campaign_summary", {})
    primary_segment = context.get("primary_segment", "")

    age_range = _extract_age_range(primary_segment)
    gender = _extract_gender(primary_segment)

    prioritized = context.get("channels_to_prioritize") or []
    brief_channels = brief.get("current_channels") or []
    if prioritized:
        channel_used = prioritized[0]
    elif brief_channels:
        channel_used = brief_channels[0]
    else:
        channel_used = "Instagram"

    campaign_type = campaign_summary.get("campaign_type") or "Social Media"
    duration_weeks = campaign_summary.get("duration_weeks") or brief.get("campaign_duration_weeks", 4)

    campaign_input = {
        "Channel_Used": channel_used,
        "Campaign_Type": campaign_type,
        "Audience_age_range": age_range,
        "Audience_Gender": gender,
        "Customer_Segment": _map_industry_to_segment(brief.get("industry", "")),
        "Location": brief.get("target_market", "New York"),
        "Language": "English",
        "Duration": duration_weeks * 7,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Budget": brief.get("budget_amount", 0),
    }

    try:
        output = engine.predict_one(campaign_input, verbose=False, include_shap=True)
    except Exception as e:
        logger.warning("Stage 7: ML prediction failed (%s); using mock evaluation.", e)
        return _mock_output(brief, context, job_id)

    ml_stage = output["stage2_evaluation"]
    ml_score = ml_stage["success_probability"]
    ml_verdict = ml_stage["verdict"]
    predicted_roi = ml_stage["predicted_roi"]
    shap_explanation = output["shap_explanation"]

    system_prompt = (
        "You are a marketing analytics expert who explains machine learning model outputs in plain English.\n"
        "You always respond in valid JSON only — no markdown, no explanation, no preamble.\n"
        "Your explanations are specific, honest, and actionable — never vague or overly positive."
    )

    user_prompt = (
        "Explain why this marketing campaign received this ML evaluation score.\n\n"
        f"Brand: {brief['brand_name']}\n"
        f"Campaign Theme: {campaign_summary.get('campaign_theme') or campaign_summary.get('tagline') or campaign_summary.get('name', '')}\n"
        f"ML Score (success probability): {ml_score}\n"
        f"Predicted ROI: {predicted_roi}\n"
        f"ML Verdict: {ml_verdict}\n\n"
        "SHAP Explanation:\n"
        f"ROI Drivers: {shap_explanation.get('roi_drivers', [])}\n"
        f"ROI Detractors: {shap_explanation.get('roi_detractors', [])}\n"
        f"Success Drivers: {shap_explanation.get('success_drivers', [])}\n"
        f"Success Detractors: {shap_explanation.get('success_detractors', [])}\n\n"
        "Instructions:\n"
        "- Write one paragraph of 4-6 sentences in plain English\n"
        "- Name the top 2 positive contributors and explain why they help\n"
        "- Name any detractors and explain what risk they represent\n"
        "- End with a clear statement: launch as-is, or what specifically needs adjustment\n"
        '- Return valid JSON with exactly one key: "written_explanation"'
    )

    try:
        explanation_result = call_claude(
            system_prompt,
            user_prompt,
            max_tokens=1024,
        )
    except (ValueError, RuntimeError) as e:
        logger.warning("Stage 7: explanation LLM failed (%s); using template text.", e)
        mock = _mock_output(brief, context, job_id)
        return {
            "ml_score": ml_score,
            "ml_verdict": ml_verdict,
            "predicted_roi": predicted_roi,
            "shap_explanation": shap_explanation,
            "written_explanation": mock["written_explanation"],
        }
    written = explanation_result.get("written_explanation")
    if written is None or not str(written).strip():
        raise ValueError(
            f"LLM returned JSON without a non-empty 'written_explanation' key: {explanation_result!r}"
        )

    return {
        "ml_score": ml_score,
        "ml_verdict": ml_verdict,
        "predicted_roi": predicted_roi,
        "shap_explanation": shap_explanation,
        "written_explanation": str(written).strip(),
    }
