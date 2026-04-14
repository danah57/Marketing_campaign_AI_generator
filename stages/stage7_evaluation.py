"""Stage 7: Evaluation Layer — ML verdict and explainability summary."""

import os
import sys
from datetime import datetime

from utils.llm_client import call_claude

USE_MOCK = True


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
        "Real Estate": "High-Income Earners",
    }
    return mapping.get(industry, "Online Shoppers")


def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
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


def _real_output(brief: dict, context: dict) -> dict:
    ml_root = os.getenv("ML_ROOT_PATH", "")
    if ml_root and ml_root not in sys.path:
        sys.path.insert(0, ml_root)

    from pipeline.inference import InferenceEngine

    stage5 = context.get("stage5", {})
    campaign_summary = stage5.get("campaign_summary", {})
    primary_segment = context.get("primary_segment", "")

    age_range = _extract_age_range(primary_segment)
    gender = _extract_gender(primary_segment)

    campaign_input = {
        "Channel_Used": campaign_summary.get("channel_mix", ["Instagram"])[0],
        "Campaign_Type": campaign_summary.get("campaign_type", "Social Media"),
        "Audience_age_range": age_range,
        "Audience_Gender": gender,
        "Customer_Segment": _map_industry_to_segment(brief.get("industry", "")),
        "Location": brief.get("target_market", "New York"),
        "Language": "English",
        "Duration": campaign_summary.get("duration_weeks", 4) * 7,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Budget": brief.get("budget_amount", 0),
    }

    engine = InferenceEngine()
    output = engine.predict_one(campaign_input, verbose=False, include_shap=True)

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
        f"Campaign Theme: {campaign_summary.get('campaign_theme', '')}\n"
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

    explanation_result = call_claude(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
    )

    return {
        "ml_score": ml_score,
        "ml_verdict": ml_verdict,
        "predicted_roi": predicted_roi,
        "shap_explanation": shap_explanation,
        "written_explanation": explanation_result.get("written_explanation", ""),
    }
