"""Stage 4: Audience Analysis — segment definition and prioritization."""

import logging

from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")


def run(brief_dict: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 4", _mock_output, _real_output, brief_dict, context, job_id)


def _mock_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand = brief_dict.get("brand_name", "the brand")
    return {
        "persona_name": "Rami El-Khatib",
        "age_range": "25-34",
        "gender_skew": "balanced",
        "occupation": "Remote product designer juggling deadlines and home café rituals",
        "pain_points": [
            "Inconsistent brew quality between mornings",
            "Overwhelming gadget choices with no guided learning path",
            "Wants premium taste without café commute time",
        ],
        "desires": [
            "A confident morning ritual that feels premium",
            "Proof the AI actually adapts to his taste",
            "Shareable moments for Instagram stories",
        ],
        "content_preferences": ["Reels", "Carousels", "Behind-the-scenes"],
        "platform_behaviour": {
            "most_active_times": "Weekday mornings 7:30–9:00am, evenings 8–10pm",
            "content_consumption": "mixed",
            "engagement_style": "active",
        },
        "buying_triggers": ["Peer creator endorsement", "Visible taste improvement in 7 days", "Bundle with app onboarding"],
        "messaging_hooks": [
            f"{brand} learns how strong you like it—then keeps the bar high",
            "Café clarity without the commute",
            "Dial in once; sip smarter every day",
        ],
        "primary_segment": "Remote professionals 25-35",
        "primary_segment_reason": "Matches brief target market and priority channels for awareness.",
    }


def _real_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand_name = brief_dict["brand_name"]
    target_market = brief_dict["target_market"]
    industry = brief_dict["industry"]
    campaign_goal = brief_dict["campaign_goal"]
    product_or_service = brief_dict["product_or_service"]

    positioning_opportunity = context.get("positioning_opportunity", "")
    channels_to_prioritize = context.get("channels_to_prioritize") or []
    tone_descriptor = context.get("tone_descriptor", "")

    channels_joined = ", ".join(channels_to_prioritize) if channels_to_prioritize else "Not yet determined"

    system_prompt = (
        "You are an audience research specialist for NexBrand AI.\n"
        "You build detailed audience personas to guide campaign targeting.\n"
        "Always respond with valid JSON only. No markdown. No explanation outside JSON."
    )

    user_prompt = f"""Build an audience persona for this campaign.

Brand: {brand_name} ({industry})
Product/Service: {product_or_service}
Target Market: {target_market}
Campaign Goal: {campaign_goal}
Positioning Opportunity: {positioning_opportunity}
Channels: {channels_joined}
Voice: {tone_descriptor}

Return a JSON object with exactly these keys:
{{
  "persona_name": "Representative name",
  "demographics": "Age, occupation, or lifestyle summary",
  "pain_points": ["array of 3 core problems"],
  "desires": ["array of 3 ultimate goals"],
  "buying_barriers": ["array of 2 main objections or hesitations"],
  "content_preferences": ["array of 2 preferred formats"],
  "platform_behaviour": {{
    "active_times": "e.g., evenings 8-10pm",
    "consumption": "short-form|long-form|mixed",
    "engagement": "passive|active|creator"
  }},
  "buying_triggers": ["array of 2 conversion triggers"],
  "messaging_hooks": ["array of 3 distinct angles to grab attention"]
}}"""

    return call_claude(system_prompt, user_prompt, max_tokens=750)
