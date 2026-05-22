"""Stage 2: Business Analysis — SWOT, positioning, market analysis, and campaign readiness."""

import json
import logging

from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")


def run(brief_dict: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 2", _mock_output, _real_output, brief_dict, context, job_id)


def _mock_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand = brief_dict.get("brand_name", "the brand")

    budget_weekly = float(
        brief_dict.get("budget_amount", 5000) or 5000
    ) / max(int(brief_dict.get("campaign_duration_weeks", 1) or 1), 1)

    return {
        "business_summary": (
            f"{brand} is positioned as a modern lifestyle-focused coffee brand "
            "designed for remote professionals seeking café-quality experiences at home. "
            "Its AI-guided brewing personalization creates a differentiated product experience "
            "that blends convenience, premium quality, and daily ritual culture."
        ),

        "swot": {
            "strengths": [
                "Differentiated AI-powered brewing personalization creates a unique product experience competitors cannot easily replicate.",
                "Strong lifestyle and ritual-based branding fits highly visual social media platforms like Instagram and TikTok.",
                "Premium positioning allows the brand to compete on experience and identity rather than price alone."
            ],
            "weaknesses": [
                "Limited brand awareness may reduce short-term conversion efficiency.",
                "Premium positioning may narrow accessibility for price-sensitive audiences.",
                "Educational content may be required to explain the AI-guided brewing experience clearly."
            ],
            "opportunities": [
                "Growing remote-work culture increases demand for elevated at-home coffee experiences.",
                "Creator partnerships in productivity and lifestyle niches can generate strong trust-based awareness.",
                "Short-form video content offers strong potential for sensory product storytelling and demonstrations."
            ],
            "threats": [
                "Large coffee brands with bigger advertising budgets may dominate paid visibility.",
                "Consumer skepticism toward AI marketing claims could reduce trust if messaging is not authentic.",
                "Economic pressure may shift consumer spending away from premium lifestyle products."
            ]
        },

        "brand_positioning": (
            f"{brand} is the premium smart-coffee experience brand for modern professionals "
            "who want café-quality rituals at home powered by personalized AI guidance."
        ),

        "market_challenges": [
            "Building awareness in a highly saturated coffee and lifestyle market.",
            "Balancing premium branding with approachable messaging.",
            "Educating audiences about the practical value of AI-powered personalization."
        ],

        "growth_opportunities": [
            "Expand through influencer-led productivity and remote-work content.",
            "Develop community-driven coffee ritual campaigns across TikTok and Instagram.",
            "Leverage UGC and sensory-focused content to increase trust and engagement."
        ],

        "competitive_advantage": (
            "The combination of AI personalization, premium ritual branding, and lifestyle-focused storytelling "
            "creates a differentiated identity that is difficult for traditional coffee brands to imitate authentically."
        ),

        "campaign_readiness_score": 8.4,

        "campaign_readiness_reasoning": (
            "The brand has a clear USP, strong visual storytelling potential, and a realistic budget structure "
            "for a focused multi-platform awareness campaign."
        ),

        "tone_descriptor": "warm, confident, craft-forward",

        "tone_guidelines": [
            "Lead with emotional lifestyle moments before technical features.",
            "Use confident but approachable language that feels modern and authentic.",
            "Avoid aggressive sales language or discount-heavy messaging."
        ],

        "budget_tier": "small",

        "budget_weekly": budget_weekly,

        "recommended_focus": (
            "Prioritize short-form lifestyle storytelling and creator collaborations "
            "to establish emotional connection and product differentiation."
        ),
    }


def _real_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand_name = brief_dict["brand_name"]
    product_or_service = brief_dict["product_or_service"]
    industry = brief_dict["industry"]
    sub_industry = brief_dict.get("sub_industry")
    unique_selling_point = brief_dict["unique_selling_point"]
    company_size = brief_dict["company_size"]
    campaign_goal = brief_dict["campaign_goal"]
    campaign_goal_details = brief_dict.get("campaign_goal_details")
    budget_amount = brief_dict["budget_amount"]
    budget_currency = brief_dict["budget_currency"]
    campaign_duration_weeks = brief_dict["campaign_duration_weeks"]
    has_previous_campaigns = brief_dict["has_previous_campaigns"]
    previous_campaign_description = brief_dict.get("previous_campaign_description")
    brand_tone = brief_dict.get("brand_tone")

    system_prompt = (
        "You are a senior brand strategist and growth marketing consultant.\n"
        "Analyze the provided business data and return ONLY a valid JSON object matching the requested schema.\n"
        "No markdown, no backticks, no text outside the JSON.\n"
        "Ensure SWOT and strategic insights are realistic, specific, and data-driven. Avoid vague buzzwords."
    )

    brand_tone_block = (
        json.dumps(brand_tone, indent=2)
        if brand_tone
        else "Not specified — infer from industry and company size"
    )

    user_prompt = f"""
Analyze this business for strategic campaign planning.
Brand: {brand_name}
Product/Service: {product_or_service}
Industry: {industry} > {sub_industry or 'N/A'}
Company Size: {company_size}
USP: {unique_selling_point}
Campaign Goal:{campaign_goal} — {campaign_goal_details or 'No additional details'}
Budget:{budget_amount} {budget_currency} over {campaign_duration_weeks} weeks
Previous Campaigns:{"Yes — " + previous_campaign_description if has_previous_campaigns and previous_campaign_description else "None"}
Brand Tone Profile:{brand_tone_block}

Return a JSON object matching this structure exactly:
{{
  "business_summary": "2-3 sentence strategic business overview",
  "swot": {{
    "strengths": ["array of 3 specific items"],
    "weaknesses": ["array of 3 specific items"],
    "opportunities": ["array of 3 specific items"],
    "threats": ["array of 3 specific items"]
  }},
  "brand_positioning": "1-2 sentence positioning statement",
  "market_challenges": ["array of 3 items"],
  "growth_opportunities": ["array of 3 items"],
  "competitive_advantage": "What makes this business difficult to replicate",
  "campaign_readiness_score": 0.0,
  "campaign_readiness_reasoning": "one sentence explanation",
  "tone_descriptor": "3 adjectives describing brand voice",
  "tone_guidelines": ["array of 3 items"],
  "budget_tier": "micro|small|medium|large|enterprise",
  "budget_weekly": 0.0,
  "recommended_focus": "one sentence explaining campaign priority"
}}
"""
    return call_claude(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=2048,
    )

#     """Stage 2: Business Analysis — SWOT, brand positioning, challenges, growth opportunities."""

# import json
# import logging

# from utils.claude_client import call_claude
# from utils.llm_runtime import use_mock_llm

# logger = logging.getLogger("campaign_model.llm")


# def run(brief_dict: dict, context: dict, job_id: str) -> dict:
#     if use_mock_llm():
#         return _mock_output(brief_dict, context)
#     return _real_output(brief_dict, context)


# def _mock_output(brief_dict: dict, context: dict, job_id: str) -> dict:
#     brand = brief_dict.get("brand_name", "the brand")
#     return {
#         "business_summary": (
#             f"{brand} targets remote professionals who want café-quality coffee at home with an AI-guided brewing companion."
#         ),
#         "core_strengths": [
#             "Differentiated AI taste learning vs static competitors",
#             "Strong ritual and lifestyle story for social channels",
#             "Clear premium positioning without discount vocabulary",
#         ],
#         "campaign_readiness_score": 8.2,
#         "campaign_readiness_reasoning": "Solid USP, defined channels, and realistic budget for a six-week awareness push.",
#         "tone_descriptor": "warm, confident, craft-forward",
#         "tone_guidelines": [
#             "Lead with sensory coffee moments, not specs",
#             "Use playful confidence without sounding gimmicky",
#             "Avoid discount or 'cheap' framing entirely",
#         ],
#         "budget_tier": "small",
#         "budget_weekly": float(brief_dict.get("budget_amount", 5000) or 5000)
#         / max(int(brief_dict.get("campaign_duration_weeks", 1) or 1), 1),
#         "recommended_focus": "Own the remote-work morning ritual on Instagram and TikTok with proof-led demos.",
#     }


# def _real_output(brief_dict: dict, context: dict, job_id: str) -> dict:
#     brand_name = brief_dict["brand_name"]
#     product_or_service = brief_dict["product_or_service"]
#     industry = brief_dict["industry"]
#     sub_industry = brief_dict.get("sub_industry")
#     unique_selling_point = brief_dict["unique_selling_point"]
#     company_size = brief_dict["company_size"]
#     campaign_goal = brief_dict["campaign_goal"]
#     campaign_goal_details = brief_dict.get("campaign_goal_details")
#     budget_amount = brief_dict["budget_amount"]
#     budget_currency = brief_dict["budget_currency"]
#     campaign_duration_weeks = brief_dict["campaign_duration_weeks"]
#     has_previous_campaigns = brief_dict["has_previous_campaigns"]
#     previous_campaign_description = brief_dict.get("previous_campaign_description")
#     brand_tone = brief_dict.get("brand_tone")

#     system_prompt = (
#         "You are a senior marketing strategist AI for AI platform.\n"
#         "You analyze brand briefs and produce structured strategic analysis.\n"
#         "Always respond with valid JSON only. No markdown, no explanation outside JSON.\n"
#         "Your tone analysis must respect the brand_tone profile when provided."
#     )

#     brand_tone_block = (
#         json.dumps(brand_tone, indent=2)
#         if brand_tone
#         else "Not specified — infer from industry and company size"
#     )

#     user_prompt = f"""Analyze this brand for campaign planning.

# Brand: {brand_name}
# Product/Service: {product_or_service}
# Industry: {industry} > {sub_industry or 'N/A'}
# Company Size: {company_size}
# USP: {unique_selling_point}
# Campaign Goal: {campaign_goal} — {campaign_goal_details or 'No details'}
# Budget: {budget_amount} {budget_currency} over {campaign_duration_weeks} weeks
# Previous Campaigns: {"Yes — " + previous_campaign_description if has_previous_campaigns and previous_campaign_description else "None"}

# Brand Tone Profile:
# {brand_tone_block}

# Return a JSON object with exactly these keys:
# {{
#   "business_summary": "2-3 sentence brand overview",
#   "core_strengths": ["strength1", "strength2", "strength3"],
#   "campaign_readiness_score": <float 1-10>,
#   "campaign_readiness_reasoning": "one sentence",
#   "tone_descriptor": "3 adjectives describing the brand voice based on tone profile",
#   "tone_guidelines": ["guideline1", "guideline2", "guideline3"],
#   "budget_tier": "micro | small | medium | large | enterprise",
#   "budget_weekly": <float>,
#   "recommended_focus": "one sentence on what this campaign should prioritize"
# }}"""

#     try:
#         result = call_claude(system_prompt, user_prompt, max_tokens=1024)
#     except (ValueError, RuntimeError) as e:
#         logger.error(f"Stage 2 AI call failed: {e}")
#         raise RuntimeError(f"Stage 2 failed — AI response error: {e}") from e
#     return result
