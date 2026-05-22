"""Stage 5: Strategic Framework — messaging and channel strategy blueprint."""

import logging

from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")


def run(brief_dict: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 5", _mock_output, _real_output, brief_dict, context, job_id)


def _mock_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    return {
        "campaign_summary": {
            "campaign_type": "Social Media",
            "channel_mix": ["Instagram", "TikTok"],
            "tone_of_voice": "Honest, direct, and quietly confident — never preachy or activist",
            "campaign_theme": "Proof Not Promises",
            "duration_weeks": 8,
            "target_audience": "Eco-conscious males aged 25-34 in the US",
        },
        "positioning_statement": (
            "EcoWear is the sustainable clothing brand for people who want proof, not promises — "
            "affordable, transparent, and built for everyday life rather than outdoor expeditions or luxury wardrobes."
        ),
        "core_message": (
            "We show you exactly where your clothes come from, who made them, and what it cost the planet — "
            "because you deserve to know before you buy."
        ),
        "campaign_hooks": [
            "We filmed our entire supply chain so you never have to take our word for it",
            "The sustainable clothing brand that finally made something for him",
            "Same planet. Better wardrobe. Half the price of Patagonia.",
        ],
        "content_pillars": [
            {
                "pillar": "Supply Chain Transparency",
                "description": "Documentary-style content showing the real journey from raw material to finished garment.",
                "examples": [
                    "Meet the farmer who grew your cotton — a 60-second TikTok",
                    "The true cost of making one EcoWear t-shirt — Instagram carousel",
                    "Carbon calculation breakdown for our bestselling hoodie — Instagram Story",
                ],
            },
            {
                "pillar": "Real People Real Wardrobes",
                "description": "UGC-style content featuring actual customers styling EcoWear in their everyday lives.",
                "examples": [
                    "5 ways I style my EcoWear hoodie for the office — TikTok",
                    "My first month wearing only sustainable brands — Instagram Reel",
                    "Why I switched from Patagonia to EcoWear — customer story post",
                ],
            },
            {
                "pillar": "Myth Busting",
                "description": "Direct comparison content debunking greenwashing myths and positioning EcoWear as the honest alternative.",
                "examples": [
                    "What sustainable fashion brands won't show you — TikTok",
                    "Greenwashing vs real sustainability — what to look for — Instagram carousel",
                    "We checked our competitors' claims so you don't have to — blog post",
                ],
            },
        ],
        "funnel": {
            "awareness": {
                "goal": (
                    "Reach eco-conscious males 25-34 across Instagram and TikTok and introduce the EcoWear brand "
                    "through supply chain transparency content."
                ),
                "tactics": [
                    "TikTok supply chain documentary series — 4 videos over 8 weeks",
                    "Instagram Reels myth-busting series targeting sustainable fashion keywords",
                ],
                "kpi": "Reach 500,000 unique users within the primary segment over 8 weeks",
            },
            "consideration": {
                "goal": (
                    "Convert reached users into engaged followers and email subscribers who actively seek more "
                    "information about EcoWear."
                ),
                "tactics": [
                    "Instagram carousel deep-dives on product sustainability credentials",
                    "Email capture via a free Sustainable Fashion Guide lead magnet",
                ],
                "kpi": "Achieve 2,500 new email subscribers and 1,000 new Instagram followers",
            },
            "conversion": {
                "goal": (
                    "Drive first purchases from the most engaged segment of the awareness audience using a launch offer."
                ),
                "tactics": [
                    "Retargeting ads on Instagram for users who watched 75% of TikTok content",
                    "Email welcome sequence with a 15% first-purchase discount for new subscribers",
                ],
                "kpi": "Generate 150 first-time purchases at an average order value of $85",
            },
        },
        "kpis": [
            "500,000 unique reach within primary segment over 8 weeks",
            "2,500 new email subscribers from lead magnet campaign",
            "150 first-time purchases with average order value of $85",
            "1,000 new Instagram followers from Reels campaign",
            "Average TikTok video completion rate above 45%",
        ],
        "budget_allocation": {
            "paid_ads": 45,
            "content_creation": 25,
            "influencer": 20,
            "tools_and_software": 10,
        },
    }


def _real_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand_name = brief_dict["brand_name"]
    campaign_goal = brief_dict["campaign_goal"]
    budget_amount = brief_dict["budget_amount"]
    budget_currency = brief_dict["budget_currency"]
    campaign_duration_weeks = brief_dict["campaign_duration_weeks"]

    business_summary = context.get("business_summary", "")
    core_strengths = context.get("core_strengths") or []
    tone_guidelines = context.get("tone_guidelines") or []
    budget_tier = context.get("budget_tier", "")
    budget_weekly = context.get("budget_weekly", 0.0)
    recommended_focus = context.get("recommended_focus", "")

    content_gaps = context.get("content_gaps") or []
    positioning_opportunity = context.get("positioning_opportunity", "")
    channels_to_prioritize = context.get("channels_to_prioritize") or []
    competitor_weaknesses_to_exploit = context.get("competitor_weaknesses_to_exploit") or []

    persona_name = context.get("persona_name", "")
    pain_points = context.get("pain_points") or []
    desires = context.get("desires") or []
    messaging_hooks = context.get("messaging_hooks") or []
    platform_behaviour = context.get("platform_behaviour") or {}
    if not isinstance(platform_behaviour, dict):
        platform_behaviour = {}

    pat = platform_behaviour.get("most_active_times", "evenings")
    pcc = platform_behaviour.get("content_consumption", "mixed")

    system_prompt = (
        "You are a campaign strategist. Synthesize brand, competitive, and audience data into an "
        "execution blueprint. Return ONLY a valid JSON object matching the requested schema. No markdown."
    )

    user_prompt = f"""Create a full campaign strategy for {brand_name}.
Goal: {campaign_goal}
Duration: {campaign_duration_weeks} weeks
Budget: {budget_amount} {budget_currency} ({budget_tier} tier, {budget_weekly}/week)

Context:
- Summary: {business_summary}
- Strengths: {", ".join(str(s) for s in core_strengths)}
- Focus: {recommended_focus}
- Tone: {"; ".join(str(t) for t in tone_guidelines)}
- Angle: {positioning_opportunity}
- Gaps: {", ".join(str(g) for g in content_gaps)}
- Channels: {", ".join(str(c) for c in channels_to_prioritize)}
- Exploits: {", ".join(str(w) for w in competitor_weaknesses_to_exploit)}
- Persona: {persona_name} (Pains: {", ".join(str(p) for p in pain_points)} | Desires: {", ".join(str(d) for d in desires)})
- Hooks: {", ".join(str(h) for h in messaging_hooks)}
- Activity: Active {pat}, consumes {pcc} content
Return a JSON object with exactly these keys:
{{
  "campaign_summary": {{
    "name": "Creative name",
    "tagline": "One-line hook/tagline",
    "duration_weeks": {campaign_duration_weeks},
    "total_budget": {budget_amount}
  }},
  "positioning_statement": "For [persona] who [pain], [brand] offers [solution] unlike [competitors].",
  "core_message": "Single most important message of the campaign",
  "campaign_hooks": ["array of 3 distinct ad/post hooks"],
  "content_pillars": [
    {{"pillar": "Name", "description": "Core themes", "share_percentage": 0}}
  ],
  "funnel": {{
    "awareness": ["2-3 specific top-of-funnel tactics/content mechanics"],
    "consideration": ["2-3 middle-of-funnel nurture tactics"],
    "conversion": ["2-3 conversion/offer tactics"]
  }},
  "kpis": [
    {{"metric": "e.g., Conversion Rate", "target": "e.g., 2.5%", "source": "e.g., Shopify Analytics"}}
  ],
  "budget_allocation_percentage": {{
    "paid_ads": 0,
    "content_creation": 0,
    "influencer": 0,
    "tools_and_software": 0
  }}
}}"""

    return call_claude(system_prompt, user_prompt, max_tokens=1500)
