"""Stage 5: Strategic Framework — messaging and channel strategy blueprint."""

import json

from utils.llm_client import call_claude

USE_MOCK = True


def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
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
            "total": 15000,
            "currency": "USD",
            "breakdown": {
                "Instagram": {"amount": 6000, "percentage": 40},
                "TikTok": {"amount": 5250, "percentage": 35},
                "Email Marketing": {"amount": 1500, "percentage": 10},
                "Content Creation": {"amount": 2250, "percentage": 15},
            },
        },
    }


def _real_output(brief: dict, context: dict) -> dict:
    primary = context.get("primary_segment", "")
    segments = context.get("segments", [])
    message_angle = ""
    for s in segments:
        if s.get("name") == primary:
            message_angle = s.get("message_angle", "")
            break

    system_prompt = (
        "You are a senior marketing strategist with 20 years of experience designing campaign strategies for consumer brands.\n"
        "You produce full strategic blueprints that are specific, creative, and grounded in real audience and competitive intelligence.\n"
        "You always respond in valid JSON only — no markdown, no explanation, no preamble.\n"
        "Every element must directly reflect the brand, audience, competitive gaps, and business context provided.\n"
        "Never produce generic strategies. Every hook, pillar, and tactic must feel purpose-built for this exact brand."
    )

    user_prompt = (
        "Design a full campaign strategy for this brand.\n\n"
        f"Brand: {brief['brand_name']}\n"
        f"Product: {brief['product_or_service']}\n"
        f"Campaign Goal: {brief['campaign_goal']}\n"
        f"Budget: {brief['budget_amount']} {brief['budget_currency']}\n"
        f"Duration: {brief['campaign_duration_weeks']} weeks\n"
        f"USP: {brief['unique_selling_point']}\n\n"
        "Business Positioning:\n"
        f"{context.get('brand_positioning', '')}\n\n"
        "Primary Audience Segment:\n"
        f"Name: {primary}\n"
        f"Reason: {context.get('primary_segment_reason', '')}\n"
        f"Message Angle: {message_angle}\n\n"
        "Competitive Differentiation:\n"
        f"{context.get('recommended_differentiation', '')}\n\n"
        "Market Gaps to Target:\n"
        f"{context.get('market_gaps', [])}\n\n"
        "Instructions:\n"
        "- Design a strategy that directly exploits the market gaps and differentiation identified above\n"
        "- The campaign theme must feel creative and ownable — not a generic tagline\n"
        "- Produce exactly 3 content pillars, each with 3 content examples\n"
        f"- Budget allocation must sum exactly to {brief['budget_amount']} {brief['budget_currency']}\n"
        "- All percentages in budget breakdown must sum to exactly 100\n"
        "- Return valid JSON matching exactly the output structure specified"
    )

    result = call_claude(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
    json.dumps(result)
    return result
