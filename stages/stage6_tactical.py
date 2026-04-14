"""Stage 6: Tactical Planning — execution-level content and channel actions."""

import json

from utils.llm_client import call_claude

USE_MOCK = True


def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
    return {
        "ad_copy_examples": [
            {
                "channel": "TikTok",
                "format": "60-second documentary video",
                "headline": "We filmed our entire supply chain. Here is what we found.",
                "body": "Most brands tell you they are sustainable. We decided to show you instead. From the organic cotton farm in Texas to the carbon-neutral factory in Portugal — every step, on camera, no edits.",
                "cta": "Follow to see the full series",
            },
            {
                "channel": "Instagram",
                "format": "Carousel — 6 slides",
                "headline": "The true cost of making one EcoWear t-shirt",
                "body": "Slide 1: The cotton. Slide 2: The farmer. Slide 3: The factory. Slide 4: The shipping. Slide 5: The carbon offset. Slide 6: The price you pay. Nothing hidden.",
                "cta": "Swipe to see where your money actually goes",
            },
            {
                "channel": "Instagram",
                "format": "Reel — 15 seconds",
                "headline": "Sustainable fashion finally made for him",
                "body": "Not outdoor gear. Not luxury basics. Just clean, honest everyday clothing — with the receipts to prove it.",
                "cta": "Shop EcoWear — link in bio",
            },
            {
                "channel": "TikTok",
                "format": "30-second myth-busting video",
                "headline": "What sustainable fashion brands won't show you",
                "body": "We checked three of our competitors' sustainability claims so you don't have to. Here is what we found — and here is what we do differently.",
                "cta": "Comment your brand and we will check it next",
            },
        ],
        "cta_recommendations": [
            "Follow to see the full supply chain series — drives account growth from documentary content",
            "Get the free Sustainable Fashion Guide — drives email capture for consideration nurture",
            "Shop now and get 15% off your first order — drives conversion from retargeting audiences",
        ],
        "creative_direction": (
            "Raw, documentary-style visual language — natural lighting, real locations, no studio polish. "
            "The aesthetic should feel like a Vice documentary crossed with a Patagonia field journal. "
            "Color palette is earthy and muted: cream, olive, slate, raw cotton white. Typography is clean sans-serif. "
            "No stock photography. Every piece of creative should feel like it was captured on location, not designed in an agency."
        ),
        "posting_frequency": {
            "Instagram": "4 posts per week — 2 Reels, 1 Carousel, 1 Story sequence",
            "TikTok": "3 videos per week — 1 documentary segment, 1 myth-busting video, 1 community response",
        },
        "ab_test_suggestions": [
            {
                "element": "TikTok hook — first 3 seconds",
                "variant_a": "Open with the farm — visual-first, no voiceover for 3 seconds",
                "variant_b": "Open with a bold text claim — 'Most brands lie about sustainability'",
                "success_metric": "Video completion rate at 45 seconds",
            },
            {
                "element": "Instagram Reel CTA",
                "variant_a": "Shop EcoWear — link in bio",
                "variant_b": "Get the free Sustainable Fashion Guide — link in bio",
                "success_metric": "Click-through rate from profile link",
            },
            {
                "element": "Email subject line for welcome sequence",
                "variant_a": "Here is exactly where your EcoWear order comes from",
                "variant_b": "The sustainable fashion guide no brand wanted us to publish",
                "success_metric": "Email open rate at 48 hours",
            },
        ],
    }


def _real_output(brief: dict, context: dict) -> dict:
    s5 = context.get("stage5") or {}
    cs = s5.get("campaign_summary") or {}
    primary = context.get("primary_segment", "")
    segments = context.get("segments", [])
    message_angle = ""
    for s in segments:
        if s.get("name") == primary:
            message_angle = s.get("message_angle", "")
            break

    system_prompt = (
        "You are a senior creative strategist and media planner with deep expertise in social media advertising and content marketing.\n"
        "You produce precise, platform-native tactical plans with real ad copy, not placeholder text.\n"
        "You always respond in valid JSON only — no markdown, no explanation, no preamble.\n"
        "Every piece of ad copy must sound like it was written by a human copywriter for this specific brand — never generic.\n"
        "A/B test suggestions must be specific and testable with a clear measurable success metric."
    )

    user_prompt = (
        "Produce a full tactical execution plan for this campaign.\n\n"
        f"Brand: {brief['brand_name']}\n"
        f"Product: {brief['product_or_service']}\n"
        f"Campaign Theme: {cs.get('campaign_theme', '')}\n"
        f"Tone of Voice: {cs.get('tone_of_voice', '')}\n"
        f"Channel Mix: {cs.get('channel_mix', [])}\n"
        f"Core Message: {s5.get('core_message', '')}\n"
        f"Campaign Hooks: {s5.get('campaign_hooks', [])}\n"
        f"Content Pillars: {s5.get('content_pillars', [])}\n"
        f"Primary Audience: {primary}\n"
        f"Message Angle: {message_angle}\n\n"
        "Instructions:\n"
        "- Write exactly 4 ad copy examples — one per channel/format combination from the channel mix\n"
        "- Each ad copy must feel native to its platform — TikTok copy sounds different from Instagram copy\n"
        "- Write exactly 3 CTA recommendations — one per funnel stage (awareness, consideration, conversion)\n"
        "- Creative direction must be one specific paragraph — visual style, mood, color palette, and what to avoid\n"
        "- Posting frequency must cover every channel in the channel mix\n"
        "- Write exactly 3 A/B test suggestions — each with a specific measurable success metric\n"
        "- Return valid JSON matching exactly the output structure specified"
    )

    result = call_claude(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
    json.dumps(result)
    return result
