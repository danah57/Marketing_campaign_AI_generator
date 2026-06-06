"""Stage 6: Tactical Planning — execution-level content and channel actions."""

import json
import logging

from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")


def run(brief_dict: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 6", _mock_output, _real_output, brief_dict, context, job_id)


def _mock_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    return {
        "platform_content": {
            "Instagram": {
                "format_recommendation": "reel",
                "optimal_posting_time": "Tuesday–Thursday, 7–9pm",
                "posts": [
                    {
                        "variant": "A",
                        "caption": (
                            "Mock caption A for Instagram. Bold opening line that stops the scroll. "
                            "This campaign is built for results. #MockCampaign #NexBrandAI"
                        ),
                        "hashtags": ["#MockCampaign", "#NexBrandAI", "#Marketing"],
                        "visual_direction": (
                            "Bright, high-contrast product shot with minimal text overlay. "
                            "Person using the product in a real-life setting."
                        ),
                        "cta": "Tap the link in bio to learn more.",
                        "virality_score": 7.5,
                        "virality_reasoning": (
                            "Strong opening hook + clear CTA + relevant hashtags pushes score above average."
                        ),
                    },
                    {
                        "variant": "B",
                        "caption": (
                            "Mock caption B — different angle. What if your biggest problem had a simple solution? "
                            "We built it. #MockCampaign #NexBrandAI"
                        ),
                        "hashtags": ["#MockCampaign", "#NexBrandAI", "#Innovation"],
                        "visual_direction": (
                            "Before/after split screen. Left side shows the problem, right side shows the product solving it."
                        ),
                        "cta": "Comment YES if you want to know more.",
                        "virality_score": 8.1,
                        "virality_reasoning": (
                            "Question hook + emotional contrast + engagement-bait CTA earns high virality score."
                        ),
                    },
                ],
            },
            "Facebook": {
                "format_recommendation": "carousel",
                "optimal_posting_time": "Wednesday, 12–2pm",
                "posts": [
                    {
                        "variant": "A",
                        "caption": (
                            "Mock Facebook caption A. Longer form, storytelling tone. "
                            "Here is why this product changes everything for people like you. Share with someone who needs this."
                        ),
                        "hashtags": ["#MockCampaign"],
                        "visual_direction": (
                            "3-slide carousel: slide 1 is problem statement, slide 2 is product feature, "
                            "slide 3 is social proof quote."
                        ),
                        "cta": "Share this with a friend who needs it.",
                        "virality_score": 6.8,
                        "virality_reasoning": (
                            "Share-based CTA boosts social spread potential but no strong emotional hook in opening line."
                        ),
                    },
                    {
                        "variant": "B",
                        "caption": (
                            "Mock Facebook caption B. Data-led angle. 9 out of 10 customers say this changed their routine. "
                            "Here is what they found."
                        ),
                        "hashtags": ["#MockCampaign"],
                        "visual_direction": (
                            "Static image with bold statistic overlay. Clean white background, brand colours."
                        ),
                        "cta": "Click to read the full story.",
                        "virality_score": 7.2,
                        "virality_reasoning": (
                            "Social proof data in opening sentence is a strong trust signal that improves click-through."
                        ),
                    },
                ],
            },
        },
        "content_creation_checklist": [
            "Film 2 reel variations for Instagram using the visual directions above",
            "Prepare 3-slide carousel assets for Facebook",
            "Write caption copy into scheduling tool before posting",
            "Review all posts against brand tone guidelines before publishing",
        ],
        "campaign_hashtag_set": [
            "#MockCampaign",
            "#NexBrandAI",
            "#Marketing",
            "#Innovation",
            "#ContentStrategy",
        ],
        "posting_frequency": {
            "Instagram": 4,
            "Facebook": 3,
        },
    }


def _real_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand_name = brief_dict["brand_name"]
    current_channels = list(brief_dict.get("current_channels") or [])
    campaign_duration_weeks = brief_dict["campaign_duration_weeks"]
    budget_currency = brief_dict["budget_currency"]

    core_message = context.get("core_message", "")
    campaign_hooks = context.get("campaign_hooks") or []
    content_pillars = context.get("content_pillars") or []
    tone_guidelines = context.get("tone_guidelines") or []
    positioning_statement = context.get("positioning_statement", "")
    persona_name = context.get("persona_name", "")
    messaging_hooks = context.get("messaging_hooks") or []
    platform_behaviour = context.get("platform_behaviour") or {}
    if not isinstance(platform_behaviour, dict):
        platform_behaviour = {}

    channels_to_prioritize = list(context.get("channels_to_prioritize") or [])
    merged: list[str] = []
    seen: set[str] = set()
    for ch in [*current_channels, *channels_to_prioritize]:
        if ch and ch not in seen:
            seen.add(str(ch))
            merged.append(str(ch))
    active_channels = merged if merged else ["Instagram", "Facebook"]

    budget_allocation = context.get("budget_allocation") or {}

    system_prompt = (
        "You are a senior social media content strategist nd media planner with deep expertise in social media advertising and content creation.\n"
        "You create platform-specific content with psychological hooks and measurable virality potential.\n"
        "Always respond with valid JSON only. No markdown. No explanation outside JSON.\n"
        "\n"
        "Virality Score rules (1-10):\n"
        "- +2 if caption opens with a question or bold statement\n"
        "- +2 if there is a clear emotional hook (curiosity, aspiration, urgency, humor)\n"
        "- +1 if it includes a strong call to action\n"
        "- +1 if it uses social proof or data\n"
        "- +1 if it has platform-native format (reel hook, carousel tease, thread opener)\n"
        "- -1 if caption is over the platform character limit\n"
        "- -1 if it has no hashtags (Instagram/TikTok)\n"
        "Score is a float. Explain the score in one sentence."
    )

    active_json = json.dumps(active_channels)
    tone_joined = "; ".join(str(t) for t in tone_guidelines)
    hooks_joined = ", ".join(str(h) for h in messaging_hooks)

    user_prompt = f"""Generate a tactical content plan for {brand_name}.

Brand: {brand_name}
Core Message: {core_message}
Positioning: {positioning_statement}
Tone: {tone_joined}
Target Persona: {persona_name}
Hooks: {hooks_joined}
Duration: {campaign_duration_weeks}w ({budget_currency})
Platforms: {", ".join(active_channels)}
Data Profiles:
Pillars: {json.dumps(content_pillars)}
Behavior: {json.dumps(platform_behaviour)}
Allocation: {json.dumps(budget_allocation)}

For EACH platform in {active_json}, generate 2 variants (A and B).
Respect strict native length rules (IG: 2200 chars, X: 280, LI: 3000, TikTok: script outline, FB: 500).
Keep captions concise (under 400 chars unless platform requires more). One sentence per visual_direction and conversion_rationale.

Return a JSON object matching this structure exactly:
{{
  "platform_content": {{
    "<platform_name>": {{
      "format_recommendation": "reel|carousel|static|thread|story",
      "optimal_posting_time": "e.g., Tuesday 7-9pm",
      "posts": [
        {{
          "variant": "A|B",
          "caption": "Full native caption text with hook",
          "hashtags": ["3-5 hyper-relevant tags"],
          "visual_direction": "Visual or video direction details",
          "cta": "Call to action text",
          "engagement_triggers": ["2 psychological triggers used, e.g., Curiosity Loop"],
          "conversion_rationale": "One sentence explaining why this hooks the target persona"
        }}
      ]
    }}
  }},
  "content_creation_checklist": ["3 core creative production steps"],
  "campaign_hashtag_set": ["5 core master campaign hashtags"]
}}"""

    return call_claude(system_prompt, user_prompt, max_tokens=4096)
