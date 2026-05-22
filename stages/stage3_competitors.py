"""Stage 3: Competitor Analysis — competitor mapping and market gaps."""
import logging
import json
from utils.claude_client import call_claude
from utils.llm_runtime import run_llm_stage

logger = logging.getLogger("campaign_model.llm")


def run(brief_dict: dict, context: dict, job_id: str) -> dict:
    return run_llm_stage("Stage 3", _mock_output, _real_output, brief_dict, context, job_id)


def _mock_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    return {
        "competitor_summary": (
            "Pod-led incumbents own convenience; premium design brands own aesthetics; few combine AI guidance with ritual storytelling."
        ),
        "content_gaps": [
            "Short-form proof of repeatable morning ritual outcomes",
            "Honest comparisons that respect pods without trashing users",
            "Localized Gulf and Egypt creator angles for remote workers",
        ],
        "positioning_opportunity": (
            "Win as the smart ritual brand that teaches taste improvement over time—not another hardware-only SKU."
        ),
        "channels_to_prioritize": ["Instagram", "TikTok", "Facebook"],
        "topics_to_avoid": ["Aggressive pod-shaming", "Unverified health claims", "Deep discount framing"],
        "competitor_weaknesses_to_exploit": [
            "Limited personalization narratives from pod systems",
            "Premium competitors lack accessible AI onboarding",
            "Few authentic regional home-barista stories in MENA feeds",
        ],
    }


def enrich_competitor(competitor: dict, industry: str) -> dict:
    """
    Heuristically infers strengths, weaknesses, and positioning tags 
    locally using deterministic rule-based mapping to slash API token expenses.
    """
    platforms = [p.lower() for p in competitor.get("platforms", []) if isinstance(p, str)]
    notes = competitor.get("notes", "").strip()
    
    strengths = []
    weaknesses = []
    
    # Platform-specific heuristic evaluations
    if "instagram" in platforms:
        strengths.append("strong visual branding and aesthetic consistency")
        weaknesses.append("high dependence on organic platform algorithm visibility")
    if "tiktok" in platforms:
        strengths.append("strong short-form trend engagement and rapid reach")
        weaknesses.append("low customer retention and long-term audience loyalty")
    if "youtube" in platforms:
        strengths.append("high long-form educational authority and deep product explanations")
        weaknesses.append("high content production costs and slower deployment cycles")
    if "linkedin" in platforms:
        strengths.append("established professional B2B authority and domain positioning")
        weaknesses.append("limited emotional lifestyle storytelling or creative mass-appeal")
    if "twitter" in platforms or "x" in platforms:
        strengths.append("agile real-time audience engagement and public commentary")
        weaknesses.append("high conversational volatility and shorter content shelf-life")
        
    # General fallback thresholds if no platforms match
    if not strengths:
        strengths = ["established market presence", "existing baseline customer segment"]
        weaknesses = ["limited digital market differentiation", "passive multichannel strategy deployment"]
        
    # Infer Positioning Tags from Industry + Notes keywords
    notes_lower = notes.lower()
    if any(k in notes_lower for k in ["premium", "luxury", "high-end", "expensive"]):
        positioning_tag = f"premium {industry.lower()} positioning"
    elif any(k in notes_lower for k in ["cheap", "affordable", "budget", "low cost"]):
        positioning_tag = f"value-driven {industry.lower()} accessibility"
    elif any(k in notes_lower for k in ["fast", "quick", "convenient", "easy"]):
        positioning_tag = f"convenience-optimized {industry.lower()} focus"
    else:
        positioning_tag = f"standard {industry.lower()} incumbent positioning"
        
    return {
        "name": competitor.get("name", "Unknown Competitor"),
        "platforms": competitor.get("platforms", []),
        "notes": notes if notes else "No additional background context provided.",
        "inferred_strengths": strengths,
        "inferred_weaknesses": weaknesses,
        "positioning_tag": positioning_tag
    }


def _format_competitors_block(competitors: list | None, industry: str) -> str:
    if not competitors:
        return "No specified market competitors provided."
        
    lines = []
    for c in competitors:
        if not isinstance(c, dict) or not c.get("name"):
            continue
            
        # Enrich each object dynamically in the backend data processor layer
        enriched = enrich_competitor(c, industry)
        
        block = (
            f"- Competitor: {enriched['name']}\n"
            f"  Platforms: {', '.join(enriched['platforms']) if enriched['platforms'] else 'None'}\n"
            f"  Strategic Tag: {enriched['positioning_tag']}\n"
            f"  Context Notes: {enriched['notes']}\n"
            f"  Inferred Strengths: {', '.join(enriched['inferred_strengths'])}\n"
            f"  Inferred Weaknesses: {', '.join(enriched['inferred_weaknesses'])}"
        )
        lines.append(block)
        
    return "\n\n".join(lines) if lines else "No specified market competitors provided."


def _real_output(brief_dict: dict, context: dict, job_id: str) -> dict:
    brand_name = brief_dict["brand_name"]
    industry = brief_dict["industry"]
    current_channels = brief_dict.get("current_channels") or []
    competitors = brief_dict.get("competitors")

    tone_descriptor = context.get("tone_descriptor", "")
    recommended_focus = context.get("recommended_focus", "")

    # Execute backend dynamic enrichment matching matrix
    formatted_competitors_block = _format_competitors_block(competitors, industry)

    system_prompt = (
        "You are a competitive intelligence analyst. Return ONLY a valid JSON object "
        "matching the requested schema. No markdown, backticks, or text outside JSON."
    )

    channels_joined = ", ".join(current_channels) if current_channels else "None"

    user_prompt = f"""
Brand: {brand_name} ({industry})
Tone: {tone_descriptor}
Focus: {recommended_focus}
Channels: {channels_joined}

Competitor Intelligence Profiles:
{formatted_competitors_block}

Return a JSON object matching this structure exactly:
{{
  "competitor_summary": "2 sentence landscape overview",
  "content_gaps": ["array of 3 specific content gaps"],
  "positioning_opportunity": "one clear differentiation sentence",
  "channels_to_prioritize": ["array of 2 target channels"],
  "topics_to_avoid": ["array of 2 saturated/irrelevant topics"],
  "competitor_weaknesses_to_exploit": ["array of 2 exploitable weaknesses"]
}}
"""

    return call_claude(system_prompt, user_prompt, max_tokens=700)