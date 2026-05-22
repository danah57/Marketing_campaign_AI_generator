import json
import logging
from services.influencer_loader import (
    load_influencer_candidates,
    parse_engagement,
    parse_followers,
)
from utils.claude_client import call_claude
from utils.llm_runtime import use_mock_llm

logger = logging.getLogger("campaign_model.llm")

# [Scoring Engine, Ranking, and Selection remain unchanged for deterministic execution safety]
def compute_influencer_score(inf, campaign):
    score = 0.0
    reasons = []
    if inf["primaryPlatform"] in campaign.get("channels", []):
        score += 2.0
        reasons.append("Platform match")
    engagement = inf.get("engagementRate", 0) or 0
    score += min(engagement * 1.5, 3.0)
    if engagement > 5: reasons.append("High engagement")
    elif engagement > 2: reasons.append("Moderate engagement")
    followers = inf.get("followersCount", 0) or 0
    score += min(followers / 200000, 2.0)
    if followers > 200000: reasons.append("Strong reach")
    overlap = set(inf.get("categories") or []) & set(campaign.get("categories") or [])
    if overlap:
        score += len(overlap) * 1.5
        reasons.append(f"Categories: {', '.join(overlap)}")
    if inf.get("audienceAgeRange") == campaign.get("age_range"):
        score += 1.5
        reasons.append("Age match")
    if inf.get("audienceGender") == campaign.get("gender"):
        score += 1.0
        reasons.append("Gender match")
    return round(score, 2), reasons

def rank_influencers(influencers, campaign):
    ranked = []
    for inf in influencers:
        score, reasons = compute_influencer_score(inf, campaign)
        ranked.append({**inf, "fit_score": score, "fit_reasoning": reasons})
    return sorted(ranked, key=lambda x: x["fit_score"], reverse=True)

def select_top_influencers(ranked, top_k=3, threshold=5.0):
    return [inf for inf in ranked if inf["fit_score"] >= threshold][:top_k]

def _candidates_from_brief(brief_dict: dict) -> list[dict]:
    raw = brief_dict.get("influencer_candidates") or []
    if not raw:
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            row = dict(item)
        else:
            row = item.model_dump() if hasattr(item, "model_dump") else {}
        if row.get("id") is not None:
            out.append(
                {
                    "id": row["id"],
                    "primaryPlatform": row.get("primaryPlatform"),
                    "followersCount": parse_followers(row.get("followersCount")),
                    "engagementRate": parse_engagement(row.get("engagementRate")),
                    "categories": row.get("categories") or [],
                    "contentTypes": row.get("contentTypes") or [],
                    "collaborationTypes": row.get("collaborationTypes") or [],
                    "audienceAgeRange": row.get("audienceAgeRange"),
                    "audienceGender": row.get("audienceGender"),
                    "audienceLocation": row.get("audienceLocation"),
                    "interests": row.get("interests") or [],
                }
            )
    return out


def run(brief_dict: dict, context: dict, job_id: str) -> dict:
    candidates = _candidates_from_brief(brief_dict)
    if not candidates:
        candidates = load_influencer_candidates()
    if not candidates:
        return {
            "influencer_matches": [],
            "influencer_strategy_note": "",
            "influencer_stage_skipped": True
        }

    campaign = {
        "channels": context.get("channels_to_prioritize") or brief_dict.get("current_channels") or [],
        "categories": brief_dict.get("industry_categories") or [],
        "age_range": context.get("age_range"),
        "gender": context.get("gender_skew")
    }

    selected = select_top_influencers(rank_influencers(candidates, campaign))
    
    # Prune elements for Claude processing to avoid wasting tokens
    llm_candidates = []
    matches = []
    
    for inf in selected:
        # Build basic return dictionary structure
        match_obj = {
            "influencer_id": inf["id"],
            "fit_score": inf["fit_score"],
            "fit_reasoning": " | ".join(inf["fit_reasoning"]),
            "suggested_collaboration_type": (inf.get("collaborationTypes") or ["Sponsored Post"])[0],
            "suggested_budget_usd": 0.0,
            "outreach_message": ""
        }
        matches.append(match_obj)
        
        # Only hand over what Claude strictly needs to read
        llm_candidates.append({
            "id": inf["id"],
            "platform": inf.get("primaryPlatform"),
            "reasons": inf["fit_reasoning"]
        })

    if use_mock_llm() or not matches:
        return {
            "influencer_matches": matches,
            "influencer_strategy_note": "Selection optimized by platform parameters.",
            "influencer_stage_skipped": False
        }

    system_prompt = (
        "You are an influencer relations specialist. Generate personalized outreach copy "
        "and campaign summaries. Return ONLY a valid JSON object matching the requested schema. No markdown."
    )

    user_prompt = f"""
Brand: {brief_dict.get('brand_name')}
Campaign Config: {json.dumps(campaign)}
Selected Creators: {json.dumps(llm_candidates)}

Return a JSON object matching this structure exactly:
{{
  "strategy_note": "A 2-3 sentence macro overview explaining why this collective group fits the campaign direction",
  "outreach_templates": {{
    "creator_id_here": "A personalized, high-converting 2-3 sentence outreach message tailored specifically to their platform and matching reasons"
  }}
}}
"""

    try:
        # Request data processing with optimized ceiling boundary limits
        explanation = call_claude(system_prompt, user_prompt, max_tokens=800)
        
        # Parse returned data structures back smoothly into match components
        strategy_note = explanation.get("strategy_note", "")
        outreach_map = explanation.get("outreach_templates", {})
        
        for m in matches:
            str_id = str(m["influencer_id"])
            if str_id in outreach_map:
                m["outreach_message"] = outreach_map[str_id]
            else:
                # Safe functional fallback string if processing keys misalign
                m["outreach_message"] = f"Hi! We love your content and think you'd be a perfect fit for our campaign. Let's collaborate!"

    except Exception as e:
        logger.warning(f"LLM Influencer optimization processing failed: {e}")
        strategy_note = "Selection compiled using structural engine rules."

    return {
        "influencer_matches": matches,
        "influencer_strategy_note": strategy_note,
        "influencer_stage_skipped": False
    }