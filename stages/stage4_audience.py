"""Stage 4: Audience Analysis — segment definition and prioritization."""

import json

from utils.llm_client import call_claude

USE_MOCK = True


def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
    return {
        "segments": [
            {
                "name": "Conscious Millennial Males",
                "demographics": "Males aged 25-34, urban US cities, mid-income $45K-$75K, college-educated professionals.",
                "psychographics": "Value environmental responsibility and personal integrity, follow sustainability content online, make deliberate purchase decisions based on brand ethics.",
                "pain_points": [
                    "Most sustainable fashion brands ignore them entirely or feel too feminine in aesthetic.",
                    "They distrust greenwashing claims and have no easy way to verify a brand's actual sustainability.",
                    "Premium sustainable options like Patagonia are out of their everyday fashion budget range.",
                ],
                "preferred_channels": ["Instagram", "TikTok"],
                "message_angle": "Finally a sustainable clothing brand built for men who care — with the proof to back it up.",
            },
            {
                "name": "Eco-Conscious Millennial Women",
                "demographics": "Females aged 25-35, US suburban and urban, mid-income $40K-$70K, lifestyle and wellness oriented.",
                "psychographics": "Actively reduce environmental footprint across food, transport, and fashion, follow slow fashion creators, share values-aligned brands with their networks.",
                "pain_points": [
                    "Reformation and similar brands feel aspirational but not affordable for everyday wardrobe building.",
                    "Fast fashion sustainability lines feel dishonest and make it harder to identify brands they can actually trust.",
                    "They want to advocate for brands they believe in but need compelling content to share with their communities.",
                ],
                "preferred_channels": ["Instagram", "Email"],
                "message_angle": "Sustainable fashion that fits your everyday life and your values — at a price that makes sense.",
            },
            {
                "name": "Gen Z First-Time Sustainables",
                "demographics": "Males and females aged 18-24, US college towns and cities, lower income $20K-$40K, digitally native.",
                "psychographics": "Climate anxiety is a daily reality, actively seek brands that align with their identity, highly influenced by creators and peer recommendations on TikTok.",
                "pain_points": [
                    "Budget constraints make sustainable fashion feel like a luxury they cannot justify yet.",
                    "Overwhelmed by the number of brands claiming sustainability with no clear way to compare them.",
                    "Want to participate in sustainable fashion culture but lack entry-level options with strong brand identity.",
                ],
                "preferred_channels": ["TikTok", "Instagram"],
                "message_angle": "Your first sustainable wardrobe staple — real impact, real price, real proof.",
            },
        ],
        "primary_segment": "Conscious Millennial Males",
        "primary_segment_reason": (
            "This segment is the largest underserved gap in the sustainable fashion market, directly validated by the competitor analysis, "
            "and aligns with EcoWear's USP of accessible pricing and transparency — making it the highest-return focus for a $15,000 awareness campaign."
        ),
    }


def _real_output(brief: dict, context: dict) -> dict:
    s2 = context.get("stage2") or {}
    s3 = context.get("stage3") or {}
    positioning = context.get("brand_positioning") or s2.get("brand_positioning", "")
    growth_ops = context.get("growth_opportunities") or s2.get("growth_opportunities", [])
    market_gaps = context.get("market_gaps") or s3.get("market_gaps", [])
    differentiation = context.get("recommended_differentiation") or s3.get("recommended_differentiation", "")

    system_prompt = (
        "You are a senior audience strategist specializing in consumer segmentation.\n"
        "You produce precise, actionable persona cards grounded in real consumer behavior.\n"
        "You always respond in valid JSON only — no markdown, no explanation, no preamble.\n"
        "Every insight must be specific to the brand, market, and competitive context provided.\n"
        "Never produce generic demographic buckets — each segment must feel like a real person with real motivations."
    )

    user_prompt = (
        "Produce full audience persona cards for this brand's campaign.\n\n"
        f"Brand: {brief['brand_name']}\n"
        f"Product: {brief['product_or_service']}\n"
        f"Industry: {brief['industry']}\n"
        f"Market: {brief['target_market']}\n"
        f"Campaign Goal: {brief['campaign_goal']}\n"
        f"USP: {brief['unique_selling_point']}\n\n"
        "Business Analysis:\n"
        f"Positioning: {positioning}\n"
        f"Growth Opportunities: {growth_ops}\n\n"
        "Competitive Intelligence:\n"
        f"Market Gaps: {market_gaps}\n"
        f"Recommended Differentiation: {differentiation}\n\n"
        "Instructions:\n"
        "- Produce exactly 3 audience segments\n"
        "- Each segment must have: name, demographics, psychographics, pain_points (3 items), preferred_channels, message_angle\n"
        "- Identify one primary_segment and explain in one sentence why it gives the best return for this campaign goal and budget\n"
        "- Return valid JSON matching exactly this structure:\n"
        "{\n"
        '  "segments": [...],\n'
        '  "primary_segment": "...",\n'
        '  "primary_segment_reason": "..."\n'
        "}"
    )

    result = call_claude(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
    json.dumps(result)
    return result
