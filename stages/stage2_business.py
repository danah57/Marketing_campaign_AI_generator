"""Stage 2: Business Analysis — SWOT, brand positioning, challenges, growth opportunities."""

from utils.llm_client import call_claude

USE_MOCK = True

def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
    return {
        "swot": {
            "strengths": [
                "EcoWear's 100% organic materials and carbon-neutral shipping directly address the top purchase drivers for eco-conscious consumers in the US fashion market.",
                "Operating as a small brand allows EcoWear to move faster, tell an authentic story, and build community-level loyalty that large competitors cannot replicate.",
                "The sustainable fashion segment is growing at 9.7% annually, positioning EcoWear in a high-momentum market with strong tailwinds.",
            ],
            "weaknesses": [
                "As a small company, EcoWear has limited budget headroom compared to established sustainable brands like Patagonia, restricting paid media scale.",
                "Brand awareness is currently low, meaning EcoWear must invest heavily in top-of-funnel activity before conversion campaigns can be effective.",
                "Organic and carbon-neutral supply chains typically carry higher production costs, which may compress margins and limit promotional flexibility.",
            ],
            "opportunities": [
                "Growing consumer distrust of greenwashing creates an opening for EcoWear to differentiate through radical transparency in its supply chain.",
                "Instagram and TikTok creator communities around sustainability are highly engaged and under-monetized, offering cost-efficient awareness at scale.",
                "Expanding into corporate gifting and B2B sustainable merchandise represents an untapped revenue stream that aligns with EcoWear's existing USP.",
            ],
            "threats": [
                "Patagonia and other premium sustainable brands have deep brand equity and customer loyalty that will be difficult to compete with on awareness alone.",
                "Fast fashion brands are increasingly launching sustainability sub-lines, blurring consumer perception of what genuine sustainability means.",
                "Rising costs of organic raw materials and logistics could erode EcoWear's price competitiveness in a cost-sensitive consumer segment.",
            ],
        },
        "brand_positioning": (
            "EcoWear is the transparent, accessible sustainable clothing brand for US "
            "consumers who want to make environmentally responsible fashion choices "
            "without compromising on style or paying luxury prices."
        ),
        "key_challenges": [
            "Building brand awareness from a low base against well-funded competitors within a $15,000 USD campaign budget requires precise channel targeting and strong organic amplification.",
            "Communicating the authentic difference between EcoWear's genuine sustainability credentials and the greenwashing of fast fashion competitors in a crowded content landscape.",
            "Converting awareness into lasting brand affinity among eco-conscious millennials who are highly informed, skeptical of marketing, and loyal to established sustainable brands.",
        ],
        "growth_opportunities": [
            "Leveraging user-generated content from early customers as social proof to reduce paid acquisition costs and build authentic community around the EcoWear brand.",
            "Partnering with micro-influencers in the sustainability and slow fashion niches on Instagram and TikTok where audience trust is high and sponsorship costs are low.",
            "Launching a brand storytelling content series documenting EcoWear's supply chain transparency to build SEO authority and differentiate from competitors who avoid this level of disclosure.",
        ],
    }


def _real_output(brief: dict, context: dict) -> dict:
    system_prompt = (
        "You are a senior marketing strategist with 20 years of experience in brand "
        "positioning and business analysis.\n"
        "You receive a business brief and produce a structured strategic analysis.\n"
        "You always respond in valid JSON only — no markdown, no explanation, no preamble.\n"
        "Your analysis is specific, actionable, and grounded in the actual business details provided.\n"
        "Never produce generic filler content. Every insight must directly reference the brand, "
        "product, market, or goal provided."
    )

    industry = brief["industry"]
    if brief.get("sub_industry"):
        industry = f"{industry} — {brief['sub_industry']}"

    campaign_goal = brief["campaign_goal"]
    if brief.get("campaign_goal_details"):
        campaign_goal = f"{campaign_goal} — {brief['campaign_goal_details']}"

    previous_campaigns = "None"
    if brief.get("has_previous_campaigns") and brief.get("previous_campaign_description"):
        previous_campaigns = f"Yes — {brief['previous_campaign_description']}"

    user_prompt = (
        "Analyze this business and produce a full strategic analysis.\n\n"
        "Business Brief:\n"
        f"- Brand: {brief['brand_name']}\n"
        f"- Product/Service: {brief['product_or_service']}\n"
        f"- Industry: {industry}\n"
        f"- Market: {brief['target_market']}\n"
        f"- Company Size: {brief['company_size']}\n"
        f"- Campaign Goal: {campaign_goal}\n"
        f"- Budget: {brief['budget_amount']} {brief['budget_currency']}\n"
        f"- Unique Selling Point: {brief['unique_selling_point']}\n"
        f"- Current Channels: {', '.join(brief['current_channels'])}\n"
        f"- Previous Campaigns: {previous_campaigns}\n\n"
        "Return a JSON object with exactly this structure:\n"
        "{\n"
        '  "swot": {\n'
        '    "strengths": ["...", "...", "..."],\n'
        '    "weaknesses": ["...", "...", "..."],\n'
        '    "opportunities": ["...", "...", "..."],\n'
        '    "threats": ["...", "...", "..."]\n'
        "  },\n"
        '  "brand_positioning": "A single precise sentence describing where this brand sits in its market",\n'
        '  "key_challenges": ["...", "...", "..."],\n'
        '  "growth_opportunities": ["...", "...", "..."]\n'
        "}\n\n"
        "Rules:\n"
        "- Each SWOT list must have exactly 3 items\n"
        "- Each item must be a complete sentence, not a label\n"
        "- key_challenges must have exactly 3 items\n"
        "- growth_opportunities must have exactly 3 items\n"
        "- Every item must be specific to this brand — never generic"
    )

    return call_claude(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
