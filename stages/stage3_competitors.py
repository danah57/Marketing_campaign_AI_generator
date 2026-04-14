"""Stage 3: Competitor Analysis — competitor mapping and market gaps."""

import json

from utils.llm_client import call_claude

USE_MOCK = True


def run(brief: dict, context: dict, job_id: str) -> dict:
    if USE_MOCK:
        return _mock_output(brief, context)
    return _real_output(brief, context)


def _mock_output(brief: dict, context: dict) -> dict:
    return {
        "competitors": [
            {
                "name": "Patagonia",
                "they_are_better_at": "Decades of environmental credibility and brand trust that eco-conscious consumers default to without question.",
                "we_are_better_at": "Accessible everyday fashion pricing that Patagonia's premium positioning completely ignores.",
            },
            {
                "name": "Everlane",
                "they_are_better_at": "Radical price transparency and a highly optimized DTC funnel with strong millennial brand recognition.",
                "we_are_better_at": "Genuine end-to-end sustainability credentials versus Everlane's recent trust erosion from labor controversies.",
            },
            {
                "name": "Reformation",
                "they_are_better_at": "Fashion-forward aesthetic and strong influencer presence that makes sustainability feel aspirational rather than dutiful.",
                "we_are_better_at": "Gender-inclusive range targeting male eco-conscious consumers that Reformation almost entirely ignores.",
            },
        ],
        "market_gaps": [
            "No major sustainable fashion brand is effectively owning the male eco-conscious consumer aged 25-34 with fashion-forward positioning.",
            "Most sustainable brands communicate through guilt or activism — no one is making sustainability feel effortless and positive.",
            "Supply chain transparency is claimed by every competitor but visually documented and verified by almost none of them.",
        ],
        "recommended_differentiation": (
            "Position EcoWear as the accessible, visually transparent sustainable "
            "brand for everyday fashion — not outdoor gear, not luxury basics — so "
            "it occupies a space none of the three main competitors can credibly claim."
        ),
    }


def _real_output(brief: dict, context: dict) -> dict:
    competitors = brief.get("competitors") or []
    if competitors:
        competitor_block = "\n".join(
            [
                f"- {c['name']}: {c.get('website', 'no website')} — {c.get('notes', 'no notes')}"
                for c in competitors
            ]
        )
    else:
        competitor_block = (
            "None provided — identify the top 3 competitors in this market from your own knowledge."
        )

    system_prompt = (
        "You are a competitive intelligence analyst.\n"
        "Your only job is to identify what competitors do better than the client brand "
        "and where the client has a clear advantage.\n"
        "You always respond in valid JSON only — no markdown, no explanation, no preamble.\n"
        "Be specific and direct. One sentence per field. No filler."
    )

    user_prompt = (
        "Analyze the competitive landscape for this brand.\n\n"
        f"Brand: {brief['brand_name']}\n"
        f"Product: {brief['product_or_service']}\n"
        f"Industry: {brief['industry']}\n"
        f"Market: {brief['target_market']}\n"
        f"USP: {brief['unique_selling_point']}\n\n"
        "User-provided competitors:\n"
        f"{competitor_block}\n\n"
        "Instructions:\n"
        "- For each user-provided competitor, fill in they_are_better_at and we_are_better_at\n"
        "- Add 2 additional competitors you know from this market with the same fields\n"
        "- Identify exactly 3 market gaps — underserved segments or unmet needs competitors are missing\n"
        "- Write one recommended_differentiation sentence that tells this brand exactly how to win\n"
        "- Return valid JSON matching exactly this structure:\n"
        "{\n"
        '  "competitors": [\n'
        "    {\n"
        '      "name": "...",\n'
        '      "they_are_better_at": "...",\n'
        '      "we_are_better_at": "..."\n'
        "    }\n"
        "  ],\n"
        '  "market_gaps": ["...", "...", "..."],\n'
        '  "recommended_differentiation": "..."\n'
        "}"
    )

    result = call_claude(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
    json.dumps(result)
    return result
