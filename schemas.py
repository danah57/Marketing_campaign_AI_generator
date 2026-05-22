from typing import Literal

from pydantic import BaseModel, Field


class Competitor(BaseModel):
    name: str
    website: str | None = None
    notes: str | None = None


IndustryLiteral = Literal[
    'E-commerce & Retail',
    'Fashion & Beauty',
    'Food & Beverage',
    'Media & Content Creation',
    'Fitness & Wellness',
    'Home & Local Services',
    'Education & Coaching',
    'Travel & Hospitality',
    'Real Estate',
    'Healthcare & Wellness',
    'Finance & Business',
    'Technology & Apps',
    'Other',
]

CompanySizeLiteral = Literal["Solo", "Small", "Mid-size", "Enterprise"]
CampaignGoalLiteral = Literal["Awareness", "Leads", "Sales", "Retention", "Re-engagement"]
CurrencyLiteral = Literal["USD", "EUR", "GBP", "EGP", "SAR", "AED"]
ChannelLiteral = Literal[
    "Instagram",
    "TikTok",
    "YouTube",
    "Facebook",
    "Website",
    "None",
]


class BrandToneProfile(BaseModel):
    tone_formality: int = Field(ge=1, le=5, description="1=very casual, 5=very formal")
    tone_playfulness: int = Field(ge=1, le=5, description="1=serious, 5=very playful")
    tone_boldness: int = Field(ge=1, le=5, description="1=subtle, 5=very bold")
    preferred_vocabulary: list[str] | None = None
    avoided_vocabulary: list[str] | None = None


class InfluencerProfile(BaseModel):
    id: int
    bio: str | None = None
    primaryPlatform: str | None = None
    followersCount: str | None = None
    engagementRate: str | None = None
    categories: list[str] | None = None
    contentTypes: list[str] | None = None
    collaborationTypes: list[str] | None = None
    audienceAgeRange: str | None = None
    audienceGender: str | None = None
    audienceLocation: str | None = None
    interests: list[str] | None = None
    socialMediaLinks: dict | None = None


class InfluencerMatch(BaseModel):
    influencer_id: int
    fit_score: float
    fit_reasoning: str
    suggested_collaboration_type: str
    suggested_budget_usd: float
    outreach_message: str


class CampaignBrief(BaseModel):
    job_id: str | None = None
    brand_name: str
    product_or_service: str
    industry: IndustryLiteral
    sub_industry: str | None = None
    target_market: str
    company_size: CompanySizeLiteral
    campaign_goal: CampaignGoalLiteral
    campaign_goal_details: str | None = None
    budget_amount: float = Field(gt=0)
    budget_currency: CurrencyLiteral
    campaign_duration_weeks: int = Field(ge=1, le=52)
    unique_selling_point: str
    current_channels: list[ChannelLiteral] = Field(default_factory=list)
    competitors: list[Competitor] | None = Field(default_factory=list)
    has_previous_campaigns: bool
    previous_campaign_description: str | None = None
    brand_tone: BrandToneProfile | None = None
    start_date: str | None = None


class GenerateResponse(BaseModel):
    strategy: dict
    calendar: dict
    influencer_matches: list[InfluencerMatch] | None = Field(default_factory=list)
    influencer_strategy_note: str | None = None
    influencer_stage_skipped: bool = True


class StageCheckpointResponse(BaseModel):
    job_id: str
    stage: str | int
    status: str  # "complete" | "cached" | "failed"
    output: dict
