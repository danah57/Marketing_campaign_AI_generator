from typing import Literal

from pydantic import BaseModel, Field


class Competitor(BaseModel):
    name: str
    website: str | None = None
    notes: str | None = None


IndustryLiteral = Literal[
    "E-commerce",
    "F&B",
    "Fashion",
    "Technology",
    "Health & Wellness",
    "Education",
    "Real Estate",
    "Finance",
    "Travel",
    "Beauty",
    "Sports",
    "Other",
]

CompanySizeLiteral = Literal["Solo", "Small", "Mid-size", "Enterprise"]
CampaignGoalLiteral = Literal["Awareness", "Leads", "Sales", "Retention", "Re-engagement"]
CurrencyLiteral = Literal["USD", "EUR", "GBP", "EGP", "SAR", "AED"]
ChannelLiteral = Literal[
    "Instagram",
    "TikTok",
    "YouTube",
    "Facebook",
    "Google Ads",
    "Email",
    "Website",
    "None",
]


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
    current_channels: list[ChannelLiteral] = Field(min_length=1)
    competitors: list[Competitor] | None = Field(default_factory=list)
    has_previous_campaigns: bool
    previous_campaign_description: str | None = None


class GenerateResponse(BaseModel):
    strategy: dict
    calendar: dict
