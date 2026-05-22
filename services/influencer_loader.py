import json
import logging
import os
import re
from pathlib import Path

from sqlalchemy import create_engine, text

logger = logging.getLogger("campaign_model.llm")

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "influencer_cache.json"
_engine = None


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:12345@localhost:5432/gradproject",
    ).strip()


def _get_engine():
    global _engine
    if _engine is None:
        connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "3"))
        _engine = create_engine(
            _database_url(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": connect_timeout},
        )
    return _engine


def parse_followers(value):
    if not value:
        return 0.0

    value = str(value).upper().replace(" ", "")

    if "–" in value or "-" in value:
        parts = re.split(r"[–-]", value)
        nums = [parse_followers(p) for p in parts]
        return sum(nums) / len(nums)

    if "M" in value:
        return float(value.replace("M", "")) * 1_000_000

    if "K" in value:
        return float(value.replace("K", "")) * 1_000

    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_engagement(value):
    if not value:
        return 0.0

    value = str(value).replace("%", "").strip()

    try:
        return float(value)
    except ValueError:
        return 0.0


def _normalize_row(r: dict) -> dict:
    return {
        "id": r["id"],
        "primaryPlatform": r.get("primaryPlatform"),
        "followersCount": parse_followers(r.get("followersCount")),
        "engagementRate": parse_engagement(r.get("engagementRate")),
        "categories": r.get("categories") or [],
        "contentTypes": r.get("contentTypes") or [],
        "collaborationTypes": r.get("collaborationTypes") or [],
        "audienceAgeRange": r.get("audienceAgeRange"),
        "audienceGender": r.get("audienceGender"),
        "audienceLocation": r.get("audienceLocation"),
        "interests": r.get("interests") or [],
    }


def _load_from_cache(limit: int = 20) -> list[dict]:
    if not _CACHE_PATH.is_file():
        logger.warning("Influencer cache not found at %s", _CACHE_PATH)
        return []
    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read influencer cache: %s", e)
        return []
    if not isinstance(raw, list):
        return []
    return [_normalize_row(r) for r in raw[:limit] if isinstance(r, dict) and "id" in r]


def _load_from_db(limit: int = 20) -> list[dict]:
    query = text("""
    SELECT
        id,
        "primaryPlatform",
        "followersCount",
        "engagementRate",
        categories,
        "contentTypes",
        "collaborationTypes",
        "audienceAgeRange",
        "audienceGender",
        "audienceLocation",
        interests
    FROM "InfluencerProfiles"
    WHERE "isOnboarded" = true
    LIMIT :limit
""")

    with _get_engine().connect() as conn:
        result = conn.execute(query, {"limit": limit}).mappings()
        rows = result.all()

    return [_normalize_row(dict(r)) for r in rows]


def load_influencer_candidates(limit: int = 20) -> list[dict]:
    """Load influencers from cache file first, then PostgreSQL with a short connect timeout."""
    cached = _load_from_cache(limit)
    if cached:
        return cached

    use_db = os.getenv("INFLUENCER_USE_DB", "true").strip().lower() not in ("0", "false", "no", "off")
    if not use_db:
        return []

    try:
        rows = _load_from_db(limit)
        if rows:
            return rows
        logger.info("DB returned no influencers.")
    except Exception as e:
        logger.warning("DB influencer load failed (%s).", e)

    return []


def save_to_file():
    data = load_influencer_candidates(limit=20)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} influencers to cache")


if __name__ == "__main__":
    save_to_file()
