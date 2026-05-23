import json
import logging
import os
import re
from pathlib import Path

from sqlalchemy import create_engine, text

logger = logging.getLogger("campaign_model.llm")

_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "influencer_cache.json"
)

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
        nums = [parse_followers(p) for p in parts if p]

        if nums:
            return sum(nums) / len(nums)

        return 0.0

    # Millions
    if "M" in value:
        try:
            return float(value.replace("M", "")) * 1_000_000
        except ValueError:
            return 0.0

    # Thousands
    if "K" in value:
        try:
            return float(value.replace("K", "")) * 1_000
        except ValueError:
            return 0.0

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


def _row_id(r: dict) -> int | None:
    """DB rows use userId; cache / Node payloads may use id."""
    for key in ("userId", "id"):
        val = r.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def _normalize_row(r: dict) -> dict | None:
    row_id = _row_id(r)
    if row_id is None:
        return None

    followers_raw = r.get("followersCount")
    if isinstance(followers_raw, (int, float)):
        followers = float(followers_raw)
    else:
        followers = parse_followers(followers_raw)

    engagement_raw = r.get("engagementRate")
    if isinstance(engagement_raw, (int, float)):
        engagement = float(engagement_raw)
    else:
        engagement = parse_engagement(engagement_raw)

    return {
        "id": row_id,
        "primaryPlatform": r.get("primaryPlatform"),
        "followersCount": followers,
        "engagementRate": engagement,
        "categories": r.get("categories") or [],
        "contentTypes": r.get("contentTypes") or [],
        "collaborationTypes": r.get("collaborationTypes") or [],
        "audienceAgeRange": r.get("audienceAgeRange"),
        "audienceGender": r.get("audienceGender"),
        "audienceLocation": r.get("audienceLocation"),
        "interests": r.get("interests") or [],
    }


def _load_from_cache() -> list[dict]:
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

    out: list[dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        try:
            row = _normalize_row(r)
        except Exception as e:
            logger.warning("Skipping invalid cache row: %s", e)
            continue
        if row is not None:
            out.append(row)
    return out


def _load_from_db() -> list[dict]:
    """
    Load ALL onboarded influencers from database.
    No LIMIT applied.
    """

    query = text("""
        SELECT
            "userId",
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
    """)

    with _get_engine().connect() as conn:
        result = conn.execute(query).mappings()
        rows = result.all()

    logger.info("Loaded %s influencers from database", len(rows))

    out: list[dict] = []
    for r in rows:
        row = _normalize_row(dict(r))
        if row is not None:
            out.append(row)
    return out


def load_influencer_candidates() -> list[dict]:
    """
    Load influencers from cache first.
    If cache is empty, load ALL influencers from PostgreSQL.
    """

    # Try cache first
    cached = _load_from_cache()

    if cached:
        logger.info("Loaded %s influencers from cache", len(cached))
        return cached

    # Check if DB loading enabled
    use_db = (
        os.getenv("INFLUENCER_USE_DB", "true")
        .strip()
        .lower()
        not in ("0", "false", "no", "off")
    )

    if not use_db:
        logger.warning("Database loading disabled")
        return []

    # Load from DB
    try:
        rows = _load_from_db()

        if rows:
            logger.info("Loaded %s influencers from DB", len(rows))
            return rows

        logger.info("DB returned no influencers")

    except Exception as e:
        logger.exception("DB influencer load failed: %s", e)

    return []


def save_to_file():
    """
    Save ALL influencers from DB to local cache file.
    """

    try:
        data = _load_from_db()
    except Exception as e:
        logger.exception("Could not load influencers from DB: %s", e)
        data = []

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} influencers to cache")


if __name__ == "__main__":
    save_to_file()