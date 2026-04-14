"""
Phase 5 — End-to-end inference pipeline for NEW campaigns.

Loads Phase 2/3/4 artifacts (no retraining), predicts Stage 1 metric intervals,
assembles Stage 2 features, predicts ROI + success probability, and adds SHAP
explanations in plain language.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.preprocessing import CampaignPreprocessor

ARTIFACTS_DIR = ROOT / "artifacts"
STAGE1_DIR = ARTIFACTS_DIR / "stage1"
STAGE2_DIR = ARTIFACTS_DIR / "stage2"
REPORTS_DIR = ROOT / "reports"

EPS = 1e-9


STAGE1_TARGETS = {
    "clicks": "Clicks",
    "impressions": "Impressions",
    "conversion_rate": "Conversion_Rate",
    "engagement_score": "Engagement_Score",
    "acquisition_cost": "Acquisition_Cost",
}

TARGET_TO_RAW_COL = {
    "clicks": "Clicks",
    "impressions": "Impressions",
    "conversion_rate": "Conversion_Rate",
    "engagement_score": "Engagement_Score",
    "acquisition_cost": "Acquisition_Cost",
}

TARGET_TO_STAGE2_SCALED = {
    "clicks": "Clicks_scaled",
    "impressions": "Impressions_scaled",
    "conversion_rate": "Conversion_Rate_scaled",
    "engagement_score": "Engagement_Score_scaled",
    "acquisition_cost": "Acquisition_Cost_scaled",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _format_output_for_console(output: Dict[str, Any]) -> str:
    lines = [
        "=== Campaign inference ===",
        f"Channel: {output['campaign_summary']['channel']} | "
        f"Type: {output['campaign_summary']['type']} | "
        f"Location: {output['campaign_summary']['location']}",
        f"Audience: {output['campaign_summary']['audience']} | "
        f"Duration (days): {output['campaign_summary']['duration_days']}",
    ]
    if "budget" in output["campaign_summary"]:
        lines.append(f"Budget: {output['campaign_summary']['budget']}")
    s2 = output["stage2_evaluation"]
    lines.append(
        f"Predicted ROI: {s2['predicted_roi']:.4f} | "
        f"Success probability: {s2['success_probability']:.4f} | "
        f"Verdict: {s2['verdict']}"
    )
    lines.append("--- Stage 1 (q50 / confidence) ---")
    for k, v in output["stage1_predictions"].items():
        lines.append(f"  {k}: q50={v['q50']:.4g} ({v['confidence']})")
    sh = output["shap_explanation"]
    lines.append("--- SHAP (top drivers / detractors) ---")
    lines.append(f"  ROI drivers: {sh['roi_drivers']}")
    lines.append(f"  ROI detractors: {sh['roi_detractors']}")
    lines.append(f"  Success drivers: {sh['success_drivers']}")
    lines.append(f"  Success detractors: {sh['success_detractors']}")
    if output["confidence_warning"]:
        lines.append(f"Warning: {output['confidence_warning']}")
    lines.append("")
    return "\n".join(lines)


def _combine_confidence_warnings(
    low_conf_targets: List[str],
    clip_ceiling_warnings: List[str],
) -> Optional[str]:
    parts: List[str] = []
    if low_conf_targets:
        parts.append(
            "One or more metric predictions are LOW confidence: "
            f"[{', '.join(low_conf_targets)}]"
        )
    if clip_ceiling_warnings:
        parts.append("; ".join(clip_ceiling_warnings))
    if not parts:
        return None
    return " | ".join(parts)


def _ranking_at_most_one_adjacent_swap(expected_best_to_worst: List[str], actual_best_to_worst: List[str]) -> bool:
    """True if actual order equals expected or differs by exactly one swap of neighbors."""
    if actual_best_to_worst == expected_best_to_worst:
        return True
    n = len(expected_best_to_worst)
    for i in range(n - 1):
        cand = list(expected_best_to_worst)
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        if actual_best_to_worst == cand:
            return True
    return False


def _strictly_increasing_floats(vals: List[float], abs_tol: float = 1e-9) -> bool:
    return all(vals[i] < vals[i + 1] - abs_tol for i in range(len(vals) - 1))


# Phase 6 anti-gaming neutral baseline (preprocessing schema).
PHASE6_NEUTRAL_BASELINE: Dict[str, Any] = {
    "Channel_Used": "Facebook",
    "Campaign_Type": "Display",
    "Audience_age_range": "All Ages",
    "Audience_Gender": "Men",
    "Customer_Segment": "Foodies",
    "Location": "Chicago",
    "Language": "English",
    "Duration": 30,
    "Date": "2024-06-15",
    "Budget": 10000,
}

PHASE6_RANDOM_CHANNELS = ["Facebook", "Instagram", "Tiktok", "Website", "YouTube", "Google Ads"]
PHASE6_RANDOM_TYPES = ["Influencer", "Search", "Display", "Social Media"]
PHASE6_RANDOM_AGES = ["18-24", "25-34", "35-44", "45-54", "All Ages"]
PHASE6_RANDOM_GENDERS = ["Men", "Woman"]
PHASE6_RANDOM_SEGMENTS = [
    "Tech Enthusiasts",
    "Fashionistas",
    "Foodies",
    "Health and Wellness",
    "Outdoor Adventurers",
]
PHASE6_RANDOM_LOCS = ["New York", "Egypt", "saudi-Arabia", "Emirates", "Houston", "Chicago", "Miami"]
PHASE6_RANDOM_LANGS = ["English", "Spanish", "French", "German", "Arabic"]
PHASE6_RANDOM_DATES = ["2024-01-10", "2024-03-15", "2024-06-15", "2024-09-20", "2024-11-01"]


def _phase6_sample_random_campaign(rng: random.Random) -> Dict[str, Any]:
    return {
        "Channel_Used": rng.choice(PHASE6_RANDOM_CHANNELS),
        "Campaign_Type": rng.choice(PHASE6_RANDOM_TYPES),
        "Audience_age_range": rng.choice(PHASE6_RANDOM_AGES),
        "Audience_Gender": rng.choice(PHASE6_RANDOM_GENDERS),
        "Customer_Segment": rng.choice(PHASE6_RANDOM_SEGMENTS),
        "Location": rng.choice(PHASE6_RANDOM_LOCS),
        "Language": rng.choice(PHASE6_RANDOM_LANGS),
        "Duration": rng.choice([15, 30, 45, 60]),
        "Date": rng.choice(PHASE6_RANDOM_DATES),
        "Budget": int(rng.randint(5_000, 20_000)),
    }


def _canon_value(v: Any) -> str:
    return str(v).strip()


def normalize_campaign_input(campaign: Dict[str, Any]) -> pd.DataFrame:
    """Normalize user-facing campaign schema into preprocessing schema."""
    c = dict(campaign)

    # Key normalization
    if "Audience_Gender" in c and "Target_Audience_gender" not in c:
        c["Target_Audience_gender"] = c.pop("Audience_Gender")
    if "Audience_age_range" not in c and "Audience_Age_Range" in c:
        c["Audience_age_range"] = c.pop("Audience_Age_Range")

    # Value normalization
    ch_map = {
        "google ads": "Tiktok",
        "tiktok": "Tiktok",
        "tik tok": "Tiktok",
        "website": "Website",
        "youtube": "YouTube",
        "instagram": "Instagram",
        "facebook": "Facebook",
    }
    loc_map = {
        "new york": "New York",
        "egypt": "Egypt",
        "saudi-arabia": "Saudi-Arabia",
        "saudi arabia": "Saudi-Arabia",
        "emirates": "Emirates",
        "houston": "Houston",
        # keep unknowns as-is; encoder handle_unknown=ignore
        "chicago": "Chicago",
        "miami": "Miami",
    }
    lang_map = {
        "english": "English",
        "spanish": "Spanish",
        "french": "French",
        "german": "German",
        "arabic": "Arabic",
    }
    seg_map = {
        "tech enthusiasts": "Tech Enthusiasts",
        "fashionistas": "Fashion",
        "fashion": "Fashion",
        "foodies": "Food_Beverages",
        "food_beverages": "Food_Beverages",
        "health and wellness": "Health and Wellness",
        "outdoor adventurers": "Other",
        "other": "Other",
    }
    age_map = {
        "18-24": "18-24",
        "25-34": "25-34",
        "35-44": "35-44",
        "45-54": "45-54",
        "all ages": "All Ages",
        "all_ages": "All Ages",
    }
    gender_map = {
        "men": "Men",
        "man": "Men",
        "woman": "Woman",
        "women": "Woman",
    }
    type_map = {
        "influencer": "Influencer",
        "search": "Search",
        "display": "Display",
        "social media": "Social Media",
    }

    c["Campaign_Type"] = type_map.get(_canon_value(c.get("Campaign_Type", "")).lower(), _canon_value(c.get("Campaign_Type", "")))
    c["Channel_Used"] = ch_map.get(_canon_value(c.get("Channel_Used", "")).lower(), _canon_value(c.get("Channel_Used", "")))
    c["Location"] = loc_map.get(_canon_value(c.get("Location", "")).lower(), _canon_value(c.get("Location", "")))
    c["Language"] = lang_map.get(_canon_value(c.get("Language", "")).lower(), _canon_value(c.get("Language", "")))
    c["Customer_Segment"] = seg_map.get(_canon_value(c.get("Customer_Segment", "")).lower(), _canon_value(c.get("Customer_Segment", "")))
    c["Audience_age_range"] = age_map.get(_canon_value(c.get("Audience_age_range", "")).lower(), _canon_value(c.get("Audience_age_range", "")))
    c["Target_Audience_gender"] = gender_map.get(_canon_value(c.get("Target_Audience_gender", "")).lower(), _canon_value(c.get("Target_Audience_gender", "")))

    # Duration integer -> "X days"
    dur = c.get("Duration")
    if isinstance(dur, (int, float, np.integer, np.floating)):
        c["Duration"] = f"{int(dur)} days"
    elif isinstance(dur, str) and "day" not in dur.lower():
        try:
            c["Duration"] = f"{int(float(dur))} days"
        except Exception:
            c["Duration"] = dur

    # Keep only fields used by preprocessing for inference.
    keep = [
        "Campaign_Type",
        "Target_Audience_gender",
        "Audience_age_range",
        "Duration",
        "Channel_Used",
        "Location",
        "Language",
        "Customer_Segment",
        "Date",
    ]
    normalized = {k: c.get(k) for k in keep}
    return pd.DataFrame([normalized])


class InferenceEngine:
    def __init__(self):
        self.pre = CampaignPreprocessor.load_artifacts(ARTIFACTS_DIR)
        self.numeric_pipe = _load_pickle(ARTIFACTS_DIR / "numeric_scaler.pkl")
        self.roi_scaler = _load_pickle(ARTIFACTS_DIR / "roi_scaler.pkl")
        self.stage1_clip_bounds = _load_json(ARTIFACTS_DIR / "stage1_clip_bounds.json")
        self.stage1_tiers = _load_json(STAGE1_DIR / "stage1_confidence_tiers.json")
        self.stage1_feature_cols = _load_json(STAGE1_DIR / "stage1_feature_columns.json")
        self.stage2_feature_cols = _load_json(STAGE2_DIR / "stage2_feature_columns.json")

        self.stage2_roi_model = _load_pickle(STAGE2_DIR / "stage2_roi_model.pkl")
        self.stage2_success_model = _load_pickle(STAGE2_DIR / "stage2_success_model.pkl")
        self.shap_roi_explainer = _load_pickle(STAGE2_DIR / "shap_explainer_roi.pkl")
        self.shap_success_explainer = _load_pickle(STAGE2_DIR / "shap_explainer_success.pkl")

        # Load 15 Stage 1 models
        self.stage1_models = {}
        for target in STAGE1_TARGETS.keys():
            for q in ("q10", "q50", "q90"):
                self.stage1_models[(target, q)] = _load_pickle(STAGE1_DIR / f"{target}_{q}.pkl")

        # numeric_scaler.pkl is a Pipeline(imputer, scaler). Use scaler stats.
        self.num_scale_cols = list(self.numeric_pipe.feature_names_in_)
        scaler = self.numeric_pipe.named_steps["scaler"]
        self.num_means = {c: float(m) for c, m in zip(self.num_scale_cols, scaler.mean_)}
        self.num_scales = {c: float(s) for c, s in zip(self.num_scale_cols, scaler.scale_)}

    def _inverse_scale_numeric(self, raw_col: str, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        return z * self.num_scales[raw_col] + self.num_means[raw_col]

    def _scale_numeric(self, raw_col: str, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        return (x - self.num_means[raw_col]) / self.num_scales[raw_col]

    def _inverse_roi(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        return z * float(self.roi_scaler.scale_[0]) + float(self.roi_scaler.mean_[0])

    def _plain_phrase(self, feature: str, shap_val: float) -> str:
        pos = shap_val > 0
        if feature == "Conversion_Rate_scaled":
            return f"Conversion rate is {'high' if pos else 'low'}"
        if feature == "Cost_per_Click_scaled":
            return f"Cost per click is {'high' if pos else 'low'}"
        if feature == "Acquisition_Cost_scaled":
            return f"Acquisition cost is {'high' if pos else 'low'}"
        if feature == "Clicks_scaled":
            return f"Expected clicks are {'high' if pos else 'low'}"
        if feature == "Impressions_scaled":
            return f"Expected impressions are {'high' if pos else 'low'}"
        if feature == "CTR_scaled":
            return f"Click-through rate is {'high' if pos else 'low'}"
        if feature == "Engagement_Score_scaled":
            return f"Engagement score is {'high' if pos else 'low'}"
        if feature.startswith("Channel_Used_"):
            return f"Channel choice is {'helping' if pos else 'hurting'} the score"
        if feature.startswith("Campaign_Type_"):
            return f"Campaign type is {'helping' if pos else 'hurting'} the score"
        if feature.startswith("Target_Audience_gender_"):
            g = feature[len("Target_Audience_gender_") :]
            return f"Targeting {g} is {'helping' if pos else 'hurting'} the score"
        clean = feature.replace("_scaled", "").replace("_", " ")
        return f"{clean} is {'high' if pos else 'low'}"

    def _shap_plain_list_deduped(
        self,
        shap_row: np.ndarray,
        feature_names: List[str],
        *,
        positive: bool,
        k: int = 3,
    ) -> List[str]:
        """Top contributors by |SHAP|, mapped to plain language; skip duplicate phrases (order preserved)."""
        pairs = list(zip(feature_names, shap_row))
        if positive:
            pool = [p for p in pairs if p[1] > 0]
        else:
            pool = [p for p in pairs if p[1] < 0]
        pool.sort(key=lambda x: abs(x[1]), reverse=True)
        seen: set[str] = set()
        out: List[str] = []
        for feat, val in pool:
            phrase = self._plain_phrase(feat, val)
            if phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _coerce_campaign_row(campaign_input: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        if isinstance(campaign_input, pd.DataFrame):
            if len(campaign_input) != 1:
                raise ValueError("DataFrame input must contain exactly one row.")
            return campaign_input.iloc[0].to_dict()
        if isinstance(campaign_input, Mapping):
            return dict(campaign_input)
        raise TypeError("campaign_input must be a dict or a one-row pandas DataFrame.")

    def predict_one(
        self,
        campaign_input: Union[Dict[str, Any], pd.DataFrame],
        *,
        verbose: bool = True,
        include_shap: bool = True,
    ) -> Dict[str, Any]:
        raw_campaign = self._coerce_campaign_row(campaign_input)
        # Step 1 — preprocess pre-execution input
        input_df = normalize_campaign_input(raw_campaign)
        pre_df = self.pre.transform_inference_pre_execution(input_df)

        # Step 2 — Stage 1 quantile predictions
        missing_stage1_features = [c for c in self.stage1_feature_cols if c not in pre_df.columns]
        if missing_stage1_features:
            raise ValueError(f"Missing Stage1 feature columns after preprocessing: {missing_stage1_features}")

        X_stage1 = pre_df[self.stage1_feature_cols].astype(np.float32)

        stage1_orig: Dict[str, Dict[str, float]] = {}
        stage1_scaled_q50: Dict[str, float] = {}
        low_conf_targets: List[str] = []
        clip_ceiling_warnings: List[str] = []

        for t in STAGE1_TARGETS.keys():
            pred_scaled = {}
            for q in ("q10", "q50", "q90"):
                pred_scaled[q] = self.stage1_models[(t, q)].predict(X_stage1)[0]

            raw_col = TARGET_TO_RAW_COL[t]
            pred_orig = {q: float(self._inverse_scale_numeric(raw_col, np.array([pred_scaled[q]]))[0]) for q in ("q10", "q50", "q90")}

            # clip in original units
            bounds = self.stage1_clip_bounds[STAGE1_TARGETS[t]]
            lo, hi = float(bounds[0]), float(bounds[1])
            pred_orig = {q: float(np.clip(pred_orig[q], lo, hi)) for q in ("q10", "q50", "q90")}

            if math.isclose(pred_orig["q90"], hi, rel_tol=0.0, abs_tol=1e-9):
                clip_ceiling_warnings.append(
                    f"{STAGE1_TARGETS[t]} upper bound is clipped — true uncertainty may be higher"
                )

            # rescale q50 for Stage 2
            q50_scaled = float(self._scale_numeric(raw_col, np.array([pred_orig["q50"]]))[0])
            stage1_scaled_q50[t] = q50_scaled

            conf = self.stage1_tiers.get(STAGE1_TARGETS[t], "LOW")
            if conf == "LOW":
                low_conf_targets.append(STAGE1_TARGETS[t])

            stage1_orig[t] = {
                "q10": pred_orig["q10"],
                "q50": pred_orig["q50"],
                "q90": pred_orig["q90"],
                "confidence": conf,
                "interval_width": pred_orig["q90"] - pred_orig["q10"],
            }

        # derived metrics from q50 in original, then scale
        ctr_orig = stage1_orig["clicks"]["q50"] / (stage1_orig["impressions"]["q50"] + EPS)
        cpc_orig = stage1_orig["acquisition_cost"]["q50"] / (stage1_orig["clicks"]["q50"] + EPS)
        ctr_scaled = float(self._scale_numeric("CTR", np.array([ctr_orig]))[0])
        cpc_scaled = float(self._scale_numeric("Cost_per_Click", np.array([cpc_orig]))[0])

        # Step 3 — assemble Stage 2 vector
        stage2_df = pre_df.copy()
        stage2_df["Clicks_scaled"] = stage1_scaled_q50["clicks"]
        stage2_df["Impressions_scaled"] = stage1_scaled_q50["impressions"]
        stage2_df["Conversion_Rate_scaled"] = stage1_scaled_q50["conversion_rate"]
        stage2_df["Engagement_Score_scaled"] = stage1_scaled_q50["engagement_score"]
        stage2_df["Acquisition_Cost_scaled"] = stage1_scaled_q50["acquisition_cost"]
        stage2_df["CTR_scaled"] = ctr_scaled
        stage2_df["Cost_per_Click_scaled"] = cpc_scaled

        ordered_stage2 = list(self.stage2_feature_cols)
        missing_stage2_cols = [c for c in ordered_stage2 if c not in stage2_df.columns]
        if missing_stage2_cols:
            raise ValueError(
                f"Missing Stage2 feature columns before prediction: {missing_stage2_cols}"
            )

        # Enforce exact column order from stage2_feature_columns.json (do not rely on DataFrame order).
        row_vec = stage2_df.iloc[0][ordered_stage2].to_numpy(dtype=np.float64)
        X_stage2 = pd.DataFrame([row_vec], columns=ordered_stage2).astype(np.float32)
        actual_order = list(X_stage2.columns)
        if actual_order != ordered_stage2:
            raise ValueError(
                "Stage 2 feature column order mismatch vs stage2_feature_columns.json: "
                f"expected {ordered_stage2}, got {actual_order}"
            )

        # Step 4 — Stage 2 predictions
        roi_scaled = float(self.stage2_roi_model.predict(X_stage2)[0])
        predicted_roi = float(self._inverse_roi(np.array([roi_scaled]))[0])
        success_probability = float(self.stage2_success_model.predict_proba(X_stage2)[0, 1])

        if predicted_roi >= 6.5 and success_probability >= 0.75:
            verdict = "LAUNCH"
        elif predicted_roi >= 5.0 and success_probability >= 0.50:
            verdict = "REVISE"
        else:
            verdict = "DROP"

        # Step 5 — SHAP
        if include_shap:
            shap_roi = self.shap_roi_explainer.shap_values(X_stage2, check_additivity=False)
            if isinstance(shap_roi, list):
                shap_roi = shap_roi[0]
            shap_roi_row = np.asarray(shap_roi)[0]

            shap_success = self.shap_success_explainer.shap_values(X_stage2, check_additivity=False)
            if isinstance(shap_success, list):
                shap_success = shap_success[-1]
            shap_success_row = np.asarray(shap_success)[0]

            roi_drivers = self._shap_plain_list_deduped(
                shap_roi_row, self.stage2_feature_cols, positive=True, k=3
            )
            roi_detractors = self._shap_plain_list_deduped(
                shap_roi_row, self.stage2_feature_cols, positive=False, k=3
            )
            success_drivers = self._shap_plain_list_deduped(
                shap_success_row, self.stage2_feature_cols, positive=True, k=3
            )
            success_detractors = self._shap_plain_list_deduped(
                shap_success_row, self.stage2_feature_cols, positive=False, k=3
            )
        else:
            roi_drivers = []
            roi_detractors = []
            success_drivers = []
            success_detractors = []

        # Step 6 — output object
        audience = f"{input_df.iloc[0]['Target_Audience_gender']} {input_df.iloc[0]['Audience_age_range']}"
        duration_days = int(str(input_df.iloc[0]["Duration"]).split()[0])
        summary = {
            "channel": input_df.iloc[0]["Channel_Used"],
            "type": input_df.iloc[0]["Campaign_Type"],
            "audience": audience,
            "duration_days": duration_days,
            "location": input_df.iloc[0]["Location"],
        }
        if raw_campaign.get("Budget") is not None:
            summary["budget"] = raw_campaign.get("Budget")
        output = {
            "campaign_summary": summary,
            "stage1_predictions": {
                "clicks": {
                    "q10": stage1_orig["clicks"]["q10"],
                    "q50": stage1_orig["clicks"]["q50"],
                    "q90": stage1_orig["clicks"]["q90"],
                    "confidence": stage1_orig["clicks"]["confidence"],
                },
                "impressions": {
                    "q10": stage1_orig["impressions"]["q10"],
                    "q50": stage1_orig["impressions"]["q50"],
                    "q90": stage1_orig["impressions"]["q90"],
                    "confidence": stage1_orig["impressions"]["confidence"],
                },
                "conversion_rate": {
                    "q10": stage1_orig["conversion_rate"]["q10"],
                    "q50": stage1_orig["conversion_rate"]["q50"],
                    "q90": stage1_orig["conversion_rate"]["q90"],
                    "confidence": stage1_orig["conversion_rate"]["confidence"],
                },
                "engagement_score": {
                    "q10": stage1_orig["engagement_score"]["q10"],
                    "q50": stage1_orig["engagement_score"]["q50"],
                    "q90": stage1_orig["engagement_score"]["q90"],
                    "confidence": stage1_orig["engagement_score"]["confidence"],
                },
                "acquisition_cost": {
                    "q10": stage1_orig["acquisition_cost"]["q10"],
                    "q50": stage1_orig["acquisition_cost"]["q50"],
                    "q90": stage1_orig["acquisition_cost"]["q90"],
                    "confidence": stage1_orig["acquisition_cost"]["confidence"],
                },
            },
            "stage2_evaluation": {
                "predicted_roi": predicted_roi,
                "success_probability": success_probability,
                "verdict": verdict,
            },
            "shap_explanation": {
                "roi_drivers": roi_drivers,
                "roi_detractors": roi_detractors,
                "success_drivers": success_drivers,
                "success_detractors": success_detractors,
            },
            "confidence_warning": _combine_confidence_warnings(
                low_conf_targets, clip_ceiling_warnings
            ),
        }
        if verbose:
            print(_format_output_for_console(output))
        return output


def run_phase6_validation() -> str:
    """
    Phase 6 validation suite + bug-fix confirmation. Writes reports/phase6_validation_report.txt.
    Raises RuntimeError on MAJOR flags or broken Phase 5 ranking (after report is saved).
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "phase6_validation_report.txt"
    engine = InferenceEngine()
    lines: List[str] = []
    major_flags: List[str] = []
    minor_flags: List[str] = []

    # --- Section 1: Phase 5 campaigns (SHAP on to verify dedupe; clipping warnings visible) ---
    lines.append("=== Section 1: Bug fix confirmation (Phase 5 campaigns) ===")
    lines.append(
        "SHAP enabled; plain-language lists deduplicate repeated phrases. "
        "confidence_warning combines LOW-tier + q90 ceiling clips when applicable."
    )
    lines.append("")

    phase5_payloads = {
        "Campaign A": {
            "Channel_Used": "Google Ads",
            "Campaign_Type": "Search",
            "Audience_age_range": "25-34",
            "Audience_Gender": "Men",
            "Customer_Segment": "Tech Enthusiasts",
            "Location": "New York",
            "Language": "English",
            "Duration": 60,
            "Date": "2024-06-15",
            "Budget": 15000,
        },
        "Campaign B": {
            "Channel_Used": "YouTube",
            "Campaign_Type": "Influencer",
            "Audience_age_range": "18-24",
            "Audience_Gender": "Men",
            "Customer_Segment": "Fashionistas",
            "Location": "Miami",
            "Language": "Spanish",
            "Duration": 15,
            "Date": "2024-01-10",
            "Budget": 8000,
        },
        "Campaign C": {
            "Channel_Used": "Instagram",
            "Campaign_Type": "Social Media",
            "Audience_age_range": "all_ages",
            "Audience_Gender": "Woman",
            "Customer_Segment": "Foodies",
            "Location": "Chicago",
            "Language": "English",
            "Duration": 30,
            "Date": "2024-09-20",
            "Budget": 12000,
        },
    }
    p5_out: Dict[str, Dict[str, Any]] = {}
    for name, payload in phase5_payloads.items():
        p5_out[name] = engine.predict_one(payload, verbose=False, include_shap=True)
        lines.append(f"--- {name} ---")
        lines.append(json.dumps(p5_out[name], indent=2))
        lines.append("")

    roi_a = p5_out["Campaign A"]["stage2_evaluation"]["predicted_roi"]
    roi_b = p5_out["Campaign B"]["stage2_evaluation"]["predicted_roi"]
    roi_c = p5_out["Campaign C"]["stage2_evaluation"]["predicted_roi"]
    lines.append("Ranking check (predicted_roi): A > C > B required")
    lines.append(f"  A={roi_a:.6f}, C={roi_c:.6f}, B={roi_b:.6f}")
    ranking_ok = roi_a > roi_c > roi_b
    lines.append("  PASS" if ranking_ok else "  FAIL")
    lines.append("")
    if not ranking_ok:
        major_flags.append("Phase 5 ranking A > C > B broken after bug fixes")

    # --- Section 2: Channel sweep ---
    lines.append("=== Section 2: Channel sweep ===")
    lines.append("Expected ROI order (high to low): Website > Facebook > Instagram > YouTube > Tiktok")
    lines.append("PASS if actual order matches expected OR differs by at most one adjacent swap.")
    lines.append("")
    exp_channels = ["Website", "Facebook", "Instagram", "YouTube", "Tiktok"]
    ch_results: List[Tuple[str, float]] = []
    for ch in exp_channels:
        pl = {**PHASE6_NEUTRAL_BASELINE, "Channel_Used": ch}
        roi = engine.predict_one(pl, verbose=False, include_shap=False)["stage2_evaluation"]["predicted_roi"]
        ch_results.append((ch, float(roi)))
    ch_results.sort(key=lambda x: -x[1])
    actual_ch = [x[0] for x in ch_results]
    lines.append("channel\tpredicted_roi")
    for ch, r in ch_results:
        lines.append(f"{ch}\t{r:.6f}")
    ch_ok = _ranking_at_most_one_adjacent_swap(exp_channels, actual_ch)
    lines.append(f"Actual order (by ROI): {' > '.join(actual_ch)}")
    lines.append(f"Result: {'PASS' if ch_ok else 'UNEXPECTED (more than 1 adjacent swap)'}")
    lines.append("")
    if not ch_ok:
        minor_flags.append("Channel sweep ordering UNEXPECTED")

    # --- Section 3: Campaign type sweep ---
    lines.append("=== Section 3: Campaign type sweep ===")
    lines.append("Expected ROI order: Search > Social Media > Display > Influencer")
    lines.append("")
    exp_types = ["Search", "Social Media", "Display", "Influencer"]
    ty_results: List[Tuple[str, float]] = []
    for ty in exp_types:
        pl = {**PHASE6_NEUTRAL_BASELINE, "Campaign_Type": ty}
        roi = engine.predict_one(pl, verbose=False, include_shap=False)["stage2_evaluation"]["predicted_roi"]
        ty_results.append((ty, float(roi)))
    ty_results.sort(key=lambda x: -x[1])
    actual_ty = [x[0] for x in ty_results]
    lines.append("campaign_type\tpredicted_roi")
    for ty, r in ty_results:
        lines.append(f"{ty}\t{r:.6f}")
    ty_ok = _ranking_at_most_one_adjacent_swap(exp_types, actual_ty)
    lines.append(f"Actual order (by ROI): {' > '.join(actual_ty)}")
    lines.append(f"Result: {'PASS' if ty_ok else 'UNEXPECTED (more than 1 adjacent swap)'}")
    lines.append("")
    if not ty_ok:
        minor_flags.append("Campaign type sweep ordering UNEXPECTED")

    # --- Section 4: Duration ---
    lines.append("=== Section 4: Duration effect ===")
    lines.append("Expected: strictly increasing predicted_roi for Duration 15, 30, 45, 60.")
    lines.append("")
    durs = [15, 30, 45, 60]
    drois: List[float] = []
    for d in durs:
        pl = {**PHASE6_NEUTRAL_BASELINE, "Duration": d}
        drois.append(
            float(
                engine.predict_one(pl, verbose=False, include_shap=False)["stage2_evaluation"][
                    "predicted_roi"
                ]
            )
        )
    lines.append("duration\tpredicted_roi")
    for d, r in zip(durs, drois):
        lines.append(f"{d}\t{r:.6f}")
    dur_ok = _strictly_increasing_floats(drois)
    lines.append(f"Result: {'PASS' if dur_ok else 'UNEXPECTED (not strictly increasing)'}")
    lines.append("")
    if not dur_ok:
        minor_flags.append("Duration monotonicity UNEXPECTED")

    # --- Section 5: Audience sweep (age range) ---
    lines.append("=== Section 5: Audience sweep ===")
    lines.append(
        "Vary Audience_age_range only (baseline otherwise). "
        "PASS if predicted ROI is not flat across all age buckets (model responds to audience)."
    )
    lines.append("")
    ages = ["18-24", "25-34", "35-44", "45-54", "All Ages"]
    arois: List[float] = []
    for ag in ages:
        pl = {**PHASE6_NEUTRAL_BASELINE, "Audience_age_range": ag}
        arois.append(
            float(
                engine.predict_one(pl, verbose=False, include_shap=False)["stage2_evaluation"][
                    "predicted_roi"
                ]
            )
        )
    lines.append("audience_age_range\tpredicted_roi")
    for ag, r in zip(ages, arois):
        lines.append(f"{ag}\t{r:.6f}")
    spread = max(arois) - min(arois)
    aud_ok = spread > 1e-5
    lines.append(f"ROI spread (max-min): {spread:.6f}")
    lines.append(f"Result: {'PASS' if aud_ok else 'UNEXPECTED (flat ROI across ages)'}")
    lines.append("")
    if not aud_ok:
        minor_flags.append("Audience age sweep: ROI invariant (flat)")

    # --- Section 6: Diversity (100 random) ---
    lines.append("=== Section 6: Diversity check (100 random campaigns, SHAP off) ===")
    rng = random.Random(42)
    verdicts: List[str] = []
    launch_channels: List[str] = []
    for _ in range(100):
        camp = _phase6_sample_random_campaign(rng)
        out = engine.predict_one(camp, verbose=False, include_shap=False)
        v = out["stage2_evaluation"]["verdict"]
        verdicts.append(v)
        if v == "LAUNCH":
            launch_channels.append(out["campaign_summary"]["channel"])

    vc = Counter(verdicts)
    lines.append(f"Distinct verdict values seen: {len(vc)}")
    lines.append("Verdict counts:")
    for v in ("LAUNCH", "REVISE", "DROP"):
        lines.append(f"  {v}: {vc.get(v, 0)}")
    all_three_verdicts = all(vc.get(v, 0) > 0 for v in ("LAUNCH", "REVISE", "DROP"))
    lines.append(f"All three verdicts represented: {'yes' if all_three_verdicts else 'no'}")
    if not all_three_verdicts:
        major_flags.append("Diversity: not all three verdicts in 100 random draws")

    bias_flag = False
    if launch_channels:
        lcc = Counter(launch_channels)
        top_ch, top_n = lcc.most_common(1)[0]
        share = top_n / len(launch_channels)
        lines.append("LAUNCH verdicts: channel distribution")
        for ch, cnt in lcc.most_common():
            lines.append(f"  {ch}: {cnt} ({cnt / len(launch_channels):.1%} of LAUNCH)")
        lines.append(f"Largest LAUNCH share (single channel): {top_ch} at {share:.1%}")
        if share > 0.60:
            bias_flag = True
            minor_flags.append(
                f"Potential channel bias: {top_ch} in {share:.1%} of LAUNCH verdicts (>60%)"
            )
        lines.append(f"Channel bias (>60% one channel among LAUNCH): {'FLAG' if bias_flag else 'PASS'}")
    else:
        lines.append("No LAUNCH verdicts in sample — channel bias check N/A")

    lines.append("")
    lines.append(f"Section 6 summary: verdict diversity={'PASS' if all_three_verdicts else 'FLAG'}")
    lines.append("")

    # --- Section 7: Overall ---
    lines.append("=== Section 7: Overall verdict ===")
    if major_flags and minor_flags:
        overall = "MAJOR FLAGS"
        lines.append("Verdict: MAJOR FLAGS — do not proceed to integration")
        lines.append("Major:")
        for m in major_flags:
            lines.append(f"  - {m}")
        lines.append("Minor (also noted):")
        for m in minor_flags:
            lines.append(f"  - {m}")
    elif major_flags:
        overall = "MAJOR FLAGS"
        lines.append("Verdict: MAJOR FLAGS — do not proceed to integration")
        for m in major_flags:
            lines.append(f"  - {m}")
    elif minor_flags:
        overall = "MINOR FLAGS"
        lines.append("Verdict: MINOR FLAGS — pipeline works; note flagged behaviors for users")
        for m in minor_flags:
            lines.append(f"  - {m}")
    else:
        overall = "ALL PASS"
        lines.append("Verdict: ALL PASS — pipeline is ready for integration")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")

    if major_flags:
        raise RuntimeError(
            "Phase 6 MAJOR FLAGS or ranking failure — see phase6_validation_report.txt"
        )
    return overall


def run_phase5_test() -> Dict[str, Dict[str, Any]]:
    engine = InferenceEngine()
    campaigns = {
        "Campaign A": {
            "Channel_Used": "Google Ads",
            "Campaign_Type": "Search",
            "Audience_age_range": "25-34",
            "Audience_Gender": "Men",
            "Customer_Segment": "Tech Enthusiasts",
            "Location": "New York",
            "Language": "English",
            "Duration": 60,
            "Date": "2024-06-15",
            "Budget": 15000,
        },
        "Campaign B": {
            "Channel_Used": "YouTube",
            "Campaign_Type": "Influencer",
            "Audience_age_range": "18-24",
            "Audience_Gender": "Men",
            "Customer_Segment": "Fashionistas",
            "Location": "Miami",
            "Language": "Spanish",
            "Duration": 15,
            "Date": "2024-01-10",
            "Budget": 8000,
        },
        "Campaign C": {
            "Channel_Used": "Instagram",
            "Campaign_Type": "Social Media",
            "Audience_age_range": "all_ages",
            "Audience_Gender": "Woman",
            "Customer_Segment": "Foodies",
            "Location": "Chicago",
            "Language": "English",
            "Duration": 30,
            "Date": "2024-09-20",
            "Budget": 12000,
        },
    }

    outputs = {
        name: engine.predict_one(payload, verbose=False, include_shap=True)
        for name, payload in campaigns.items()
    }

    roi_a = outputs["Campaign A"]["stage2_evaluation"]["predicted_roi"]
    roi_b = outputs["Campaign B"]["stage2_evaluation"]["predicted_roi"]
    roi_c = outputs["Campaign C"]["stage2_evaluation"]["predicted_roi"]

    lines = []
    lines.append("=== Phase 5 Inference Test ===")
    lines.append("")
    for name in ("Campaign A", "Campaign B", "Campaign C"):
        lines.append(f"--- {name} ---")
        lines.append(json.dumps(outputs[name], indent=2))
        lines.append("")

    lines.append("Ranking check (predicted_roi):")
    lines.append(f"A={roi_a:.6f}, B={roi_b:.6f}, C={roi_c:.6f}")
    if roi_a > roi_c > roi_b:
        lines.append("PASS: A > C > B")
    elif roi_a > roi_b:
        lines.append("PASS (minimum required): A > B")
    else:
        lines.append("FAIL: A is not higher than B")

    if not (roi_a > roi_b):
        lines.append("")
        lines.append("STOP CONDITION TRIGGERED: Campaign A does not score higher than Campaign B.")
        lines.append("Campaign A output:")
        lines.append(json.dumps(outputs["Campaign A"], indent=2))
        lines.append("Campaign B output:")
        lines.append(json.dumps(outputs["Campaign B"], indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "phase5_inference_test.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if not (roi_a > roi_b):
        print(report_path.read_text(encoding="utf-8"))
        raise RuntimeError("Campaign A predicted ROI is not higher than Campaign B.")

    print(report_path.read_text(encoding="utf-8"))
    return outputs


if __name__ == "__main__":
    run_phase6_validation()

