"""
Phase 3 — Stage 1: quantile LightGBM models for campaign performance metrics.

Loads splits from /data and numeric_scaler from /artifacts (do not modify Phase 2).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm", file=sys.stderr)
    raise

from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
STAGE1_DIR = ARTIFACTS_DIR / "stage1"

OHE_PREFIXES = (
    "Campaign_Type_",
    "Target_Audience_gender_",
    "Audience_age_range_",
    "Channel_Used_",
    "Location_",
    "Language_",
    "Customer_Segment_",
)

STAGE1_EXTRA_NUM = [
    "Duration_scaled",
    "month_scaled",
    "quarter_scaled",
    "day_of_week_scaled",
    "season_scaled",
]

TARGETS = [
    ("Clicks", "Clicks_scaled", 3),
    ("Impressions", "Impressions_scaled", 4),
    ("Conversion_Rate", "Conversion_Rate_scaled", 1),
    ("Engagement_Score", "Engagement_Score_scaled", 5),
    ("Acquisition_Cost", "Acquisition_Cost_scaled", 2),
]

QUANTILES = [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]

CLIP_BOUNDS = {
    "Clicks": [100, 1000],
    "Impressions": [1000, 10000],
    "Conversion_Rate": [0.01, 0.15],
    "Engagement_Score": [1, 10],
    "Acquisition_Cost": [5000, 20000],
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}


def is_ohe_column(name: str) -> bool:
    return name.startswith(OHE_PREFIXES)


def build_feature_columns(all_columns) -> list[str]:
    ohe = [c for c in all_columns if is_ohe_column(c)]
    ohe_sorted = sorted(ohe)
    missing = [c for c in STAGE1_EXTRA_NUM if c not in all_columns]
    if missing:
        raise ValueError(f"Missing Stage 1 numeric features: {missing}")
    return ohe_sorted + STAGE1_EXTRA_NUM


def load_numeric_scaler(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def column_inverse_scaled(scaler_pipeline, col_index: int, z: np.ndarray) -> np.ndarray:
    """Map one column from scaled space back to original (post-impute) scale."""
    scaler = scaler_pipeline.named_steps["scaler"]
    mean = scaler.mean_[col_index]
    scale = scaler.scale_[col_index]
    z = np.asarray(z, dtype=np.float64).ravel()
    return z * scale + mean


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def interval_coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    lo = np.minimum(q_lo, q_hi)
    hi = np.maximum(q_lo, q_hi)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def tier_for_target(r2: float, coverage: float) -> str:
    if r2 >= 0.7 and coverage >= 0.80:
        return "HIGH"
    if r2 >= 0.5 and coverage >= 0.70:
        return "MEDIUM"
    return "LOW"


def overall_verdict(tiers: dict) -> str:
    ok_med = sum(1 for t in tiers.values() if t in ("HIGH", "MEDIUM"))
    all_ok = all(t in ("HIGH", "MEDIUM") for t in tiers.values())
    if all_ok:
        return "RELIABLE"
    if ok_med >= 3:
        return "ACCEPTABLE"
    return "WEAK"


def train_one_quantile(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    alpha: float,
) -> lgb.LGBMRegressor:
    """Train fixed n_estimators trees (spec); optional val set for logging only."""
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        **LGBM_PARAMS,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    with open(ARTIFACTS_DIR / "stage1_clip_bounds.json", "w", encoding="utf-8") as f:
        json.dump(CLIP_BOUNDS, f, indent=2)

    scaler_path = ARTIFACTS_DIR / "numeric_scaler.pkl"
    if not scaler_path.exists():
        print(f"Missing {scaler_path}", file=sys.stderr)
        return 1

    num_pipe = load_numeric_scaler(scaler_path)

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    feature_cols = build_feature_columns(train.columns)
    with open(STAGE1_DIR / "stage1_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    X_train = train[feature_cols].astype(np.float32)
    X_val = val[feature_cols].astype(np.float32)
    X_test = test[feature_cols].astype(np.float32)

    models: dict[tuple, lgb.LGBMRegressor] = {}
    test_metrics: dict = {}
    tiers: dict = {}
    stop_r2 = False

    for target_name, target_col, scaler_idx in TARGETS:
        y_tr = train[target_col].to_numpy()
        y_va = val[target_col].to_numpy()
        y_te = test[target_col].to_numpy()

        preds_q = {}
        for qname, alpha in QUANTILES:
            m = train_one_quantile(X_train, y_tr, X_val, y_va, alpha)
            models[(target_name, qname)] = m
            preds_q[qname] = m.predict(X_test)

        q10 = preds_q["q10"]
        q50 = preds_q["q50"]
        q90 = preds_q["q90"]

        r2 = r2_score(y_te, q50)
        rmse_scaled = float(np.sqrt(mean_squared_error(y_te, q50)))
        y_te_orig = column_inverse_scaled(num_pipe, scaler_idx, y_te)
        q50_orig = column_inverse_scaled(num_pipe, scaler_idx, q50)
        rmse_orig = float(np.sqrt(mean_squared_error(y_te_orig, q50_orig)))

        cov = interval_coverage(y_te, q10, q90)
        pin = {qname: pinball_loss(y_te, preds_q[qname], alpha) for qname, alpha in QUANTILES}

        test_metrics[target_name] = {
            "R2_q50": r2,
            "RMSE_scaled": rmse_scaled,
            "RMSE_original": rmse_orig,
            "pinball_q10": pin["q10"],
            "pinball_q50": pin["q50"],
            "pinball_q90": pin["q90"],
            "interval_coverage_pct": 100.0 * cov,
            "poorly_calibrated": cov < 0.70,
        }

        if r2 < 0:
            stop_r2 = True

        tiers[target_name] = tier_for_target(r2, cov)

    with open(STAGE1_DIR / "stage1_confidence_tiers.json", "w", encoding="utf-8") as f:
        json.dump(tiers, f, indent=2)

    verdict = overall_verdict(tiers)

    lines = []
    lines.append("=== PHASE 3 STAGE 1 REPORT ===")
    lines.append("")
    for target_name, _, _ in TARGETS:
        m = test_metrics[target_name]
        lines.append(f"--- {target_name} ---")
        lines.append(f"  R2 (q50, test): {m['R2_q50']:.6f}")
        lines.append(f"  RMSE scaled (q50): {m['RMSE_scaled']:.6f}")
        lines.append(f"  RMSE original units (q50): {m['RMSE_original']:.6f}")
        lines.append(f"  Pinball q10: {m['pinball_q10']:.6f}")
        lines.append(f"  Pinball q50: {m['pinball_q50']:.6f}")
        lines.append(f"  Pinball q90: {m['pinball_q90']:.6f}")
        lines.append(f"  Interval coverage (q10-q90): {m['interval_coverage_pct']:.2f}%")
        cal_note = "FLAG: coverage < 70% (poorly calibrated)" if m["poorly_calibrated"] else "OK"
        lines.append(f"  Calibration: {cal_note}")
        lines.append(f"  Confidence tier: {tiers[target_name]}")
        lines.append("")

    lines.append("=== Overall Stage 1 reliability verdict ===")
    lines.append(f"  {verdict}")
    lines.append("  RELIABLE: all targets HIGH or MEDIUM")
    lines.append("  ACCEPTABLE: at least 3 targets HIGH or MEDIUM")
    lines.append("  WEAK: fewer than 3 targets at MEDIUM or better")
    lines.append("")

    report_path = ROOT / "reports" / "phase3_stage1_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if stop_r2:
        lines.append("STOP: One or more targets have R2 < 0 on test (worse than mean).")
        lines.append("Model PKL files were NOT saved.")
        lines.append("")
        lines.append("=== NOTES ===")
        lines.append(
            "- See script docstring: weak test R2 means pre-exec features barely predict "
            "post-exec metrics on this CSV; interval coverage may still be near 80%."
        )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 2

    name_map = {
        "Clicks": "clicks",
        "Impressions": "impressions",
        "Conversion_Rate": "conversion_rate",
        "Engagement_Score": "engagement_score",
        "Acquisition_Cost": "acquisition_cost",
    }
    for target_name, _, _ in TARGETS:
        base = name_map[target_name]
        for qname, _ in QUANTILES:
            path = STAGE1_DIR / f"{base}_{qname}.pkl"
            with open(path, "wb") as f:
                pickle.dump(models[(target_name, qname)], f)

    if verdict == "WEAK":
        lines.append("STOP: Overall verdict is WEAK — do not proceed to Phase 4 until improved.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if verdict != "WEAK" else 3


if __name__ == "__main__":
    raise SystemExit(main())
