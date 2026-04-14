"""
Reusable campaign preprocessing for training and inference.

Fits on training split only (no leakage). Persists OneHotEncoder, numeric
imputer+scaler pipeline, and ROI scaler for inverse transforms.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default ROI threshold for success label (multiplier scale, not %). Override via CLI.
DEFAULT_SUCCESS_ROI_THRESHOLD = 5.0

EPS = 1e-9

CAT_COLS: List[str] = [
    "Campaign_Type",
    "Target_Audience_gender",
    "Audience_age_range",
    "Channel_Used",
    "Location",
    "Language",
    "Customer_Segment",
]

NUM_SCALE_COLS: List[str] = [
    "Duration",
    "Conversion_Rate",
    "Acquisition_Cost",
    "Clicks",
    "Impressions",
    "Engagement_Score",
    "CTR",
    "Cost_per_Click",
    "month",
    "quarter",
    "day_of_week",
    "season",
    "ROI",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_currency_series(s: pd.Series) -> pd.Series:
    def _one(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        t = str(x).replace(",", "").replace("$", "").strip()
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.map(_one)


def parse_duration_series(s: pd.Series) -> pd.Series:
    def _one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).replace(" days", "").replace("days", "").strip()
        try:
            return int(float(t))
        except ValueError:
            return np.nan

    return s.map(_one)


def month_to_season(m: int) -> int:
    """0=winter Dec–Feb, 1=spring Mar–May, 2=summer Jun–Aug, 3=fall Sep–Nov."""
    if m in (12, 1, 2):
        return 0
    if m in (3, 4, 5):
        return 1
    if m in (6, 7, 8):
        return 2
    return 3


def parse_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: parse raw CSV columns."""
    out = df.copy()
    if "Acquisition_Cost" in out.columns:
        out["Acquisition_Cost"] = parse_currency_series(out["Acquisition_Cost"])
    if "Duration" in out.columns:
        out["Duration"] = parse_duration_series(out["Duration"])
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: calendar features; drop Date."""
    out = df.copy()
    if "Date" not in out.columns:
        return out
    dt = out["Date"]
    out["month"] = dt.dt.month.astype(float)
    out["quarter"] = dt.dt.quarter.astype(float)
    out["day_of_week"] = dt.dt.dayofweek.astype(float)
    out["season"] = dt.dt.month.map(month_to_season).astype(float)
    out = out.drop(columns=["Date"])
    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: CTR / Cost_per_Click if sources exist."""
    out = df.copy()
    if "Clicks" in out.columns and "Impressions" in out.columns:
        out["CTR"] = out["Clicks"].astype(float) / (out["Impressions"].astype(float) + EPS)
    else:
        out["CTR"] = np.nan

    if "Acquisition_Cost" in out.columns and "Clicks" in out.columns:
        out["Cost_per_Click"] = out["Acquisition_Cost"].astype(float) / (
            out["Clicks"].astype(float) + EPS
        )
    else:
        out["Cost_per_Click"] = np.nan
    return out


def add_success_label(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Step 6 (logical): binary label from raw ROI before scaling (NA ROI => NA label)."""
    out = df.copy()
    if "ROI" not in out.columns:
        out["success_label"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        return out
    roi = pd.to_numeric(out["ROI"], errors="coerce")
    labels = []
    for r in roi:
        if pd.isna(r):
            labels.append(pd.NA)
        else:
            labels.append(int(float(r) > threshold))
    out["success_label"] = pd.Series(labels, dtype="Int64", index=out.index)
    return out


def drop_campaign_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("Campaign_ID", "Company"):
        if col in out.columns:
            out = out.drop(columns=[col])
    return out



class CampaignPreprocessor:
    """
    Fit on training DataFrame (parsed + engineered; Campaign_ID and Company dropped).
    Transform applies the same steps for any split or inference batch.
    """

    def __init__(self, success_roi_threshold: float = DEFAULT_SUCCESS_ROI_THRESHOLD):
        self.success_roi_threshold = success_roi_threshold
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.numeric_preprocessor: Optional[Pipeline] = None
        self.roi_scaler: Optional[StandardScaler] = None
        self._ohe_feature_names: Optional[np.ndarray] = None

    def engineer_up_to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse, date features, ratios, success label, drop Campaign_ID and Company. No OHE/scale."""
        z = parse_raw_columns(df)
        z = add_date_features(z)
        z = add_ratio_features(z)
        z = add_success_label(z, self.success_roi_threshold)
        z = drop_campaign_id(z)
        return z

    def fit(self, df_train_raw: pd.DataFrame) -> "CampaignPreprocessor":
        train_eng = self.engineer_up_to_categorical(df_train_raw)

        missing_cat = [c for c in CAT_COLS if c not in train_eng.columns]
        if missing_cat:
            raise ValueError(f"Missing categorical columns: {missing_cat}")

        missing_num = [c for c in NUM_SCALE_COLS if c not in train_eng.columns]
        if missing_num:
            raise ValueError(f"Missing numeric columns: {missing_num}")

        try:
            self.onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float64,
            )
        except TypeError:
            self.onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
                dtype=np.float64,
            )
        X_cat = train_eng[CAT_COLS].astype(str)
        self.onehot_encoder.fit(X_cat)
        self._ohe_feature_names = self.onehot_encoder.get_feature_names_out(CAT_COLS)

        self.numeric_preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        X_num = train_eng[NUM_SCALE_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
        self.numeric_preprocessor.fit(X_num)

        self.roi_scaler = StandardScaler()
        self.roi_scaler.fit(train_eng[["ROI"]].astype(float))

        return self

    def transform(
        self,
        df_raw: pd.DataFrame,
        *,
        include_raw_roi: bool = True,
    ) -> pd.DataFrame:
        if self.onehot_encoder is None or self.numeric_preprocessor is None:
            raise RuntimeError("Preprocessor is not fitted; call fit() first.")

        eng = self.engineer_up_to_categorical(df_raw)

        for c in CAT_COLS:
            if c not in eng.columns:
                eng[c] = "missing"
        X_cat = eng[CAT_COLS].astype(str)
        X_ohe = self.onehot_encoder.transform(X_cat)
        ohe_df = pd.DataFrame(X_ohe, columns=self._ohe_feature_names, index=eng.index)

        for c in NUM_SCALE_COLS:
            if c not in eng.columns:
                eng[c] = np.nan
        X_num_t = self.numeric_preprocessor.transform(
            eng[NUM_SCALE_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        num_df = pd.DataFrame(
            X_num_t,
            columns=[f"{c}_scaled" for c in NUM_SCALE_COLS],
            index=eng.index,
        )

        out = pd.concat([ohe_df, num_df, eng[["success_label"]]], axis=1)
        if include_raw_roi and "ROI" in eng.columns:
            out["ROI_raw"] = eng["ROI"].astype(float).values
        return out

    def transform_inference_pre_execution(
        self,
        df_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For new campaigns without post metrics: pass rows without Clicks/Impressions/etc.
        Ratios become NaN and are imputed from training medians.
        """
        return self.transform(df_raw, include_raw_roi=False)

    def save_artifacts(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "onehot_encoder.pkl", "wb") as f:
            pickle.dump(self.onehot_encoder, f)
        with open(artifacts_dir / "numeric_scaler.pkl", "wb") as f:
            pickle.dump(self.numeric_preprocessor, f)
        with open(artifacts_dir / "roi_scaler.pkl", "wb") as f:
            pickle.dump(self.roi_scaler, f)
        meta = {
            "success_roi_threshold": self.success_roi_threshold,
            "cat_cols": CAT_COLS,
            "num_scale_cols": NUM_SCALE_COLS,
            "ohe_feature_names": list(self._ohe_feature_names) if self._ohe_feature_names is not None else [],
        }
        with open(artifacts_dir / "preprocessor_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load_artifacts(cls, artifacts_dir: Path) -> "CampaignPreprocessor":
        artifacts_dir = Path(artifacts_dir)
        with open(artifacts_dir / "preprocessor_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        pre = cls(success_roi_threshold=meta["success_roi_threshold"])
        with open(artifacts_dir / "onehot_encoder.pkl", "rb") as f:
            pre.onehot_encoder = pickle.load(f)
        with open(artifacts_dir / "numeric_scaler.pkl", "rb") as f:
            pre.numeric_preprocessor = pickle.load(f)
        with open(artifacts_dir / "roi_scaler.pkl", "rb") as f:
            pre.roi_scaler = pickle.load(f)
        pre._ohe_feature_names = np.array(meta["ohe_feature_names"])
        return pre


def split_train_val_test(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70% / 15% / 15% stratified by index only (shuffle split)."""
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=random_state, shuffle=True
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=random_state, shuffle=True
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def check_success_label_balance(labels: pd.Series, name: str) -> Tuple[bool, str]:
    pos_rate = labels.mean()
    if pos_rate < 0.10:
        return False, f"{name}: positive rate {pos_rate:.4f} < 10% — STOP, lower threshold or check data."
    if pos_rate > 0.90:
        return False, f"{name}: positive rate {pos_rate:.4f} > 90% — STOP, raise threshold or check data."
    return True, f"{name}: positive rate {pos_rate:.4f} OK."


def check_zero_variance(df: pd.DataFrame, name: str) -> Tuple[bool, List[str]]:
    bad = []
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce")
        v = col.var(ddof=0)
        if not np.isfinite(v) or v <= 1e-14:
            bad.append(c)
    if bad:
        return False, bad
    return True, []


def run_full_pipeline(
    csv_path: Path,
    *,
    data_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    threshold: float,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df_train, df_val, df_test = split_train_val_test(df)

    pre = CampaignPreprocessor(success_roi_threshold=threshold)
    pre.fit(df_train)

    train_t = pre.transform(df_train)
    val_t = pre.transform(df_val)
    test_t = pre.transform(df_test)

    # Global label balance (before split)
    full_eng = pre.engineer_up_to_categorical(df)
    ok_bal, bal_msg = check_success_label_balance(full_eng["success_label"], "full_data")
    if not ok_bal:
        raise RuntimeError(bal_msg)

    ok_bal_tr, msg_tr = check_success_label_balance(train_t["success_label"], "train")
    if not ok_bal_tr:
        raise RuntimeError(msg_tr)

    for split_name, split_df in [("train", train_t), ("val", val_t), ("test", test_t)]:
        ok_z, zbad = check_zero_variance(split_df, split_name)
        if not ok_z:
            raise RuntimeError(
                f"Zero-variance columns in {split_name}: {zbad} — STOP before proceeding."
            )

    pre.save_artifacts(artifacts_dir)

    train_t.to_csv(data_dir / "train.csv", index=False)
    val_t.to_csv(data_dir / "val.csv", index=False)
    test_t.to_csv(data_dir / "test.csv", index=False)

    # Phase 2 report
    lines = []
    lines.append("=== PHASE 2 FEATURE REPORT ===")
    lines.append(f"Source CSV: {csv_path}")
    lines.append(f"SUCCESS_ROI_THRESHOLD: {threshold}")
    lines.append("Company column: dropped before encoding (not a model feature).")
    lines.append("")
    lines.append("=== Final column names (processed CSVs) ===")
    lines.append(", ".join(train_t.columns.tolist()))
    lines.append("")
    lines.append("=== Split shapes ===")
    lines.append(f"train: {train_t.shape}")
    lines.append(f"val:   {val_t.shape}")
    lines.append(f"test:  {test_t.shape}")
    lines.append("")
    lines.append("=== success_label balance ===")
    lines.append(
        f"full (engineered): mean={full_eng['success_label'].mean():.4f}, counts:\n{full_eng['success_label'].value_counts().sort_index().to_string()}"
    )
    lines.append(
        f"train: mean={train_t['success_label'].mean():.4f}\n{train_t['success_label'].value_counts().sort_index().to_string()}"
    )
    lines.append("")
    lines.append("=== Value ranges (train) min / max per column ===")
    for c in train_t.columns:
        col = pd.to_numeric(train_t[c], errors="coerce")
        lines.append(f"  {c}: min={col.min():.6g}, max={col.max():.6g}")
    lines.append("")
    lines.append("=== Artifacts ===")
    lines.append(str(artifacts_dir / "onehot_encoder.pkl"))
    lines.append(str(artifacts_dir / "numeric_scaler.pkl"))
    lines.append(str(artifacts_dir / "roi_scaler.pkl"))
    lines.append(str(artifacts_dir / "preprocessor_meta.pkl"))
    lines.append("")
    lines.append("OK: success_label balance and variance checks passed.")

    report_path = reports_dir / "phase2_feature_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


def main():
    root = _project_root()
    parser = argparse.ArgumentParser(description="Phase 2 preprocessing pipeline")
    parser.add_argument(
        "--csv",
        type=Path,
        default=root.parent / "marketing_campaign_dataset_v2.csv",
        help="Path to raw marketing_campaign_dataset_v2.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SUCCESS_ROI_THRESHOLD,
        help="ROI > threshold => success_label 1",
    )
    args = parser.parse_args()

    run_full_pipeline(
        Path(args.csv),
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        reports_dir=root / "reports",
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
