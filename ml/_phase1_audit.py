"""Phase 1 one-off data audit — run: python _phase1_audit.py"""
import numpy as np
import pandas as pd
from pathlib import Path

path = Path(r"d:\SocialMedia Marketing\AI\marketing_campaign_dataset.csv")
df = pd.read_csv(path, low_memory=False)

lines = []

def log(msg=""):
    lines.append(msg)
    print(msg)

log("=== ROWS & COLUMNS ===")
log(f"Rows: {len(df):,}, Columns: {df.shape[1]}")
log("")

log("=== COLUMN TYPES (pandas infer) ===")
for c in df.columns:
    log(f"  {c}: {df[c].dtype}")
log("")

log("=== MISSING VALUES (count & pct) ===")
miss = df.isna().sum()
miss_pct = 100 * miss / len(df)
m = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct.round(4)})
m = m[m["missing_count"] > 0].sort_values("missing_count", ascending=False)
if m.empty:
    log("  No missing values detected.")
else:
    log(m.to_string())
log("")

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
log("=== MODELING NOTE ===")
log("  Company exists in raw CSV; pipeline drops it (not one-hot encoded).")
log("")
log("=== CATEGORICAL UNIQUE COUNTS (nunique) ===")
for c in cat_cols:
    nu = df[c].nunique(dropna=True)
    log(f"  {c}: {nu} unique")
    if nu <= 20:
        vc = df[c].value_counts(dropna=False).head(15)
        log(f"    value_counts (up to 15): {vc.to_dict()}")
log("")
log("  (Full cardinality above; full value_counts when nunique <= 20)")
log("")

num_cols = ["Conversion_Rate", "ROI", "Clicks", "Impressions", "Engagement_Score"]


def parse_money(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


acq = df["Acquisition_Cost"].map(parse_money)

log("=== NUMERIC SUMMARY ===")
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    log(f"\n{c}:")
    log(s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
log("\nAcquisition_Cost (parsed from currency string):")
log(acq.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
log("")

log("=== DATA QUALITY FLAGS ===")
log(f"Duplicate Campaign_ID rows (dupes after first): {df['Campaign_ID'].duplicated().sum():,}")
if df["Campaign_ID"].duplicated().any():
    both = df["Campaign_ID"].duplicated(keep=False).sum()
    log(f"  Rows involved in duplicate Campaign_ID groups: {both:,}")
imp = pd.to_numeric(df["Impressions"], errors="coerce")
clk = pd.to_numeric(df["Clicks"], errors="coerce")
log(f"Impressions == 0: {(imp == 0).sum():,}; Impressions < 0: {(imp < 0).sum():,}")
log(f"Clicks == 0: {(clk == 0).sum():,}; Clicks > Impressions: {(clk > imp).sum():,}")
dt = pd.to_datetime(df["Date"], errors="coerce")
log(f"Date parse failures (NaT): {dt.isna().sum():,}")
if dt.notna().any():
    log(f"  Date range: {dt.min()} to {dt.max()}")
roi = pd.to_numeric(df["ROI"], errors="coerce")
log(f"ROI missing: {roi.isna().sum():,}; ROI < 0: {(roi < 0).sum():,}")

out_dir = Path(r"d:\SocialMedia Marketing\AI\ml\reports")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "phase1_data_audit.txt"
out_file.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved: {out_file}")
