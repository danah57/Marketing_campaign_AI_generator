"""
Phase 4 — Stage 2 Evaluator

Trains:
  - ROI regression (xgb.XGBRegressor) on REAL ROI_scaled from train.csv
  - Success probability classifier (xgb.XGBClassifier) on REAL success_label from train.csv

Never uses Stage 1 predictions during training.

Outputs artifacts + reports and then stops (no inference pipeline).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import shap
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
STAGE2_DIR = ARTIFACTS_DIR / "stage2"
REPORTS_DIR = ROOT / "reports"

STAGE2_DIR.mkdir(parents=True, exist_ok=True)


def _load_numeric_roi_scaler(roi_scaler_path: Path):
    with open(roi_scaler_path, "rb") as f:
        # StandardScaler: mean_ and scale_ are 1D arrays
        return pickle.load(f)


def inverse_transform_roi(roi_scaler, z_scaled: np.ndarray) -> np.ndarray:
    z = np.asarray(z_scaled, dtype=np.float64).ravel()
    mean = float(roi_scaler.mean_[0])
    scale = float(roi_scaler.scale_[0])
    return z * scale + mean


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = f1_score(y_true, y_pred)
    return {
        "threshold": thr,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> int:
    # Load preprocessed splits (from Phase 2)
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Feature columns = all engineered columns except targets
    exclude = {"ROI_scaled", "ROI_raw", "success_label"}
    feature_cols = [c for c in train.columns if c not in exclude]

    # Targets
    y_roi_train = train["ROI_scaled"].to_numpy()
    y_roi_val = val["ROI_scaled"].to_numpy()
    y_roi_test = test["ROI_scaled"].to_numpy()

    y_cls_train = train["success_label"].to_numpy().astype(int)
    y_cls_val = val["success_label"].to_numpy().astype(int)
    y_cls_test = test["success_label"].to_numpy().astype(int)

    X_train = train[feature_cols].astype(np.float32)
    X_val = val[feature_cols].astype(np.float32)
    X_test = test[feature_cols].astype(np.float32)

    # ROI inverse transform
    roi_scaler_path = ARTIFACTS_DIR / "roi_scaler.pkl"
    roi_scaler = _load_numeric_roi_scaler(roi_scaler_path)

    # --- Tune ROI regressor ---
    roi_depth_candidates = [4, 6, 8]
    best_roi = None
    best_roi_depth = None
    best_val_rmse = float("inf")

    roi_base_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "random_state": 42,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }

    for depth in roi_depth_candidates:
        model = XGBRegressor(max_depth=depth, **roi_base_params)
        model.fit(X_train, y_roi_train)
        pred_val_scaled = model.predict(X_val)
        pred_val_orig = inverse_transform_roi(roi_scaler, pred_val_scaled)
        y_val_orig = inverse_transform_roi(roi_scaler, y_roi_val)
        val_rmse = rmse(y_val_orig, pred_val_orig)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_roi = model
            best_roi_depth = depth

    # --- Evaluate ROI on test ---
    pred_test_roi_scaled = best_roi.predict(X_test)
    y_test_orig = inverse_transform_roi(roi_scaler, y_roi_test)
    pred_test_roi_orig = inverse_transform_roi(roi_scaler, pred_test_roi_scaled)

    roi_r2 = float(r2_score(y_test_orig, pred_test_roi_orig))
    roi_rmse = rmse(y_test_orig, pred_test_roi_orig)
    roi_mae = float(mean_absolute_error(y_test_orig, pred_test_roi_orig))

    if roi_r2 < 0.50:
        # Stop: still write report, but do not save artifacts
        report_path = REPORTS_DIR / "phase4_stage2_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            f"STOP: ROI test R2 below 0.50 (R2={roi_r2:.6f}). No artifacts saved.\n",
            encoding="utf-8",
        )
        print(report_path.read_text(encoding="utf-8"))
        return 2

    # --- Tune Success classifier ---
    cls_depth_candidates = [4, 6, 8]
    best_cls = None
    best_cls_depth = None
    best_val_auc = -float("inf")

    cls_base_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "eval_metric": "auc",
        "random_state": 42,
        "tree_method": "hist",
        "use_label_encoder": False,
        "objective": "binary:logistic",
    }

    for depth in cls_depth_candidates:
        model = XGBClassifier(max_depth=depth, **cls_base_params)
        model.fit(X_train, y_cls_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_cls_val, val_prob))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_cls = model
            best_cls_depth = depth

    # --- Evaluate classifier on test ---
    test_prob = best_cls.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_cls_test, test_prob))
    if test_auc < 0.70:
        report_path = REPORTS_DIR / "phase4_stage2_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            f"STOP: Success AUC-ROC below 0.70 (AUC={test_auc:.6f}). No artifacts saved.\n",
            encoding="utf-8",
        )
        print(report_path.read_text(encoding="utf-8"))
        return 2

    cm_05 = compute_confusion(y_cls_test, test_prob, thr=0.5)
    cm_40 = None
    if cm_05["recall"] < 0.70:
        cm_40 = compute_confusion(y_cls_test, test_prob, thr=0.40)

    top5_features_roi_shap = []
    top5_features_cls_shap = []

    # --- SHAP (TEST set only) ---
    # Use TreeExplainer for both models.
    explainer_roi = shap.TreeExplainer(best_roi)
    shap_values_roi = explainer_roi.shap_values(X_test, check_additivity=False)
    if isinstance(shap_values_roi, list):
        shap_values_roi = shap_values_roi[0]

    explainer_cls = shap.TreeExplainer(best_cls)
    shap_values_cls = explainer_cls.shap_values(X_test, check_additivity=False)
    if isinstance(shap_values_cls, list):
        # For binary classification, shap can be a list: [for class 0, for class 1]
        shap_values_cls = shap_values_cls[-1]

    shap_abs_roi = np.abs(shap_values_roi)
    shap_abs_cls = np.abs(shap_values_cls)

    mean_abs_roi = shap_abs_roi.mean(axis=0)
    mean_abs_cls = shap_abs_cls.mean(axis=0)

    feature_names = list(feature_cols)
    # Top 15 per model for global report
    top15_roi_idx = np.argsort(mean_abs_roi)[::-1][:15]
    top15_cls_idx = np.argsort(mean_abs_cls)[::-1][:15]

    global_report_path = REPORTS_DIR / "stage2_shap_global.txt"
    global_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(global_report_path, "w", encoding="utf-8") as f:
        f.write("=== Stage2 SHAP Global Importance ===\n\n")
        f.write("ROI model (regressor) top 15 mean(|SHAP|):\n")
        for idx in top15_roi_idx:
            f.write(f"- {feature_names[idx]}: {mean_abs_roi[idx]:.8f}\n")
        f.write("\nSuccess model (classifier) top 15 mean(|SHAP|):\n")
        for idx in top15_cls_idx:
            f.write(f"- {feature_names[idx]}: {mean_abs_cls[idx]:.8f}\n")

    # Sample explanations for 3 test rows
    idx_high = int(np.argmax(pred_test_roi_orig))
    idx_low = int(np.argmin(pred_test_roi_orig))
    idx_boundary = int(np.argmin(np.abs(test_prob - 0.5)))

    def top_shap_for_row(shap_row: np.ndarray, idxs: np.ndarray, k: int = 8):
        abs_row = np.abs(shap_row)
        top_idx = np.argsort(abs_row)[::-1][:k]
        items = []
        for j in top_idx:
            items.append(
                (
                    feature_names[j],
                    float(shap_row[j]),
                )
            )
        return items

    sample_report_path = REPORTS_DIR / "stage2_shap_samples.txt"
    with open(sample_report_path, "w", encoding="utf-8") as f:
        f.write("=== Stage2 SHAP Sample Explanations ===\n\n")
        cases = [
            ("Highest predicted ROI", idx_high),
            ("Lowest predicted ROI", idx_low),
            ("Closest to decision boundary (p~0.5)", idx_boundary),
        ]
        for label, idx in cases:
            f.write(f"--- {label} (test row index: {idx}) ---\n")
            f.write(f"True ROI (orig): {y_test_orig[idx]:.4f}\n")
            f.write(f"Pred ROI (orig): {pred_test_roi_orig[idx]:.4f}\n")
            f.write(f"True success_label: {int(y_cls_test[idx])}\n")
            f.write(f"Pred success probability: {float(test_prob[idx]):.6f}\n\n")

            f.write("ROI model (regressor) top SHAP contributions:\n")
            for feat, val in top_shap_for_row(shap_values_roi[idx], None):
                direction = "increases" if val > 0 else "decreases"
                f.write(f"- {feat}: SHAP={val:.6f} -> tends to {direction} predicted ROI\n")
            f.write("\nSuccess model (classifier) top SHAP contributions:\n")
            for feat, val in top_shap_for_row(shap_values_cls[idx], None):
                direction = "increases" if val > 0 else "decreases"
                f.write(f"- {feat}: SHAP={val:.6f} -> tends to {direction} success probability\n")
            f.write("\n")

    # Save explainers
    with open(STAGE2_DIR / "shap_explainer_roi.pkl", "wb") as f:
        pickle.dump(explainer_roi, f)
    with open(STAGE2_DIR / "shap_explainer_success.pkl", "wb") as f:
        pickle.dump(explainer_cls, f)

    # Save models + feature columns list
    with open(STAGE2_DIR / "stage2_roi_model.pkl", "wb") as f:
        pickle.dump(best_roi, f)
    with open(STAGE2_DIR / "stage2_success_model.pkl", "wb") as f:
        pickle.dump(best_cls, f)

    with open(STAGE2_DIR / "stage2_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Top 5 features based on global SHAP
    top5_roi_idx = np.argsort(mean_abs_roi)[::-1][:5]
    top5_cls_idx = np.argsort(mean_abs_cls)[::-1][:5]

    top5_features_roi_shap = [(feature_names[i], float(mean_abs_roi[i])) for i in top5_roi_idx]
    top5_features_cls_shap = [(feature_names[i], float(mean_abs_cls[i])) for i in top5_cls_idx]

    # --- Final report ---
    report_path = REPORTS_DIR / "phase4_stage2_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    verdict = "WEAK"
    if roi_r2 > 0.85 and test_auc > 0.90:
        verdict = "STRONG"
    elif roi_r2 > 0.70 and test_auc > 0.80:
        verdict = "GOOD"

    lines = []
    lines.append("=== Phase 4 Stage 2 Report ===")
    lines.append("")

    lines.append("Best model selection:")
    lines.append(f"- ROI regressor: best max_depth={best_roi_depth}, best val RMSE={best_val_rmse:.6f} (orig ROI units)")
    lines.append(f"- Success classifier: best max_depth={best_cls_depth}, best val AUC-ROC={best_val_auc:.6f}")
    lines.append("")

    lines.append("TARGET A — ROI regression (test, original ROI units):")
    lines.append(f"- R2: {roi_r2:.6f}")
    lines.append(f"- RMSE: {roi_rmse:.6f}")
    lines.append(f"- MAE: {roi_mae:.6f}")
    lines.append("")

    lines.append("TARGET B — Success classifier (test):")
    lines.append(f"- AUC-ROC: {test_auc:.6f}")
    lines.append("Threshold=0.50:")
    lines.append(f"- Precision: {cm_05['precision']:.6f}")
    lines.append(f"- Recall: {cm_05['recall']:.6f}")
    lines.append(f"- F1: {cm_05['f1']:.6f}")
    lines.append(f"- Confusion matrix [ [TN, FP], [FN, TP] ]: {cm_05['confusion_matrix']}")
    if cm_40 is not None:
        lines.append("")
        lines.append("Threshold=0.40 (since recall<0.70 at 0.50):")
        lines.append(f"- Precision: {cm_40['precision']:.6f}")
        lines.append(f"- Recall: {cm_40['recall']:.6f}")
        lines.append(f"- F1: {cm_40['f1']:.6f}")
        lines.append(f"- Confusion matrix [ [TN, FP], [FN, TP] ]: {cm_40['confusion_matrix']}")
    lines.append("")

    lines.append("Top SHAP features:")
    lines.append("ROI model top 5 (by mean(|SHAP|)):")
    for feat, score in top5_features_roi_shap:
        lines.append(f"- {feat}: {score:.8f}")
    lines.append("Success model top 5 (by mean(|SHAP|)):")
    for feat, score in top5_features_cls_shap:
        lines.append(f"- {feat}: {score:.8f}")
    lines.append("")

    lines.append(f"Verdict: {verdict}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

