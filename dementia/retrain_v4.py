"""
Retrain the AD/Dementia model on PER-CLIP features.

dementia_master_dataset.csv's 163 feature columns (115 ac_* + li_*/syn_*/
freq_*/read_*/sem_* = 48) are exactly the output of
extract_pd_acoustic_features() + extract_pd_linguistic_features() from
feature_extraction.py -- the same raw per-clip features used by the PD
model, with no subject-level _mean/_std/_range aggregation.

This replaces the old dementia_fusion_v3_SVM.py model, whose 30
SelectKBest features were all subject-aggregation _std/_range columns
that single-clip inference always sets to 0.0 (constant ~0.88 output
for every input).

Saves dementia/results_v3/best_model.joblib in the same
{pipeline, model_name, feature_columns, n_features, cv_metrics} format.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dementia_master_dataset.csv")
OUT_PATH = os.path.join(BASE_DIR, "results_v3", "best_model.joblib")

NON_FEATURE_COLS = {"subject", "file", "label", "whisper_transcript", "whisper_word_timestamps"}

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

X = df[feature_cols].values
y = (df["label"] == "Dementia").astype(int).values
groups = df["subject"].values

n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
scale_pos_weight = n_neg / n_pos

print(f"Loaded {len(df)} clips from {df['subject'].nunique()} subjects, {len(feature_cols)} features")
print(f"Class balance: {n_pos} Dementia / {n_neg} No Dementia (scale_pos_weight={scale_pos_weight:.3f})")


def make_classifier(name: str):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42,
            eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        )
    raise ValueError(name)


CLASSIFIER_NAMES = ["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost"]
K_VALUES = [10, 20, 30, 40, 50]
SEEDS = [0, 1, 2]


def cv_metrics_for(name: str, k: int) -> dict:
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(mutual_info_classif, k=k)),
                ("clf", make_classifier(name)),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            y_test = y[test_idx]
            accs.append(accuracy_score(y_test, pred))
            precs.append(precision_score(y_test, pred, zero_division=0))
            recs.append(recall_score(y_test, pred, zero_division=0))
            f1s.append(f1_score(y_test, pred, zero_division=0))
            aucs.append(roc_auc_score(y_test, proba))
    return {
        "Accuracy": float(np.mean(accs)),
        "Precision": float(np.mean(precs)),
        "Recall": float(np.mean(recs)),
        "F1": float(np.mean(f1s)),
        "ROC-AUC": float(np.mean(aucs)),
        "ROC-AUC-std": float(np.std(aucs)),
    }


results = []
for name in CLASSIFIER_NAMES:
    for k in K_VALUES:
        metrics = cv_metrics_for(name, k)
        results.append((name, k, metrics))
        print(f"{name:20s} k={k:3d}  ROC-AUC={metrics['ROC-AUC']:.3f} +- {metrics['ROC-AUC-std']:.3f}  "
              f"Acc={metrics['Accuracy']:.3f}  F1={metrics['F1']:.3f}")

best_name, best_k, best_metrics = max(results, key=lambda r: r[2]["ROC-AUC"])
print(f"\nBest: {best_name} k={best_k} ROC-AUC={best_metrics['ROC-AUC']:.3f}")

del best_metrics["ROC-AUC-std"]

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(mutual_info_classif, k=best_k)),
    ("clf", make_classifier(best_name)),
])
final_pipeline.fit(X, y)

selected_mask = final_pipeline.named_steps["select"].get_support()
selected_features = [c for c, m in zip(feature_cols, selected_mask) if m]
print(f"\nSelected {len(selected_features)} features:")
for f in selected_features:
    print(f"  {f}")

joblib.dump({
    "pipeline": final_pipeline,
    "model_name": best_name,
    "feature_columns": feature_cols,
    "n_features": best_k,
    "cv_metrics": best_metrics,
}, OUT_PATH)
print(f"\nSaved to {OUT_PATH}")
