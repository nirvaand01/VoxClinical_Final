"""
=============================================================
FUSION MODEL — Acoustic + Linguistic Parkinson's Detection
              (curated master dataset, GridSearchCV)
Disease  : Parkinson's Disease (PD)
Modality : Acoustic + Linguistic (lexical, syntactic [spaCy],
           semantic/frequency/readability [NLTK] -- all
           pre-extracted in the master dataset)
Dataset  : parkinsons_master_dataset.csv
           36 clips / 36 subjects (15 PD / 21 HC)
Features : curated ac_* (acoustic) + li_*/syn_*/freq_*/sem_*/read_*
           (linguistic) -- top 15 selected via SelectKBest (MI)
Models   : XGBoost, Random Forest, SVM -- each tuned with GridSearchCV
           (hyperparameters selected via Stratified 5-Fold CV,
           scoring=ROC-AUC; final honest evaluation via LOO-CV,
           the standard scheme for this small N)
=============================================================
"""

import os
import warnings
from functools import partial

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "parkinsons_master_dataset.csv")
OUT  = os.path.join(BASE, "results_fusion_master")
os.makedirs(OUT, exist_ok=True)


# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)

EXCLUDE = {"subject", "file", "label", "whisper_transcript", "whisper_word_timestamps"}
feat_cols = [c for c in df.columns if c not in EXCLUDE]
ac_cols   = [c for c in feat_cols if c.startswith("ac_")]
li_cols   = [c for c in feat_cols if c not in ac_cols]
feat_modality = {c: "Acoustic" for c in ac_cols}
feat_modality.update({c: "Linguistic" for c in li_cols})

X = df[feat_cols].values.astype(float)
y = (df["label"] == "PD").astype(int).values

n_pos, n_neg = int(y.sum()), int((y == 0).sum())
spw = n_neg / n_pos

print(f"Dataset: {len(df)} subjects")
print(f"  PD: {n_pos} | HC: {n_neg}")
print(f"  Acoustic features  : {len(ac_cols)}")
print(f"  Linguistic features: {len(li_cols)}")
print(f"  Total features     : {len(feat_cols)}")


# ── Pipeline & CV ──────────────────────────────────────────────────────────
N_FEATURES = 15
grid_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # for hyperparameter search
final_cv = LeaveOneOut()                                                # for honest small-N evaluation


# functools.partial (not a local function) so the fitted pipeline -- including
# this SelectKBest -- can be pickled/unpickled outside this script.
mi_score = partial(mutual_info_classif, random_state=42)


def make_pipe(estimator):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mi_score, k=N_FEATURES)),
        ("clf",    estimator),
    ])


# ── Models + hyperparameter grids ────────────────────────────────────────
MODEL_GRIDS = {
    "XGBoost": (
        xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42),
        {
            "clf__n_estimators":  [100, 200],
            "clf__max_depth":     [2, 3, 4],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample":     [0.8, 1.0],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {
            "clf__n_estimators":     [200, 400],
            "clf__max_depth":        [3, 5, None],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    ),
    "SVM": (
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        {
            "clf__C":     [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1],
        },
    ),
}


# ── Grid search per model ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GRID SEARCH (Stratified 5-Fold CV, scoring=ROC-AUC)")
print("=" * 60)

gs_results = {}
for name, (estimator, grid) in MODEL_GRIDS.items():
    print(f"\n{name} ...")
    gs = GridSearchCV(make_pipe(estimator), grid, cv=grid_cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y)
    gs_results[name] = gs
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV AUC : {gs.best_score_:.4f}")


# ── Final LOO-CV evaluation of each tuned model ──────────────────────────
print("\n" + "=" * 60)
print("  FINAL EVALUATION (tuned pipelines, Leave-One-Out CV)")
print("=" * 60)

comparison_rows = []
fold_predictions = {}
for name, gs in gs_results.items():
    tuned = clone(gs.best_estimator_)
    y_pred  = cross_val_predict(tuned, X, y, cv=final_cv, method="predict")
    y_proba = cross_val_predict(tuned, X, y, cv=final_cv, method="predict_proba")[:, 1]
    fold_predictions[name] = (y_pred, y_proba)

    row = {
        "Model":       name,
        "Accuracy":    accuracy_score(y, y_pred),
        "Precision":   precision_score(y, y_pred, zero_division=0),
        "Recall":      recall_score(y, y_pred, zero_division=0),
        "F1":          f1_score(y, y_pred, zero_division=0),
        "ROC-AUC":     roc_auc_score(y, y_proba),
        "GridCV_AUC":  gs.best_score_,
        "Best Params": str(gs.best_params_),
    }
    comparison_rows.append(row)
    print(f"\n  {name}:")
    for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        print(f"    {k:10s}: {row[k]:.4f}")

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

# Select by ROC-AUC among XGBoost/Random Forest. SVC(probability=True,
# class_weight="balanced") is known to produce predict()/predict_proba() that
# disagree on small/imbalanced folds (Platt-scaling quirk) -- it can show a
# near-perfect AUC while its predictions are degenerate (e.g. all one class).
# It's still reported/plotted for comparison, just not eligible for "best".
eligible = comp_df[comp_df["Model"] != "SVM"]
best_name = eligible.loc[eligible["ROC-AUC"].idxmax(), "Model"]
best_gs   = gs_results[best_name]
y_pred, y_proba = fold_predictions[best_name]
best_auc = roc_auc_score(y, y_proba)

print(f"\n*** Best model: {best_name} (LOO ROC-AUC={best_auc:.4f}) ***")
print(f"\nClassification report ({best_name}):")
print(classification_report(y, y_pred, target_names=["HC", "PD"]))


# ── Plots: model comparison, confusion matrix, ROC ───────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
comp_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
    kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)
ax.set_title("Parkinson's Fusion (Master Dataset) — Model Comparison (Tuned, LOO-CV)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y, y_pred), display_labels=["HC", "PD"]
).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Best Model ({best_name}) — LOO Confusion Matrix")
RocCurveDisplay.from_predictions(y, y_proba, ax=axes[1], name=best_name)
axes[1].set_title(f"ROC Curve (AUC={best_auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ── Feature importance (permutation importance on tuned pipeline) ───────
print("\nFitting best tuned pipeline on full data for feature importance ...")
final_pipe = clone(best_gs.best_estimator_).fit(X, y)

# ── Save the trained model ────────────────────────────────────────────────
model_path = os.path.join(OUT, "best_model.joblib")
joblib.dump({
    "pipeline":        final_pipe,
    "model_name":      best_name,
    "best_params":     best_gs.best_params_,
    "feature_columns": feat_cols,
    "n_features":      N_FEATURES,
    "cv_metrics":      comp_df[comp_df["Model"] == best_name].iloc[0].to_dict(),
}, model_path)
print(f"Saved trained pipeline -> {model_path}")

perm = permutation_importance(
    final_pipe, X, y, n_repeats=30, random_state=42, scoring="roc_auc", n_jobs=-1
)
imp_df = pd.DataFrame({
    "feature":    feat_cols,
    "importance": perm.importances_mean,
    "modality":   [feat_modality[f] for f in feat_cols],
}).sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

print(f"\nTop 15 features (permutation importance, {best_name}):")
print(imp_df.head(15).to_string(index=False))

palette = {"Acoustic": "#1565C0", "Linguistic": "#E64A19"}
top = imp_df.head(25)
fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(data=top, y="feature", x="importance", hue="modality",
            dodge=False, palette=palette, ax=ax)
ax.set_title(f"Parkinson's Fusion — Top 25 Feature Importances ({best_name})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

modal_imp = imp_df.assign(importance=imp_df["importance"].clip(lower=0)) \
                   .groupby("modality")["importance"].sum()
print(f"\nImportance by modality:\n{modal_imp.to_string()}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(modal_imp.values, labels=modal_imp.index, autopct="%1.1f%%",
       colors=[palette[m] for m in modal_imp.index], startangle=90)
ax.set_title("Feature Importance by Modality")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "modality_pie.png"), dpi=150)
plt.close()


# ── Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Best model : {best_name}")
print(f"  Params     : {best_gs.best_params_}")
print(f"  Accuracy   : {accuracy_score(y, y_pred):.4f}")
print(f"  Precision  : {precision_score(y, y_pred, zero_division=0):.4f}")
print(f"  Recall     : {recall_score(y, y_pred, zero_division=0):.4f}")
print(f"  F1-Score   : {f1_score(y, y_pred, zero_division=0):.4f}")
print(f"  ROC-AUC    : {best_auc:.4f}")
print(f"\nAll results saved to: {OUT}")
print("Done.")
