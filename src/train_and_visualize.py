"""Train 4 models on features_v2_all.csv (17 features) and visualize results."""
import csv, copy, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score,
)
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "outputs" / "features_v2_all.csv"
FIG = BASE / "outputs" / "figures_v2"
FIG.mkdir(exist_ok=True)

FEATURES = [
    "has_reentrancy_eth", "num_reentrancy_detectors",
    "cei_violation", "num_external_calls",
    "has_low_level_call", "has_dangerous_call",
    "num_functions", "has_modifier",
    "has_reentrancy_guard", "has_emit_after_call",
    "has_mutex_pattern", "num_require_assert",
    "has_state_set_before_call", "num_modifiers_applied",
    "has_internal_ext_call", "has_interface_cast",
    "lines_of_code",
]

with open(CSV, "r") as f:
    rows = list(csv.DictReader(f))
X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
y = np.array([int(r["true_label"]) for r in rows])
print(f"Dataset: {len(y)} samples, {X.shape[1]} features", flush=True)
print(f"Safe={sum(y==0)}, Reentrant={sum(y==1)}", flush=True)

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=2, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in MODELS.items():
    print(f"\nTraining {name}...", flush=True)
    all_true, all_pred, all_prob = [], [], []
    fold_m = {"acc": [], "prec": [], "rec": [], "f1": []}

    for train_idx, test_idx in cv.split(X, y):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", copy.deepcopy(model))])
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        probs = pipe.predict_proba(X[test_idx])[:, 1]

        all_true.extend(y[test_idx])
        all_pred.extend(preds)
        all_prob.extend(probs)

        fold_m["acc"].append(accuracy_score(y[test_idx], preds))
        fold_m["prec"].append(precision_score(y[test_idx], preds, zero_division=0))
        fold_m["rec"].append(recall_score(y[test_idx], preds, zero_division=0))
        fold_m["f1"].append(f1_score(y[test_idx], preds, zero_division=0))

    # Feature importance (only RF and XGBoost)
    imp = None
    if hasattr(model, "feature_importances_"):
        pipe_full = Pipeline([("scaler", StandardScaler()), ("clf", copy.deepcopy(model))])
        pipe_full.fit(StandardScaler().fit_transform(X), y)
        imp = pipe_full.named_steps["clf"].feature_importances_

    results[name] = {
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "y_prob": np.array(all_prob),
        "metrics": {k: (np.mean(v), np.std(v)) for k, v in fold_m.items()},
        "importances": imp,
    }

    acc_m, acc_s = results[name]["metrics"]["acc"]
    prec_m, prec_s = results[name]["metrics"]["prec"]
    rec_m, rec_s = results[name]["metrics"]["rec"]
    f1_m, f1_s = results[name]["metrics"]["f1"]
    print(f"  Acc={acc_m:.4f}+/-{acc_s:.4f}  Prec={prec_m:.4f}  Rec={rec_m:.4f}  F1={f1_m:.4f}", flush=True)

# =========================================
# 1. Confusion Matrices (2x2 grid)
# =========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, res) in zip(axes.flat, results.items()):
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Reentrant"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    f1_val = res["metrics"]["f1"][0]
    acc_val = res["metrics"]["acc"][0]
    ax.set_title(f"{name}\nAcc={acc_val:.1%}  F1={f1_val:.1%}", fontsize=11)
fig.suptitle("Confusion Matrices — 5-Fold CV (365 contracts, 17 features)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrices.png", dpi=150)
plt.close()
print("\nSaved confusion_matrices.png", flush=True)

# =========================================
# 2. ROC Curves
# =========================================
COLORS = {"Random Forest": "#2ecc71", "XGBoost": "#3498db",
          "Logistic Regression": "#e67e22", "SVM (RBF)": "#9b59b6"}
fig, ax = plt.subplots(figsize=(8, 7))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — 5-Fold CV", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "roc_curves.png", dpi=150)
plt.close()
print("Saved roc_curves.png", flush=True)

# =========================================
# 3. Metrics Comparison Bar Chart
# =========================================
metric_names = ["Accuracy", "Precision", "Recall", "F1"]
metric_keys = ["acc", "prec", "rec", "f1"]
x = np.arange(len(metric_names))
width = 0.18
fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, res) in enumerate(results.items()):
    means = [res["metrics"][k][0] for k in metric_keys]
    stds = [res["metrics"][k][1] for k in metric_keys]
    ax.bar(x + i * width, means, width, yerr=stds, label=name,
           color=COLORS[name], capsize=4, alpha=0.85)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — 5-Fold CV (17 features)", fontsize=14, fontweight="bold")
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(metric_names, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0.65, 1.0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "metrics_comparison.png", dpi=150)
plt.close()
print("Saved metrics_comparison.png", flush=True)

# =========================================
# 4. Feature Importance (RF)
# =========================================
imp = results["Random Forest"]["importances"]
if imp is not None:
    sorted_idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if i >= len(imp) - 5 else "#3498db" for i in range(len(imp))]
    sorted_colors = [colors[i] for i in sorted_idx]
    ax.barh(range(len(imp)), imp[sorted_idx], color=sorted_colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([FEATURES[i] for i in sorted_idx], fontsize=10)
    ax.set_xlabel("Mean Importance (Gini)", fontsize=12)
    ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png", flush=True)

# =========================================
# 5. Summary Table
# =========================================
print("\n" + "=" * 72, flush=True)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}", flush=True)
print("-" * 72, flush=True)
for name, res in results.items():
    m = res["metrics"]
    print(f"{name:<25} {m['acc'][0]:>10.4f} {m['prec'][0]:>10.4f} {m['rec'][0]:>10.4f} {m['f1'][0]:>10.4f}", flush=True)

    cm = confusion_matrix(res["y_true"], res["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"{'':>25} TP={tp}  FP={fp}  FN={fn}  TN={tn}", flush=True)
print("=" * 72, flush=True)
print("\nDone!", flush=True)
