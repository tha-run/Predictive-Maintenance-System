"""
========================================================
  Industrial IoT Predictive Maintenance System
  Dataset :Predictive Maintenance Dataset (AI4I 2020)
  Model   : RandomForestClassifier
  Goal    : Predict machine failure from sensor data
========================================================
"""

# ── Standard & Third-Party Imports ───────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── Global Paths ──────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "data.csv")
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl")

# Stored after preprocessing so predict_machine_failure() can reuse them
_feature_columns: list = []
_model: RandomForestClassifier = None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ▸ Load Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    print("\n" + "=" * 60)
    print("  STEP 1 ▸ Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(path)

    # ── Basic Inspection ──────────────────────────────────────────────────────
    print(f"\n  Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n  Missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  → None found ✓")

    print("\n  Target class distribution (Machine failure):")
    target_col = _detect_target(df)
    dist = df[target_col].value_counts().sort_index()
    for label, count in dist.items():
        pct = count / len(df) * 100
        tag = "No Failure" if label == 0 else "Failure"
        print(f"    {label} ({tag}) : {count:,}  ({pct:.1f} %)")

    return df


def _detect_target(df: pd.DataFrame) -> str:
    """Auto-detect the target column (case-insensitive match for 'failure')."""
    for col in df.columns:
        if "failure" in col.lower() and "type" not in col.lower():
            return col
    raise ValueError("Could not auto-detect a target column containing 'failure'.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ▸ Preprocess Data
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame):
    """
    • Drop irrelevant ID / index columns
    • One-hot encode categorical features (e.g. Machine Type)
    • Separate features X and target y
    """
    global _feature_columns

    print("\n" + "=" * 60)
    print("  STEP 2 ▸ Preprocessing")
    print("=" * 60)

    # Drop obvious non-feature columns (ids, UDI, product ids, etc.)
    drop_candidates = [c for c in df.columns if c.lower() in {"udi", "product id", "productid"}]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)
        print(f"\n  Dropped non-feature columns : {drop_candidates}")

    target_col = _detect_target(df)

    # Separate target before encoding
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Also drop any secondary failure-type flag columns (e.g. TWF, HDF, …)
    secondary = [c for c in X.columns if c.lower() in {"twf", "hdf", "pwf", "osf", "rnf"}]
    if secondary:
        X = X.drop(columns=secondary)
        print(f"  Dropped secondary failure flags : {secondary}")

    # One-hot encode categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"\n  One-hot encoding categorical columns : {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    else:
        print("\n  No categorical columns detected — skipping encoding.")

    _feature_columns = X.columns.tolist()
    print(f"\n  Final feature count : {len(_feature_columns)}")
    print(f"  Features : {_feature_columns}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ▸ Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80 / 20 split to preserve class proportions."""
    print("\n" + "=" * 60)
    print("  STEP 3 ▸ Train-Test Split  (80 / 20, stratified)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"\n  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ▸ Train Model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest with class_weight='balanced' to handle
    the typical class imbalance in failure datasets.
    """
    global _model

    print("\n" + "=" * 60)
    print("  STEP 4 ▸ Training RandomForestClassifier")
    print("=" * 60)
 
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        
    )

    print("\n  Fitting model …")
    model.fit(X_train, y_train)
    print("  Training complete ✓")

    _model = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 ▸ Evaluate Model
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Compute and print:
      Accuracy · Precision · Recall · F1 · ROC-AUC
      Confusion Matrix · Classification Report

    Returns predictions and probabilities for downstream plotting.
    """
    print("\n" + "=" * 60)
    print("  STEP 5 ▸ Model Evaluation")
    print("=" * 60)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.3).astype(int)  # probability of class 1

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)

    # ── Pretty-print metrics ──────────────────────────────────────────────────
    print(f"""
  ┌─────────────────────────────────────────┐
  │  Metric          │  Value               │
  ├──────────────────┼──────────────────────┤
  │  Accuracy        │  {acc:.4f}               │
  │  Precision       │  {precision:.4f}               │
  │  Recall          │  {recall:.4f}               │
  │  F1 Score        │  {f1:.4f}               │
  │  ROC-AUC         │  {roc_auc:.4f}               │
  └──────────────────┴──────────────────────┘
    """)

    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

    # ── Feature Importance ────────────────────────────────────────────────────
    print("  Top-10 Feature Importances (descending):")
    importance_df = pd.DataFrame({
        "Feature"   : _feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print(importance_df.head(10).to_string(index=False))

    return y_pred, y_prob, cm, importance_df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 ▸ Plot Results
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    importance_df: pd.DataFrame,
):
    """Generate three publication-quality plots saved as PNG files."""

    print("\n" + "=" * 60)
    print("  STEP 6 ▸ Plotting Results")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Industrial IoT — Predictive Maintenance Dashboard", fontsize=15, fontweight="bold")

    # ── Plot 1 ▸ Confusion Matrix Heatmap ────────────────────────────────────
    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
        linewidths=0.5,
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=11)
    ax1.set_ylabel("True Label", fontsize=11)

    # ── Plot 2 ▸ ROC Curve ────────────────────────────────────────────────────
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)

    ax2.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax2.plot([0, 1], [0, 1], color="#BDBDBD", lw=1.5, linestyle="--", label="Random Classifier")
    ax2.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax2.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    # ── Plot 3 ▸ Feature Importance ───────────────────────────────────────────
    ax3 = axes[2]
    top_n   = min(10, len(importance_df))
    top_imp = importance_df.head(top_n)

    bars = ax3.barh(
        top_imp["Feature"][::-1],
        top_imp["Importance"][::-1],
        color=sns.color_palette("Blues_d", top_n),
        edgecolor="white",
    )
    ax3.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Importance Score", fontsize=11)

    for bar, val in zip(bars, top_imp["Importance"][::-1]):
        ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "dashboard.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard saved → {plot_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 ▸ Save Model
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model: RandomForestClassifier, path: str = MODEL_PATH):
    """Persist the trained model to disk using joblib."""
    print("\n" + "=" * 60)
    print("  STEP 7 ▸ Saving Model")
    print("=" * 60)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"\n  Model saved → {path}  ({size_kb:.1f} KB) ✓")


# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME INFERENCE ▸ predict_machine_failure()
# ─────────────────────────────────────────────────────────────────────────────
def predict_machine_failure(input_dict: dict) -> float:
    """
    Simulate real-time inference for a single machine reading.

    Parameters
    ----------
    input_dict : dict
        Raw sensor readings, e.g.:
        {
            "Air temperature [K]"     : 298.1,
            "Process temperature [K]" : 308.6,
            "Rotational speed [rpm]"  : 1551,
            "Torque [Nm]"             : 42.8,
            "Tool wear [min]"         : 108,
            "Type"                    : "M"
        }

    Returns
    -------
    float
        Probability of machine failure (0.0 – 1.0).
    """
    if _model is None:
        raise RuntimeError("Model has not been trained yet. Run main() first.")

    # Build a single-row DataFrame aligned to training features
    sample_df = pd.DataFrame([input_dict])

    # One-hot encode categorical columns (same logic as preprocess_data)
    cat_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        sample_df = pd.get_dummies(sample_df, columns=cat_cols, drop_first=False)

    # Align columns with the training feature set (fill missing dummies with 0)
    sample_df = sample_df.reindex(columns=_feature_columns, fill_value=0)

    prob = _model.predict_proba(sample_df)[0, 1]
    return float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Pipeline ──────────────────────────────────────────────────────────────
    df                              = load_data()
    X, y                            = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model                           = train_model(X_train, y_train)
    y_pred, y_prob, cm, imp_df      = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred, y_prob, cm, imp_df)
    save_model(model)

    # ── Real-time Inference Demo ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEMO ▸ Real-Time Inference")
    print("=" * 60)

    # Take a random sample from the test set and reconstruct its raw dict
    sample_index = X_test.sample(1, random_state=0).index[0]
    sample_input = X_test.loc[[sample_index]].to_dict(orient="records")[0]

    failure_prob = predict_machine_failure(sample_input)
    true_label   = y_test.loc[sample_index]

    print(f"\n  True label         : {'Failure ⚠' if true_label == 1 else 'No Failure ✓'}")
    print(f"  Predicted prob.    : {failure_prob:.4f}")
    print(f"  Predicted label    : {'Failure ⚠' if failure_prob >= 0.5 else 'No Failure ✓'}")

    print("\n" + "=" * 60)
    print("  Pipeline complete ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
