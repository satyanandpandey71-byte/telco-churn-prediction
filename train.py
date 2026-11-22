"""
Customer Churn Prediction - training pipeline (IBM Telco dataset)

Steps
-----
1. Load + clean data
2. Build a preprocessing ColumnTransformer (scale numerics, one-hot categoricals)
3. Compare Logistic Regression, Random Forest, XGBoost with
   (a) class weights and (b) SMOTE, using stratified 5-fold cross-validation
4. Track every candidate in MLflow
5. Pick the best by CV ROC-AUC, evaluate on a held-out test set
6. Choose a decision threshold from out-of-fold predictions (max F1)
7. Explain the model with SHAP and save plots
8. Save the full pipeline (joblib) for the Streamlit app

Usage
-----
    python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import argparse
import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODEL_DIR = "models"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges has blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if c not in NUMERIC_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_candidates(X_train, y_train):
    """Return {name: sklearn/imblearn Pipeline} for every model/imbalance combo."""
    neg, pos = np.bincount(y_train)
    spw = neg / pos  # scale_pos_weight for XGBoost

    def lr():
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    def rf(**kw):
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **kw,
        )

    def xgb(**kw):
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **kw,
        )

    def pipe(model, smote=False):
        steps = [("prep", build_preprocessor(X_train))]
        if smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))
        return ImbPipeline(steps)

    lr_w = lr().set_params(class_weight="balanced")
    return {
        "logreg_classweight": pipe(lr_w),
        "logreg_smote": pipe(lr(), smote=True),
        "rf_classweight": pipe(rf(class_weight="balanced")),
        "rf_smote": pipe(rf(), smote=True),
        "xgb_classweight": pipe(xgb(scale_pos_weight=spw)),
        "xgb_smote": pipe(xgb(), smote=True),
    }


def cross_validate_all(candidates, X_train, y_train, cv):
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }
    results = {}
    for name, pipeline in candidates.items():
        print(f"  CV -> {name}")
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        results[name] = {
            m: float(np.mean(scores[f"test_{m}"])) for m in scoring
        }
        results[name].update(
            {f"{m}_std": float(np.std(scores[f"test_{m}"])) for m in scoring}
        )

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("imbalance_strategy", "smote" if "smote" in name else "class_weight")
            mlflow.log_params(
                {f"model__{k}": v for k, v in pipeline.named_steps["model"].get_params().items()
                 if isinstance(v, (int, float, str, bool, type(None)))}
            )
            mlflow.log_metrics({f"cv_{k}": v for k, v in results[name].items()})
    return pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])


def evaluate(pipeline, X_test, y_test, threshold):
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print("\nTest-set report (threshold = %.3f)" % threshold)
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    return metrics, proba


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def explain_with_shap(pipeline, X_train, X_test, out_dir):
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    X_bg = pd.DataFrame(prep.transform(X_train.sample(200, random_state=RANDOM_STATE)),
                        columns=feature_names)
    X_exp = pd.DataFrame(prep.transform(X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)),
                         columns=feature_names)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_bg)
        sv = explainer.shap_values(X_exp)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_exp)
        if isinstance(sv, list):  # older shap + RandomForest -> list per class
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:  # newer shap + RandomForest -> (n, f, classes)
            sv = sv[:, :, 1]

    # Beeswarm
    plt.figure()
    shap.summary_plot(sv, X_exp, max_display=15, show=False)
    plt.tight_layout()
    path_beeswarm = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path_beeswarm, dpi=150, bbox_inches="tight")
    plt.close()

    # Mean |SHAP| bar
    plt.figure()
    shap.summary_plot(sv, X_exp, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    path_bar = os.path.join(out_dir, "shap_importance.png")
    plt.savefig(path_bar, dpi=150, bbox_inches="tight")
    plt.close()

    importance = (
        pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\nTop churn drivers (mean |SHAP|):")
    print(importance.round(4).to_string())
    return [path_beeswarm, path_bar]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(args.data)
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features, churn rate = {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("telco-churn")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = build_candidates(X_train, y_train)
    print("\nCross-validating candidates...")
    cv_table = cross_validate_all(candidates, X_train, y_train, cv)
    print("\nCV results (sorted by ROC-AUC):")
    print(cv_table[["roc_auc", "pr_auc", "recall", "precision", "f1"]].round(4))
    cv_table.to_csv(os.path.join(MODEL_DIR, "cv_results.csv"))

    best_name = cv_table.index[0]
    best_pipe = candidates[best_name]
    print(f"\nBest model: {best_name}")

    # Threshold from out-of-fold training predictions (no test leakage)
    oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof)

    best_pipe.fit(X_train, y_train)
    metrics, _ = evaluate(best_pipe, X_test, y_test, threshold)

    plots = explain_with_shap(best_pipe, X_train, X_test, MODEL_DIR)

    # Save artifacts for the app
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, "churn_model.joblib"))
    meta = {
        "model_name": best_name,
        "threshold": threshold,
        "test_metrics": metrics,
        "feature_columns": list(X.columns),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with mlflow.start_run(run_name=f"FINAL_{best_name}"):
        mlflow.log_param("model_name", best_name)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items()})
        for p in plots:
            mlflow.log_artifact(p)
        mlflow.log_artifact(os.path.join(MODEL_DIR, "cv_results.csv"))
        mlflow.sklearn.log_model(
            best_pipe, name="model", serialization_format="cloudpickle"
        )

    print(f"\nSaved model + metadata to ./{MODEL_DIR}/")
    print("Run `mlflow ui` to browse experiments, `streamlit run app.py` to launch the app.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the IBM Telco churn CSV",
    )
    main(parser.parse_args())