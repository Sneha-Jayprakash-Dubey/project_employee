import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.burnout_model import burnout_risk


def train_and_save_model(
    data_path="dataset/employee_data.csv",
    model_path="productivity_model.pkl",
    scaler_path="scaler.pkl",
    features_path="model_features.pkl",
    classifier_path="burnout_classifier.pkl",
    metrics_path="model/metrics.json",
):
    """Train a productivity regression model and a burnout risk classifier."""
    model_version = 2

    df = pd.read_csv(data_path)

    # Drop columns that are not used for prediction
    drop_cols = [c for c in ["Employee_ID", "Job_Title", "Hire_Date"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Scale monthly salary to thousands (e.g. 20,000 -> 20) to reduce magnitude impact
    if "Monthly_Salary" in df.columns:
        df["Monthly_Salary"] = df["Monthly_Salary"] / 1000.0

    # Derived / engineered features
    df["Total_Effort"] = df["Work_Hours_Per_Week"] + df["Overtime_Hours"]
    df["Effort_per_Project"] = df["Total_Effort"] / df["Projects_Handled"].replace(0, 1)
    df["Workload_Index"] = df["Total_Effort"] / df["Team_Size"].replace(0, 1)

    # Legacy naming used in the webapp
    df["Satisfaction_Level"] = df["Employee_Satisfaction_Score"]

    # Create a burnout label based on the same logic used in the app
    df["Burnout_Risk_Label"] = df.apply(
        lambda r: burnout_risk(
            float(r.get("Work_Hours_Per_Week", 0)),
            float(r.get("Overtime_Hours", 0)),
            float(r.get("Employee_Satisfaction_Score", 0)),
        ),
        axis=1,
    )

    # One-hot encode categoricals (keep all resulting columns)
    df = pd.get_dummies(
        df,
        columns=["Department", "Gender", "Education_Level"],
        prefix=["Department", "Gender", "Education_Level"],
    )

    # Split targets for regression + classification
    target_reg = "Performance_Score"
    target_clf = "Burnout_Risk_Label"

    X = df.drop(columns=[target_reg, target_clf])
    y_reg = df[target_reg]
    y_clf = df[target_clf]

    # Encode label for classification
    label_enc = LabelEncoder()
    y_clf_enc = label_enc.fit_transform(y_clf)

    X_train, X_test, y_train, y_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf_enc, test_size=0.2, random_state=42, stratify=y_clf_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try multiple regressors and keep the best by test R2.
    reg_candidates = {
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42, n_estimators=350, learning_rate=0.05, max_depth=3, subsample=0.9
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
    }
    best_reg_name = None
    best_reg_r2 = float("-inf")
    model = None
    for name, candidate in reg_candidates.items():
        candidate.fit(X_train_scaled, y_train)
        score = r2_score(y_test, candidate.predict(X_test_scaled))
        if score > best_reg_r2:
            best_reg_r2 = score
            best_reg_name = name
            model = candidate

    # Try multiple classifiers and keep the best by accuracy.
    clf_candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }
    best_clf_name = None
    best_clf_acc = float("-inf")
    classifier = None
    for name, candidate in clf_candidates.items():
        candidate.fit(X_train_scaled, y_clf_train)
        score = accuracy_score(y_clf_test, candidate.predict(X_test_scaled))
        if score > best_clf_acc:
            best_clf_acc = score
            best_clf_name = name
            classifier = candidate

    # Regression metrics
    y_pred = model.predict(X_test_scaled)
    final_r2 = r2_score(y_test, y_pred)
    regression_metrics = {
        "best_model": best_reg_name,
        "r2_score": round(final_r2, 4),
        "accuracy_pct": round(final_r2 * 100, 2),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "mape": round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2),
    }

    # Classification metrics
    y_clf_pred = classifier.predict(X_test_scaled)
    cm = confusion_matrix(y_clf_test, y_clf_pred)
    classification_metrics = {
        "best_model": best_clf_name,
        "accuracy": round(accuracy_score(y_clf_test, y_clf_pred), 4),
        "confusion_matrix": cm.tolist(),
        "labels": label_enc.classes_.tolist(),
        "report": classification_report(
            y_clf_test, y_clf_pred, target_names=label_enc.classes_, output_dict=True
        ),
    }

    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X.columns), features_path)
    joblib.dump(classifier, classifier_path)
    joblib.dump(label_enc, classifier_path + ".labels")

    with open(metrics_path, "w") as f:
        json.dump({
            "metadata": {"model_version": model_version},
            "regression": regression_metrics,
            "classification": classification_metrics,
        }, f)

    print("Model training complete")
    print("Regression metrics:", regression_metrics)
    print("Classification metrics:", classification_metrics)


if __name__ == "__main__":
    train_and_save_model()
