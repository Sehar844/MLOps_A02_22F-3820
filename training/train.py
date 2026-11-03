
import argparse
import json
import os
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="data/raw/breast_cancer.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts"
    )
    parser.add_argument(
        "--model-version",
        default=None
    )
    args = parser.parse_args()

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    random_state = params["random_state"]
    test_size = params["test_size"]
    cv_folds = params["cv_folds"]

    model_version = (
        args.model_version or params["model_version"]
    )

    df = pd.read_csv(args.data_path)

    feature_names = [
        column for column in df.columns
        if column != "target"
    ]

    X = df[feature_names]
    y = df["target"]

    assert len(feature_names) == 30
    assert not df.isnull().any().any()
    assert set(y.unique()) == {0, 1}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["logistic_regression"]["C"],
                max_iter=params["logistic_regression"]["max_iter"],
                random_state=random_state
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=params["random_forest"]["n_estimators"],
                max_depth=params["random_forest"]["max_depth"],
                min_samples_split=params["random_forest"]["min_samples_split"],
                min_samples_leaf=params["random_forest"]["min_samples_leaf"],
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    }

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "malignant_recall": "recall",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc"
    }

    results = {}

    print("\nCross-validation results:")

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results[name] = {
            "mean_malignant_recall": float(
                scores["test_malignant_recall"].mean()
            ),
            "mean_accuracy": float(
                scores["test_accuracy"].mean()
            ),
            "mean_roc_auc": float(
                scores["test_roc_auc"].mean()
            )
        }

        print(f"\n{name}")
        print(
            "Mean malignant recall:",
            round(results[name]["mean_malignant_recall"], 4)
        )
        print(
            "Mean accuracy:",
            round(results[name]["mean_accuracy"], 4)
        )
        print(
            "Mean ROC-AUC:",
            round(results[name]["mean_roc_auc"], 4)
        )

    best_name = max(
        results,
        key=lambda name: results[name]["mean_malignant_recall"]
    )

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "malignant_recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        )
    }

    print("\nSelected model:", best_name)
    print("\nHeld-out test metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    card_path = output_dir / "model_card.json"

    joblib.dump(best_model, model_path)

    model_card = {
        "model_version": model_version,
        "model_family": best_name,
        "features": feature_names,
        "metrics": {
            "cross_validation": results,
            "test": test_metrics
        },
        "random_state": random_state,
        "sklearn_version": sklearn.__version__,
        "git_commit": get_git_commit(),
        "dataset_rows": len(df),
        "training_rows": len(X_train),
        "test_rows": len(X_test)
    }

    with open(card_path, "w") as file:
        json.dump(model_card, file, indent=4)

    print("\nSaved model:", model_path)
    print("Saved model card:", card_path)

    if test_metrics["malignant_recall"] < 0.95:
        raise SystemExit(
            "ERROR: Test malignant recall is below 0.95"
        )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
