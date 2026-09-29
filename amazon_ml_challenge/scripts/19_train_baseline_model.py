from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


POSITIVE_FILE = Path("output/match_feature_sample.csv")
RANDOM_NEGATIVE_FILE = Path(
    "output/positive_negative_feature_sample.csv"
)
HARD_NEGATIVE_FILE = Path(
    "output/hard_negative_feature_sample.csv"
)

MODEL_FILE = Path(
    "output/entity_match_baseline.joblib"
)

FEATURES = [
    "country_same",
    "name_exact",
    "name_compact_exact",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_overlap",
    "name_containment",
    "address_exact",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_overlap",
    "address_containment",
]


def load_file(path, label=None):

    print(f"Loading {path}")

    df = pd.read_csv(path)

    if label is not None:
        df["label"] = label

    # Some older feature files don't contain country_same.
    # The model can still train without it.
    available_features = [
        column
        for column in FEATURES
        if column in df.columns
    ]

    missing_features = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_features:
        print(
            "  Missing features: "
            + ", ".join(missing_features)
        )

    for column in available_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    return df

def main():

    print("=" * 72)
    print("BASELINE ENTITY MATCH MODEL")
    print("=" * 72)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    positive = load_file(
        POSITIVE_FILE,
        label=1,
    )

    random_negative = load_file(
        RANDOM_NEGATIVE_FILE,
        label=0,
    )

    hard_negative = load_file(
        HARD_NEGATIVE_FILE,
        label=0,
    )

    print("\nDataset sizes:")
    print(
        f"Positive:        {len(positive):,}"
    )
    print(
        f"Random negative: {len(random_negative):,}"
    )
    print(
        f"Hard negative:   {len(hard_negative):,}"
    )

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    datasets = [
        positive,
        random_negative,
        hard_negative,
    ]

    common_features = [
        feature
        for feature in FEATURES
        if all(
            feature in dataset.columns
            for dataset in datasets
        )
    ]

    missing_from_common = [
        feature
        for feature in FEATURES
        if feature not in common_features
    ]

    if missing_from_common:
        print(
            "\nFeatures excluded because they are "
            "missing from at least one dataset:"
        )
        for feature in missing_from_common:
            print(f"  - {feature}")

    print("\nFeatures used:")
    for feature in common_features:
        print(f"  - {feature}")

    data = pd.concat(
        datasets,
        ignore_index=True,
    )

    data = data.dropna(
        subset=common_features + ["label"]
    )

    print(
        f"\nTotal training rows: "
        f"{len(data):,}"
    )

    print(
        "\nClass distribution:"
    )
    print(
        data["label"].value_counts()
    )

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    X = data[common_features]
    y = data["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining logistic regression...")

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("MODEL PERFORMANCE")
    print("=" * 72)

    print(
        f"ROC AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"Average precision: "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    # ------------------------------------------------------------
    # Threshold analysis
    # ------------------------------------------------------------

    print("=" * 72)
    print("THRESHOLD ANALYSIS")
    print("=" * 72)

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
        0.97,
        0.99,
    ]

    print(
        f"{'threshold':>10} "
        f"{'selected':>10} "
        f"{'precision':>10} "
        f"{'recall':>10} "
        f"{'F0.5':>10}"
    )

    for threshold in thresholds:

        pred = (
            probabilities >= threshold
        )

        tp = np.sum(
            pred & (y_test.to_numpy() == 1)
        )

        fp = np.sum(
            pred & (y_test.to_numpy() == 0)
        )

        fn = np.sum(
            (~pred) & (y_test.to_numpy() == 1)
        )

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        beta = 0.5

        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (
                (1 + beta**2)
                * precision
                * recall
                / (
                    beta**2 * precision
                    + recall
                )
            )

        print(
            f"{threshold:10.2f} "
            f"{np.sum(pred):10,} "
            f"{precision:10.4f} "
            f"{recall:10.4f} "
            f"{f05:10.4f}"
        )

    # ------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = pd.DataFrame(
        {
            "feature": common_features,
            "coefficient": (
                classifier.coef_[0]
            ),
        }
    )

    coefficients[
        "abs_coefficient"
    ] = coefficients[
        "coefficient"
    ].abs()

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False,
    )

    print("\n" + "=" * 72)
    print("FEATURE COEFFICIENTS")
    print("=" * 72)

    print(
        coefficients[
            [
                "feature",
                "coefficient",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    MODEL_FILE.parent.mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_FILE,
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()