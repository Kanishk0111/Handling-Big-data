from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("output/candidate_feature_sample.csv")

MODEL_OUTPUT_PATH = Path(
    "output/candidate_model.joblib"
)

METRICS_OUTPUT_PATH = Path(
    "output/candidate_model_validation.csv"
)

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
BETA = 0.5


MODEL_FEATURES = [
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


# ============================================================
# Threshold evaluation
# ============================================================

def evaluate_thresholds(y_true, probabilities):

    rows = []

    thresholds = np.arange(
        0.05,
        0.991,
        0.01,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f05 = fbeta_score(
            y_true,
            predictions,
            beta=BETA,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f0.5": f05,
                "predicted_positive": int(
                    predictions.sum()
                ),
            }
        )

    result = pd.DataFrame(rows)

    best_index = result["f0.5"].idxmax()

    return (
        result,
        result.loc[best_index],
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("TRAIN CANDIDATE-CONDITIONED MATCH MODEL")
    print("=" * 72)

    # --------------------------------------------------------
    # Load candidate feature dataset
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = (
        [
            "s1_id",
            "source_id",
            "label",
        ]
        + MODEL_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=[
            "s1_id",
            "label",
        ]
        + MODEL_FEATURES
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Prepare X / y / groups
    # --------------------------------------------------------

    X = df[
        MODEL_FEATURES
    ].astype(float)

    y = df[
        "label"
    ].astype(int).to_numpy()

    groups = df[
        "s1_id"
    ].astype(str).to_numpy()

    print()
    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Positive: {int(y.sum()):,}"
    )

    print(
        f"Negative: {int((y == 0).sum()):,}"
    )

    print(
        f"S1 groups: "
        f"{df['s1_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Entity-level train/validation split
    #
    # CRITICAL:
    # Same S1 must never occur in both sets.
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_index, validation_index = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_index
    ]

    X_validation = X.iloc[
        validation_index
    ]

    y_train = y[
        train_index
    ]

    y_validation = y[
        validation_index
    ]

    train_groups = set(
        groups[train_index]
    )

    validation_groups = set(
        groups[validation_index]
    )

    assert train_groups.isdisjoint(
        validation_groups
    )

    print()
    print("=" * 72)
    print("ENTITY-LEVEL SPLIT")
    print("=" * 72)

    print(
        f"Train rows: "
        f"{len(train_index):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_index):,}"
    )

    print(
        f"Train S1 entities: "
        f"{len(train_groups):,}"
    )

    print(
        f"Validation S1 entities: "
        f"{len(validation_groups):,}"
    )

    # ========================================================
    # MODEL 1
    # Logistic Regression
    # ========================================================

    logistic = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print()
    print(
        "TRAINING LOGISTIC REGRESSION..."
    )

    logistic.fit(
        X_train,
        y_train,
    )

    logistic_probability = (
        logistic.predict_proba(
            X_validation
        )[:, 1]
    )

    logistic_auc = roc_auc_score(
        y_validation,
        logistic_probability,
    )

    logistic_ap = (
        average_precision_score(
            y_validation,
            logistic_probability,
        )
    )

    (
        logistic_thresholds,
        logistic_best,
    ) = evaluate_thresholds(
        y_validation,
        logistic_probability,
    )

    print()
    print("=" * 72)
    print("LOGISTIC REGRESSION")
    print("=" * 72)

    print(
        f"ROC AUC: "
        f"{logistic_auc:.6f}"
    )

    print(
        f"Average precision: "
        f"{logistic_ap:.6f}"
    )

    print(
        f"Best validation F0.5: "
        f"{logistic_best['f0.5']:.6f}"
    )

    print(
        f"Threshold: "
        f"{logistic_best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{logistic_best['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{logistic_best['recall']:.6f}"
    )

    print(
        f"Predicted positive: "
        f"{int(logistic_best['predicted_positive']):,}"
    )

    # ========================================================
    # MODEL 2
    # Histogram Gradient Boosting
    # ========================================================

    gradient_boosting = (
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    )

    print()
    print(
        "TRAINING HISTOGRAM "
        "GRADIENT BOOSTING..."
    )

    gradient_boosting.fit(
        X_train,
        y_train,
    )

    gb_probability = (
        gradient_boosting.predict_proba(
            X_validation
        )[:, 1]
    )

    gb_auc = roc_auc_score(
        y_validation,
        gb_probability,
    )

    gb_ap = (
        average_precision_score(
            y_validation,
            gb_probability,
        )
    )

    (
        gb_thresholds,
        gb_best,
    ) = evaluate_thresholds(
        y_validation,
        gb_probability,
    )

    print()
    print("=" * 72)
    print(
        "HISTOGRAM "
        "GRADIENT BOOSTING"
    )
    print("=" * 72)

    print(
        f"ROC AUC: "
        f"{gb_auc:.6f}"
    )

    print(
        f"Average precision: "
        f"{gb_ap:.6f}"
    )

    print(
        f"Best validation F0.5: "
        f"{gb_best['f0.5']:.6f}"
    )

    print(
        f"Threshold: "
        f"{gb_best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{gb_best['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{gb_best['recall']:.6f}"
    )

    print(
        f"Predicted positive: "
        f"{int(gb_best['predicted_positive']):,}"
    )

    # ========================================================
    # Save threshold results
    # ========================================================

    logistic_thresholds[
        "model"
    ] = "logistic"

    gb_thresholds[
        "model"
    ] = "hist_gradient_boosting"

    threshold_results = pd.concat(
        [
            logistic_thresholds,
            gb_thresholds,
        ],
        ignore_index=True,
    )

    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_results.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # Save models
    # ========================================================

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "logistic": logistic,
            "hist_gradient_boosting": (
                gradient_boosting
            ),
            "features": MODEL_FEATURES,
            "random_state": RANDOM_STATE,
        },
        MODEL_OUTPUT_PATH,
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(
        MODEL_OUTPUT_PATH
    )

    print(
        METRICS_OUTPUT_PATH
    )

    print()
    print("=" * 72)
    print("IMPORTANT")
    print("=" * 72)

    print(
        "These are pair-level metrics on the "
        "sampled candidate dataset."
    )

    print(
        "Do NOT freeze the production threshold "
        "from this run."
    )

    print(
        "The next step is full candidate-set "
        "validation with macro F0.5 per S1."
    )


if __name__ == "__main__":
    main()