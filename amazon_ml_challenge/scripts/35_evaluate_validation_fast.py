from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

SCORED_FILE = OUTPUT_DIR / "full_validation_scored_pairs_v2.csv"
TRUTH_FILE = OUTPUT_DIR / "validation_truth_pairs.csv"

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.73,
    0.75,
    0.77,
    0.80,
    0.82,
    0.85,
    0.87,
    0.90,
    0.92,
    0.94,
    0.95,
    0.97,
    0.99,
]


def f05(precision, recall):
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


print("=" * 80)
print("FAST VALIDATION THRESHOLD EVALUATION")
print("=" * 80)

print(f"Loading: {SCORED_FILE}")

scored = pd.read_csv(SCORED_FILE)

print(f"Rows: {len(scored):,}")
print(f"S1 entities: {scored['s1_id'].nunique():,}")

required = {"s1_id", "label", "probability"}
missing = required - set(scored.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# ---------------------------------------------------------------------
# Entity coding
# ---------------------------------------------------------------------
#
# Convert S1 IDs to integer group codes once.
# This lets numpy.bincount calculate per-S1 counts extremely quickly.
#

s1_values = scored["s1_id"].to_numpy()

unique_s1, s1_codes = np.unique(
    s1_values,
    return_inverse=True,
)

labels = scored["label"].to_numpy(dtype=np.int8)
probabilities = scored["probability"].to_numpy(dtype=np.float32)

n_entities = len(unique_s1)

print(f"Encoded entities: {n_entities:,}")

# True number of matches for every S1.
true_counts = np.bincount(
    s1_codes,
    weights=labels,
    minlength=n_entities,
).astype(np.int32)

print(f"Total true pairs represented: {true_counts.sum():,}")

# ---------------------------------------------------------------------
# Evaluate thresholds
# ---------------------------------------------------------------------

results = []

for threshold in THRESHOLDS:

    mask = probabilities >= threshold

    # Number predicted for each S1.
    predicted_counts = np.bincount(
        s1_codes[mask],
        minlength=n_entities,
    ).astype(np.int32)

    # Number of correctly predicted true pairs for each S1.
    tp_counts = np.bincount(
        s1_codes[mask & (labels == 1)],
        minlength=n_entities,
    ).astype(np.int32)

    fp_counts = predicted_counts - tp_counts
    fn_counts = true_counts - tp_counts

    # Entity-level precision / recall.
    precision = np.zeros(n_entities, dtype=np.float64)
    recall = np.zeros(n_entities, dtype=np.float64)

    has_predictions = predicted_counts > 0
    has_truth = true_counts > 0

    precision[has_predictions] = (
        tp_counts[has_predictions]
        / predicted_counts[has_predictions]
    )

    recall[has_truth] = (
        tp_counts[has_truth]
        / true_counts[has_truth]
    )

    # Challenge-specific F0.5.
    #
    # Empty truth + empty prediction = perfect score.
    #
    f_scores = np.zeros(n_entities, dtype=np.float64)

    denom = 0.25 * precision + recall
    valid = denom > 0

    f_scores[valid] = (
        1.25
        * precision[valid]
        * recall[valid]
        / denom[valid]
    )

    empty_correct = (
        (true_counts == 0)
        & (predicted_counts == 0)
    )

    f_scores[empty_correct] = 1.0

    macro_f05 = f_scores.mean()

    total_tp = int(tp_counts.sum())
    total_fp = int(fp_counts.sum())
    total_fn = int(fn_counts.sum())

    pair_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    pair_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    results.append(
        {
            "threshold": threshold,
            "macro_f05": macro_f05,
            "pair_precision": pair_precision,
            "pair_recall": pair_recall,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "predicted_pairs": int(predicted_counts.sum()),
            "s1_with_predictions": int((predicted_counts > 0).sum()),
            "s1_empty_correct": int(empty_correct.sum()),
        }
    )

    print(
        f"threshold={threshold:.2f} "
        f"macro_F0.5={macro_f05:.6f} "
        f"pair_precision={pair_precision:.6f} "
        f"pair_recall={pair_recall:.6f} "
        f"predicted={predicted_counts.sum():,}"
    )


results_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "validation_threshold_results_v2.csv"
results_df.to_csv(output_file, index=False)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)

print(
    results_df.sort_values(
        "macro_f05",
        ascending=False,
    ).to_string(index=False)
)

best = results_df.loc[
    results_df["macro_f05"].idxmax()
]

print()
print("=" * 80)
print("BEST MACRO F0.5")
print("=" * 80)

print(f"Threshold:       {best['threshold']:.2f}")
print(f"Macro F0.5:      {best['macro_f05']:.6f}")
print(f"Pair precision:  {best['pair_precision']:.6f}")
print(f"Pair recall:     {best['pair_recall']:.6f}")
print(f"Predicted pairs: {int(best['predicted_pairs']):,}")

print()
print(f"Saved: {output_file}")