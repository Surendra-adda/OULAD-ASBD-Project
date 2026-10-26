
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("Loading OULAD student activity data...")

activity = pd.read_csv(PROCESSED / "student_activity.csv")

required = {
    "code_module",
    "code_presentation",
    "id_student",
    "activity_type",
    "total_clicks",
    "final_result",
}

if not required.issubset(activity.columns):
    raise ValueError(
        f"student_activity.csv is missing columns: "
        f"{required - set(activity.columns)}"
    )

activity["total_clicks"] = pd.to_numeric(
    activity["total_clicks"], errors="coerce"
).fillna(0)

activity = activity.dropna(
    subset=[
        "code_module",
        "code_presentation",
        "id_student",
        "activity_type",
        "final_result",
    ]
)

# Build one row per student-course presentation.
# Resource-type click counts become separate model features.
student_keys = [
    "code_module",
    "code_presentation",
    "id_student",
]

resource_features = (
    activity.pivot_table(
        index=student_keys,
        columns="activity_type",
        values="total_clicks",
        aggfunc="sum",
        fill_value=0,
    )
)

resource_features.columns = [
    f"clicks_{str(col)}" for col in resource_features.columns
]

total_clicks = (
    activity.groupby(student_keys)["total_clicks"]
    .sum()
    .rename("total_clicks")
)

activity_type_count = (
    activity.groupby(student_keys)["activity_type"]
    .nunique()
    .rename("distinct_activity_types")
)

# Outcomes are labels, not input features.
labels = (
    activity.groupby(student_keys)["final_result"]
    .agg(lambda values: values.mode().iloc[0])
    .rename("final_result")
)

data = pd.concat(
    [
        resource_features,
        total_clicks,
        activity_type_count,
        labels,
    ],
    axis=1,
).reset_index()

# Binary classification:
# 1 = Pass or Distinction
# 0 = Fail or Withdrawn
data["target"] = data["final_result"].map(
    {
        "Pass": 1,
        "Distinction": 1,
        "Fail": 0,
        "Withdrawn": 0,
    }
)

data = data.dropna(subset=["target"]).copy()
data["target"] = data["target"].astype(int)

# Exclude identifiers and all outcome-label columns from features.
excluded = set(
    student_keys + ["final_result", "target"]
)

feature_columns = [
    col for col in data.columns if col not in excluded
]

X = data[feature_columns].fillna(0)
y = data["target"]

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Insufficient examples for both target classes."
    )

print("\n=== CLASSIFICATION DATASET ===")
print("Student-course records:", len(data))
print("Feature count:", len(feature_columns))
print("Features:", feature_columns)
print("\nTarget distribution:")
print(
    y.map({1: "Pass/Distinction", 0: "Fail/Withdrawn"})
    .value_counts()
)


# Split by student so one student's records cannot appear
# in both training and testing sets.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=data["id_student"])
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Verify that there is no student overlap.
train_students = set(data.iloc[train_idx]["id_student"])
test_students = set(data.iloc[test_idx]["id_student"])

assert train_students.isdisjoint(test_students), (
    "Student overlap detected between train and test sets!"
)

print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Unique training students:", len(train_students))
print("Unique testing students:", len(test_students))
print("Student overlap check: PASSED")


models = {
    "Majority baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

results = []
prediction_frames = []

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
    }
    results.append(metrics)

    prediction_frames.append(
        pd.DataFrame(
            {
                "model": name,
                "actual": y_test.to_numpy(),
                "predicted": predictions,
            }
        )
    )

    print(pd.Series(metrics).to_string())
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Fail/Withdrawn",
                "Pass/Distinction",
            ],
            zero_division=0,
        )
    )

    if name in ["Decision Tree", "Logistic Regression"]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=[
                "Fail/Withdrawn",
                "Pass/Distinction",
            ],
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{name}: Confusion Matrix")
        fig.tight_layout()
        fig.savefig(
            OUTPUTS / (
                "confusion_matrix_"
                + name.lower().replace(" ", "_")
                + ".png"
            ),
            dpi=160,
        )
        plt.close(fig)

# Save evaluation metrics and held-out predictions.
results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUTS / "classification_metrics.csv",
    index=False,
)

pd.concat(prediction_frames, ignore_index=True).to_csv(
    OUTPUTS / "classification_test_predictions.csv",
    index=False,
)

# Export feature importance for interpretation.
tree = models["Decision Tree"]
importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": tree.feature_importances_,
    }
).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUTS / "classification_feature_importance.csv",
    index=False,
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))

print("\nTop Decision Tree features:")
print(importance.head(10).to_string(index=False))

print("\nClassification analysis complete.")
print("Results saved in:", OUTPUTS)
