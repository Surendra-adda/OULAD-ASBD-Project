
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

activity = pd.read_csv(DATA / "student_activity.csv")

keys = ["code_module", "code_presentation", "id_student"]

# Each student should have one outcome within each course presentation.
outcome_check = activity.groupby(keys)["final_result"].nunique()
if (outcome_check > 1).any():
    raise ValueError("Conflicting outcomes found for a student-course record.")

# Correct student-level engagement comparison.
student = (
    activity.groupby(keys + ["final_result"], as_index=False)
    .agg(total_clicks=("total_clicks", "sum"))
)

summary = (
    student.groupby("final_result")["total_clicks"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\n=== STUDENT-LEVEL ENGAGEMENT ===")
print(summary)
summary.to_csv(OUT / "student_level_engagement.csv")

# Create one transaction per student-course presentation.
# Items are the activity types with at least one recorded click.
baskets = (
    activity.loc[activity["total_clicks"] > 0]
    .groupby(keys)["activity_type"]
    .agg(lambda x: sorted(set(x)))
    .reset_index(name="activity_types")
)

# Attach the student's outcome as a separate item.
outcomes = student[keys + ["final_result"]]
baskets = baskets.merge(
    outcomes,
    on=keys,
    how="left",
    validate="one_to_one",
)

# Save a transparent transaction file for later frequent-pattern mining.
baskets["items"] = baskets.apply(
    lambda row: row["activity_types"] + [f"outcome={row['final_result']}"],
    axis=1,
)

baskets[ keys + ["final_result", "items"] ].to_json(
    OUT / "activity_transactions.jsonl",
    orient="records",
    lines=True,
)

print("\n=== TRANSACTION DATA ===")
print("Student-course transactions:", len(baskets))
print("Average distinct activity types:",
      round(baskets["activity_types"].str.len().mean(), 2))
print("Transactions saved to outputs/activity_transactions.jsonl")
