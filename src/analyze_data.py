
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading processed datasets...")

activity = pd.read_csv(DATA / "student_activity.csv")
daily = pd.read_csv(DATA / "daily_activity.csv")
outcomes = pd.read_csv(DATA / "student_outcomes.csv")

# --------------------------------------------------
# 1. Validate the processed data
# --------------------------------------------------
print("\n=== DATA VALIDATION ===")
print("Student activity rows:", len(activity))
print("Daily activity rows:", len(daily))
print("Outcome records:", len(outcomes))
print("Missing outcome labels:", activity["final_result"].isna().sum())
print("Missing activity types:", activity["activity_type"].isna().sum())

print("\nOutcome counts:")
print(outcomes["final_result"].value_counts(dropna=False))

# --------------------------------------------------
# 2. Overall engagement by student outcome
# --------------------------------------------------
print("\n=== MEAN CLICKS BY OUTCOME ===")

outcome_summary = (
    activity.groupby("final_result", dropna=False)
    .agg(
        total_clicks=("total_clicks", "sum"),
        mean_clicks_per_student_resource_type=(
            "total_clicks", "mean"
        ),
        median_clicks_per_student_resource_type=(
            "total_clicks", "median"
        ),
        student_resource_groups=("id_student", "size"),
    )
    .sort_values("total_clicks", ascending=False)
)

print(outcome_summary.round(2))
outcome_summary.to_csv(OUTPUT / "outcome_summary.csv")

# Plot the mean clicks per student-resource-type group.
ax = (
    activity.groupby("final_result")["total_clicks"]
    .mean()
    .sort_values(ascending=False)
    .plot(kind="bar", figsize=(8, 5))
)
ax.set_title("Mean Clicks per Student-Resource-Type Group")
ax.set_xlabel("Final outcome")
ax.set_ylabel("Mean total clicks")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "mean_clicks_by_outcome.png", dpi=150)
plt.close()

# --------------------------------------------------
# 3. Resource-type usage
# --------------------------------------------------
print("\n=== RESOURCE-TYPE USAGE ===")

resource_summary = (
    activity.groupby("activity_type")
    .agg(
        total_clicks=("total_clicks", "sum"),
        interaction_records=("interaction_records", "sum"),
        student_resource_groups=("id_student", "size"),
    )
    .sort_values("total_clicks", ascending=False)
)

print(resource_summary)
resource_summary.to_csv(OUTPUT / "resource_type_summary.csv")

ax = resource_summary["total_clicks"].plot(
    kind="bar", figsize=(10, 5)
)
ax.set_title("Total Clicks by Learning Resource Type")
ax.set_xlabel("Resource type")
ax.set_ylabel("Total clicks")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "clicks_by_resource_type.png", dpi=150)
plt.close()

# --------------------------------------------------
# 4. Daily engagement trend
# --------------------------------------------------
print("\n=== DAILY ENGAGEMENT ===")

daily["date"] = pd.to_numeric(daily["date"], errors="coerce")

daily_trend = (
    daily.groupby("date", as_index=False)["total_clicks"]
    .sum()
    .sort_values("date")
)

daily_trend.to_csv(OUTPUT / "daily_engagement_trend.csv", index=False)

plt.figure(figsize=(11, 5))
plt.plot(daily_trend["date"], daily_trend["total_clicks"])
plt.title("Daily Learning Platform Activity")
plt.xlabel("Days relative to course presentation start")
plt.ylabel("Total clicks")
plt.tight_layout()
plt.savefig(OUTPUT / "daily_engagement_trend.png", dpi=150)
plt.close()

print("\nAnalysis complete.")
print("Tables and charts saved in:", OUTPUT)
