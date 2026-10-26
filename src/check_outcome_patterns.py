
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

records = []

with open(
    OUTPUT / "activity_transactions.jsonl",
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Each transaction's "items" includes activity types and one outcome label.
df["activity_types"] = df["items"].apply(
    lambda items: [
        item for item in items
        if not item.startswith("outcome=")
    ]
)

print("=== OUTCOME BASE RATES ===")
base_rates = df["final_result"].value_counts(normalize=True)
print((base_rates * 100).round(2).astype(str) + "%")

# Count the presence of each activity type in each outcome group.
rows = []

for outcome in sorted(df["final_result"].dropna().unique()):
    group = df[df["final_result"] == outcome]
    group_size = len(group)

    counts = {}
    for items in group["activity_types"]:
        for item in set(items):
            counts[item] = counts.get(item, 0) + 1

    for activity_type, count in counts.items():
        prevalence = count / group_size

        rows.append({
            "outcome": outcome,
            "activity_type": activity_type,
            "students_with_activity": count,
            "outcome_group_size": group_size,
            "prevalence": prevalence,
        })

results = pd.DataFrame(rows)

overall = (
    results.groupby("activity_type")
    .apply(
        lambda g: (
            g["students_with_activity"].sum()
            / g["outcome_group_size"].sum()
        ),
        include_groups=False,
    )
    .rename("overall_prevalence")
    .reset_index()
)

results = results.merge(overall, on="activity_type")
results["lift"] = (
    results["prevalence"] / results["overall_prevalence"]
)

for outcome in sorted(results["outcome"].unique()):
    print(f"\n=== {outcome} ===")
    print(
        results[results["outcome"] == outcome]
        .sort_values("lift", ascending=False)
        .head(8)[
            [
                "activity_type",
                "students_with_activity",
                "prevalence",
                "overall_prevalence",
                "lift",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

results.to_csv(
    OUTPUT / "activity_prevalence_by_outcome.csv",
    index=False,
)

print("\nSaved activity prevalence analysis.")
