
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "activity_transactions.jsonl"
OUTPUT_FILE = ROOT / "outputs" / "outcome_activity_patterns.csv"

records = []

with INPUT_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        items = row.get("items", [])

        outcome_items = [
            item for item in items
            if item.startswith("outcome=")
        ]
        if not outcome_items:
            continue

        outcome = outcome_items[0].replace("outcome=", "")
        activity_types = {
            item for item in items
            if not item.startswith("outcome=")
        }

        records.append({
            "outcome": outcome,
            "activity_types": activity_types
        })

if not records:
    raise ValueError(
        f"No usable transactions found in {INPUT_FILE}"
    )

total_students = len(records)
overall_counts = Counter()
outcome_counts = Counter()
outcome_activity_counts = defaultdict(Counter)

for record in records:
    outcome = record["outcome"]
    activities = record["activity_types"]

    outcome_counts[outcome] += 1

    for activity in activities:
        overall_counts[activity] += 1
        outcome_activity_counts[outcome][activity] += 1

rows = []

for outcome, group_size in outcome_counts.items():
    outcome_prevalence = group_size / total_students

    for activity, count in outcome_activity_counts[outcome].items():
        group_prevalence = count / group_size
        overall_prevalence = overall_counts[activity] / total_students

        lift = (
            group_prevalence / overall_prevalence
            if overall_prevalence else 0
        )

        rows.append({
            "outcome": outcome,
            "activity_type": activity,
            "students_with_activity_and_outcome": count,
            "students_in_outcome_group": group_size,
            "outcome_group_prevalence": group_prevalence,
            "overall_activity_prevalence": overall_prevalence,
            "outcome_prevalence": outcome_prevalence,
            "lift": lift
        })

result = pd.DataFrame(rows)

# Exclude extremely rare activity types from the main interpretation.
result["reliable_for_interpretation"] = (
    result["students_with_activity_and_outcome"] >= 30
)

result = result.sort_values(
    ["outcome", "lift"],
    ascending=[True, False]
)

result.to_csv(OUTPUT_FILE, index=False)

print(f"Transactions analysed: {total_students:,}")
print("\nOutcome distribution:")
for outcome, count in outcome_counts.most_common():
    print(f"{outcome:12} {count:6,} ({count / total_students:.2%})")

print("\nTop activity associations for each outcome:")
for outcome in sorted(outcome_counts):
    print(f"\n{outcome}")
    subset = result[
        (result["outcome"] == outcome)
        & result["reliable_for_interpretation"]
    ].head(5)

    if subset.empty:
        print("No activity types met the minimum count.")
        continue

    print(
        subset[
            [
                "activity_type",
                "students_with_activity_and_outcome",
                "outcome_group_prevalence",
                "overall_activity_prevalence",
                "lift"
            ]
        ].to_string(
            index=False,
            formatters={
                "outcome_group_prevalence": "{:.1%}".format,
                "overall_activity_prevalence": "{:.1%}".format,
                "lift": "{:.2f}".format
            }
        )
    )

print(f"\nSaved full results to: {OUTPUT_FILE}")
