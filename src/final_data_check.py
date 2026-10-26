
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

files = [
    ROOT / "data" / "processed" / "student_outcomes.csv",
    ROOT / "data" / "processed" / "student_activity.csv",
    ROOT / "data" / "processed" / "daily_activity.csv",
    ROOT / "outputs" / "activity_prevalence_by_outcome.csv",
    ROOT / "outputs" / "course_outcome_percentages.csv",
    ROOT / "outputs" / "benchmark_summary.csv",
]

print("FINAL DATA QUALITY CHECK")

for path in files:
    print(f"\n--- {path.name} ---")

    if not path.exists():
        print("MISSING FILE")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Missing cells:", int(df.isna().sum().sum()))

    if "final_result" in df.columns:
        print("Outcome counts:")
        print(df["final_result"].value_counts(dropna=False).to_string())

print("\nCheck complete.")
