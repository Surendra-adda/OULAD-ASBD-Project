
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

students = pd.read_csv(DATA / "student_outcomes.csv")
activity = pd.read_csv(DATA / "student_activity.csv")

# Overall outcome distribution for each course presentation
course_counts = (
    students.groupby(
        ["code_module", "code_presentation", "final_result"]
    )
    .size()
    .unstack(fill_value=0)
)

course_rates = course_counts.div(
    course_counts.sum(axis=1), axis=0
) * 100

print("=== OUTCOME PERCENTAGES BY COURSE PRESENTATION ===")
print(course_rates.round(1).to_string())

course_rates.to_csv(OUTPUT / "course_outcome_percentages.csv")

# Compare the distribution of student clicks by outcome and module
student = (
    activity.groupby(
        [
            "code_module",
            "code_presentation",
            "id_student",
            "final_result",
        ],
        as_index=False,
    )
    .agg(total_clicks=("total_clicks", "sum"))
)

print("\n=== MEDIAN CLICKS BY OUTCOME AND MODULE ===")

median_clicks = (
    student.groupby(["code_module", "final_result"])["total_clicks"]
    .median()
    .unstack()
)

print(median_clicks.round(1).to_string())
median_clicks.to_csv(OUTPUT / "median_clicks_by_module_outcome.csv")

# Plot outcome rates for each module presentation
plot_data = course_rates.reset_index()

for outcome in course_rates.columns:
    plt.figure(figsize=(10, 5))
    plt.bar(
        plot_data["code_module"] + "-" + plot_data["code_presentation"],
        plot_data[outcome],
    )
    plt.title(f"{outcome}: Percentage by Course Presentation")
    plt.xlabel("Course presentation")
    plt.ylabel("Students (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT / f"course_outcome_{outcome.lower()}.png",
        dpi=150,
    )
    plt.close()

print("\nSaved course-level analysis and charts to:", OUTPUT)
