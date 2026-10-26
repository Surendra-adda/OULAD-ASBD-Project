
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

runs = pd.DataFrame([
    {"run": 1, "algorithm": "Apriori", "seconds": 5.656012},
    {"run": 1, "algorithm": "FP-Growth", "seconds": 0.379975},
    {"run": 2, "algorithm": "Apriori", "seconds": 6.972838},
    {"run": 2, "algorithm": "FP-Growth", "seconds": 0.489156},
    {"run": 3, "algorithm": "Apriori", "seconds": 6.028840},
    {"run": 3, "algorithm": "FP-Growth", "seconds": 0.421566},
])

runs.to_csv(OUTPUT / "benchmark_three_runs.csv", index=False)

summary = (
    runs.groupby("algorithm")["seconds"]
    .agg(["mean", "median", "std", "min", "max"])
    .reset_index()
)

summary.to_csv(OUTPUT / "benchmark_summary.csv", index=False)

means = summary.set_index("algorithm")["mean"]
speedup = means["Apriori"] / means["FP-Growth"]

print("Three-run benchmark summary:")
print(summary.round(4).to_string(index=False))
print(f"\nSpeedup based on mean runtimes: {speedup:.2f}x")
print("Saved benchmark_three_runs.csv and benchmark_summary.csv")
