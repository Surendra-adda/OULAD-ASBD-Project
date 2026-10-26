
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rules = pd.read_csv(ROOT / "outputs" / "outcome_association_rules.csv")

print("Total outcome rules:", len(rules))
print("\nRules per outcome:")
print(rules["consequent_text"].value_counts(dropna=False))

print("\nConfidence range per outcome:")
print(
    rules.groupby("consequent_text")["confidence"]
    .agg(["count", "min", "median", "max"])
    .round(3)
)

print("\nUnique antecedents per outcome:")
print(
    rules.groupby("consequent_text")["antecedents_text"]
    .nunique()
)
