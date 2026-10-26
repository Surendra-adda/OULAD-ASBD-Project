
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

rules_file = OUTPUT / "outcome_association_rules.csv"
rules = pd.read_csv(rules_file)

if rules.empty:
    print("No association rules available.")
    raise SystemExit(0)

# Keep shorter rules, which are easier to explain.
rules["antecedent_length"] = rules["antecedents_text"].apply(
    lambda text: len(text.split(", "))
)

rules = rules[rules["antecedent_length"] <= 2].copy()

# Support count makes the scale of a rule easier to understand.
transaction_count = 29228
rules["support_count_approx"] = (
    rules["support"] * transaction_count
).round().astype(int)

# Avoid presenting many nearly identical rules.
rules = rules.sort_values(
    ["consequent_text", "lift", "confidence", "support"],
    ascending=[True, False, False, False],
)

# Show the best concise rules for each outcome.
selected = (
    rules.groupby("consequent_text", group_keys=False)
    .head(10)
    .reset_index(drop=True)
)

selected.to_csv(OUTPUT / "concise_outcome_rules.csv", index=False)

for outcome, group in selected.groupby("consequent_text"):
    print(f"\n=== {outcome} ===")
    print(
        group[
            [
                "antecedents_text",
                "support_count_approx",
                "confidence",
                "lift",
            ]
        ].round(3).to_string(index=False)
    )

print("\nSaved concise rules to outputs/concise_outcome_rules.csv")
