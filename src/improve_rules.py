
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

INPUT_FILE = OUTPUT / "outcome_association_rules.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing {INPUT_FILE}. Run src/mine_patterns.py first."
    )

rules = pd.read_csv(INPUT_FILE)

if rules.empty:
    print("No association rules found.")
    raise SystemExit(0)

# Keep positively associated rules with reasonable confidence.
rules = rules[
    (rules["lift"] > 1.0)
    & (rules["confidence"] >= 0.25)
].copy()

if rules.empty:
    print("No rules meet the selected criteria.")
    raise SystemExit(0)

rules["outcome"] = rules["consequent_text"].str.replace(
    "outcome=", "", regex=False
)

def parse_antecedent(text):
    return {
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    }

rules["antecedent_set"] = rules["antecedents_text"].apply(
    parse_antecedent
)
rules["antecedent_length"] = rules["antecedent_set"].apply(len)

# Prefer stronger rules, then shorter antecedents.
rules = rules.sort_values(
    ["outcome", "lift", "confidence", "support", "antecedent_length"],
    ascending=[True, False, False, False, True],
)

# Remove a longer rule when a shorter rule with the same outcome
# has equal or better confidence and lift.
selected = []

for outcome, group in rules.groupby("outcome", sort=True):
    kept = []

    for _, row in group.iterrows():
        antecedent = row["antecedent_set"]

        redundant = any(
            prior["antecedent_set"].issubset(antecedent)
            and prior["antecedent_set"] != antecedent
            and prior["confidence"] >= row["confidence"] - 1e-9
            and prior["lift"] >= row["lift"] - 1e-9
            for prior in kept
        )

        if not redundant:
            kept.append(row)
            selected.append(row)

reduced = pd.DataFrame(selected)

# Remove internal helper columns from saved output.
reduced = reduced.drop(
    columns=["antecedent_set", "antecedent_length"],
    errors="ignore",
)

# Keep up to 10 representative rules per outcome.
top_rules = (
    reduced.sort_values(
        ["outcome", "lift", "confidence", "support"],
        ascending=[True, False, False, False],
    )
    .groupby("outcome", group_keys=False)
    .head(10)
)

reduced.to_csv(
    OUTPUT / "reduced_association_rules.csv",
    index=False,
)
top_rules.to_csv(
    OUTPUT / "top_association_rules.csv",
    index=False,
)

print("Positive-association rules:", len(rules))
print("Rules after redundancy reduction:", len(reduced))
print("Shortlisted rules:", len(top_rules))

print("\nShortlisted rules by outcome:")
print(
    top_rules["outcome"].value_counts().to_string()
    if not top_rules.empty else "No rules available."
)

print("\nTop rules:")
if not top_rules.empty:
    print(
        top_rules[
            [
                "antecedents_text",
                "outcome",
                "support",
                "confidence",
                "lift",
            ]
        ].round(3).to_string(index=False)
    )

print("\nSaved:")
print(OUTPUT / "reduced_association_rules.csv")
print(OUTPUT / "top_association_rules.csv")
