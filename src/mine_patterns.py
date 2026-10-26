
from pathlib import Path
import json
import time

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    apriori,
    fpgrowth,
    association_rules,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
TRANSACTION_FILE = OUTPUT / "activity_transactions.jsonl"

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.25

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading activity transactions...")

transactions = []
with open(TRANSACTION_FILE, "r", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        transactions.append(row["items"])

print("Number of transactions:", len(transactions))
print("Minimum support:", MIN_SUPPORT)
print("Minimum confidence:", MIN_CONFIDENCE)

# --------------------------------------------------
# 1. Convert transactions to a one-hot Boolean matrix
# --------------------------------------------------
encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=encoder.columns_)

print("Transaction matrix shape:", basket.shape)

# --------------------------------------------------
# 2. Apriori: measure execution time
# --------------------------------------------------
print("\nRunning Apriori...")
apriori_start = time.perf_counter()

apriori_sets = apriori(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    low_memory=True,
)

apriori_seconds = time.perf_counter() - apriori_start

apriori_sets["itemset_length"] = (
    apriori_sets["itemsets"].apply(len)
)
apriori_sets["itemsets_text"] = (
    apriori_sets["itemsets"].apply(
        lambda items: ", ".join(sorted(items))
    )
)

apriori_sets.sort_values(
    ["support", "itemset_length"],
    ascending=[False, True],
).to_csv(OUTPUT / "apriori_itemsets.csv", index=False)

print("Apriori frequent itemsets:", len(apriori_sets))
print(f"Apriori runtime: {apriori_seconds:.3f} seconds")

# --------------------------------------------------
# 3. FP-Growth: measure execution time
# --------------------------------------------------
print("\nRunning FP-Growth...")
fp_start = time.perf_counter()

fp_sets = fpgrowth(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True,
)

fp_seconds = time.perf_counter() - fp_start

fp_sets["itemset_length"] = (
    fp_sets["itemsets"].apply(len)
)
fp_sets["itemsets_text"] = fp_sets["itemsets"].apply(
    lambda items: ", ".join(sorted(items))
)

fp_sets.sort_values(
    ["support", "itemset_length"],
    ascending=[False, True],
).to_csv(OUTPUT / "fpgrowth_itemsets.csv", index=False)

print("FP-Growth frequent itemsets:", len(fp_sets))
print(f"FP-Growth runtime: {fp_seconds:.3f} seconds")

# --------------------------------------------------
# 4. Verify itemsets AND support values
# --------------------------------------------------
def get_itemset_supports(frame):
    return {
        frozenset(items): round(float(support), 10)
        for items, support in zip(
            frame["itemsets"],
            frame["support"],
        )
    }


apriori_results = get_itemset_supports(apriori_sets)
fp_results = get_itemset_supports(fp_sets)

itemsets_match = apriori_results == fp_results

print(
    "Frequent itemsets and support values match:",
    itemsets_match,
)

if not itemsets_match:
    apriori_only = set(apriori_results) - set(fp_results)
    fp_only = set(fp_results) - set(apriori_results)

    print("Itemsets found only by Apriori:", len(apriori_only))
    print("Itemsets found only by FP-Growth:", len(fp_only))

# --------------------------------------------------
# 5. Save performance comparison
# --------------------------------------------------
performance = pd.DataFrame([
    {
        "algorithm": "Apriori",
        "transactions": len(transactions),
        "features": basket.shape[1],
        "min_support": MIN_SUPPORT,
        "frequent_itemsets": len(apriori_sets),
        "runtime_seconds": round(apriori_seconds, 6),
    },
    {
        "algorithm": "FP-Growth",
        "transactions": len(transactions),
        "features": basket.shape[1],
        "min_support": MIN_SUPPORT,
        "frequent_itemsets": len(fp_sets),
        "runtime_seconds": round(fp_seconds, 6),
    },
])

performance.to_csv(
    OUTPUT / "algorithm_performance.csv",
    index=False,
)

print("\n=== ALGORITHM PERFORMANCE ===")
print(performance.to_string(index=False))

if fp_seconds > 0:
    ratio = apriori_seconds / fp_seconds
    if ratio > 1:
        print(f"FP-Growth was {ratio:.2f}x as fast in this run.")
    elif ratio < 1:
        print(f"Apriori was {1 / ratio:.2f}x as fast in this run.")
    else:
        print("Both algorithms had approximately equal runtimes.")

# --------------------------------------------------
# 6. Generate association rules
# --------------------------------------------------
print("\nGenerating association rules...")

rules = pd.DataFrame()

if fp_sets.empty:
    print("No frequent itemsets found at this support threshold.")
else:
    rules = association_rules(
        fp_sets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )

    def is_outcome_consequent(items):
        return (
            len(items) == 1
            and next(iter(items)).startswith("outcome=")
        )

    if not rules.empty:
        rules = rules[
            rules["consequents"].apply(is_outcome_consequent)
        ].copy()

        rules["antecedents_text"] = rules["antecedents"].apply(
            lambda items: ", ".join(sorted(items))
        )
        rules["consequent_text"] = rules["consequents"].apply(
            lambda items: ", ".join(sorted(items))
        )

        rules = rules[
            [
                "antecedents_text",
                "consequent_text",
                "support",
                "confidence",
                "lift",
            ]
        ].sort_values(
            ["lift", "confidence", "support"],
            ascending=[False, False, False],
        )

rules.to_csv(
    OUTPUT / "outcome_association_rules.csv",
    index=False,
)

print("Outcome rules found:", len(rules))

if not rules.empty:
    print("\nRules per outcome:")
    print(rules["consequent_text"].value_counts().to_string())

    print("\nTop 15 outcome association rules:")
    print(rules.head(15).round(3).to_string(index=False))
else:
    print("No rules met the selected thresholds.")

print("\nPattern mining complete. Results saved in:", OUTPUT)
print("Performance results:", OUTPUT / "algorithm_performance.csv")
