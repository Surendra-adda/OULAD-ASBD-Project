
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = [
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
]

print("=" * 65)
print("OULAD DATASET INSPECTION")
print("=" * 65)

for filename in FILES:
    filepath = DATA_DIR / filename

    if not filepath.exists():
        print(f"\nMISSING FILE: {filename}")
        continue

    # Read only a small sample, not the entire file
    sample = pd.read_csv(filepath, nrows=5)

    print(f"\nFile: {filename}")
    print(f"Size: {filepath.stat().st_size / (1024 ** 2):.2f} MB")
    print("Columns:", list(sample.columns))
    print("\nFirst 3 rows:")
    print(sample.head(3).to_string(index=False))

print("\nInspection complete.")
