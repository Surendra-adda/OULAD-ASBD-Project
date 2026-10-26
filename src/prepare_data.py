
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TEMP = PROCESSED / "temp_chunks"

PROCESSED.mkdir(parents=True, exist_ok=True)

# Start with clean temporary output from any previous run.
if TEMP.exists():
    shutil.rmtree(TEMP)
TEMP.mkdir(parents=True, exist_ok=True)

try:
    print("Loading learning-resource metadata...")

    vle = pd.read_csv(RAW / "vle.csv", na_values=["?"])
    vle = vle[
        [
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
        ]
    ].drop_duplicates()

    vle["activity_type"] = vle["activity_type"].fillna("unknown")

    print("Loading student outcomes...")

    outcomes = pd.read_csv(RAW / "studentInfo.csv", na_values=["?"])
    outcomes = outcomes[
        [
            "code_module",
            "code_presentation",
            "id_student",
            "final_result",
        ]
    ]

    if outcomes.duplicated(
        ["code_module", "code_presentation", "id_student"]
    ).any():
        raise ValueError("Duplicate student outcome keys found.")

    outcomes.to_csv(PROCESSED / "student_outcomes.csv", index=False)

    print("Processing studentVle.csv in chunks...")

    student_file = RAW / "studentVle.csv"
    chunk_size = 200_000
    chunk_number = 0

    for chunk in pd.read_csv(
        student_file,
        chunksize=chunk_size,
        na_values=["?"],
    ):
        chunk["sum_click"] = pd.to_numeric(
            chunk["sum_click"], errors="coerce"
        ).fillna(0)

        chunk = chunk.merge(
            vle,
            on=["id_site", "code_module", "code_presentation"],
            how="left",
            validate="many_to_one",
        )

        chunk["activity_type"] = chunk["activity_type"].fillna("unknown")

        student_keys = [
            "code_module",
            "code_presentation",
            "id_student",
            "activity_type",
        ]

        # Keep the date until all chunks are combined so that a date
        # appearing in multiple chunks is counted only once.
        student_date_keys = student_keys + ["date"]

        student_summary = (
            chunk.groupby(
                student_date_keys,
                as_index=False,
                dropna=False,
            )
            .agg(
                total_clicks=("sum_click", "sum"),
                interaction_records=("sum_click", "size"),
            )
        )

        daily_keys = [
            "code_module",
            "code_presentation",
            "date",
            "activity_type",
        ]

        daily_summary = (
            chunk.groupby(
                daily_keys,
                as_index=False,
                dropna=False,
            )
            .agg(total_clicks=("sum_click", "sum"))
        )

        student_summary.to_csv(
            TEMP / f"student_{chunk_number:03d}.csv",
            index=False,
        )

        daily_summary.to_csv(
            TEMP / f"daily_{chunk_number:03d}.csv",
            index=False,
        )

        chunk_number += 1

        if chunk_number % 5 == 0:
            print(f"Processed {chunk_number} chunks...")

    print(f"Processed {chunk_number} chunks in total.")
    print("Combining partial summaries...")

    # Combine student/date summaries and calculate distinct dates
    # only after all chunks have been read.
    student_parts = sorted(TEMP.glob("student_*.csv"))
    student_frames = [pd.read_csv(f) for f in student_parts]

    student_activity = pd.concat(student_frames, ignore_index=True)

    student_activity = (
        student_activity.groupby(
            [
                "code_module",
                "code_presentation",
                "id_student",
                "activity_type",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            total_clicks=("total_clicks", "sum"),
            interaction_records=("interaction_records", "sum"),
            active_days=("date", "nunique"),
        )
    )

    student_activity = student_activity.merge(
        outcomes,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        validate="many_to_one",
    )

    if student_activity["final_result"].isna().any():
        raise ValueError(
            "Some activity records did not match student outcomes."
        )

    student_activity.to_csv(
        PROCESSED / "student_activity.csv",
        index=False,
    )

    # Combine daily summaries.
    daily_parts = sorted(TEMP.glob("daily_*.csv"))
    daily_frames = [pd.read_csv(f) for f in daily_parts]

    daily_activity = pd.concat(daily_frames, ignore_index=True)

    daily_activity = (
        daily_activity.groupby(
            [
                "code_module",
                "code_presentation",
                "date",
                "activity_type",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(total_clicks=("total_clicks", "sum"))
    )

    daily_activity.to_csv(
        PROCESSED / "daily_activity.csv",
        index=False,
    )

    print("\nData preparation complete!")
    print("Student activity rows:", len(student_activity))
    print("Daily activity rows:", len(daily_activity))
    print("Outcome categories:")
    print(outcomes["final_result"].value_counts(dropna=False))
    print("\nFiles saved in:", PROCESSED)

finally:
    # Clean up intermediate files even if processing raises an error.
    if TEMP.exists():
        shutil.rmtree(TEMP)
