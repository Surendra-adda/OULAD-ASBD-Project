# Learning Engagement Pattern Mining and Course Improvement Intelligence Using OULAD

An educational data-mining project that explores how students interact
with online learning resources and how those engagement patterns relate
to final course outcomes. The project combines data preparation,
exploratory analysis, association-rule mining, model evaluation, and an
interactive Streamlit dashboard.



## Project objectives

-   Prepare student outcome and virtual learning environment (VLE)
    activity data for analysis.
-   Explore student outcomes, course presentations, and engagement
    patterns.
-   Use Apriori and FP-Growth to discover frequent activity combinations
    and association rules.
-   Compare algorithms using recorded runtime measurements.
-   Evaluate baseline and machine-learning classifiers for a binary
    outcome task.
-   Present the methods, results, and limitations through an interactive
    dashboard.

## Dataset

This project uses the **Open University Learning Analytics Dataset
(OULAD)**, an anonymised dataset containing information about courses,
students, assessments, and interactions with the Open University's
virtual learning environment. The dataset covers seven modules and
multiple course presentations. See the official dataset pages:

-   [Open University dataset
    information](https://analyse.kmi.open.ac.uk/open_dataset)
-   [UCI Machine Learning Repository:
    OULAD](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)

Please obtain the original data from the official source and follow its
attribution and licence requirements. The large raw `studentVle.csv`
file is intentionally excluded from this Git repository because it is
over GitHub's normal per-file size limit. Keep it locally in `data/raw/`
if you need to rerun the preprocessing pipeline.

### Expected local data

The dashboard uses processed CSVs produced by the project. Depending on
the project version, these include:

-   `data/processed/student_outcomes.csv`
-   `data/processed/student_activity.csv`
-   `data/processed/daily_activity.csv`
-   `data/processed/outcome_activity_patterns.csv`

The dashboard may also read saved association-mining, benchmark, and
classification outputs when those files are present. Keep the output
filenames and paths expected by the code unchanged unless you update the
corresponding code too.

## Main workflow

1.  **Data preparation** --- read the raw OULAD CSV files, connect VLE
    records to activity metadata, and aggregate student activity into
    analysis-ready tables.
2.  **Exploratory data analysis (EDA)** --- inspect outcome counts,
    engagement levels, activity types, and course-presentation
    differences.
3.  **Association mining** --- represent each student-course record as a
    transaction of activity types and outcome, then mine frequent
    itemsets and rules using Apriori and FP-Growth.
4.  **Algorithm benchmark** --- compare the recorded runtime of the
    mining algorithms under the same selected data and thresholds.
5.  **Classification** --- compare a majority-class baseline with
    Decision Tree and Logistic Regression models for a binary outcome.
6.  **Dashboard** --- explore results, methods, formulas, charts, and
    available datasets in a Streamlit interface.

## Association mining: methods and recorded findings

Association rules describe items that occur together in the same
transaction. In this project, activity types can appear in the rule
antecedent, while an outcome such as `Pass` or `Withdrawn` can appear in
the consequent.

The main measures are:

-   **Support**: the fraction of all transactions containing the
    itemset.
-   **Confidence**: the fraction of transactions containing the
    antecedent that also contain the consequent.
-   **Lift**: confidence divided by the overall prevalence of the
    consequent. A lift above 1 indicates a positive association relative
    to that baseline; it does not establish causation.

Recorded project results include:

-   29,228 student-course transactions in the activity population.
-   41,319 frequent itemsets found by each of Apriori and FP-Growth,
    with matching itemsets/supports in the recorded comparison.
-   A recorded three-run mean runtime of approximately 6.219 seconds for
    Apriori and 0.430 seconds for FP-Growth, or about **14.46×** speedup
    for FP-Growth in that benchmark.
-   14,383 generated outcome rules under the recorded settings: 14,367
    with `Pass` as the consequent and 16 with `Withdrawn` as the
    consequent.
-   A shortlisted example rule: `folder, quiz, url → Pass` with support
    approximately 0.049, confidence approximately 0.749, and lift
    approximately 1.770.
-   A shortlisted example rule: `externalquiz → Withdrawn` with support
    approximately 0.043, confidence approximately 0.259, and lift
    approximately 1.056.

These figures depend on the transaction construction, selected activity
population, minimum support/confidence thresholds, implementation, and
runtime environment. Rules may overlap or be redundant. A rule not
appearing in a shortlist does not prove that no relationship exists.

The project also includes activity-type outcome-pattern analysis. Those
single-activity lift/prevalence summaries are a different analysis from
multi-item association rules; do not treat them as the same output.

## Classification: task and recorded results

The recorded binary classification target groups:

-   **Positive class (1):** `Pass` or `Distinction`
-   **Negative class (0):** `Fail` or `Withdrawn`

The recorded evaluation uses a student-grouped train/test split to avoid
the same student appearing in both partitions. The activity features are
aggregated over the course, so this is a **retrospective classification
experiment**, not a demonstration that the model can predict risk early
enough for intervention.

Recorded test results:

  Model                   Accuracy   Precision    Recall   F1-score
  --------------------- ---------- ----------- --------- ----------
  Majority baseline         52.20%      52.20%   100.00%     68.60%
  Decision Tree             82.26%      79.32%    89.31%     84.02%
  Logistic Regression       78.23%      83.73%    72.37%     77.64%

These values describe the recorded split and current feature
construction. Re-running the scripts may produce different results if
the data, random seed, dependencies, or parameters differ. Feature
importance and association measures are not causal explanations.

## Repository layout

The following is a guide to the intended structure; actual filenames may
vary by project version.

``` text
OULAD_ASBD_Project/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/                 # Original OULAD data; do not commit large files
│   └── processed/           # Analysis-ready CSV files
├── outputs/                 # Saved rules, metrics, benchmarks, and plots
├── src/                     # Data preparation and analysis scripts
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Use Python 3.10 or a compatible version supported by the installed
packages.

1.  Clone the repository:

    ``` bash
    git clone YOUR_GITHUB_REPOSITORY_URL
    cd OULAD_ASBD_Project
    ```

2.  Create and activate a virtual environment.

    **Windows PowerShell:**

    ``` powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    **macOS/Linux:**

    ``` bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Ensure the processed CSV files are in the paths expected by
    `dashboard/app.py`.

5.  Start the dashboard from the repository root:

    ``` bash
    streamlit run dashboard/app.py
    ```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Reproducing the analysis

The dashboard can use the processed CSVs already included in the
repository and does not normally need the large raw VLE file just to
display the existing results.

To rebuild the data or rerun the analysis from scratch:

1.  Download OULAD from the official source.
2.  Place its raw CSV files under the local `data/raw/` directory in the
    locations expected by the preparation script.
3.  Install all dependencies required by the analysis scripts.
4.  Run the scripts in the intended order: data preparation first, then
    exploratory/association analysis and classification.
5.  Check the paths and parameters in each script before execution;
    filenames and outputs should match the version committed to the
    repository.

Do not assume the complete pipeline can be rerun using only the three
dashboard packages. Ensure `requirements.txt` includes the dependencies
imported by all scripts you intend to support.

## Limitations and responsible interpretation

-   The dataset is anonymised historical learning data from a particular
    institution and set of course presentations; results may not
    generalise to other settings.
-   Association rules show co-occurrence patterns, not cause-and-effect
    relationships.
-   Rules are sensitive to transaction definitions and minimum
    support/confidence thresholds.
-   The recorded classification uses full-course aggregated activity and
    final outcomes; it is not an early prediction experiment.
-   Total clicks and resource-specific click features may overlap in
    meaning.
-   Model metrics depend on the selected split, features, and
    parameters.
-   A model's outputs should not be used to label or penalise individual
    learners without validation, fairness assessment, context, and
    appropriate human oversight.

## Technologies

-   Python
-   pandas and NumPy (where used by the project scripts)
-   Streamlit
-   Plotly
-   scikit-learn (classification analysis)
-   Association-rule mining implementations used in the project

See `requirements.txt` for the actual package list.

## Acknowledgement and citation

The dataset was created by Jakub Kuzilek, Martin Hlosta, and Zdenek
Zdrahal.

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning
Analytics dataset*. Scientific Data, 4, 170171.
https://doi.org/10.1038/sdata.2017.171

Also cite the dataset record used for access:

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). *Open University Learning
Analytics Dataset*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5KK69

## Author

Add your name, institution, course/project details, and contact or
portfolio links here before publishing.
