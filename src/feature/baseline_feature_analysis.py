from pathlib import Path
from sys import path
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# Define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
path.append(str(PROJECT_ROOT / "src" / "data"))

from data_loader import CICIDS2017DataLoader


REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results" / "feature_analysis"


def prepare_numeric_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate numeric features from class labels.

    Non-numeric columns such as Label and Source File are
    automatically excluded from the feature dataset.
    """

    # Verify that the label column exists
    if "Label" not in df.columns:
        raise ValueError(
            "The Label column was not found in the dataset."
        )

    # Keep only numeric columns as candidate features
    feature_df = df.select_dtypes(
        include=[np.number]
    ).copy()

    labels = df["Label"].astype(str).copy()

    # Detect features containing only one unique value
    constant_columns = [
        column
        for column in feature_df.columns
        if feature_df[column].nunique(dropna=False) <= 1
    ]

    # Remove constant features because they provide no useful information
    if constant_columns:
        print(
            "\nConstant features detected and excluded "
            "from the analysis:"
        )

        for column in constant_columns:
            print(f"- {column}")

        feature_df.drop(
            columns=constant_columns,
            inplace=True,
        )

    return feature_df, labels


def calculate_basic_metrics(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate basic quality metrics for each numeric feature.
    """

    rows = []
    number_of_rows = len(feature_df)

    for feature in feature_df.columns:
        series = feature_df[feature]

        unique_values = int(
            series.nunique(dropna=True)
        )

        unique_ratio = (
            unique_values / number_of_rows
            if number_of_rows > 0
            else 0.0
        )

        zero_ratio = float(
            series.eq(0).sum() / number_of_rows
        )

        variance = float(
            series.var()
        )

        rows.append(
            {
                "feature": feature,
                "unique_values": unique_values,
                "unique_ratio": unique_ratio,
                "zero_ratio": zero_ratio,
                "variance": variance,
            }
        )

    return pd.DataFrame(rows)


def calculate_mutual_information(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate the mutual information between each feature
    and the attack label.

    A higher score usually means that the feature contains
    more useful information for distinguishing traffic classes.
    """

    # Convert text labels into numeric class identifiers
    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(
        labels
    )

    # Estimate the information shared by each feature and the labels
    mutual_information = mutual_info_classif(
        feature_df,
        encoded_labels,
        discrete_features=False,
        random_state=random_state,
    )

    return pd.DataFrame(
        {
            "feature": feature_df.columns,
            "mutual_information": mutual_information,
        }
    )


def calculate_redundancy_scores(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate the redundancy of each feature.

    max_absolute_correlation:
        The highest absolute correlation between the feature
        and any other feature.

    mean_absolute_correlation:
        The average absolute correlation between the feature
        and all other features.
    """

    # Calculate the absolute feature correlation matrix
    correlation_matrix = feature_df.corr().abs()

    rows = []

    for feature in correlation_matrix.columns:
        correlations = (
            correlation_matrix[feature]
            .drop(index=feature)
            .dropna()
        )

        if correlations.empty:
            maximum_correlation = 0.0
            mean_correlation = 0.0
        else:
            maximum_correlation = float(
                correlations.max()
            )

            mean_correlation = float(
                correlations.mean()
            )

        rows.append(
            {
                "feature": feature,
                "max_absolute_correlation": (
                    maximum_correlation
                ),
                "mean_absolute_correlation": (
                    mean_correlation
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_source_stability(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Estimate feature stability across different source files.

    source_mean_cv is the coefficient of variation of the
    feature means across traffic files.

    A higher value indicates that the feature distribution
    changes more significantly across different days or scenarios.
    """

    # Return unavailable values if source information is missing
    if "Source File" not in df.columns:
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "source_mean_cv": np.nan,
            }
        )

    # Calculate the mean of each feature in every source file
    source_means = (
        df.groupby("Source File")[feature_columns]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
    )

    rows = []

    for feature in feature_columns:
        means = source_means[feature].dropna()

        if means.empty:
            coefficient_of_variation = 0.0
        else:
            overall_mean = float(
                means.mean()
            )

            standard_deviation = float(
                means.std()
            )

            # Avoid division by zero for features with very small means
            denominator = abs(overall_mean) + 1e-12

            coefficient_of_variation = (
                standard_deviation / denominator
            )

        rows.append(
            {
                "feature": feature,
                "source_mean_cv": coefficient_of_variation,
            }
        )

    return pd.DataFrame(rows)


def normalize_series(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize a numeric series to the range from 0 to 1.
    """

    minimum = series.min()
    maximum = series.max()

    # Return zeros when valid normalization is not possible
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


def build_baseline_feature_score(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a baseline feature score.

    This is not the final adaptive feature-selection algorithm.
    It provides an initial ranking that can later be compared
    with the proposed adaptive scoring method.
    """

    result = metrics.copy()

    # Normalize feature-label relevance
    result["mi_normalized"] = normalize_series(
        result["mutual_information"]
    )

    # Apply logarithmic scaling before normalizing variance
    result["variance_normalized"] = normalize_series(
        np.log1p(
            result["variance"].clip(lower=0)
        )
    )

    # Convert redundancy into a penalty
    result["redundancy_penalty"] = normalize_series(
        result["max_absolute_correlation"]
    )

    # Features dominated by zero values receive a penalty
    result["zero_penalty"] = (
        result["zero_ratio"]
        .clip(lower=0, upper=1)
    )

    # Features that vary greatly between files receive a penalty
    stability_values = (
        result["source_mean_cv"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    result["instability_penalty"] = normalize_series(
        stability_values
    )

    # Calculate the initial weighted feature score
    result["baseline_score"] = (
        0.50 * result["mi_normalized"]
        + 0.15 * result["variance_normalized"]
        - 0.15 * result["redundancy_penalty"]
        - 0.10 * result["zero_penalty"]
        - 0.10 * result["instability_penalty"]
    )

    return (
        result.sort_values(
            by="baseline_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    # Create output directories if they do not exist
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load cleaned CIC-IDS2017 data
    loader = CICIDS2017DataLoader()

    print("Loading datasets for baseline feature analysis...")

    # Load up to 20,000 records from each cleaned file
    df = loader.load_all(
        sample_per_file=20_000,
    )

    print(f"\nDataset shape: {df.shape}")

    # Prepare candidate numeric features and labels
    feature_df, labels = prepare_numeric_features(
        df
    )

    print(
        "Number of numeric features used for analysis: "
        f"{len(feature_df.columns)}"
    )

    # Calculate basic feature metrics
    print("\nCalculating basic feature metrics...")

    basic_metrics = calculate_basic_metrics(
        feature_df
    )

    # Calculate feature relevance to attack labels
    print("Calculating mutual information...")

    mutual_information = calculate_mutual_information(
        feature_df=feature_df,
        labels=labels,
    )

    # Calculate feature redundancy
    print("Calculating feature redundancy...")

    redundancy_metrics = calculate_redundancy_scores(
        feature_df
    )

    # Calculate feature stability across source files
    print("Calculating cross-source feature stability...")

    stability_metrics = calculate_source_stability(
        df=df,
        feature_columns=feature_df.columns.tolist(),
    )

    # Merge all feature metrics into one table
    combined_metrics = (
        basic_metrics
        .merge(
            mutual_information,
            on="feature",
            how="left",
        )
        .merge(
            redundancy_metrics,
            on="feature",
            how="left",
        )
        .merge(
            stability_metrics,
            on="feature",
            how="left",
        )
    )

    # Generate the baseline feature ranking
    scored_features = build_baseline_feature_score(
        combined_metrics
    )

    metrics_path = (
        RESULTS_DIR / "baseline_feature_metrics.csv"
    )

    ranking_path = (
        RESULTS_DIR / "baseline_feature_ranking.csv"
    )

    summary_path = (
        REPORTS_DIR
        / "baseline_feature_analysis_summary.json"
    )

    # Save raw feature metrics
    combined_metrics.to_csv(
        metrics_path,
        index=False,
    )

    # Save ranked feature results
    scored_features.to_csv(
        ranking_path,
        index=False,
    )

    # Create a compact JSON summary
    summary = {
        "sample_rows": int(len(df)),
        "numeric_features_analyzed": int(
            len(feature_df.columns)
        ),
        "top_10_features": (
            scored_features
            .head(10)["feature"]
            .tolist()
        ),
        "bottom_10_features": (
            scored_features
            .tail(10)["feature"]
            .tolist()
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    display_columns = [
        "feature",
        "mutual_information",
        "max_absolute_correlation",
        "zero_ratio",
        "source_mean_cv",
        "baseline_score",
    ]

    # Display the highest-ranked features
    print("\nTop 15 baseline features:")

    print(
        scored_features[
            display_columns
        ].head(15).to_string(index=False)
    )

    # Display the lowest-ranked features
    print("\nBottom 10 baseline features:")

    print(
        scored_features[
            display_columns
        ].tail(10).to_string(index=False)
    )

    # Display generated output locations
    print("\nGenerated files:")
    print(f"- {metrics_path}")
    print(f"- {ranking_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()