from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# 定义项目根目录
# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 把项目根目录加入 Python 模块搜索路径
# Add the project root directory to the Python module search path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import CICIDS2017DataLoader


# 定义报告输出目录
# Define the report output directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# 定义 EDA 结果输出目录
# Define the EDA results output directory
RESULTS_DIR = PROJECT_ROOT / "results" / "eda"


def calculate_class_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the number and percentage of samples in each class.

    Args:
        df: Input dataset containing the Label column.

    Returns:
        A DataFrame containing class names, sample counts,
        and percentages.

    Raises:
        ValueError: If the Label column is missing or the
            dataset is empty.
    """

    # 检查数据集是否为空
    # Verify that the dataset is not empty
    if df.empty:
        raise ValueError(
            "The input dataset is empty."
        )

    # 检查标签列是否存在
    # Verify that the Label column exists
    if "Label" not in df.columns:
        raise ValueError(
            "The Label column was not found in the dataset."
        )

    # 统计每个类别的样本数量
    # Count the number of samples in each class
    counts = df["Label"].value_counts()

    # 构建类别分布结果表
    # Build the class-distribution result table
    result = pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percentage": (
                counts.values
                / len(df)
                * 100
            ),
        }
    )

    return result


def calculate_feature_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numeric features.

    Args:
        df: Input dataset.

    Returns:
        A DataFrame containing descriptive statistics,
        missing-value counts, unique-value counts,
        and zero-value percentages.

    Raises:
        ValueError: If the dataset contains no numeric features.
    """

    # 选择所有数值特征
    # Select all numeric features
    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    # 检查是否存在数值特征
    # Verify that numeric features exist
    if numeric_df.empty:
        raise ValueError(
            "No numeric features were found in the dataset."
        )

    # 计算基础描述性统计
    # Calculate basic descriptive statistics
    statistics = (
        numeric_df.describe().T
    )

    # 统计每个特征的缺失值数量
    # Count missing values for each feature
    statistics["missing_values"] = (
        numeric_df.isna().sum()
    )

    # 统计每个特征的唯一值数量
    # Count unique values for each feature
    statistics["unique_values"] = (
        numeric_df.nunique()
    )

    # 计算每个特征中零值所占的百分比
    # Calculate the percentage of zero values in each feature
    statistics["zero_percentage"] = (
        numeric_df.eq(0).sum()
        / len(numeric_df)
        * 100
    )

    # 将特征名称从索引转换为普通列
    # Convert feature names from the index into a column
    statistics.reset_index(
        inplace=True
    )

    statistics.rename(
        columns={
            "index": "feature",
        },
        inplace=True,
    )

    return statistics


def find_high_correlations(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Find highly correlated pairs of numeric features.

    Args:
        df: Input dataset.
        threshold: Minimum absolute correlation required
            for a feature pair to be included.

    Returns:
        A DataFrame containing highly correlated feature pairs.

    Raises:
        ValueError: If the correlation threshold is outside
            the range from zero to one.
    """

    # 验证相关性阈值范围
    # Validate the correlation threshold
    if not 0 <= threshold <= 1:
        raise ValueError(
            "The correlation threshold must be between 0 and 1."
        )

    # 选择所有数值特征
    # Select all numeric features
    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    # 当数值特征少于两个时返回空结果
    # Return an empty result when fewer than two features exist
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "absolute_correlation",
            ]
        )

    # 计算数值特征之间的绝对相关系数矩阵
    # Calculate the absolute correlation matrix
    correlation_matrix = (
        numeric_df.corr().abs()
    )

    # 创建相关矩阵上三角区域的布尔掩码
    # Create a Boolean mask for the upper triangle
    upper_triangle_mask = np.triu(
        np.ones(
            correlation_matrix.shape,
            dtype=bool,
        ),
        k=1,
    )

    # 只保留相关矩阵的上三角部分
    # Keep only the upper triangle of the matrix
    upper_triangle = correlation_matrix.where(
        upper_triangle_mask
    )

    # 创建高度相关特征对列表
    # Create a list for highly correlated feature pairs
    pairs = []

    # 查找绝对相关系数达到阈值的特征对
    # Find feature pairs meeting the correlation threshold
    for feature_2 in upper_triangle.columns:
        correlated_features = upper_triangle.index[
            upper_triangle[feature_2] >= threshold
        ]

        for feature_1 in correlated_features:
            pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "absolute_correlation": float(
                        upper_triangle.loc[
                            feature_1,
                            feature_2,
                        ]
                    ),
                }
            )

    # 如果没有高度相关的特征对，则返回空表
    # Return an empty table when no correlated pairs are found
    if not pairs:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "absolute_correlation",
            ]
        )

    # 按绝对相关系数从高到低排序
    # Sort feature pairs by absolute correlation
    result = pd.DataFrame(
        pairs
    ).sort_values(
        by="absolute_correlation",
        ascending=False,
    )

    return result.reset_index(
        drop=True
    )


def calculate_source_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate label distributions for each source file.

    Args:
        df: Input dataset containing Source File and Label columns.

    Returns:
        A DataFrame containing source-file and label counts.

    Raises:
        ValueError: If the Source File column exists but the
            Label column is missing.
    """

    # 如果没有来源文件列，则返回空结果
    # Return an empty result if the Source File column is missing
    if "Source File" not in df.columns:
        return pd.DataFrame(
            columns=[
                "Source File",
                "Label",
                "count",
            ]
        )

    # 检查标签列是否存在
    # Verify that the Label column exists
    if "Label" not in df.columns:
        raise ValueError(
            "The Label column was not found in the dataset."
        )

    # 按来源文件和标签统计样本数量
    # Count samples by source file and label
    result = (
        df.groupby(
            [
                "Source File",
                "Label",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    return result


def build_eda_summary(
    df: pd.DataFrame,
    class_distribution: pd.DataFrame,
    high_correlations: pd.DataFrame,
) -> dict:
    """
    Build a compact summary of the exploratory data analysis.

    Args:
        df: Input dataset.
        class_distribution: Calculated class-distribution table.
        high_correlations: Detected high-correlation feature pairs.

    Returns:
        A dictionary containing the main EDA results.

    Raises:
        ValueError: If the class-distribution table is empty.
    """

    # 检查类别分布结果是否为空
    # Verify that the class-distribution result is not empty
    if class_distribution.empty:
        raise ValueError(
            "The class distribution is empty."
        )

    # 获取所有数值特征名称
    # Obtain all numeric feature names
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    # 获取样本数量最多的类别
    # Identify the largest class
    largest_class_row = (
        class_distribution.iloc[0]
    )

    # 获取样本数量最少的类别
    # Identify the smallest class
    smallest_class_row = (
        class_distribution.iloc[-1]
    )

    # 构建 EDA 摘要
    # Build the EDA summary
    summary = {
        "sample_rows": int(
            len(df)
        ),
        "number_of_columns": int(
            len(df.columns)
        ),
        "number_of_numeric_features": int(
            len(numeric_columns)
        ),
        "number_of_labels": int(
            df["Label"].nunique()
        ),
        "number_of_high_correlation_pairs": int(
            len(high_correlations)
        ),
        "largest_class": str(
            largest_class_row["label"]
        ),
        "largest_class_count": int(
            largest_class_row["count"]
        ),
        "largest_class_percentage": float(
            largest_class_row["percentage"]
        ),
        "smallest_class": str(
            smallest_class_row["label"]
        ),
        "smallest_class_count": int(
            smallest_class_row["count"]
        ),
        "smallest_class_percentage": float(
            smallest_class_row["percentage"]
        ),
    }

    return summary


def main() -> None:
    # 创建报告输出目录
    # Create the report output directory
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 创建 EDA 结果输出目录
    # Create the EDA results output directory
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 创建 CIC-IDS2017 数据加载器
    # Create the CIC-IDS2017 data loader
    loader = CICIDS2017DataLoader()

    # 显示数据加载状态
    # Display the dataset loading status
    print(
        "Loading cleaned datasets "
        "for exploratory analysis..."
    )

    # 每个文件最多随机抽取 20,000 条记录
    # Randomly sample up to 20,000 records from each file
    df = loader.load_all(
        sample_per_file=20_000,
    )

    # 检查加载后的数据是否为空
    # Verify that the loaded dataset is not empty
    if df.empty:
        raise ValueError(
            "The loaded dataset is empty."
        )

    # 检查标签列是否存在
    # Verify that the Label column exists
    if "Label" not in df.columns:
        raise ValueError(
            "The Label column was not found in the loaded dataset."
        )

    # 显示 EDA 使用的数据规模
    # Display the shape of the dataset used for EDA
    print(
        f"\nEDA dataset shape: {df.shape}"
    )

    # 计算类别分布
    # Calculate the class distribution
    print(
        "\nCalculating class distribution..."
    )

    class_distribution = (
        calculate_class_distribution(df)
    )

    # 计算数值特征的描述性统计
    # Calculate descriptive statistics for numeric features
    print(
        "Calculating feature statistics..."
    )

    feature_statistics = (
        calculate_feature_statistics(df)
    )

    # 查找高度相关的特征对
    # Find highly correlated feature pairs
    print(
        "Calculating feature correlations..."
    )

    high_correlations = find_high_correlations(
        df,
        threshold=0.95,
    )

    # 计算不同来源文件中的标签分布
    # Calculate label distributions across source files
    print(
        "Calculating source-label distribution..."
    )

    source_distribution = (
        calculate_source_distribution(df)
    )

    # 定义类别分布结果文件
    # Define the class-distribution output file
    class_distribution_path = (
        RESULTS_DIR
        / "class_distribution.csv"
    )

    # 定义特征统计结果文件
    # Define the feature-statistics output file
    feature_statistics_path = (
        RESULTS_DIR
        / "feature_statistics.csv"
    )

    # 定义高度相关特征对结果文件
    # Define the high-correlation output file
    high_correlations_path = (
        RESULTS_DIR
        / "high_correlation_pairs.csv"
    )

    # 定义来源文件标签分布结果文件
    # Define the source-label distribution output file
    source_distribution_path = (
        RESULTS_DIR
        / "source_label_distribution.csv"
    )

    # 保存类别分布结果
    # Save the class-distribution results
    class_distribution.to_csv(
        class_distribution_path,
        index=False,
    )

    # 保存数值特征统计结果
    # Save numeric feature statistics
    feature_statistics.to_csv(
        feature_statistics_path,
        index=False,
    )

    # 保存高度相关特征对
    # Save highly correlated feature pairs
    high_correlations.to_csv(
        high_correlations_path,
        index=False,
    )

    # 保存来源文件标签分布
    # Save the source-file label distribution
    source_distribution.to_csv(
        source_distribution_path,
        index=False,
    )

    # 构建 EDA 摘要
    # Build the EDA summary
    summary = build_eda_summary(
        df=df,
        class_distribution=class_distribution,
        high_correlations=high_correlations,
    )

    # 定义 EDA 摘要文件路径
    # Define the EDA summary output path
    summary_path = (
        REPORTS_DIR
        / "eda_summary.json"
    )

    # 将 EDA 摘要保存为 JSON 文件
    # Save the EDA summary as a JSON file
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

    # 显示类别分布
    # Display the class distribution
    print(
        "\nClass distribution:"
    )

    print(
        class_distribution.to_string(
            index=False
        )
    )

    # 显示高度相关特征对的数量
    # Display the number of highly correlated pairs
    print(
        "\nNumber of highly correlated "
        f"feature pairs: {len(high_correlations)}"
    )

    # 显示相关性最高的前十组特征
    # Display the top ten correlated feature pairs
    if not high_correlations.empty:
        print(
            "\nTop 10 correlated feature pairs:"
        )

        print(
            high_correlations
            .head(10)
            .to_string(
                index=False
            )
        )
    else:
        print(
            "\nNo feature pairs exceeded "
            "the correlation threshold."
        )

    # 显示生成的输出文件
    # Display the generated output files
    print(
        "\nGenerated files:"
    )

    print(
        f"- {summary_path}"
    )

    print(
        f"- {class_distribution_path}"
    )

    print(
        f"- {feature_statistics_path}"
    )

    print(
        f"- {high_correlations_path}"
    )

    print(
        f"- {source_distribution_path}"
    )


if __name__ == "__main__":
    main()