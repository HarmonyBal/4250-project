from pathlib import Path
import json

import numpy as np
import pandas as pd


# 定义项目根目录
# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 定义数据集目录
# Define the dataset directory
DATASETS_DIR = PROJECT_ROOT / "datasets"

# 定义清洗后数据的输出目录
# Define the output directory for cleaned datasets
PROCESSED_DIR = DATASETS_DIR / "processed"

# 定义报告输出目录
# Define the report output directory
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_machine_learning_directory() -> Path:
    """
    Locate the original CIC-IDS2017 MachineLearningCSV directory.

    Returns:
        The path to a MachineLearningCSV directory containing CSV files.

    Raises:
        FileNotFoundError: If no valid MachineLearningCSV directory
            can be found.
    """

    # 递归查找名称中包含 MachineLearningCSV 的目录
    # Recursively search for directories containing MachineLearningCSV
    candidates = [
        candidate_path
        for candidate_path in DATASETS_DIR.rglob("*")
        if (
            candidate_path.is_dir()
            and "machinelearningcsv" in candidate_path.name.lower()
            and "processed" not in candidate_path.parts
        )
    ]

    # 找到第一个实际包含 CSV 文件的候选目录
    # Return the first candidate that contains CSV files
    for candidate in candidates:
        csv_files = list(
            candidate.rglob("*.csv")
        )

        if csv_files:
            return candidate

    raise FileNotFoundError(
        "No MachineLearningCSV directory containing CSV files "
        "was found."
    )


def normalize_label(label: object) -> str:
    """
    Normalize CIC-IDS2017 class labels.

    Args:
        label: Original label value.

    Returns:
        A normalized label string.
    """

    # 将标签转换为字符串并删除前后空格
    # Convert the label to a string and remove surrounding whitespace
    normalized_label = str(label).strip()

    # 修复部分官方 CSV 中 Web Attack 标签的异常字符
    # Fix malformed Web Attack labels in some official CSV files
    if (
        "Web Attack" in normalized_label
        and "Brute Force" in normalized_label
    ):
        return "Web Attack - Brute Force"

    if (
        "Web Attack" in normalized_label
        and "XSS" in normalized_label
    ):
        return "Web Attack - XSS"

    if (
        "Web Attack" in normalized_label
        and "Sql Injection" in normalized_label
    ):
        return "Web Attack - Sql Injection"

    return normalized_label


def clean_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame loaded from one CIC-IDS2017 CSV file.

    Args:
        df: Original DataFrame.

    Returns:
        A tuple containing the cleaned DataFrame and a cleaning report.

    Raises:
        ValueError: If the Label column is missing.
    """

    # 记录清洗前的原始行数
    # Record the original number of rows
    original_rows = len(df)

    # 去掉列名前后的空格
    # Remove leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # 检查标签列是否存在
    # Verify that the Label column exists
    if "Label" not in df.columns:
        raise ValueError(
            "The Label column was not found in the dataset."
        )

    # 统一标签格式
    # Normalize class labels
    df["Label"] = df["Label"].apply(
        normalize_label
    )

    # 将正无穷和负无穷替换为缺失值
    # Replace positive and negative infinity with missing values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # 统计包含缺失值或无穷值的行数
    # Count rows containing missing or infinite values
    rows_with_missing_values = int(
        df.isna().any(axis=1).sum()
    )

    # 删除包含缺失值的记录
    # Remove rows containing missing values
    df.dropna(
        inplace=True
    )

    # 记录删除缺失值后的行数
    # Record the number of rows after missing-value removal
    rows_after_missing_removal = len(df)

    # 统计完全重复的记录数量
    # Count completely duplicated records
    duplicate_rows = int(
        df.duplicated().sum()
    )

    # 删除完全重复的流量记录
    # Remove completely duplicated traffic records
    df.drop_duplicates(
        inplace=True
    )

    # 重置清洗后数据的索引
    # Reset the index of the cleaned dataset
    df.reset_index(
        drop=True,
        inplace=True,
    )

    # 构建单个文件的清洗报告
    # Build the cleaning report for the current file
    report = {
        "original_rows": int(original_rows),
        "rows_with_missing_or_infinite_values": (
            rows_with_missing_values
        ),
        "rows_after_missing_removal": int(
            rows_after_missing_removal
        ),
        "duplicate_rows_removed": duplicate_rows,
        "final_rows": int(len(df)),
    }

    return df, report


def clean_file(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Read, clean, and save one CIC-IDS2017 CSV file.

    Args:
        input_path: Path to the original CSV file.
        output_path: Path where the cleaned CSV file will be saved.

    Returns:
        A dictionary containing the cleaning statistics.
    """

    # 显示当前正在处理的文件
    # Display the file currently being processed
    print(
        f"\nProcessing: {input_path.name}"
    )

    # 读取原始 CSV 文件
    # Read the original CSV file
    df = pd.read_csv(
        input_path,
        low_memory=False,
        encoding="utf-8",
    )

    # 清洗当前数据文件
    # Clean the current dataset
    cleaned_df, report = clean_dataframe(
        df
    )

    # 将清洗后的数据保存为新的 CSV 文件
    # Save the cleaned dataset as a new CSV file
    cleaned_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    # 在报告中记录输入和输出文件名
    # Record the input and output filenames in the report
    report["input_file"] = input_path.name
    report["output_file"] = output_path.name

    # 显示当前文件的清洗统计
    # Display cleaning statistics for the current file
    print(
        f"Original rows: "
        f"{report['original_rows']:,}"
    )

    print(
        "Rows removed due to missing or infinite values: "
        f"{report['rows_with_missing_or_infinite_values']:,}"
    )

    print(
        "Duplicate rows removed: "
        f"{report['duplicate_rows_removed']:,}"
    )

    print(
        f"Final rows: "
        f"{report['final_rows']:,}"
    )

    return report


def main() -> None:
    # 查找原始 MachineLearningCSV 数据目录
    # Locate the original MachineLearningCSV data directory
    source_dir = find_machine_learning_directory()

    # 创建清洗后数据目录
    # Create the processed dataset directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 创建报告目录
    # Create the report directory
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 递归查找原始目录中的所有 CSV 文件
    # Recursively find all CSV files in the source directory
    csv_files = sorted(
        source_dir.rglob("*.csv")
    )

    # 检查是否找到了 CSV 文件
    # Verify that CSV files were found
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in: {source_dir}"
        )

    # 显示输入目录、输出目录和文件数量
    # Display the input directory, output directory, and file count
    print(
        f"Source dataset directory: {source_dir}"
    )

    print(
        f"Processed dataset directory: {PROCESSED_DIR}"
    )

    print(
        f"Number of CSV files found: {len(csv_files)}"
    )

    # 保存所有文件的清洗报告
    # Store cleaning reports for all files
    cleaning_reports = []

    # 依次清洗每一个 CSV 文件
    # Clean each CSV file sequentially
    for input_path in csv_files:
        # 为清洗后的文件添加 cleaned_ 前缀
        # Add the cleaned_ prefix to the output filename
        output_name = (
            f"cleaned_{input_path.name}"
        )

        output_path = (
            PROCESSED_DIR / output_name
        )

        # 清洗并保存当前文件
        # Clean and save the current file
        report = clean_file(
            input_path=input_path,
            output_path=output_path,
        )

        cleaning_reports.append(
            report
        )

    # 定义清洗报告文件路径
    # Define the cleaning report output path
    report_path = (
        REPORTS_DIR / "cleaning_report.json"
    )

    # 将所有清洗报告保存为 JSON
    # Save all cleaning reports as JSON
    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cleaning_reports,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 计算所有文件清洗前的总行数
    # Calculate the total number of original rows
    total_original = sum(
        item["original_rows"]
        for item in cleaning_reports
    )

    # 计算所有文件清洗后的总行数
    # Calculate the total number of cleaned rows
    total_final = sum(
        item["final_rows"]
        for item in cleaning_reports
    )

    # 计算总共删除的记录数量
    # Calculate the total number of removed rows
    total_removed = (
        total_original - total_final
    )

    # 显示全部文件的最终清洗结果
    # Display the final cleaning summary
    print(
        "\nAll files have been processed successfully."
    )

    print(
        f"Total original rows: {total_original:,}"
    )

    print(
        f"Total cleaned rows: {total_final:,}"
    )

    print(
        f"Total rows removed: {total_removed:,}"
    )

    print(
        f"Cleaning report: {report_path}"
    )


if __name__ == "__main__":
    main()