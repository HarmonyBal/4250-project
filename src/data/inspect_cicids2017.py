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

# 定义报告输出目录
# Define the report output directory
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_machine_learning_directory() -> Path:
    """
    Automatically locate the CIC-IDS2017 MachineLearningCSV directory.

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
            and "machinelearningcsv"
            in candidate_path.name.lower()
        )
    ]

    # 检查是否找到了候选目录
    # Verify that candidate directories were found
    if not candidates:
        raise FileNotFoundError(
            "No MachineLearningCSV directory was found.\n"
            f"Current search location: {DATASETS_DIR}"
        )

    # 优先选择实际包含 CSV 文件的目录
    # Prefer a directory that actually contains CSV files
    for candidate in candidates:
        csv_files = list(
            candidate.rglob("*.csv")
        )

        if csv_files:
            return candidate

    raise FileNotFoundError(
        "MachineLearningCSV directories were found, "
        "but none of them contained CSV files."
    )


def inspect_csv(
    file_path: Path,
) -> dict:
    """
    Inspect the structure and data quality of one CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A dictionary containing dataset structure and quality statistics.
    """

    # 显示当前正在检查的文件
    # Display the file currently being inspected
    print(
        f"Inspecting: {file_path.name}"
    )

    # 尝试读取 CSV 文件
    # Attempt to read the CSV file
    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )
    except Exception as exc:
        # 当文件读取失败时返回错误信息
        # Return error information when the file cannot be read
        return {
            "file": file_path.name,
            "status": "error",
            "error": str(exc),
        }

    # 去掉 CIC-IDS2017 部分列名前后的空格
    # Remove surrounding whitespace from column names
    df.columns = (
        df.columns.str.strip()
    )

    # 提取数值类型列用于无穷值统计
    # Select numeric columns for infinity-value analysis
    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    # 初始化标签统计结果
    # Initialize label-count results
    labels = {}

    # 当标签列存在时统计标签分布
    # Calculate label distribution when the Label column exists
    if "Label" in df.columns:
        labels = (
            df["Label"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .to_dict()
        )

    # 构建当前文件的检查结果
    # Build the inspection result for the current file
    return {
        "file": file_path.name,
        "status": "ok",
        "rows": int(
            len(df)
        ),
        "columns": int(
            len(df.columns)
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "positive_infinity": int(
            np.isposinf(
                numeric_df
            ).sum().sum()
        ),
        "negative_infinity": int(
            np.isneginf(
                numeric_df
            ).sum().sum()
        ),
        "label_column_found": (
            "Label" in df.columns
        ),
        "label_counts": labels,
    }


def main() -> None:
    # 创建报告目录
    # Create the report directory
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 查找原始 MachineLearningCSV 数据目录
    # Locate the original MachineLearningCSV data directory
    data_dir = (
        find_machine_learning_directory()
    )

    # 递归查找数据目录中的所有 CSV 文件
    # Recursively find all CSV files in the dataset directory
    csv_files = sorted(
        data_dir.rglob("*.csv")
    )

    # 显示数据目录和找到的文件数量
    # Display the dataset directory and file count
    print(
        f"\nDataset directory: {data_dir}"
    )

    print(
        f"Number of CSV files found: "
        f"{len(csv_files)}\n"
    )

    # 检查所有 CSV 文件
    # Inspect all CSV files
    results = [
        inspect_csv(file_path)
        for file_path in csv_files
    ]

    # 创建用于保存文件级摘要的列表
    # Create a list for file-level summaries
    summary_rows = []

    # 创建用于累计所有标签数量的字典
    # Create a dictionary for aggregated label counts
    all_label_counts: dict[str, int] = {}

    # 整理每个文件的检查结果
    # Process the inspection result from each file
    for result in results:
        # 排除详细标签分布，仅保留文件级摘要
        # Exclude detailed label counts from the file-level summary
        summary_rows.append(
            {
                key: value
                for key, value in result.items()
                if key != "label_counts"
            }
        )

        # 将当前文件的标签数量累加到整体分布
        # Add the current file's label counts to the overall distribution
        for label, count in result.get(
            "label_counts",
            {},
        ).items():
            all_label_counts[label] = (
                all_label_counts.get(
                    label,
                    0,
                )
                + int(count)
            )

    # 创建文件级检查摘要 DataFrame
    # Create the file-level inspection summary DataFrame
    summary_df = pd.DataFrame(
        summary_rows
    )

    # 创建整体标签分布 DataFrame
    # Create the overall label-distribution DataFrame
    label_df = pd.DataFrame(
        sorted(
            all_label_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ),
        columns=[
            "label",
            "count",
        ],
    )

    # 定义各报告文件的输出路径
    # Define output paths for all reports
    summary_path = (
        REPORTS_DIR
        / "dataset_inventory.csv"
    )

    label_path = (
        REPORTS_DIR
        / "label_distribution.csv"
    )

    details_path = (
        REPORTS_DIR
        / "dataset_inspection_details.json"
    )

    # 保存文件级检查摘要
    # Save the file-level inspection summary
    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # 保存整体标签分布
    # Save the overall label distribution
    label_df.to_csv(
        label_path,
        index=False,
    )

    # 保存每个文件的详细检查结果
    # Save detailed inspection results for every file
    with details_path.open("w", encoding="utf-8",) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 显示文件检查结果
    # Display file inspection results
    print("\nFile inspection results:")

    print(summary_df.to_string(index=False))

    # 显示整体标签分布
    # Display the overall label distribution
    print("\nOverall label distribution:")

    print(label_df.to_string(index=False))

    # 显示生成的报告文件路径
    # Display the generated report paths
    print("\nGenerated reports:")

    print(f"- {summary_path}")
    print(f"- {label_path}")
    print(f"- {details_path}")


if __name__ == "__main__":
    main()