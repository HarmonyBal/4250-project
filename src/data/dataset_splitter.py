from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import CICIDS2017DataLoader


# 定义项目根目录
# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 定义报告输出目录
# Define the report output directory
REPORTS_DIR = PROJECT_ROOT / "reports"


class CICIDS2017DatasetSplitter:
    """
    Split CIC-IDS2017 data into training, validation, and test sets.
    """

    def __init__(
        self,
        loader: CICIDS2017DataLoader,
    ) -> None:
        """
        Initialize the dataset splitter.

        Args:
            loader: A configured CIC-IDS2017 data loader.
        """

        # 保存数据加载器实例
        # Store the dataset loader instance
        self.loader = loader

    def random_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.70,
        validation_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        rare_class_threshold: int = 7,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Randomly split the dataset.

        Classes with enough samples are split using stratification.
        Rare classes are handled separately to prevent stratified
        splitting errors.

        Args:
            df: Input dataset containing the Label column.
            train_size: Proportion assigned to the training set.
            validation_size: Proportion assigned to the validation set.
            test_size: Proportion assigned to the test set.
            random_state: Random seed used for reproducible splitting.
            rare_class_threshold: Classes with fewer samples than this
                threshold are treated as rare classes.

        Returns:
            A tuple containing the training, validation, and test
            DataFrames.

        Raises:
            ValueError: If the split proportions do not sum to one,
                or if the Label column is missing.
        """

        # 计算三个数据集比例之和
        # Calculate the sum of the split proportions
        total_size = (
            train_size
            + validation_size
            + test_size
        )

        # 检查切分比例之和是否等于 1
        # Verify that the split proportions sum to one
        if abs(total_size - 1.0) > 1e-9:
            raise ValueError(
                "The sum of train_size, validation_size, "
                "and test_size must equal 1."
            )

        # 检查标签列是否存在
        # Verify that the Label column exists
        if "Label" not in df.columns:
            raise ValueError(
                "The Label column was not found in the dataset."
            )

        # 统计每个类别的样本数量
        # Count the number of samples in each class
        label_counts = (
            df["Label"].value_counts()
        )

        # 找出样本数量低于阈值的稀有类别
        # Identify rare classes below the sample threshold
        rare_labels = label_counts[
            label_counts < rare_class_threshold
        ].index

        # 将普通类别和稀有类别分开
        # Separate common classes from rare classes
        common_df = df[
            ~df["Label"].isin(rare_labels)
        ].copy()

        rare_df = df[
            df["Label"].isin(rare_labels)
        ].copy()

        # 显示随机切分前检测到的稀有类别
        # Display rare classes detected before random splitting
        print(
            "\nRare classes detected before random splitting:"
        )

        if len(rare_labels) == 0:
            print(
                "No rare classes were detected."
            )
        else:
            print(
                label_counts
                .loc[rare_labels]
                .to_string()
            )

        # 对样本充足的类别执行第一次分层切分
        # Perform the first stratified split for common classes
        common_train, common_temporary = (
            train_test_split(
                common_df,
                train_size=train_size,
                random_state=random_state,
                stratify=common_df["Label"],
            )
        )

        # 计算临时数据中验证集所占的相对比例
        # Calculate the relative validation ratio
        relative_validation_size = (
            validation_size
            / (
                validation_size
                + test_size
            )
        )

        # 将临时数据继续分层切分为验证集和测试集
        # Split the temporary set into validation and test sets
        common_validation, common_test = (
            train_test_split(
                common_temporary,
                train_size=relative_validation_size,
                random_state=random_state,
                stratify=common_temporary["Label"],
            )
        )

        # 创建稀有类别训练集片段列表
        # Create a list for rare-class training subsets
        rare_train_parts = []

        # 创建稀有类别验证集片段列表
        # Create a list for rare-class validation subsets
        rare_validation_parts = []

        # 创建稀有类别测试集片段列表
        # Create a list for rare-class test subsets
        rare_test_parts = []

        # 单独处理每一个稀有类别
        # Handle each rare class separately
        for label, group in rare_df.groupby(
            "Label"
        ):
            # 随机打乱当前稀有类别的数据
            # Randomly shuffle the current rare class
            group = group.sample(
                frac=1,
                random_state=random_state,
            ).reset_index(
                drop=True
            )

            count = len(group)

            if count == 1:
                # 只有一条样本时优先放入训练集
                # Place a single sample in the training set
                rare_train_parts.append(
                    group
                )

            elif count == 2:
                # 两条样本时一条放训练集，一条放测试集
                # Place one sample in training and one in testing
                rare_train_parts.append(
                    group.iloc[:1]
                )

                rare_test_parts.append(
                    group.iloc[1:]
                )

            else:
                # 至少三条样本时确保三个集合中各有一条
                # Ensure all three sets receive at least one sample
                train_count = max(
                    1,
                    int(count * train_size),
                )

                validation_count = max(
                    1,
                    int(
                        count * validation_size
                    ),
                )

                # 确保测试集至少保留一条样本
                # Ensure that at least one sample remains for testing
                if (
                    train_count
                    + validation_count
                    >= count
                ):
                    train_count = count - 2
                    validation_count = 1

                test_start = (
                    train_count
                    + validation_count
                )

                # 添加当前类别的训练集部分
                # Add the training subset for the current class
                rare_train_parts.append(
                    group.iloc[
                        :train_count
                    ]
                )

                # 添加当前类别的验证集部分
                # Add the validation subset for the current class
                rare_validation_parts.append(
                    group.iloc[
                        train_count:test_start
                    ]
                )

                # 添加当前类别的测试集部分
                # Add the test subset for the current class
                rare_test_parts.append(
                    group.iloc[
                        test_start:
                    ]
                )

        # 创建三个数据集的初始片段列表
        # Create the initial subset lists
        train_parts = [
            common_train
        ]

        validation_parts = [
            common_validation
        ]

        test_parts = [
            common_test
        ]

        # 将稀有类别训练数据加入训练集
        # Add rare-class subsets to the training set
        if rare_train_parts:
            train_parts.extend(
                rare_train_parts
            )

        # 将稀有类别验证数据加入验证集
        # Add rare-class subsets to the validation set
        if rare_validation_parts:
            validation_parts.extend(
                rare_validation_parts
            )

        # 将稀有类别测试数据加入测试集
        # Add rare-class subsets to the test set
        if rare_test_parts:
            test_parts.extend(
                rare_test_parts
            )

        # 合并并打乱训练集
        # Combine and shuffle the training set
        train_df = pd.concat(
            train_parts,
            ignore_index=True,
        ).sample(
            frac=1,
            random_state=random_state,
        )

        # 合并并打乱验证集
        # Combine and shuffle the validation set
        validation_df = pd.concat(
            validation_parts,
            ignore_index=True,
        ).sample(
            frac=1,
            random_state=random_state,
        )

        # 合并并打乱测试集
        # Combine and shuffle the test set
        test_df = pd.concat(
            test_parts,
            ignore_index=True,
        ).sample(
            frac=1,
            random_state=random_state,
        )

        # 重置三个数据集的索引并返回
        # Reset indices and return the three datasets
        return (
            train_df.reset_index(
                drop=True
            ),
            validation_df.reset_index(
                drop=True
            ),
            test_df.reset_index(
                drop=True
            ),
        )

    def day_based_split(
        self,
        train_keywords: list[str],
        validation_keywords: list[str],
        test_keywords: list[str],
        sample_per_file: int | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Split the dataset using filename or day keywords.

        Example:
            Training: Monday, Tuesday, Wednesday
            Validation: Thursday
            Testing: Friday

        Args:
            train_keywords: Keywords used to select training files.
            validation_keywords: Keywords used to select validation files.
            test_keywords: Keywords used to select test files.
            sample_per_file: Optional number of rows sampled from
                each matching file.

        Returns:
            A tuple containing the training, validation, and test
            DataFrames.
        """

        # 加载训练集对应的文件
        # Load files assigned to the training set
        train_df = self._load_keywords(
            keywords=train_keywords,
            sample_per_file=sample_per_file,
        )

        # 加载验证集对应的文件
        # Load files assigned to the validation set
        validation_df = self._load_keywords(
            keywords=validation_keywords,
            sample_per_file=sample_per_file,
        )

        # 加载测试集对应的文件
        # Load files assigned to the test set
        test_df = self._load_keywords(
            keywords=test_keywords,
            sample_per_file=sample_per_file,
        )

        return (
            train_df,
            validation_df,
            test_df,
        )

    def _load_keywords(
        self,
        keywords: list[str],
        sample_per_file: int | None,
    ) -> pd.DataFrame:
        """
        Load files matching a group of keywords without loading
        the same file more than once.

        Args:
            keywords: Filename keywords used for matching.
            sample_per_file: Optional number of rows sampled from
                each matching file.

        Returns:
            A combined DataFrame containing all matching files.

        Raises:
            ValueError: If no matching files are found.
        """

        # 创建匹配文件列表
        # Create a list for matching files
        matching_files = []

        # 遍历所有清洗后的数据文件
        # Iterate through all cleaned dataset files
        for file_path in self.loader.list_files():
            # 检查文件名是否匹配任意关键词
            # Check whether the filename matches any keyword
            if any(
                keyword.lower()
                in file_path.name.lower()
                for keyword in keywords
            ):
                matching_files.append(
                    file_path
                )

        # 去重并按文件名排序
        # Remove duplicates and sort the matching files
        matching_files = sorted(
            set(matching_files)
        )

        # 检查是否找到匹配文件
        # Verify that matching files were found
        if not matching_files:
            raise ValueError(
                "No files matched the provided keywords: "
                f"{keywords}"
            )

        # 创建用于保存各文件数据的列表
        # Create a list for storing loaded DataFrames
        dataframes = []

        # 逐个加载匹配的文件
        # Load each matching file
        for file_path in matching_files:
            # 显示当前正在加载的划分文件
            # Display the split file currently being loaded
            print(
                f"Loading split file: {file_path.name}"
            )

            # 加载当前文件
            # Load the current file
            df = self.loader.load_file(
                file_path=file_path,
                sample_size=sample_per_file,
            )

            # 添加来源文件列
            # Add the source-file column
            df["Source File"] = (
                file_path.name
            )

            # 将当前文件加入待合并列表
            # Add the current DataFrame to the merge list
            dataframes.append(
                df
            )

        # 合并所有匹配文件
        # Combine all matching files
        return pd.concat(
            dataframes,
            ignore_index=True,
        )

    @staticmethod
    def create_split_summary(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_name: str,
    ) -> dict:
        """
        Create a summary of a dataset split.

        Args:
            train_df: Training dataset.
            validation_df: Validation dataset.
            test_df: Test dataset.
            split_name: Name used to identify the split strategy.

        Returns:
            A dictionary containing row counts, column counts,
            label distributions, and source-file information.
        """

        # 将三个数据集组织到统一字典中
        # Organize the three datasets in one dictionary
        datasets = {
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
        }

        # 初始化切分摘要
        # Initialize the split summary
        summary = {
            "split_name": split_name,
            "datasets": {},
        }

        # 分别统计每个数据集的信息
        # Calculate statistics for each dataset
        for name, dataset in datasets.items():
            summary["datasets"][name] = {
                "rows": int(
                    len(dataset)
                ),
                "columns": int(
                    len(dataset.columns)
                ),
                "label_counts": (
                    dataset["Label"]
                    .value_counts()
                    .to_dict()
                ),
                "source_files": (
                    sorted(
                        dataset[
                            "Source File"
                        ]
                        .unique()
                        .tolist()
                    )
                    if (
                        "Source File"
                        in dataset.columns
                    )
                    else []
                ),
            }

        return summary


def main() -> None:
    # 创建报告目录
    # Create the report directory
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 创建数据加载器
    # Create the dataset loader
    loader = (
        CICIDS2017DataLoader()
    )

    # 创建数据切分器
    # Create the dataset splitter
    splitter = (
        CICIDS2017DatasetSplitter(
            loader
        )
    )

    # 测试时每个文件最多读取 10,000 行
    # Load up to 10,000 rows per file during testing
    sample_df = loader.load_all(
        sample_per_file=10_000
    )

    # 执行带稀有类别处理的随机切分
    # Perform random splitting with rare-class handling
    (
        random_train,
        random_validation,
        random_test,
    ) = splitter.random_split(
        sample_df
    )

    # 生成随机切分摘要
    # Generate the random-split summary
    random_summary = (
        splitter.create_split_summary(
            train_df=random_train,
            validation_df=random_validation,
            test_df=random_test,
            split_name=(
                "stratified_random_split"
            ),
        )
    )

    # 显示随机切分结果
    # Display random-split results
    print(
        "\nRandom split:"
    )

    print(
        f"Train:      "
        f"{len(random_train):,}"
    )

    print(
        f"Validation: "
        f"{len(random_validation):,}"
    )

    print(
        f"Test:       "
        f"{len(random_test):,}"
    )

    # 根据日期关键词进行切分
    # Perform day-based splitting
    (
        day_train,
        day_validation,
        day_test,
    ) = splitter.day_based_split(
        train_keywords=[
            "Monday",
            "Tuesday",
            "Wednesday",
        ],
        validation_keywords=[
            "Thursday"
        ],
        test_keywords=[
            "Friday"
        ],
        sample_per_file=10_000,
    )

    # 生成日期切分摘要
    # Generate the day-based split summary
    day_summary = (
        splitter.create_split_summary(
            train_df=day_train,
            validation_df=day_validation,
            test_df=day_test,
            split_name="day_based_split",
        )
    )

    # 显示日期切分结果
    # Display day-based split results
    print(
        "\nDay-based split:"
    )

    print(
        f"Train:      "
        f"{len(day_train):,}"
    )

    print(
        f"Validation: "
        f"{len(day_validation):,}"
    )

    print(
        f"Test:       "
        f"{len(day_test):,}"
    )

    # 合并两种切分方式的摘要
    # Combine summaries from both splitting strategies
    output = {
        "random_split": random_summary,
        "day_based_split": day_summary,
    }

    # 定义切分报告文件路径
    # Define the split report output path
    report_path = (
        REPORTS_DIR
        / "dataset_split_summary.json"
    )

    # 将切分摘要保存为 JSON 文件
    # Save the split summaries as JSON
    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 显示报告生成位置
    # Display the generated report path
    print(
        f"\nDataset split report generated: "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()