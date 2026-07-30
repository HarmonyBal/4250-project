from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


# 定义项目根目录
# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 定义清洗后数据目录
# Define the cleaned dataset directory
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


class CICIDS2017DataLoader:
    """
    Load cleaned CIC-IDS2017 datasets.

    Later modules only need to use this class and do not need
    to know the exact storage location of the dataset files.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
    ) -> None:
        """
        Initialize the CIC-IDS2017 data loader.

        Args:
            data_dir: Optional custom directory containing
                cleaned CIC-IDS2017 CSV files.

        Raises:
            FileNotFoundError: If the cleaned dataset directory
                does not exist or contains no cleaned CSV files.
        """

        # 使用指定目录，否则使用项目默认的 processed 目录
        # Use the provided directory or the default processed directory
        self.data_dir = data_dir or PROCESSED_DIR

        # 检查清洗后数据目录是否存在
        # Verify that the cleaned dataset directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"The cleaned dataset directory does not exist: "
                f"{self.data_dir}\n"
                "Run clean_cicids2017.py first."
            )

        # 查找所有以 cleaned_ 开头的 CSV 文件
        # Find all CSV files beginning with cleaned_
        self.csv_files = sorted(
            self.data_dir.glob("cleaned_*.csv")
        )

        # 检查是否找到了清洗后的 CSV 文件
        # Verify that cleaned CSV files were found
        if not self.csv_files:
            raise FileNotFoundError(
                f"No cleaned_*.csv files were found in "
                f"{self.data_dir}."
            )

    def list_files(self) -> list[Path]:
        """
        Return all available cleaned CSV files.

        Returns:
            A copy of the list of cleaned CSV file paths.
        """

        # 返回副本，避免外部代码修改内部文件列表
        # Return a copy to prevent external modification
        return self.csv_files.copy()

    def load_file(
        self,
        file_path: Path,
        columns: Iterable[str] | None = None,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Load one cleaned CSV file.

        Args:
            file_path: Path to the cleaned CSV file.
            columns: Optional columns to load in order to reduce
                memory usage.
            sample_size: Optional number of rows to sample randomly.
            random_state: Random seed used for reproducible sampling.

        Returns:
            A loaded and optionally sampled DataFrame.
        """

        # 读取指定的 CSV 文件
        # Read the specified CSV file
        df = pd.read_csv(
            file_path,
            usecols=columns,
            low_memory=False,
        )

        # 当样本数量小于数据总量时进行随机抽样
        # Randomly sample rows when the requested size is smaller
        if (
            sample_size is not None
            and sample_size < len(df)
        ):
            df = df.sample(
                n=sample_size,
                random_state=random_state,
            )

        # 重置索引并返回数据
        # Reset the index and return the dataset
        return df.reset_index(
            drop=True
        )

    def load_by_keyword(
        self,
        keyword: str,
        columns: Iterable[str] | None = None,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Load cleaned files whose filenames contain a keyword.

        Examples:
            loader.load_by_keyword("Monday")
            loader.load_by_keyword("WebAttacks")

        Args:
            keyword: Keyword used to match filenames.
            columns: Optional columns to load.
            sample_size: Optional number of rows sampled from
                each matching file.

        Returns:
            A combined DataFrame containing all matching files.

        Raises:
            ValueError: If no matching files are found.
        """

        # 根据文件名关键词查找匹配文件
        # Find matching files using the filename keyword
        matches = [
            file_path
            for file_path in self.csv_files
            if keyword.lower() in file_path.name.lower()
        ]

        # 检查是否存在匹配的文件
        # Verify that matching files were found
        if not matches:
            raise ValueError(
                f"No files containing the keyword "
                f"'{keyword}' were found."
            )

        # 加载所有匹配的文件
        # Load all matching files
        dataframes = [
            self.load_file(
                file_path=file_path,
                columns=columns,
                sample_size=sample_size,
            )
            for file_path in matches
        ]

        # 合并所有匹配文件的数据
        # Combine data from all matching files
        return pd.concat(
            dataframes,
            ignore_index=True,
        )

    def load_all(
        self,
        columns: Iterable[str] | None = None,
        sample_per_file: int | None = None,
    ) -> pd.DataFrame:
        """
        Load all cleaned CIC-IDS2017 files.

        The sample_per_file parameter can limit the number of rows
        loaded from each file to avoid loading the entire dataset
        into memory at once.

        Args:
            columns: Optional columns to load.
            sample_per_file: Optional number of rows sampled from
                each cleaned CSV file.

        Returns:
            A combined DataFrame containing all loaded files.
            A Source File column is added to identify the origin
            of each row.
        """

        # 创建用于保存每个文件数据的列表
        # Create a list for storing DataFrames from each file
        dataframes = []

        # 逐个加载所有清洗后的 CSV 文件
        # Load all cleaned CSV files one by one
        for file_path in self.csv_files:
            # 显示当前正在加载的文件
            # Display the file currently being loaded
            print(
                f"Loading: {file_path.name}"
            )

            # 加载当前文件
            # Load the current file
            df = self.load_file(
                file_path=file_path,
                columns=columns,
                sample_size=sample_per_file,
            )

            # 添加来源文件列，记录每行数据来自哪个 CSV 文件
            # Add a source column identifying the original CSV file
            df["Source File"] = file_path.name

            # 将当前数据加入待合并列表
            # Add the current DataFrame to the merge list
            dataframes.append(
                df
            )

        # 合并所有文件的数据
        # Combine data from all files
        return pd.concat(
            dataframes,
            ignore_index=True,
        )

    def iterate_files(
        self,
        columns: Iterable[str] | None = None,
    ) -> Iterator[tuple[Path, pd.DataFrame]]:
        """
        Yield one cleaned DataFrame at a time.

        This method is suitable for processing large datasets
        without loading all files into memory simultaneously.

        Args:
            columns: Optional columns to load.

        Yields:
            A tuple containing the file path and loaded DataFrame.
        """

        # 逐文件返回路径和对应的数据
        # Yield each file path and its corresponding DataFrame
        for file_path in self.csv_files:
            yield (
                file_path,
                self.load_file(
                    file_path=file_path,
                    columns=columns,
                ),
            )


if __name__ == "__main__":
    # 创建数据加载器实例
    # Create a data loader instance
    loader = CICIDS2017DataLoader()

    # 显示所有可用的清洗后数据文件
    # Display all available cleaned dataset files
    print("Available cleaned dataset files:")

    for file_path in loader.list_files():
        print(
            f"- {file_path.name}"
        )

    # 从每个文件中抽取 1,000 条记录进行测试
    # Sample 1,000 rows from each file for testing
    sample = loader.load_all(
        columns=[
            "Flow Duration",
            "Total Fwd Packets",
            "Label",
        ],
        sample_per_file=1_000,
    )

    # 显示测试数据形状
    # Display the shape of the test dataset
    print("\nTest dataset shape:")
    print(sample.shape)

    # 显示测试数据的前五行
    # Display the first five rows of the test dataset
    print("\nFirst five rows:")
    print(sample.head())

    # 显示测试数据中的标签分布
    # Display the label distribution in the test dataset
    print("\nLabel distribution:")
    print(
        sample["Label"].value_counts()
    )