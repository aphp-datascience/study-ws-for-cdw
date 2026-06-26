import asyncio
import os
import re
import sys
from typing import Any, Coroutine, Dict, List, Optional, TypeVar, Union

import pyarrow.compute as pc
from pyarrow import parquet
from pyspark import SparkConf
from pyspark.sql import SparkSession
import hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from wedsak.misc.constants import PATH_DATABASE, PATH_DATA_SCRATCH, COLUMN_MAPPING

T = TypeVar("T")

user = os.getenv("USER")


# Create Spark Session
def get_spark_sql(conf_dict: Optional[Dict[str, Any]] = None, verbose: bool = False):
    # Delete pre-existing env variables
    # for k, v in os.environ.items():
    #     if re.match(r"(HDP|HADOOP|SPARK)", k):
    #         print("Deleting env variable:", k, os.getenv(k))
    #         del os.environ[k]

    # # Set python
    # PYSPARK_PYTHON = sys.executable
    # if verbose:
    #     print("PYSPARK_PYTHON:", PYSPARK_PYTHON)

    # os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON

    # HOST = os.environ["HOSTNAME"]
    # USER = os.environ["USER"]

    # Set Spark configuration #FIXME remove hardcode
    # conf = (
    #     SparkConf()
    #     # .setAppName("MyApp")
    #     # .setMaster("local[4]")
    #     # .setExecutorEnv(
    #     #     "PYSPARK_PYTHON",
    #     #     PYSPARK_PYTHON,
    #     # )
    #     # .setExecutorEnv("PYSPARK_DRIVER_PYTHON", PYSPARK_PYTHON)
    #     .set(
    #         "spark.sql.warehouse.dir",
    #         f"{PATH_DATA_SCRATCH}/spark-warehouse",
    #     )
    #     .set("spark.local.dir", f"{PATH_DATA_SCRATCH}/spark_tmp")
    #     # .set("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
    #     # .set("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
    #     .set("spark.executor.memory", "2g")
    #     .set("spark.sql.session.timeZone", "Europe/Paris")
    #     .set("spark.driver.host", HOST)
    #     .set("spark.yarn.appMasterEnv.HADOOP_USER_NAME", USER)
    #     # .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", PYSPARK_PYTHON)
    #     # .set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", PYSPARK_PYTHON)
    # )

    conf = SparkConf()
    # spark.sparkContext.setLogLevel("ERROR")
    if conf_dict is not None:
        for key, value in conf_dict.items():
            conf.set(key, value)

    # Initialize Spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sql = spark.sql
    return spark, sql


# Create Spark Session
def get_spark_sql_hive(conf_dict: Optional[Dict[str, Any]] = None):
    conf = SparkConf()
    conf.set("spark.sql.session.timeZone", "Europe/Paris")
    if conf_dict is not None:
        for key, value in conf_dict.items():
            conf.set(key, value)

    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    sql = spark.sql
    return spark, sql


def read_table(
    table_name: str,
    folder_path: Optional[str] = None,
    db: Optional[str] = None,
    select_cols: Optional[List[str]] = None,
    spark=None,
    rename_cols: bool = True,
    conf_spark: Optional[Dict[str, Any]] = None,
):

    if db is not None:
        if spark is None:
            spark, _ = get_spark_sql_hive(conf_dict=conf_spark)
        path_table = f"{db}.{table_name}"
        df = spark.read.table(path_table)
    else:
        if spark is None:
            spark, _ = get_spark_sql(conf_dict=conf_spark)
        if folder_path is None:
            folder_path = PATH_DATABASE
        path_table = os.path.join(folder_path, table_name)
        df = spark.read.parquet(path_table)

    if rename_cols:
        for key, value in COLUMN_MAPPING.get(table_name, {}).items():
            df = df.withColumnRenamed(key, value)
    if select_cols:
        df = df.select(select_cols)
    return df


class arrowConnector:
    def __init__(self, path_table=None, db=None, table=None):
        self.path_table = path_table
        self.db = db
        if db and table:
            spark, sql = get_spark_sql()
            self.path_table = (
                sql(f"desc formatted {db}.{table}")
                .filter("col_name=='Location'")
                .collect()[0]
                .data_type
            )
            self.db = os.path.dirname(self.path_table)

        if path_table:
            self.path_table = path_table

        if (db) and (table is None):
            self.db = db

    def get_pd_fragment(
        self,
        path_table=None,
        table_name=None,
        types_mapper=None,
        integer_object_nulls=True,
        date_as_object=False,
    ):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)

        # Import the parquet as ParquetDataset
        parquet_ds = parquet.ParquetDataset(self.path_table, use_legacy_dataset=False)

        # Partitions of ds
        fragments = iter(parquet_ds.fragments)

        # Set initial length
        length = 0

        # One partition
        while length < 1:
            fragment = next(fragments)

            # pyarrow.table of a fragment
            table = fragment.to_table()

            length = len(table)

        # Import to pandas the fragment
        table_pd = table.to_pandas(
            types_mapper=types_mapper,
            integer_object_nulls=integer_object_nulls,
            date_as_object=date_as_object,
        )
        return table_pd

    def count_fragments_length(self, path_table=None, table_name=None):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)
        # Import the parquet as ParquetDataset
        parquet_ds = parquet.ParquetDataset(self.path_table, use_legacy_dataset=False)

        # Partitions of ds
        fragments = iter(parquet_ds.fragments)

        lengths = []
        for fragment in fragments:
            # pyarrow.table of a fragment
            table = fragment.to_table()
            lengths.append(len(table))

        return lengths

    def get_pd_table(
        self,
        path_table=None,
        table_name=None,
        types_mapper=None,
        integer_object_nulls=True,
        date_as_object=False,
        filter_values_keep=None,
        filter_values_avoid=None,
        select_cols=Optional[List[str]],
        cast_to_tz: Optional[str] = None,
        filter_col="person_id",
    ):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)

        table = parquet.read_table(self.path_table)
        if select_cols:
            table = table.select(select_cols)
        if filter_values_keep:
            table = table.filter(pc.field(filter_col).isin(filter_values_keep))
        if filter_values_avoid:
            table = table.filter(
                pc.bit_wise_not(pc.field(filter_col).isin(filter_values_avoid))
            )

        df = table.to_pandas(
            date_as_object=date_as_object,
            types_mapper=types_mapper,
            integer_object_nulls=integer_object_nulls,
        )

        if cast_to_tz is not None:
            df = self.cast_to_tz(df, tz=cast_to_tz)
        return df

    @staticmethod
    def cast_to_tz(df, tz="Europe/Paris"):
        cols = df.select_dtypes(include=["datetime64"]).columns
        for col in cols:
            df[col] = df[col].dt.tz_localize("UTC")

            df[col] = df[col].dt.tz_convert(tz)
        return df


def get_dir_path(file):
    path_conf_file = os.path.dirname(os.path.realpath(file))
    return path_conf_file


def build_path(file, relative_path):
    """
    Function to build an absolut path.

    Parameters
    ----------
    file: main file from where we are calling. It could be __file__
    relative_path: str,
        relative path from the main file to the desired output

    Returns
    -------
    path: absolute path
    """
    dir_path = get_dir_path(file)
    path = os.path.abspath(os.path.join(dir_path, relative_path))
    return path


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Runs an asynchronous coroutine and always waits for the result,
    whether or not an event loop is already running.

    In a standard Python script (no active event loop), it uses `asyncio.run()`.
    In a notebook or environment with a running event loop, it applies a patch
    using `nest_asyncio` and runs the coroutine via `loop.run_until_complete`.

    Parameters
    ----------
    coro : Coroutine
        The coroutine to run.

    Returns
    -------
    T
        The result returned by the coroutine.
    """
    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.get_running_loop().run_until_complete(coro)
    else:
        return asyncio.run(coro)


def hash_file_or_directory(path, verbose: bool = False) -> str:
    """Hash all files in a directory"""
    hasher = hashlib.sha256()

    # Get all files sorted for consistency
    path = Path(path).expanduser()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*"))
    else:
        raise FileNotFoundError

    if verbose:
        print(files)

    for filepath in files:
        if filepath.is_file():
            # Hash filename for structure
            hasher.update(str(filepath.relative_to(path)).encode())

            # Hash file content
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)

    return hasher.hexdigest()


def save_json(
    data,
    file_name: Optional[str] = None,
    sub_folder: Optional[str] = None,
    folder_path: str = "~/wedsak/logs/",
):
    folder_path = Path(folder_path).expanduser()

    if sub_folder is not None:
        folder_path = Path(folder_path, sub_folder)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    if file_name is None:
        timestamp = (
            str(datetime.now(tz=ZoneInfo("Europe/Paris")))
            .replace(" ", "_")
            .replace(":", "-")
        )
        file_name = f"{timestamp}.json"

    path = Path(folder_path, file_name).expanduser()

    import json

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def make_unique_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        return path

    suffix = 1
    while True:
        new_path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not new_path.exists():
            return new_path
        suffix += 1
