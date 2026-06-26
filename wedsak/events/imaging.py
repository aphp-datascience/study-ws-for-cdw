from typing import List, Optional, Union

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from wedsak.misc.utils import read_table, get_spark_sql


class ImagingEventSelector:
    def __init__(
        self,
        db: Optional[str] = None,
        vocabulary_id: str = "DICOM - CID33 - Modality",
        columns: List[str] = [
            "person_id",
            "series_datetime",
        ],
        convert_to_date: bool = True,
    ):
        self.db = db
        self.columns = columns
        self.convert_to_date = convert_to_date
        if vocabulary_id == "DICOM - CID33 - Modality":
            self.concept_col = "modality_concept_id"
        elif vocabulary_id == "APHP - PACS - Modality":
            self.concept_col = "modality_source_concept_id"
        else:
            raise ValueError(
                "vocabulary_id must be either 'DICOM - CID33 - Modality' or 'APHP - PACS - Modality'"
            )

        self.columns.append(self.concept_col)

    @staticmethod
    def cast_to_date(df: sparkDataFrame):
        # Cast datetime to date
        df = df.withColumn("series_date", F.to_date(F.col("series_datetime")))
        return df

    def __call__(self, concept_codes: List[Union[int, str, pd.DataFrame]]):
        imaging_series = read_table("imaging_series", db=self.db).select(self.columns)

        if isinstance(concept_codes, int):
            imaging_series_f = imaging_series.filter(
                F.col(self.concept_col) == concept_codes
            )
        elif isinstance(concept_codes, list):
            imaging_series_f = imaging_series.filter(
                F.col(self.concept_col).isin(concept_codes)
            )
        else:
            # Cast these codes to spark
            spark, _ = get_spark_sql()
            codes_spark = spark.createDataFrame(concept_codes)
            codes_spark = codes_spark.withColumnRenamed("concept_id", self.concept_col)

            # Get procedures of these codes (should have column name code)
            imaging_series_f = imaging_series.join(
                codes_spark.hint("broadcast"),
                on=self.concept_col,
                how="inner",
            )

        if self.convert_to_date:
            imaging_series_f = self.cast_to_date(imaging_series_f)

        return imaging_series_f
