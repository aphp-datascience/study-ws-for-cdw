from typing import List, Optional, Union

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from wedsak.misc.utils import read_table, get_spark_sql
from wedsak.misc.constants import CLAIM_SOURCE_AREM

from wedsak.misc.constants import CLAIM_SOURCE_MAPPING


class ClaimEventSelector:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def get_procedures(
        codes: Optional[Union[str, List[str], tuple, pd.DataFrame]],
        db: Optional[str] = None,
        claim_source: Optional[str] = "AREM",
        drop_duplicates: bool = False,
    ) -> sparkDataFrame:
        """
        Parameters
        ----------
        db: Optional[str] = None,
            database name
        codes: Optional[Union[str, List[str], tuple, pd.DataFrame]],
            Procedures codes (CCAM)
        claim source: Optional[str] = "AREM",
            Source of data to filter, one of {None, "ORBIS","AREM"}


        Returns
        -------
        sparkDataFrame with the following columns:
        ['person_id',
        'visit_occurrence_id',
        'procedure_occurrence_id',
        'procedure_datetime',
        'procedure_source_value',
        'cdm_source',
        'visit_occurrence_source_value'
        ]
        """
        # Get spark session
        spark, _ = get_spark_sql()

        # All procedures (CCAM)
        procedures = read_table("procedure", db=db)
        procedures = procedures.withColumn(
            "procedure_date", F.to_date(F.col("procedure_datetime"))
        )

        # Keep only claim_source
        if claim_source is not None:
            claim_source_value = CLAIM_SOURCE_MAPPING.get(claim_source)
            procedures = procedures.where(F.col("cdm_source") == claim_source_value)

        if codes is not None:
            if isinstance(codes, pd.DataFrame):
                # Cast these codes to spark
                COLUMNS_CODES = ["code"]
                if "Task Number" in codes.columns:
                    COLUMNS_CODES.append("Task Number")
                codes_spark = spark.createDataFrame(codes[COLUMNS_CODES])

                # Get procedures of these codes (should have column name code)
                procedures = procedures.join(
                    codes_spark.hint("broadcast"),
                    on=(codes_spark.code == procedures.procedure_source_value),
                    how="inner",
                )
                procedures = procedures.drop("code")

            elif isinstance(codes, str):
                procedures = procedures.where(F.col("procedure_source_value") == codes)
            elif isinstance(codes, (list, tuple)):
                procedures = procedures.where(
                    F.col("procedure_source_value").isin(list(codes))
                )
            else:
                raise TypeError(
                    "`codes` should be one of {str, list, tuple, pd.Dataframe}"
                )
            if drop_duplicates:
                procedures = procedures.drop_duplicates(
                    subset=["procedure_occurrence_id"]
                )

        vo = read_table(
            "visit_occurrence",
            db=db,
            select_cols=["visit_occurrence_id", "visit_occurrence_source_value"],
        )
        procedures = procedures.join(vo, on="visit_occurrence_id", how="left")
        return procedures

    def get_conditions(
        self,
        icd10_codes: Optional[Union[str, List[str], tuple, pd.DataFrame]],
        claim_source: Optional[str] = None,
        diagnostic_types: Optional[Union[str, List[str]]] = None,
        db: Optional[str] = None,
        drop_duplicates: bool = False,
        **kwargs,
    ) -> sparkDataFrame:
        """
        Retrieve a sparkDataFrame with visits that fulfill the specified ICD10 diagnostics.

        Parameters
        ----------

        icd10_codes: Union[str, List[str], tuple, pd.DataFrame],
            Conditions codes (ICD10)
        diagnostic_types: Optional[Union[str, List[str]]] = None,
            Type of diagnostic hierarchy, one of {None, "DP", "DAS","DR"}
        claim source: Optional[str] = CLAIM_SOURCE_AREM,
            Source of data to filter, one of {None, "ORBIS", "AREM"}
        db: Optional[str] = None,
            database name


        Returns
        -------
        Returns a spark.DataFrame with the visits and ICD10 codes that fulfill the conditions.
        Visits could be duplicated if they have multiple codes.

        With the following columns:
            ['person_id',
            'visit_occurrence_id',
            'condition_occurrence_id',
            'condition_start_date',
            'condition_source_value',
            'condition_status_source_value',
            'cdm_source',
            'visit_occurrence_source_value',
        ]



        """  # noqa: E501

        COLS = [
            "person_id",
            "visit_occurrence_id",
            "condition_occurrence_id",
            "condition_start_date",
            "condition_source_value",
            "condition_status_source_value",
            "cdm_source",
            "visit_occurrence_source_value",
            "care_site_name",
        ]

        # get table
        co = read_table(table_name="condition", db=db)
        co = co.withColumn(
            "condition_start_date", F.to_date(F.col("condition_start_datetime"))
        )

        vo = read_table(
            "visit_occurrence",
            db=db,
            select_cols=[
                "visit_occurrence_id",
                "visit_occurrence_source_value",
                "care_site_id",
            ],
        )
        icd10_bc = co.join(vo, on="visit_occurrence_id", how="left")

        if claim_source is not None:
            claim_source_value = CLAIM_SOURCE_MAPPING.get(claim_source)
            icd10_bc = icd10_bc.where(F.col("cdm_source") == claim_source_value)

        cs = read_table(
            "care_site",
            db=db,
            select_cols=["care_site_id", "care_site_name"],
        )
        icd10_bc = icd10_bc.join(cs, on="care_site_id", how="left")

        icd10 = icd10_bc.select(COLS)

        icd10 = self.filter_by_diagnostic_type(
            icd10=icd10, diagnostic_types=diagnostic_types
        )

        # Filter by codes
        icd10 = self.filter_icd10_by_code(
            icd10=icd10, icd10_codes=icd10_codes, drop_duplicates=drop_duplicates
        )

        return icd10

    @staticmethod
    def filter_by_diagnostic_type(
        icd10: sparkDataFrame,
        diagnostic_types: Optional[Union[str, List[str]]] = None,
    ):
        if diagnostic_types:
            if isinstance(diagnostic_types, str):
                icd10 = icd10.where(
                    F.col("condition_status_source_value") == diagnostic_types
                )
            elif isinstance(diagnostic_types, (list, tuple)):
                icd10 = icd10.where(
                    F.col("condition_status_source_value").isin(list(diagnostic_types))
                )
            else:
                raise TypeError("diagnostic_types should be a str or list")
        return icd10

    @staticmethod
    def filter_icd10_by_code(
        icd10: sparkDataFrame,
        icd10_codes: Optional[Union[str, List[str], tuple, pd.DataFrame]],
        drop_duplicates: bool = False,
    ):
        # Extract codes from column
        # Make columns with the ICD10 code with 0 , 2, 3, 4, 5, 6 digits. Example: C & C40 & C401, I2130, I21300, I22800  # noqa: E501
        icd10 = icd10.withColumn(
            "condition_source_value_short_6",
            F.substring("condition_source_value", 1, 7),
        )
        icd10 = icd10.withColumn(
            "condition_source_value_short_5",
            F.substring("condition_source_value", 1, 6),
        )
        icd10 = icd10.withColumn(
            "condition_source_value_short_4",
            F.substring("condition_source_value", 1, 5),
        )
        icd10 = icd10.withColumn(
            "condition_source_value_short_3",
            F.substring("condition_source_value", 1, 4),
        )
        icd10 = icd10.withColumn(
            "condition_source_value_short_2",
            F.substring("condition_source_value", 1, 3),
        )

        icd10 = icd10.withColumn(
            "condition_source_value_short_0",
            F.substring("condition_source_value", 1, 1),
        )

        # Filter by codes
        if icd10_codes is not None:
            if isinstance(icd10_codes, str):
                icd10 = icd10.where(
                    (F.col("condition_source_value_short_2") == icd10_codes)
                    | (F.col("condition_source_value_short_3") == icd10_codes)
                    | (F.col("condition_source_value_short_4") == icd10_codes)
                    | (F.col("condition_source_value_short_5") == icd10_codes)
                    | (F.col("condition_source_value_short_6") == icd10_codes)
                    | (F.col("condition_source_value_short_0") == icd10_codes)
                )
            elif isinstance(icd10_codes, (list, tuple)):
                icd10 = icd10.where(
                    (F.col("condition_source_value_short_2").isin(list(icd10_codes)))
                    | (F.col("condition_source_value_short_3").isin(list(icd10_codes)))
                    | (F.col("condition_source_value_short_4").isin(list(icd10_codes)))
                    | (F.col("condition_source_value_short_5").isin(list(icd10_codes)))
                    | (F.col("condition_source_value_short_6").isin(list(icd10_codes)))
                    | (F.col("condition_source_value_short_0").isin(list(icd10_codes)))
                )

            elif isinstance(icd10_codes, pd.DataFrame):
                sparksession, _ = get_spark_sql()
                # Cast these codes to spark
                codes_spark = sparksession.createDataFrame(icd10_codes)

                # Get procedures of these codes (should have column name code)
                icd10 = icd10.join(
                    codes_spark.hint("broadcast"),
                    on=(
                        (codes_spark.code == icd10.condition_source_value_short_2)
                        | (codes_spark.code == icd10.condition_source_value_short_3)
                        | (codes_spark.code == icd10.condition_source_value_short_4)
                        | (codes_spark.code == icd10.condition_source_value_short_5)
                        | (codes_spark.code == icd10.condition_source_value_short_6)
                        | (codes_spark.code == icd10.condition_source_value_short_0)
                    ),
                    how="inner",
                )
                icd10 = icd10.drop("code")
            else:
                raise TypeError(
                    "icd_values should be one of {str, list, tuple, pd.Dataframe}"
                )

        icd10 = icd10.drop(
            "condition_source_value_short_2",
            "condition_source_value_short_3",
            "condition_source_value_short_0",
            "condition_source_value_short_4",
            "condition_source_value_short_5",
            "condition_source_value_short_6",
        )
        if drop_duplicates:
            icd10 = icd10.drop_duplicates(subset=["condition_occurrence_id"])
        return icd10

    @staticmethod
    def get_cost(
        codes: Union[str, List[str], tuple, pd.DataFrame],
        claim_source: Optional[str] = CLAIM_SOURCE_AREM,
        db: Optional[str] = None,
        drop_duplicates: bool = False,
    ) -> sparkDataFrame:
        # Get spark session
        spark, sql = get_spark_sql()

        # All GHM (cost)
        cost = read_table(
            "cost",
            db=db,
        )
        cost = cost.withColumnRenamed("cost_event_id", "visit_occurrence_id")

        # Keep only claim_source
        if claim_source:
            cost = cost.where(F.col("cdm_source") == claim_source)

        if codes is not None:
            if isinstance(codes, pd.DataFrame):
                # Cast these codes to spark
                codes_spark = spark.createDataFrame(codes)

                # Get procedures of these codes (should have column name code)
                cost = cost.join(
                    codes_spark.hint("broadcast"),
                    on=(codes_spark.code == cost.drg_source_value),
                    how="inner",
                )
                cost = cost.drop("code")

            elif isinstance(codes, str):
                cost = cost.where(F.col("drg_source_value") == codes)
            elif isinstance(codes, (list, tuple)):
                cost = cost.where(F.col("drg_source_value").isin(list(codes)))
            else:
                raise TypeError(
                    "`codes` should be one of {str, list, tuple, pd.Dataframe}"
                )
            if drop_duplicates:
                cost = cost.drop_duplicates(subset=["cost_id"])
        return cost

    def __call__(self):
        raise NotImplementedError
