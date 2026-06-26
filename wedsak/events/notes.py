from typing import List, Optional, Union

import pandas as pd
from pyspark.sql import functions as F

from wedsak.misc.utils import read_table, get_spark_sql


class NoteEventSelector:
    def __init__(
        self,
        db: Optional[str] = None,
        vocabulary_id: str = "LOINC - Document - Class",
        columns: List[str] = [
            "person_id",
            "note_date",
        ],
    ):
        self.db = db
        self.columns = columns
        if vocabulary_id == "LOINC - Document - Class":
            self.concept_col = "note_class_concept_id"
        else:
            raise ValueError("vocabulary_id must be 'LOINC - Document - Class'")
        self.columns.append(self.concept_col)

    def __call__(self, concept_codes: Union[int, List[int], pd.DataFrame]):
        notes = read_table("note", db=self.db).select(self.columns)

        if isinstance(concept_codes, int):
            notes_f = notes.filter(
                F.array_contains(F.col(self.concept_col), concept_codes)
            )
        elif isinstance(concept_codes, list):
            notes_f = notes.filter(
                F.size(
                    F.array_intersect(
                        F.col(self.concept_col),
                        F.array(*[F.lit(c) for c in concept_codes]),
                    )
                )
                > 0
            )
        else:
            # Cast these codes to spark
            spark, _ = get_spark_sql()
            codes_spark = spark.createDataFrame(concept_codes)
            codes_spark = codes_spark.withColumnRenamed("concept_id", "join_concept_id")

            # Explode the array column, join, then deduplicate
            notes_f = (
                notes.withColumn("_exploded", F.explode(F.col(self.concept_col)))
                .join(
                    codes_spark.hint("broadcast"),
                    on=F.col("_exploded") == F.col("join_concept_id"),
                    how="inner",
                )
                .drop("_exploded", "join_concept_id")
                .distinct()
            )

        return notes_f
