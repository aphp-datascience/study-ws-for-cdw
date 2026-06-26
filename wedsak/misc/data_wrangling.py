import json
import os
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Callable, List, Optional, Union

from edsnlp.utils.extensions import rgetattr
from edsnlp.utils.span_getters import get_spans, validate_span_getter
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from pyspark.sql.types import DecimalType, StructField, StructType
from pyspark.sql.window import Window

from wedsak.misc.utils import get_spark_sql


def keep_n(
    df,
    sort_column="note_datetime",
    how="first",
    partition_by=[
        "person_id",
    ],
    n: int = 1,
):
    assert isinstance(df, sparkDataFrame)
    if how == "first":
        # Filter and keep only first
        windowSpec = Window.partitionBy(partition_by).orderBy(F.col(sort_column).asc())
    else:
        # Filter and keep only last
        windowSpec = Window.partitionBy(partition_by).orderBy(F.col(sort_column).desc())

    df = df.withColumn(
        "row",
        F.row_number().over(windowSpec),
    )

    condition = F.col("row") <= n
    df_filtered = df.filter(condition)
    df_filtered = df_filtered.drop("row")

    return df_filtered


def keep_one(
    df,
    sort_column="note_datetime",
    how="first",
    partition_by=[
        "person_id",
    ],
):
    return keep_n(
        df=df, sort_column=sort_column, how=how, partition_by=partition_by, n=1
    )


def filter_by_person_set(
    df: sparkDataFrame,
    person_set: Optional[Union[sparkDataFrame, List[int]]] = None,
    method="inner",
) -> sparkDataFrame:
    if person_set is not None:
        if isinstance(person_set, list):
            person_schema = StructType(
                [
                    StructField("person_id", DecimalType(precision=38), False),
                ]
            )

            if len(person_set) > 0:
                person_data = [[Decimal(int(i))] for i in person_set]
            spark, _ = get_spark_sql()
            person_set_dedup = spark.createDataFrame(
                person_data, person_schema
            ).drop_duplicates()
        else:
            person_set_dedup = person_set.select(["person_id"]).drop_duplicates()

        df = df.join(person_set_dedup, on="person_id", how=method)
    return df


def normalize_task_name(task_name: str) -> str:
    """Normalize task names by removing special characters and converting to lowercase."""
    normalized_task_name = (
        task_name.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("-", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )

    return normalized_task_name


class SafeDict(defaultdict):
    def __missing__(self, key):
        return "{" + key + "}"


def collector(series):
    list_class_value = [
        {"value": value, "class": class_}
        for value, class_ in zip(series.event_date, series.snomed_name)
    ]
    return list_class_value


class SpanToRowConverter:
    def __init__(
        self,
        span_attributes: List[str],
        span_getter: List[str],
        doc_attributes: List[str] = ["person_id"],
        k: int = 25,
        include_context: bool = True,
        context_getter: Optional[Callable[[object, object], str]] = None,
    ):
        self.span_attributes = span_attributes
        self.span_getter = span_getter
        self.doc_attributes = doc_attributes
        self.k = k
        self.include_context = include_context
        self.context_getter = context_getter

    def __call__(
        self,
        doc,
    ):
        span_getter = validate_span_getter(self.span_getter)
        rows = []
        for span in get_spans(doc, span_getter):
            row = {
                "note_id": doc._.note_id,
                "lexical_variant": span.text,
                "label": span.label_,
                "start": span.start_char,
                "end": span.end_char,
                **{attr: rgetattr(span, f"_.{attr}") for attr in self.span_attributes},
                **{attr: rgetattr(doc, f"_.{attr}") for attr in self.doc_attributes},
            }
            if self.include_context:
                if self.context_getter is None:
                    row["context"] = doc[span.start - self.k : span.end + self.k].text
                else:
                    row["context"] = self.context_getter(doc, span)
            rows.append(row)
        return rows


# Replicate entities


def deduplicate_entities(entities):
    seen = set()
    deduplicated = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        key = (
            entity.get("begin"),
            entity.get("end"),
            entity.get("label"),
        )

        if key not in seen:
            seen.add(key)
            deduplicated.append(entity)

    return deduplicated


def process_json_file(filepath, label_list, replacement_label):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Skipping invalid JSON: {filepath}")
        return

    entities = data.get("entities")

    if not isinstance(entities, list) or not entities:
        return

    new_entities = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        label = entity.get("label")
        if label in label_list:
            new_entity = deepcopy(entity)
            new_entity["label"] = replacement_label
            new_entity["id"] = (
                "metanno-{}-"
                + str(new_entity["begin"])
                + "-"
                + str(new_entity["end"])
                + f"-{replacement_label}"
            )
            new_entities.append(new_entity)

    if not new_entities:
        return

    combined_entities = entities + new_entities
    data["entities"] = deduplicate_entities(combined_entities)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def replicate_entities(
    root_folder: str,
    label_list: List[str],
    replacement_label: str,
    exclude_files: Optional[List[str]] = ["metadata.json"],
):
    root_folder = Path(root_folder)
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json") and (
                exclude_files is None or file not in exclude_files
            ):
                filepath = os.path.join(root, file)
                process_json_file(filepath, label_list, replacement_label)
