from typing import List, Optional, Union
import numpy as np

import pandas as pd
from pyspark.sql import functions as F

from wedsak.misc.utils import read_table, get_spark_sql
from wedsak.misc.constants import USER

# Retrieve by:
# - Code ATC (concept_code)
# - Name (concept_name)


class DrugEventSelector:
    def __init__(
        self,
        db: Optional[str] = None,
        administration_or_prescription: str = "administration",
        vocabulary_id: str = "ATC",
        columns: List[str] = [
            "person_id",
            "drug_exposure_start_date",
            "drug_class_source_value",
            "drug_ucd_source_value",
            "drug_source_value",
        ],
        retrieve_by: str = "atc_code",
    ):
        """
        Retrieve drug events based on ATC codes or names.

        Parameters
        ----------
        - db: Optional database name to read tables from.
        - administration_or_prescription: Whether to retrieve drug administrations or prescriptions. Must be either "administration" or "prescription".
        - vocabulary_id: The vocabulary to use for retrieving drug events. Currently only supports "ATC".
        - columns: List of columns to select from the drug exposure table. The concept_id column corresponding to the vocabulary will be added automatically.
        - retrieve_by: Whether to retrieve drug events by ATC code or name. Must be either "atc_code" or "name".
        """
        self.db = db
        self.columns = columns
        self.vocabulary_id = vocabulary_id
        self.retrieve_by = retrieve_by

        if vocabulary_id == "ATC":
            self.concept_col = "drug_class_concept_id"
        else:
            raise ValueError("vocabulary_id must be 'ATC' ")

        if administration_or_prescription == "administration":
            self.drug_table = "drug_exposure_administration"
        elif administration_or_prescription == "prescription":
            self.drug_table = "drug_exposure_prescription"
        else:
            raise ValueError(
                "administration_or_prescription must be either 'administration' or 'prescription'"
            )

        if self.retrieve_by == "atc_code":
            self.column_to_lookup = "concept_code"
        elif self.retrieve_by == "name":
            self.column_to_lookup = "concept_first_name"
        else:
            raise ValueError("retrieve_by must be either 'atc_code' or 'name'")

        self.columns.append(self.concept_col)

    def __call__(
        self, concepts: List[Union[int, str, pd.DataFrame]], df_column: str = "ATC"
    ):
        drug = read_table(self.drug_table, db=self.db).select(self.columns)
        concept_table = read_table("concept", db=self.db)
        concept_table = concept_table.filter(
            F.col("vocabulary_id") == self.vocabulary_id
        )
        concept_table = concept_table.select(
            ["concept_id", "concept_code", "concept_name"]
        )

        if self.retrieve_by == "name":
            concept_table = concept_table.withColumn(
                "concept_split", F.split(F.col("concept_name"), ";")
            )
            concept_table = concept_table.withColumn(
                self.column_to_lookup, F.element_at(F.col("concept_split"), 1)
            )
            concept_table = concept_table.drop("concept_split")

        if isinstance(concepts, int):
            concept_f = concept_table.filter(F.col(self.column_to_lookup) == concepts)
        elif isinstance(concepts, list):
            concept_f = concept_table.filter(
                F.col(self.column_to_lookup).isin(concepts)
            )
        else:
            # Cast these codes to spark
            spark, _ = get_spark_sql()
            concepts_dict = concepts.to_dict(orient="records")
            codes_spark = spark.createDataFrame(concepts_dict)
            codes_spark = codes_spark.withColumnRenamed(
                df_column, self.column_to_lookup
            )

            # Get procedures of these codes (should have column name code)
            concept_f = concept_table.join(
                codes_spark.hint("broadcast"),
                on=self.column_to_lookup,
                how="inner",
            )

        concept_f = concept_f.drop_duplicates(subset=["concept_id"])
        concept_f = concept_f.withColumnRenamed("concept_id", self.concept_col)

        drug_events = drug.join(
            concept_f.hint("broadcast"),
            on=self.concept_col,
            how="inner",
        )

        return drug_events


class OncologyDrugs:
    def __init__(
        self,
        path=f"/export/home/{USER}/wedsak/data/drugs_onco.csv",
        normalized_columns=None,
        treatement_types=[
            "Chimiothérapie",
            "Thérapie ciblée",
            "Immunothérapie",
            "Médicaments antihormonaux utilisés en oncologie",
        ],
    ):
        """
        Class to process oncology drugs data.

        Parameters
        - path: Path to the CSV file containing oncology drugs data.
        - normalized_columns: Dictionary mapping column names to their normalized versions.
        - treatement_types: List of treatment types to include.
            One of {"Chimiothérapie", "Thérapie ciblée", "Immunothérapie", "Médicaments antihormonaux utilisés en oncologie"}
        """
        self.path = path
        self.normalized_columns = normalized_columns or dict(
            atc={"original_name": "atcnm_e", "normalized_name": "normalized_atc"},
            commercial={
                "original_name": "medicament",
                "normalized_name": "normalized_commercial_drug",
            },
            active_substance={
                "original_name": "substance_active",
                "normalized_name": "normalized_active_substance",
            },
        )
        self.treatement_types = treatement_types

        # Mapping between treatement types and their corresponding SNOMED CT concept IDs
        # retained for this project. They correspond to Procedure concepts of the SNOMED CT vocabulary.
        self.mapping_labels_sctid_snomed = {
            "Chimiothérapie": 367336001,
            "Thérapie ciblée": 1255831008,
            "Immunothérapie": 76334006,
            "Médicaments antihormonaux utilisés en oncologie": 169413002,
        }

    @staticmethod
    def normalize_column(df, column_name: str):

        series = (
            df[column_name]
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.lower()
        )
        return series

    def process(self):
        drugs_onco = pd.read_csv(self.path)
        drugs_onco = drugs_onco.replace({np.nan: None})
        for col in self.normalized_columns.values():
            drugs_onco[col["normalized_name"]] = self.normalize_column(
                drugs_onco, col["original_name"]
            )
        drugs_onco = drugs_onco.rename(
            columns={
                "macro_class": "treatement_type",
            }
        )

        drugs_onco["SNOMED CT - SCTID"] = drugs_onco["treatement_type"].replace(
            self.mapping_labels_sctid_snomed
        )

        return drugs_onco

    def get_dict_by_treatement_type(
        self,
        label_types=["atc", "active_substance", "commercial"],
    ):
        drugs_onco = self.process()
        dict_by_treatment_type = {}
        dict_by_treatment_type["all"] = dict(all=set())

        for treatement_type in self.treatement_types:
            sub_df = drugs_onco[drugs_onco["treatement_type"] == treatement_type]

            dict_by_treatment_type[treatement_type] = {
                label_type: set(
                    sub_df[
                        self.normalized_columns[label_type]["normalized_name"]
                    ].dropna()
                )
                for label_type in label_types
            }

            set_all = set()
            for key in label_types:
                set_all.update(dict_by_treatment_type[treatement_type][key])
            dict_by_treatment_type[treatement_type]["all"] = set_all
            dict_by_treatment_type["all"]["all"].update(set_all)

        return dict_by_treatment_type

    def get_regex_by_treatement_type(
        self,
        label_types=["atc", "active_substance", "commercial"],
    ):
        """Get regex patterns for each treatment type and label type (including label 'all')."""
        dict_by_treatment_type = self.get_dict_by_treatement_type(
            label_types=label_types
        )
        regex_by_treatment_type = {}
        for treatement_type, labels in dict_by_treatment_type.items():
            regex_by_treatment_type[treatement_type] = {
                label_type: "(?i)(" + ")|(".join(labels[label_type]) + ")"
                for label_type in labels.keys()
            }

        return regex_by_treatment_type

    def get_table(
        self,
        **kwargs,
    ):
        drugs_onco = self.process()
        drugs_onco = drugs_onco[
            drugs_onco["treatement_type"].isin(self.treatement_types)
        ]

        return drugs_onco
