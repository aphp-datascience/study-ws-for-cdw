from typing import Optional

import pandas as pd
from pyspark.sql import functions as F

from wedsak.events.base import ClaimEventSelector
from wedsak.misc.data_wrangling import keep_one
from wedsak.misc.utils import read_table, build_path
from wedsak.misc.constants import (
    VALUE_DP,
    VALUE_DR,
    VALUE_DAS,
    CLAIM_SOURCE_AREM,
)


class INCaCancerStays(ClaimEventSelector):
    def __init__(
        self,
        db: Optional[str] = None,
        path_to_claim_codes: str = "../data/liste_w_steps_inca_algorithm.xlsx",
        path_to_localisation_codes: str = "../data/referentiels_localisation_INCa.xlsx",
        steps=[
            1,
            2,
            3,
            4,
        ],  # 5 is not included because of the absence of the Cost table (GHM) in the database.
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        col_date: str = "visit_start_datetime",
        claim_source: str = CLAIM_SOURCE_AREM,
        localisation_level: str = "organ",  # system
    ):
        super().__init__()
        self.db = db
        self.path_to_claim_codes = path_to_claim_codes
        self.path_to_localisation_codes = path_to_localisation_codes
        self.steps = steps
        self.after_date = after_date
        self.before_date = before_date
        self.col_date = col_date
        self.claim_source = claim_source
        self.localisation_level = localisation_level

    def process(self):
        path = build_path(__file__, self.path_to_claim_codes)
        df = pd.read_excel(
            path,
        )
        df.rename(columns={"Code": "code"}, inplace=True)
        list_1_non_radiotherapy_condition = df.query("Liste_1=='x'")[["code"]]
        list_1_radiotherapy_condition = df.query("Radioth=='x'")[["code"]]
        list_2 = df.query("Liste_2=='x'")[["code"]]
        list_3 = df.query("Liste_3=='x'")[["code"]]
        list_4d = df.query("Liste_4_D=='x'")[["code"]]
        list_4a = df.query("Liste_4_A=='x'")[["code"]]
        # list_5 = df.query("Liste_5=='x'")[["code"]]

        ########
        # Step 1a
        step_1a_conditions = self.get_conditions(
            icd10_codes=list_1_non_radiotherapy_condition,
            diagnostic_types=VALUE_DP,
            db=self.db,
            claim_source=self.claim_source,
        )

        step_1a_stays = step_1a_conditions.select(
            ["visit_occurrence_id"]
        ).drop_duplicates()

        step_1a_stays = step_1a_stays.withColumn("step_1a", F.lit(True))

        # Step 1b
        step_1b_conditions_dr = self.get_conditions(
            icd10_codes=list_1_non_radiotherapy_condition,
            diagnostic_types=VALUE_DR,
            db=self.db,
            claim_source=self.claim_source,
        ).select(["visit_occurrence_id"])

        step_1b_conditions_dp = self.get_conditions(
            icd10_codes=list_1_radiotherapy_condition,
            diagnostic_types=VALUE_DP,
            db=self.db,
            claim_source=self.claim_source,
        ).select(["visit_occurrence_id"])

        step_1b_conditions = step_1b_conditions_dp.join(
            step_1b_conditions_dr, on="visit_occurrence_id", how="inner"
        )

        step_1b_stays = step_1b_conditions.drop_duplicates()
        step_1b_stays = step_1b_stays.withColumn("step_1b", F.lit(True))

        # Step 1
        step_1_stays = step_1b_stays.join(
            step_1a_stays, on=["visit_occurrence_id"], how="outer"
        )

        step_1_stays = step_1_stays.withColumn(
            "step_1", F.coalesce(F.col("step_1a"), F.col("step_1b"))
        )

        ########
        # Step 2
        step_2_stays = (
            self.get_conditions(
                icd10_codes=list_2,
                diagnostic_types=VALUE_DR,
                db=self.db,
                claim_source=self.claim_source,
            )
            .select(["visit_occurrence_id"])
            .drop_duplicates()
        )
        step_2_stays = step_2_stays.withColumn("step_2", F.lit(True))

        ########
        # Step 3
        step_3_stays = (
            self.get_conditions(
                icd10_codes=list_3,
                diagnostic_types=VALUE_DAS,
                db=self.db,
                claim_source=self.claim_source,
            )
            .select(["visit_occurrence_id"])
            .drop_duplicates()
        )
        step_3_stays = step_3_stays.withColumn("step_3", F.lit(True))

        ########
        # Step 4
        step_4d_stays = (
            self.get_conditions(
                icd10_codes=list_4d,
                diagnostic_types=VALUE_DAS,
                db=self.db,
                claim_source=self.claim_source,
            )
            .select(["visit_occurrence_id"])
            .drop_duplicates()
        )
        step_4a_stays = (
            self.get_procedures(
                codes=list_4a,
                db=self.db,
                claim_source=self.claim_source,
            )
            .select(["visit_occurrence_id"])
            .drop_duplicates()
        )

        step_4_stays = step_4a_stays.join(
            step_4d_stays, on=["visit_occurrence_id"], how="inner"
        )

        step_4_stays = step_4_stays.withColumn("step_4", F.lit(True))

        ########
        # Step 5 # FIXME  Note : Table Cost (GHM) not available.
        # step_5_stays = (
        #     self.get_cost(
        #         codes=list_5,
        #         db=self.db,
        #         claim_source=CLAIM_SOURCE_ORBIS,  # self.claim_source, # FIXME
        #     )
        #     .select(["visit_occurrence_id"])
        #     .drop_duplicates()
        # )

        # step_5_stays = step_5_stays.withColumn("step_5", F.lit(True))

        ########
        # Join all steps
        stays_cancer = step_1_stays.join(
            step_2_stays, on=["visit_occurrence_id"], how="outer"
        )
        stays_cancer = stays_cancer.join(
            step_3_stays, on=["visit_occurrence_id"], how="outer"
        )
        stays_cancer = stays_cancer.join(
            step_4_stays, on=["visit_occurrence_id"], how="outer"
        )
        # stays_cancer = stays_cancer.join(
        #     step_5_stays, on=["visit_occurrence_id"], how="outer"
        # ) # FIXME  Note : Table Cost (GHM) not available.
        stays_cancer = stays_cancer.withColumn(
            "stay_cancer", F.greatest(*[F.col(f"step_{i}") for i in self.steps])
        )
        stays_cancer = stays_cancer.fillna(False)

        # Person
        stays = read_table(
            "visit_occurrence",
            db=self.db,
            select_cols=[
                "visit_occurrence_id",
                "person_id",
                "visit_start_datetime",
                "visit_end_datetime",
                "visit_source_value",
            ],
        )
        stays_cancer = stays_cancer.join(stays, on=["visit_occurrence_id"], how="left")

        return stays_cancer

    def add_localistaion(self, stays_cancer):
        path = build_path(__file__, self.path_to_localisation_codes)
        localisation = pd.read_excel(path, usecols=["diag", "Appareil", "Organe"])
        localisation.rename(
            columns={"diag": "code", "Organe": "organ", "Appareil": "system"},
            inplace=True,
        )

        conditions = self.get_conditions(
            icd10_codes=localisation,
            diagnostic_types=None,
            db=self.db,
            claim_source=self.claim_source,
        )
        conditions = conditions.select(
            [
                "visit_occurrence_id",
                "condition_source_value",
                "condition_status_source_value",
                "system",
                "organ",
            ]
        )

        conditions = conditions.join(
            stays_cancer.select(["visit_occurrence_id"]),
            on="visit_occurrence_id",
            how="inner",
        )

        conditions = conditions.withColumn(
            "weight_code",
            F.when(F.col("condition_status_source_value") == VALUE_DP, 3).otherwise(
                F.when(F.col("condition_status_source_value") == VALUE_DR, 2).otherwise(
                    1
                )
            ),
        )

        the_other_localisation_level = "system"
        if self.localisation_level == "system":
            the_other_localisation_level = "organ"

        conditions_gp = conditions.groupBy(
            ["visit_occurrence_id", self.localisation_level]
        ).agg(
            F.sum("weight_code").alias("weight_code"),
            F.first(the_other_localisation_level).alias(the_other_localisation_level),
        )

        conditions_stay = keep_one(
            conditions_gp,
            sort_column="weight_code",
            how="last",
            partition_by=["visit_occurrence_id"],
        ).select(["visit_occurrence_id", "system", "organ", "weight_code"])

        stays_cancer = stays_cancer.join(
            conditions_stay, how="left", on="visit_occurrence_id"
        )
        stays_cancer = stays_cancer.fillna(
            "Absence de localisation primitive",
            subset=["system", "organ"],
        )

        return stays_cancer

    def add_hospital_information(self, stays_cancer):
        hospitals = read_table(
            "care_site",
            db=self.db,
            select_cols=[
                "care_site_id",
                "care_site_name",
            ],
        )
        vo = read_table(
            "visit_occurrence",
            db=self.db,
            select_cols=["visit_occurrence_id", "care_site_id"],
        )
        stays_cancer = stays_cancer.join(vo, on="visit_occurrence_id", how="left")

        stays_cancer = stays_cancer.join(hospitals, on="care_site_id", how="left")

        return stays_cancer

    def __call__(self):
        stays = self.process()
        if self.after_date:
            stays = stays.filter(F.col(self.col_date) >= self.after_date)

        if self.before_date:
            stays = stays.filter(F.col(self.col_date) <= self.before_date)

        stays = self.add_localistaion(stays)
        stays = self.add_hospital_information(stays)
        return stays
