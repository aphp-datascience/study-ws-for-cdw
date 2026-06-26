from typing import Optional

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from pyspark.sql.window import Window

from wedsak.events.cancer_stays_inca import INCaCancerStays
from wedsak.misc.data_wrangling import keep_one
from wedsak.misc.constants import CLAIM_SOURCE_AREM


class CancerCohortSelector(INCaCancerStays):
    def __init__(
        self,
        db: str,
        path_to_claim_codes: str = "../data/liste_w_steps_inca_algorithm.xlsx",
        path_to_localisation_codes: str = "../data/referentiels_localisation_INCa.xlsx",
        steps=[1, 2, 3, 4, 5],
        first_date_after: Optional[str] = None,
        first_date_before: Optional[str] = None,
        claim_source: str = CLAIM_SOURCE_AREM,
        localisation_level: str = "organ",  # system
        main_localisation_types: Optional[str] = None,
        localisation_types: Optional[str] = None,
    ):
        super().__init__(
            db=db,
            path_to_claim_codes=path_to_claim_codes,
            path_to_localisation_codes=path_to_localisation_codes,
            steps=steps,
            after_date=None,
            before_date=None,
            claim_source=claim_source,
            localisation_level=localisation_level,
        )
        self.first_date_after = first_date_after
        self.first_date_before = first_date_before
        self.main_localisation_types = main_localisation_types
        self.localisation_types = localisation_types

    @staticmethod
    def aggregate_stays(
        stays: sparkDataFrame,
        localisation_level="organ",
        first_date_after: Optional[str] = None,
        first_date_before: Optional[str] = None,
        main_localisation_types: Optional[str] = None,
        localisation_types: Optional[str] = None,
    ):
        windowSpec1 = Window.partitionBy(
            ["person_id", "visit_source_value", localisation_level]
        )
        stays = stays.withColumn(
            "weight_type_stay",
            F.count("*").over(windowSpec1),
        )

        stays = stays.withColumn(
            "weight",
            F.when(
                F.col("visit_source_value") == "hospitalisation incomplète",
                F.col("weight_code") / F.col("weight_type_stay"),
            ).otherwise(F.col("weight_code")),
        )

        windowSpec2 = Window.partitionBy(["person_id", localisation_level])
        stays = stays.withColumn(
            "weight_localisation",
            F.sum("weight").over(windowSpec2),
        )
        stays = stays.withColumn(
            "localisation_first_date",
            F.min("visit_start_datetime").over(windowSpec2),
        )
        stays = stays.withColumn(
            "n_occurrences_localisation",
            F.count("*").over(windowSpec2),
        )

        stays = stays.withColumn(
            "targeted_localisations",
            F.col(localisation_level).isin(localisation_types),
        )

        patient_level = keep_one(
            stays,
            sort_column="weight_localisation",
            how="last",
            partition_by=["person_id"],
        )
        stays_gp = stays.groupBy("person_id").agg(
            F.countDistinct("organ").alias("n_distinct_organ"),
            F.countDistinct("system").alias("n_distinct_system"),
            F.collect_set("organ").alias("all_organs"),
            F.collect_set("system").alias("all_systems"),
            F.max("targeted_localisations").alias("has_targeted_localisation"),
        )
        cohort = patient_level.join(stays_gp, on="person_id", how="left")

        if first_date_after:
            cohort = cohort.filter(F.col("localisation_first_date") >= first_date_after)

        if first_date_before:
            cohort = cohort.filter(
                F.col("localisation_first_date") <= first_date_before
            )

        if main_localisation_types is not None:
            cohort = cohort.filter(
                F.col(localisation_level).isin(main_localisation_types)
            )

        if localisation_types is not None:
            cohort = cohort.filter(F.col("has_targeted_localisation"))

        cohort = cohort.select(
            [
                "person_id",
                "system",
                "organ",
                "localisation_first_date",
                "n_distinct_organ",
                "n_distinct_system",
                "all_organs",
                "all_systems",
                "care_site_id",
                "care_site_name",
            ]
        )
        return cohort

    def __call__(self):
        stays = super().__call__()
        cohort = self.aggregate_stays(
            stays,
            self.localisation_level,
            self.first_date_after,
            self.first_date_before,
            main_localisation_types=self.main_localisation_types,
            localisation_types=self.localisation_types,
        )
        return cohort
