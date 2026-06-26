from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from confit import Cli
from typing_extensions import Annotated

from wedsak.events.base import ClaimEventSelector
from wedsak.events.imaging import ImagingEventSelector
from wedsak.events.notes import NoteEventSelector
from wedsak.events.drugs import OncologyDrugs, DrugEventSelector
from wedsak.misc.constants import PATH_LF_REFERENCE
from wedsak.misc.data_wrangling import collector, filter_by_person_set
from wedsak.misc.getters import get_tasks
from wedsak.misc.logger_utils import setup_logger

app = Cli()


@app.command(name="retrieve_external_knowledge")
def retrieve_external_knowledge(
    path_cohort: Annotated[str, typer.Option()],
    path_save_ek: Annotated[Optional[str], typer.Option()],
    path_codes_definition: Annotated[str, typer.Option()] = PATH_LF_REFERENCE,
    log_level: Annotated[str, typer.Option()] = "INFO",
    claim_source: Optional[str] = None,
    db: str = None,
):
    ## Setup logger
    logger = setup_logger(log_level, script_name="retrieve_external_knowledge")
    logger.info("Retrieve External Knowledge started")

    path_cohort = Path(path_cohort).expanduser()
    if path_save_ek is not None:
        path_save_ek = Path(path_save_ek).expanduser()
    path_codes_definition = Path(path_codes_definition).expanduser()

    print("Path Cohort:", path_cohort)
    print("Path External Knowledge:", path_save_ek)
    print("Path Codes Definition:", path_codes_definition)

    if path_cohort.suffix == ".csv":
        sample = pd.read_csv(path_cohort)
    else:
        sample = pd.read_pickle(
            path_cohort,
        )

    person_ids = list(sample.person_id)
    print("Number of lines in cohort:", len(sample))
    print("Number of unique persons:", sample.person_id.nunique())
    logger.info(f"Claim source {claim_source}")

    ##############################################################################
    # Retreive External knowledge

    ##############################################################################
    ## Medical Procedures (CCAM)
    ccam_codes = pd.read_excel(
        path_codes_definition, sheet_name="CCAM codes"
    )  # FIXME move this to ClaimEventSelector
    ccam_codes.rename(columns={"Code": "code"}, inplace=True)

    ccam = (
        ClaimEventSelector()
        .get_procedures(codes=ccam_codes, claim_source=claim_source, db=db)
        .select(
            [
                "person_id",
                "procedure_date",
                "Task Number",
                # "procedure_source_value",
                # "Label",
            ]
        )
    )

    # TODO move this to ClaimEventSelector
    ccam = ccam.drop_duplicates()
    ccam_cohort = filter_by_person_set(ccam, person_ids).toPandas()
    ccam_cohort.rename(columns={"procedure_date": "event_date"}, inplace=True)
    print("CCAM Task distribution:", ccam_cohort["Task Number"].value_counts())
    tasks_found = set(ccam_cohort["Task Number"])
    all_tasks = set(ccam_codes["Task Number"])
    difference = all_tasks.difference(tasks_found)
    print("Tasks not founded (procedure)", difference)

    ##############################################################################
    ## Diagnostics (ICD-10)
    # TODO move this to ClaimEventSelector
    icd10_codes = pd.read_excel(path_codes_definition, sheet_name="CIM10 codes")
    icd10_codes.rename(columns={"Code": "code"}, inplace=True)
    icd10 = (
        ClaimEventSelector()
        .get_conditions(icd10_codes=icd10_codes, claim_source=claim_source, db=db)
        .select(
            [
                "person_id",
                "condition_start_date",
                "Task Number",
                # "procedure_source_value",
                # "Label",
            ]
        )
    )
    # TODO move this to ClaimEventSelector
    icd10 = icd10.drop_duplicates()
    icd10_cohort = filter_by_person_set(icd10, person_ids).toPandas()
    icd10_cohort.rename(columns={"condition_start_date": "event_date"}, inplace=True)
    print("ICD10 Task distribution:", icd10_cohort["Task Number"].value_counts())

    ##############################################################################
    ## Clinical Note metadata
    cr_metadata = pd.read_excel(
        path_codes_definition,
        sheet_name="Clinical Reports metadata",
    )

    note_events = NoteEventSelector(db=db)(
        cr_metadata[["Task Number", "concept_id"]]
    ).drop("note_class_concept_id")
    note_events_cohort = (
        filter_by_person_set(note_events, person_ids).drop_duplicates().toPandas()
    )
    note_events_cohort.rename(columns={"note_date": "event_date"}, inplace=True)
    print(
        "Note Events Task distribution:",
        note_events_cohort["Task Number"].value_counts(),
    )

    ##############################################################################
    ## Imaging Metadata
    imaging_metadata = pd.read_excel(
        path_codes_definition,
        sheet_name="Imaging metadata",
    )

    # 'DICOM - CID33 - Modality' >> modality_concept_id
    # 'APHP - PACS - Modality' >> modality_source_concept_id

    imaging1 = ImagingEventSelector(vocabulary_id="DICOM - CID33 - Modality", db=db)(
        imaging_metadata[["Task Number", "concept_id"]]
    ).select(["person_id", "series_date", "Task Number"])

    imaging2 = ImagingEventSelector(vocabulary_id="APHP - PACS - Modality", db=db)(
        imaging_metadata[["Task Number", "concept_id"]]
    ).select(["person_id", "series_date", "Task Number"])

    imaging = imaging1.union(imaging2).drop_duplicates()
    imaging_events_cohort = (
        filter_by_person_set(imaging, person_ids).drop_duplicates().toPandas()
    )
    imaging_events_cohort.rename(columns={"series_date": "event_date"}, inplace=True)
    print(
        "Imaging Events Task distribution:",
        imaging_events_cohort["Task Number"].value_counts(),
    )

    ##############################################################################
    ## Drugs
    onco_drugs = OncologyDrugs().get_table()

    COLS_DRUGS = ["person_id", "drug_exposure_start_date", "SNOMED CT - SCTID"]

    drugs_admin_atc = (
        DrugEventSelector(db=db)(onco_drugs[["SNOMED CT - SCTID", "ATC"]])
        .select(COLS_DRUGS)
        .drop_duplicates()
    )

    drugs_presc_atc = (
        DrugEventSelector(db=db, administration_or_prescription="prescription")(
            onco_drugs[["SNOMED CT - SCTID", "ATC"]]
        )
        .select(COLS_DRUGS)
        .drop_duplicates()
    )

    drugs_presc_name = (
        DrugEventSelector(
            db=db, administration_or_prescription="prescription", retrieve_by="name"
        )(
            onco_drugs[["SNOMED CT - SCTID", "normalized_atc"]],
            df_column="normalized_atc",
        )
        .select(COLS_DRUGS)
        .drop_duplicates()
    )

    drugs_admin_name = (
        DrugEventSelector(
            db=db, administration_or_prescription="administration", retrieve_by="name"
        )(
            onco_drugs[["SNOMED CT - SCTID", "normalized_atc"]],
            df_column="normalized_atc",
        )
        .select(COLS_DRUGS)
        .drop_duplicates()
    )

    drugs = (
        drugs_admin_atc.union(drugs_presc_atc)
        .union(drugs_presc_name)
        .union(drugs_admin_name)
    )

    drugs_cohort = filter_by_person_set(drugs, person_ids).drop_duplicates().toPandas()
    drugs_cohort.rename(
        columns={"drug_exposure_start_date": "event_date"}, inplace=True
    )

    ## Add SNOMED name
    tasks = get_tasks()
    tasks.rename(columns={"task_id": "Task Number"}, inplace=True)

    drugs_cohort = drugs_cohort.merge(
        tasks[["Task Number", "SNOMED CT - SCTID"]],
        on="SNOMED CT - SCTID",
        how="inner",
    )
    drugs_cohort.drop(columns="SNOMED CT - SCTID", inplace=True)
    drugs_cohort = drugs_cohort[["person_id", "event_date", "Task Number"]]

    print(
        "Drugs Events Task distribution:",
        drugs_cohort["Task Number"].value_counts(),
    )

    ##############################################################################
    ## Concatenate all external Knowledge

    ccam_cohort["lf"] = "lf_ccam"
    icd10_cohort["lf"] = "lf_icd10"
    note_events_cohort["lf"] = "lf_note_md"
    imaging_events_cohort["lf"] = "lf_imaging_md"
    drugs_cohort["lf"] = "lf_drugs"
    ek = pd.concat(
        [
            ccam_cohort,
            icd10_cohort,
            note_events_cohort,
            imaging_events_cohort,
            drugs_cohort,
        ]
    )

    ## Add SNOMED name
    tasks = get_tasks()
    tasks.rename(
        columns={"task": "snomed_name", "task_id": "Task Number"}, inplace=True
    )
    tasks = tasks[["Task Number", "snomed_name"]]

    ek = ek.merge(tasks, on=["Task Number"], validate="many_to_one", how="left")

    assert not ek["snomed_name"].hasnans

    ## Event date cleaning
    print("External Knowledge event_date missing:", ek.event_date.isna().sum())
    ek.event_date = pd.to_datetime(ek.event_date, errors="coerce")
    ek.dropna(subset="event_date", inplace=True)
    ek.sort_values(by=["person_id", "event_date"], inplace=True)

    print("After dropna, event_date missing:", ek.event_date.isna().sum())
    print("Task distribution (SNOMED):", ek.snomed_name.value_counts())

    ##############################################################################
    ## Group by person_id and lf
    ek_grouped = ek.groupby(["person_id", "lf"]).apply(collector, include_groups=False)
    df_ek_grouped = ek_grouped.to_frame(name="ek").reset_index()
    df_ek_grouped = df_ek_grouped.pivot(index="person_id", columns="lf", values="ek")
    print("External Knowledge shape:", df_ek_grouped.shape)

    if path_save_ek is not None:
        path_save_ek.parent.mkdir(parents=True, exist_ok=True)
        df_ek_grouped.to_pickle(path_save_ek)
        logger.info("External Knowledge saved to: %s", path_save_ek)
    else:
        logger.warning("External Knowledge not saved.")
    return df_ek_grouped


if __name__ == "__main__":
    # typer.run(retrieve_external_knowledge)
    app()
