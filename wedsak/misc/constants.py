import os

USER = os.getenv("USER")
PATH_DATA_SCRATCH = f"/data/hdd/{USER}/"
PATH_DATABASE = f"file://{PATH_DATA_SCRATCH}/wedsak/cohort/"
PATH_DATABASE = f"file:///export/home/{USER}/cohort_work/"


COLUMN_MAPPING = {
    "condition": {
        "_encounterReferenceId": "visit_occurrence_id",
        "code__text": "condition_source_value",
        "id": "condition_occurrence_id",
        "meta__source": "cdm_source",
        "onsetDateTime": "condition_start_datetime",
        "_subjectReferenceId": "person_id",
        "extension___diagnosisType": "condition_status_source_value",
    },
    "procedure": {
        "_encounterReferenceId": "visit_occurrence_id",
        "code__text": "procedure_source_value",
        "id": "procedure_occurrence_id",
        "meta__source": "cdm_source",
        "performedDateTime": "procedure_datetime",
        "_subjectReferenceId": "person_id",
    },
}

VALUE_DAS = "DAS"
VALUE_DP = "DP"
VALUE_DR = "DR"

CLAIM_SOURCE_AREM = "https://aphp.fr/ig/fhir/eds/Endpoint/arem"
CLAIM_SOURCE_ORBIS = "https://dedalus.com/Orbis"
CLAIM_SOURCE_MAPPING = {"AREM": CLAIM_SOURCE_AREM, "ORBIS": CLAIM_SOURCE_ORBIS}

PATH_LF_REFERENCE = f"/export/home/{USER}/wedsak/data/LF_definition.xlsx"
PATH_TASKS = f"/export/home/{USER}/wedsak/data/LF_definition.xlsx"
PATH_EVALUATION_DATASET = (
    f"/export/home/{USER}/wedsak/data/annotation/dev/annotated_docs/"
)
