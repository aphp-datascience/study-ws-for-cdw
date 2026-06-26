import os
import pandas as pd
from pathlib import Path
from typing import Optional, List
import json
from wedsak.misc.data_wrangling import normalize_task_name

from edsnlp.data.converters import get_current_tokenizer
from spacy.tokenizer import Tokenizer
from edsnlp.data import read_parquet

USER = os.getenv("USER")


def get_dataset(
    path: str,
    task_name: str,
    tokenizer: Optional[Tokenizer] = None,
    span_label: str = "date",
    extra_span_attributes: Optional[List[str]] = ["instance_id"],
):
    path = Path(path, task_name).expanduser()
    if tokenizer is None:
        tokenizer = get_current_tokenizer()

    span_labels = [span_label]

    span_attributes = {task_name: task_name}
    if extra_span_attributes is not None:
        for attr in extra_span_attributes:
            span_attributes[attr] = attr

    dataset = read_parquet(
        path,
        converter="omop",
        span_setter={task_name: span_labels},
        span_attributes=span_attributes,
        tokenizer=tokenizer,
    )

    return dataset


def get_metanno_annotations(path):
    docs = []
    patients = []
    for patient_path in Path(path).glob("*/metadata.json"):
        with open(patient_path) as f:
            patient_data = json.load(f)
        patients.append(patient_data)
        for doc_path in Path(patient_path).parent.glob("*.txt"):
            with open(doc_path) as f:
                txt = f.read()
            with open(str(doc_path).replace(".txt", ".json")) as f:
                try:
                    doc = json.load(f)
                except:  # noqa: E722
                    raise Exception(
                        "Could not read " + str(doc_path).replace(".txt", ".json")
                    )
                doc["note_text"] = txt
            # doc["note_datetime"] = datetime.strptime(doc["note_datetime"], "%Y-%m-%dT%H:%M:%S")
            docs.append(doc)
    return docs, patients


def get_annotated_entities(
    path_docs: str,
    path_annotations: str,
    tasks: pd.DataFrame,
    group_id: Optional[int] = None,
):
    """Get annotated entities from the given paths and tasks DataFrame.

    Parameters
    ----------
        path_docs (str): Path to the CSV file containing annotated documents.
        path_annotations (str): Path to the CSV file containing annotated entities.
        tasks (pd.DataFrame): DataFrame containing task information.
        group_id (Optional[int]): Optional screening group ID to filter the results.

    Returns:
        pd.DataFrame: DataFrame containing the annotated entities with relevant columns.
    """
    # Load annotated documents and entities
    path_docs = Path(path_docs).expanduser()
    path_annotations = Path(path_annotations).expanduser()

    if not path_docs.exists() or not path_annotations.exists():
        raise FileNotFoundError("One of the provided paths does not exist.")

    annotated_docs = pd.read_csv(path_docs)
    if path_docs.suffix == ".csv":
        annotated_docs = pd.read_csv(path_docs)
    else:
        annotated_docs = pd.read_pickle(
            path_docs,
        )

    annotated_docs = annotated_docs.query("seen==True")
    print("Number of annotated docs", len(annotated_docs))

    if group_id is not None:
        annotated_docs = annotated_docs.query("screening_group==@group_id")

        print(f"Number of annotated docs of group {group_id}", len(annotated_docs))
    annotated_note_ids = annotated_docs.note_id.unique()

    ## Process annotated ents
    annotated_ents = pd.read_csv(path_annotations)

    annotated_ents = annotated_ents.merge(
        tasks[["Task Name (date de)", "task_id"]],
        left_on="label_label",
        right_on="Task Name (date de)",
        validate="many_to_one",
        how="inner",
    )

    annotated_ents = annotated_ents.rename(
        columns={
            "begin_date": "start",
            "end_date": "end",
            "label_label": "ground_truth_task_label",
            "task_id": "ground_truth_task_id",
        }
    )
    annotated_ents = annotated_ents[
        [
            "note_id",
            "start",
            "end",
            "ground_truth_task_label",
            "ground_truth_task_id",
            "screening_group",
        ]
    ]
    print(
        "No annotated ents in total (selected & unselected docs)", len(annotated_ents)
    )

    if group_id is not None:
        annotated_ents = annotated_ents.query("screening_group==@group_id ")
    annotated_ents = annotated_ents.drop(columns=["screening_group"])
    print("No annotated ents in docs after group selection", len(annotated_ents))
    return annotated_ents, annotated_note_ids


def get_lf_reference(
    path: str = f"/export/home/{USER}/wedsak/data/LF_definition.xlsx",
    sheet_name="LFs",
    filter_lf: Optional[str] = "Implemented=='ok'",
) -> tuple[pd.DataFrame, list[str]]:
    """Get the reference for labeling functions (LFs) from an Excel file.

    Parameters
    ----------
    path : str, default="/export/home/{USER}/wedsak/data/LF_definition.xlsx"
        Path to the Excel file containing LF definitions.
    sheet_name : str, default="LFs"
        Name of the sheet in the Excel file to read.
    filter_lf : Optional[str], default="Implemented=='ok'"
        Filter condition to apply to the LFs DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A tuple containing:
            - DataFrame with reference LFs.
            - List of unique LF names.
    """
    path = Path(path).expanduser()

    # LF reference
    df_ref_lfs = pd.read_excel(path, sheet_name=sheet_name)

    if filter_lf is not None:
        df_ref_lfs = df_ref_lfs.query(filter_lf).copy()

    # df_ref_lfs.LF_id = df_ref_lfs.LF_id.astype("int")

    ref_lfs = df_ref_lfs[["Task Number", "ref_name", "LF type"]]  # "LF_id"
    lf_names = ref_lfs.ref_name.unique().tolist()
    tasks = get_tasks(path=path, sheet_name="Tasks")

    ref_lfs = ref_lfs.merge(
        tasks[["task_id", "normalized_task_name"]],
        left_on="Task Number",
        right_on="task_id",
        validate="many_to_one",
        how="left",
    )
    return ref_lfs, lf_names


def get_tasks(
    path: str = f"/export/home/{USER}/wedsak/data/LF_definition.xlsx",
    sheet_name: str = "Tasks",
):
    """Get the tasks from an Excel file.

    Parameters
    ----------
    path : str, default="/export/home/{USER}/wedsak/data/LF_definition.xlsx"
        Path to the Excel file containing task definitions.
    sheet_name : str, default="Tasks"
        Name of the sheet in the Excel file to read.

    Returns
    -------
    pd.DataFrame
        DataFrame containing task definitions with columns 'task_id' and 'task'.
    """

    path = Path(path).expanduser()
    tasks = pd.read_excel(path, sheet_name=sheet_name)

    tasks.rename(
        columns={
            "Task Number": "task_id",
            "SNOMED CT - Fully Specified Name (FSN)": "task",
        },
        inplace=True,
    )
    tasks["normalized_task_name"] = tasks.task.apply(normalize_task_name)
    return tasks


def get_votes(path_votes: str) -> pd.DataFrame:
    """Get LF votes from a pickle file.

    Parameters
    ----------
    path_votes : str
        Path to the pickle file containing LF votes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing LF votes.
    """
    path_votes = Path(path_votes).expanduser()

    if not path_votes.exists():
        raise FileNotFoundError(f"The file {path_votes} does not exist.")

    lf_votes = pd.read_pickle(path_votes).reset_index(drop=True)
    lf_votes.reset_index(drop=False, inplace=True, names="instance_id")

    return lf_votes


def get_votes_with_annotations(
    task_id: int,
    positive_value: int,
    negative_value: int,
    path_annotations: str,
    path_votes: str,
):
    """
    Get LF votes with annotated entities for a specific task.
    Parameters
    ----------
    task_id : int
        ID of the task for which to get LF votes.
    positive_value : int
        Value to assign for positive ground truth labels.
    negative_value : int
        Value to assign for negative ground truth labels.
    path_annotations : str
        Path to the pickle file containing annotated entities.
    path_votes : str
        Path to the pickle file containing LF votes.
    Returns
    -------
    pd.DataFrame
        DataFrame containing LF votes with annotated entities and ground truth labels.
    """
    # Read votes of LF
    votes = pd.read_pickle(path_votes)
    votes.note_id = votes.note_id.astype(str)

    # Read annotations
    annotations = pd.read_pickle(path_annotations)

    # Get annotations for the specified task
    normalized_task_name = normalize_task_name(task_id_to_task_name(task_id=task_id))
    COLS = ["text", "end_char", "start_char", "note_id", normalized_task_name]
    annotations = annotations[COLS]
    annotations = annotations.rename(
        columns={
            "start_char": "start",
            "end_char": "end",
            normalized_task_name: "ground_truth_label",
        }
    )

    # Replace annotation values with negative/positive values
    annotations["ground_truth_label"] = annotations["ground_truth_label"].replace(
        {"0": negative_value, "1": positive_value, 0: negative_value, 1: positive_value}
    )

    # Merge votes with annotations
    votes = votes.merge(
        annotations,
        on=["note_id", "start", "end"],
        validate="one_to_one",
        how="left",
    )

    # Eventually fillna with negative value (if no annotation, we consider it as negative)
    votes.fillna(
        {
            "ground_truth_label": negative_value,
        },
        inplace=True,
    )

    # Print some statistics
    print(votes.ground_truth_label.value_counts(dropna=False))

    return votes


def task_id_to_task_name(task_id: int, normalized=False) -> str:
    # Read tasks
    tasks = get_tasks()

    # Get task name
    if normalized:
        col = "normalized_task_name"
    else:
        col = "task"
    task_name = tasks.loc[tasks.task_id == task_id, col].iloc[0]
    return task_name


def normalized_task_name_to_task_id(task_name: str) -> int:
    # Read tasks
    tasks = get_tasks()

    # Get task id
    task_id = tasks.loc[tasks.normalized_task_name == task_name, "task_id"].iloc[0]
    return task_id


def get_task_mapping_to_fhir_coding():
    tasks = get_tasks()
    taks_mapping_df = tasks[
        ["normalized_task_name", "task", "SNOMED CT - SCTID"]
    ].copy()

    taks_mapping_df.rename(
        columns={
            "task": "code__coding___display",
            "SNOMED CT - SCTID": "code__coding___code",
        },
        inplace=True,
    )
    taks_mapping_df["code__coding___system"] = "https://www.snomed.org/"
    taks_mapping = taks_mapping_df.set_index("normalized_task_name").to_dict(
        orient="index"
    )
    return taks_mapping
