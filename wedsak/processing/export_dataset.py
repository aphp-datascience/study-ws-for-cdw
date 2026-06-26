import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import edsnlp
import pandas as pd
from edsnlp.core.stream import Stream
from edsnlp.data.converters import get_current_tokenizer

from wedsak.misc.data_wrangling import normalize_task_name
from wedsak.misc.getters import get_tasks, task_id_to_task_name
from wedsak.misc.logger_utils import setup_logger
from wedsak.misc.utils import hash_file_or_directory, save_json
from wedsak.processing.sampling import (
    DEFAULT_SCALE,
    ONLY_NOTES_WITH_POSITIVE_EXAMPLES,
    USE_WEIGHTED_DISTANCE,
    get_negative_examples,
)
from pret.store import load_store_snapshot
import json

USER = os.getenv("USER")


def export_dataset(
    entities: pd.DataFrame,
    documents: pd.DataFrame,
    task_names: List[str],
    path: Optional[str] = None,
    doc_attributes: List[str] = ["note_datetime", "note_id", "person_id", "organ"],
    extra_span_attributes: Optional[List[str]] = None,
    span_label: str = "date",
    batch_size: int = 100,
    overwrite: bool = True,
    span_setter: Optional[Dict] = None,
    output_format: str = "parquet",
) -> Stream:
    """
    Export a datataset from spans to a edsnlp Dataset.

    Parameters
    ----------
    entities : pd.DataFrame
        DataFrame containing the entities with columns: 'note_id', 'start', 'end'.
    documents : pd.DataFrame
        DataFrame containing the documents with columns: 'note_id', 'note_text', etc.
    task_names : List[str]
        List of task names for span attributes.
    path : Optional[str], default=None
        Path to save the dataset as a parquet file. If None, the dataset will not be saved.
    doc_attributes : List[str], default=["note_datetime", "note_id", "person_id", "organ"]
        List of document attributes to include.
    span_label : str, default="date"
        Label for the spans in the dataset.
    batch_size : int, default=100
        Batch size for writing the dataset.
    overwrite : bool, default=True
        Whether to overwrite existing files.

    Returns
    -------
    edsnlp.data.Dataset
        The exported dataset ready for use in edsnlp.
    """
    entities = entities.rename(
        columns={
            "start": "start_char",
            "end": "end_char",
            # "label": "note_nlp_source_value",
        }
    )
    entities["note_nlp_source_value"] = span_label
    span_labels = [span_label]

    print("Number of entities:", len(entities))

    grouped_ents = entities.groupby("note_id").apply(
        lambda x: x.to_dict(orient="records"), include_groups=False
    )

    grouped_ents = pd.DataFrame(grouped_ents, columns=["entities"])
    print("Number of documents with entities:", len(grouped_ents))

    documents = documents.merge(
        grouped_ents,
        left_on="note_id",
        right_index=True,
        how="inner",
        validate="one_to_many",
    )
    print("Number of documents after merge:", len(documents))

    tokenizer = get_current_tokenizer()

    if span_setter is None:
        span_setter = {task_name: span_labels for task_name in task_names}

    # span_attributes = {task_name: task_name for task_name in task_names}
    span_attributes = task_names
    if extra_span_attributes is not None:
        # for attr in extra_span_attributes:
        #     span_attributes[attr] = attr
        span_attributes.extend(extra_span_attributes)

    docs = edsnlp.data.from_pandas(
        documents,
        converter="omop",
        tokenizer=tokenizer,
        doc_attributes=doc_attributes,
        span_setter=span_setter,
        span_attributes=span_attributes,
    )

    if path is not None:
        purepath = Path(
            path,
        ).expanduser()
        if output_format == "parquet":
            _ = edsnlp.data.write_parquet(
                docs,
                purepath,
                converter="omop",
                span_attributes=span_attributes,
                overwrite=overwrite,
                batch_size=batch_size,
                span_getter=span_setter,
            )
        elif output_format == "jsonl":
            _ = edsnlp.data.write_json(
                docs,
                purepath,
                converter="omop",
                span_attributes=span_attributes,
                overwrite=overwrite,
                span_getter=span_setter,
                lines=True,
            )
        elif output_format == "json":
            _ = edsnlp.data.write_json(
                docs,
                purepath,
                converter="omop",
                span_attributes=span_attributes,
                overwrite=overwrite,
                span_getter=span_setter,
                lines=False,
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        print("Dataset exported to", purepath)

    return docs


def propagate_annotations_to_hierarchy(df, replications):
    print("Propagating annotations to hierarchy with the following cases:")
    for case in replications:
        to_replace = case.get("to_replace")
        to_add = case.get("to_add")
        to_replace.append(to_add)
        print(to_replace, to_add)
        df.loc[:, to_add] = df[to_replace].any(axis=1)
    return df


def process_annotations_to_doc(
    path_annotation: Union[str, List[str]],
    date_labels: List[str] = ["date", "duration", "period"],
    positive_value="1",
    fillna_value="0",
    replications: Optional[List[Dict[str, List[str]]]] = None,
    path_export: Optional[str] = None,
    span_label: str = "date",
    context_window: int = 100,
    subset_notes_id: Optional[List[str]] = None,
    annotator_priority: Optional[List[str]] = ["EK", "AC"],
    **kwargs,
):
    """Process annotations for a specific group and export them to a dataset.

    Parameters
    ----------
    path_annotation : Union[str, List[str]]
        Path to the annotation file(s).
    date_labels : List[str], default=["date", "duration", "period"]
        List of labels that are considered as dates.
    positive_value : str, default="1"
        Value to assign to positive date labels.
    fillna_value : str, default="0"
        Value to fill NaN entries in the dataset.
    replications : Optional[List[Dict[str, List[str]]]], default=None
        List of replication cases for propagating annotations to hierarchy.
    path_docs : Optional[str], default=None
        Path to the documents file. If None, the dataset will not be merged with documents.
    path_export : Optional[str], default=None
        Path to save the exported dataset. If None, the dataset will not be saved.
    span_label : str, default="date"
        Label for the spans in the dataset.
    **kwargs : dict
        Additional keyword arguments for `export_dataset`.

    Returns
    -------
    edsnlp.data.Dataset
        The processed dataset ready for use in edsnlp.
    """

    if isinstance(path_annotation, str):
        path_annotation = [{"path": path_annotation}]

    all_annotations = []
    for couple_path_annotator in path_annotation:
        path = Path(couple_path_annotator.get("path")).expanduser()
        annotator = couple_path_annotator.get("annotator", "unknown")
        if path.suffix == ".json":
            with open(path, "r") as f:
                annotations = json.load(f)
        else:
            snapshot = load_store_snapshot(path)
        snapshot["annotator"] = annotator
        for note in snapshot["notes"]:
            note["annotator"] = annotator
        all_annotations.append(snapshot)

    # Merge all annotation files
    annotations = {"notes": [note for ann in all_annotations for note in ann["notes"]]}
    print(f"Total number of annotations loaded: {len(annotations['notes'])}")
    if subset_notes_id is not None:
        annotations["notes"] = [
            note for note in annotations["notes"] if note["note_id"] in subset_notes_id
        ]

        print(f"Number of notes in subset list: {len(subset_notes_id)}")
        print(f"Number of annotations after subsetting: {len(annotations['notes'])}")

    ents = []
    dates = []
    documents = []
    for doc in annotations["notes"]:
        if doc.get("seen", False):
            documents.append(
                {
                    "note_id": doc["note_id"],
                    "note_text": doc["note_text"],
                    "annotator": doc["annotator"],
                }
            )
            for ent in doc["entities"]:
                ent["note_id"] = doc["note_id"]
                ent["annotator"] = doc["annotator"]
                if ent["label"] in date_labels:
                    ent["context"] = doc["note_text"][
                        (ent["begin"] - context_window) : (ent["end"] + context_window)
                    ]
                    dates.append(ent)
                else:
                    ents.append(ent)

    ents = pd.DataFrame(ents)
    dates = pd.DataFrame(dates)
    documents = pd.DataFrame(documents)

    # Select prefered annotation for documents with multiple annotators based on annotator_priority
    count_annotator_by_doc = documents.groupby("note_id")["annotator"].nunique()
    id_docs_multi_annot = count_annotator_by_doc[
        count_annotator_by_doc > 1
    ].index.tolist()

    if id_docs_multi_annot and annotator_priority:
        priority_map = {name: idx for idx, name in enumerate(annotator_priority)}
        multi_docs = documents[documents["note_id"].isin(id_docs_multi_annot)].copy()
        multi_docs["_priority"] = (
            multi_docs["annotator"].map(priority_map).fillna(len(priority_map))
        )
        preferred_by_note = (
            multi_docs.sort_values(["note_id", "_priority"])
            .drop_duplicates(subset=["note_id"], keep="first")
            .set_index("note_id")["annotator"]
        )

        def _filter_by_preferred(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            mask_multi = df["note_id"].isin(id_docs_multi_annot)
            preferred = df["note_id"].map(preferred_by_note)
            return pd.concat(
                [df[~mask_multi], df[mask_multi & (df["annotator"] == preferred)]],
                ignore_index=True,
            )

        documents = _filter_by_preferred(documents)
        ents = _filter_by_preferred(ents)
        dates = _filter_by_preferred(dates)

    # Describe docs
    n_seen = documents.note_id.nunique()
    print(f"Number of seen documents: {n_seen} in path {path_annotation}")

    # Dates
    dates_ent = merge_highlight_annotations_with_dates(highlights=ents, dates=dates)
    dates_ent["value"] = True
    dates_ent_pivot = dates_ent.pivot(
        index=["note_id", "begin", "end"],
        columns="label",
        values="value",
    )

    # Ensure all task_names are in dates_ent_pivot
    tasks = get_tasks()
    task_names_fr = tasks["Task Name (date de)"].tolist()
    for task_name in task_names_fr:
        if task_name not in dates_ent_pivot.columns:
            dates_ent_pivot[task_name] = None

    # Fill NaN / None with False
    dates_ent_pivot = dates_ent_pivot.fillna(False, inplace=False)

    # Propagate annotations to hierarchy
    if replications is not None:
        dates_ent_pivot = propagate_annotations_to_hierarchy(
            dates_ent_pivot, replications
        )

    # Rename columns to match task names
    task_names = list(tasks.normalized_task_name)
    ref_col_names = {
        name: new_name
        for name, new_name in zip(
            tasks["Task Name (date de)"], tasks["normalized_task_name"]
        )
    }
    dates_ent_pivot = dates_ent_pivot.rename(columns=ref_col_names)
    dates_ent_pivot = dates_ent_pivot.reindex(columns=task_names)
    dates_ent_pivot.reset_index(drop=False, inplace=True)

    dates_ent_pivot.columns.name = ""

    # Merge with dates to have all spans and their corresponding labels (if any)
    dates_with_multilabels = dates.merge(
        dates_ent_pivot,
        on=["note_id", "begin", "end"],
        how="left",
        validate="one_to_one",
    )
    dates_with_multilabels = dates_with_multilabels.fillna(False, inplace=False)

    dates_with_multilabels = dates_with_multilabels.rename(
        columns={
            "begin": "start_char",
            "end": "end_char",
        }
    )

    dates_with_multilabels.replace(
        {True: positive_value, False: fillna_value}, inplace=True
    )
    assert len(dates_with_multilabels) == len(dates)

    ###################
    docs = export_dataset(
        dates_with_multilabels,
        documents,
        task_names=task_names,
        span_setter={"dates": [span_label]},
        path=path_export,
        span_label=span_label,
        **kwargs,
    )
    return docs, dates_with_multilabels


def data_selection_pipeline(
    path_train_labels: str,
    task_id: int,
    path_docs_train: str,
    path_train_dataset: str = f"~/scratch/{USER}/wedsak_datasets/train/",
    max_number_of_positive_examples: Optional[int] = 1000,
    prob_thresholds: Tuple[float, float] = (0.5, 0.5),
    seed: int = 123,
    sampling_params: dict = {},
    negative_on_positive_ratio: Optional[float] = 1.0,
    strategy: Optional[str] = "subsampling",
    log_level: str = "INFO",
    config_meta=None,
    **kwargs,
):
    """Data selection pipeline for a specific task.

    Parameters
    ----------
    path_train_labels : str
        Path to the training labels file.
    task_id : int
        Task ID for which to perform data selection.
    path_docs_train : str
        Path to the training documents file.
    path_train_dataset : str, default="~/scratch/{USER}/wedsak_datasets/train/"
        Path to save the training dataset.
    max_number_of_positive_examples : int, default=1000
        Maximum number of positive examples to sample.
    prob_thresholds : Tuple[float, float], default=(0.5, 0.5)
        Probability thresholds for positive and negative examples.
    seed : int, default=123
        Random seed for sampling.
    sampling_params : dict, default={}
        Additional parameters for sampling negative examples.
        Addmissible keys are 'use_weight', 'scale', 'use_only_notes_with_positive_examples'.
        Default values are defined in wedsak.processing.sampling.py.
        These are: use_weight=True, scale=500, use_only_notes_with_positive_examples=True.
    negative_on_positive_ratio : Optional[float], default=1.0
        Ratio of negative to positive examples.
    strategy : Optional[str], default="subsampling"
        Strategy for balancing the dataset ("subsampling" or "oversampling" or None).
        If None, no balancing will be performed and all examples will be used.
    log_level : str, default="INFO"
        Logging level.
    config_meta : dict, default=None
        Configuration metadata.
    """
    ## Config
    config = config_meta["resolved_config"].serialize()

    ## Setup logger
    logger = setup_logger(log_level, script_name="data_selection")
    logger.info("Data selection started")
    logger.info(f"Sampling params: {sampling_params}")
    logger.info(f"Strategy: {strategy}")

    # Get task name
    task_name = task_id_to_task_name(task_id)
    normalized_task_name = normalize_task_name(task_name)
    # Get column names
    class1 = f"{normalized_task_name}_1"

    # Sample positive examples
    path_train_labels = Path(path_train_labels).expanduser()
    labels_train: pd.DataFrame = pd.read_pickle(path_train_labels)
    positive_examples = labels_train.loc[labels_train[class1] > prob_thresholds[1]]
    if (max_number_of_positive_examples is not None) and (
        len(positive_examples) > max_number_of_positive_examples
    ):
        positive_examples = positive_examples.sample(
            n=max_number_of_positive_examples, random_state=seed
        )

    # Determine number of negative examples to sample
    if (negative_on_positive_ratio is not None) and (strategy == "subsampling"):
        max_n_negative = int(negative_on_positive_ratio * len(positive_examples))

    else:
        max_n_negative = None  # take all possible negatives

    # Sample negative examples
    negative_examples = get_negative_examples(
        labels_train,
        positive_class_col_name=class1,
        random_state=0,
        use_weight=sampling_params.get("use_weight", USE_WEIGHTED_DISTANCE),
        n=max_n_negative,
        prob_thresholds=prob_thresholds,
        scale=sampling_params.get("scale", DEFAULT_SCALE),
        use_only_notes_with_positive_examples=sampling_params.get(
            "use_only_notes_with_positive_examples", ONLY_NOTES_WITH_POSITIVE_EXAMPLES
        ),
    )

    if (negative_on_positive_ratio is not None) and (strategy == "oversampling"):
        # Calculate number of positive examples to sample
        n_positive = int(negative_on_positive_ratio * len(negative_examples))

        # over sample positive examples
        print(
            "Number of positive examples (before oversampling):", len(positive_examples)
        )
        positive_examples = positive_examples.sample(
            n=n_positive, random_state=seed, replace=True
        )

    # ### Export dataset for training
    print("Number of positive examples:", len(positive_examples))
    print(positive_examples[class1].value_counts())

    print("Number of negative examples:", len(negative_examples))
    print(negative_examples[class1].value_counts())

    # Concatenate positive & negative examples
    COLS = [
        "note_id",
        "start",
        "end",
        "lexical_variant",
        normalized_task_name,  # TODO add span attribute with hard label
    ]
    entity_dataset = pd.concat([positive_examples[COLS], negative_examples[COLS]])

    # Add instance_id
    entity_dataset.reset_index(drop=True, inplace=True)
    entity_dataset.reset_index(inplace=True, drop=False, names="instance_id")

    # Import train documents
    path_docs_train = Path(path_docs_train).expanduser()
    if path_docs_train.suffix == ".csv":
        train_documents = pd.read_csv(path_docs_train)
    else:
        train_documents = pd.read_pickle(
            path_docs_train,
        )

    # Make path task specific
    path_train_dataset_task = Path(
        path_train_dataset, normalized_task_name
    ).expanduser()
    path_train_dataset_task.parent.mkdir(parents=True, exist_ok=True)
    print("Path to train dataset (task-specific):", path_train_dataset_task)

    # Export dataset
    docs = export_dataset(
        entity_dataset,
        train_documents,
        task_names=[normalized_task_name],
        path=path_train_dataset_task,
        span_label="train_"
        + normalized_task_name,  # TODO add span attribute with hard label
        extra_span_attributes=["instance_id"],
    )

    # Save config & hash dataset
    metadata = {
        "config": config,
        "task_id": task_id,
        "task_name": task_name,
        "normalized_task_name": normalized_task_name,
        "train_labels_hash": hash_file_or_directory(path_train_labels),
        "train_dataset_hash": hash_file_or_directory(path_train_dataset_task),
        "train_dataset_n_documents": entity_dataset.note_id.nunique(),
        "train_dataset_n_entities": entity_dataset.instance_id.nunique(),
        "train_dataset_n_positive_examples": len(positive_examples),
        "train_dataset_n_negative_examples": len(negative_examples),
        "strategy": strategy,
    }
    save_json(metadata, sub_folder="data_selection")

    return docs


def export_dataset_multiple_tasks(
    path_train_labels: str,
    task_ids: List[int],
    path_docs_train: str,
    path_train_dataset: Optional[str] = None,
    strategy: Optional[str] = "one_dataset_for_all_tasks",
    log_level: str = "INFO",
    GENERAL_COLS=[
        "person_id",
        "note_id",
        "lexical_variant",
        "label",
        "start",
        "end",
        "context",
    ],
    span_label="train_date",
    span_setter="train_date",
    suffix_task_name: Optional[str] = None,
    config_meta=None,
    **kwargs,
):
    ## Config
    config = config_meta["resolved_config"].serialize() if config_meta else {}

    ## Setup logger
    logger = setup_logger(
        log_level, script_name="data_selection", route_stdout_stderr_to_logger=False
    )
    logger.info("Data selection started")
    logger.info(f"Strategy: {strategy}")

    # Merge labels for all tasks
    tasks = get_tasks()
    task_names = tasks.loc[
        tasks.task_id.isin(task_ids), "normalized_task_name"
    ].tolist()

    labels = []
    for task_id in task_ids:
        normalized_task_name = task_id_to_task_name(task_id, normalized=True)
        labels_task = pd.read_pickle(path_train_labels.format(task_id=task_id))
        if suffix_task_name is not None:
            labels_task = labels_task.rename(
                columns={
                    f"{normalized_task_name}{suffix_task_name}": normalized_task_name
                }
            )
        labels_task = labels_task[GENERAL_COLS + [normalized_task_name]]
        labels_task.replace({True: "1", False: "0"}, inplace=True)
        # logger.info(f"{normalized_task_name} - lenght: {len(labels_task)}")
        labels.append(labels_task)

    entity_dataset = labels[0]

    for tmp in labels[1:]:
        entity_dataset = entity_dataset.merge(tmp, on=GENERAL_COLS, how="left")

    logger.info(f"Entity dataset lenght {len(entity_dataset)}")

    entity_dataset.reset_index(drop=True, inplace=True)
    entity_dataset.reset_index(drop=False, inplace=True, names="instance_id")

    # Import train documents
    path_docs_train = Path(path_docs_train).expanduser()
    if path_docs_train.suffix == ".csv":
        train_documents = pd.read_csv(path_docs_train)
    else:
        train_documents = pd.read_pickle(
            path_docs_train,
        )

    # Return df if no path
    if path_train_dataset is None:
        return entity_dataset, train_documents

    path_train_dataset = Path(path_train_dataset).expanduser()
    path_train_dataset.parent.mkdir(parents=True, exist_ok=True)
    print("Path to train dataset (all tasks):", path_train_dataset)

    # Export dataset
    _ = export_dataset(
        entity_dataset,
        train_documents,
        task_names=task_names,
        path=path_train_dataset,
        span_label=span_label,
        extra_span_attributes=["instance_id"],
        span_setter=span_setter,
    )

    # Save config & hash dataset
    metadata = {
        "config": config,
        "task_ids": task_ids,
        "task_names": task_names,
        "train_labels_hash": [
            hash_file_or_directory(
                Path(path_train_labels.format(task_id=task_id)).expanduser()
            )
            for task_id in task_ids
        ],
        "train_dataset_hash": hash_file_or_directory(
            Path(path_train_dataset).expanduser()
        ),
        "train_dataset_n_documents": entity_dataset.note_id.nunique(),
        "train_dataset_n_entities": entity_dataset.instance_id.nunique(),
        "strategy": strategy,
    }
    save_json(metadata, sub_folder="data_selection")


def merge_highlight_annotations_with_dates(highlights, dates):
    dates_ent = dates.merge(
        highlights[["note_id", "label", "begin", "end"]],
        on="note_id",
        how="inner",
        suffixes=("_date", "_label"),
    )
    dates_ent = dates_ent.query(
        "(begin_label <= end_date) & (begin_date <= end_label) "
    ).copy()
    dates_ent.drop_duplicates(
        subset=["label_label", "note_id", "begin_date", "end_date"], inplace=True
    )
    dates_ent = dates_ent[
        [
            "begin_date",
            "end_date",
            "label_label",
            "note_id",
        ]
    ]
    dates_ent = dates_ent.rename(
        columns={
            "begin_date": "begin",
            "end_date": "end",
            "label_label": "label",
        }
    )
    dates_ent.reset_index(inplace=True, drop=True)
    dates_ent.reset_index(inplace=True, drop=False, names="uid")
    return dates_ent
