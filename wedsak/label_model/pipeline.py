import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from metal.analysis import lf_summary

from wedsak.misc.data_wrangling import normalize_task_name
from wedsak.misc.getters import (
    get_lf_reference,
    get_tasks,
    get_votes,
    get_votes_with_annotations,
)
from wedsak.processing.label_model import PreprocessLFVotesToLabelMatrix
from wedsak.misc.logger_utils import setup_logger
from sklearn.metrics import precision_recall_fscore_support
from wedsak.misc.utils import save_json, hash_file_or_directory
from wedsak.label_model.base import LabelModelProtocol
from wedsak.misc.constants import PATH_LF_REFERENCE, PATH_TASKS

pd.set_option("future.no_silent_downcasting", True)


def label_model_pipeline(
    task_id: int,
    # Path
    path_lf_votes_train: str,
    path_save_labels_train: str,
    path_lf_votes_dev: str,
    path_docs_dev: str,
    path_annotations_dev: str,
    # LM
    label_model: LabelModelProtocol,
    # Other paths
    path_lf_reference: str = PATH_LF_REFERENCE,
    path_tasks: str = PATH_TASKS,
    path_metrics_label_model: str = "~/wedsak/data/label_model_metrics/",
    # Parameters
    abstention_value: int = 0,
    negative_value: int = 1,
    positive_value: int = 2,
    multi_class_lfs: List[str] = [
        "lf_ccam",
        "lf_icd10",
        "lf_imaging_md",
        "lf_note_md",
        "lf_drugs",
    ],
    fillnone_as_negative: bool = True,
    min_support_dev: int = 15,
    avoid_all_abstentions: bool = True,
    rescale_probs: bool = False,
    log_level: str = "INFO",
    config_meta=None,
    **kwargs,
) -> pd.DataFrame:
    """Label Modelling for a specific task.

    Parameters
    ----------
    task_id : int
        Task ID to process.
    path_lf_votes_train : str, default="~/wedsak/data/datasets/note_nlp_train"
        Path to the LF votes for the training set.
    label_model : Union[MetalLabelModelWrapper, HyperLabelModelWrapper, MajorityVoteLabelModel]
        Label model to use for training and prediction.
    path_lf_reference : str, default=PATH_LF_REFERENCE
        Path to the Excel file containing LF definitions.
    path_tasks : str, default=PATH_TASKS
        Path to the Excel file containing task definitions.
    path_docs_dev : str, default="~/wedsak/data/annotation/dev/docs_screening.csv"
        Path to the documents for the development set.
    path_docs_train : str, default="~/wedsak/data/annotation/train/train_notes.pickle"
        Path to the documents for the training set.
    path_annotations_dev : str, default="~/wedsak/data/annotation/dev/dates_screening.csv"
        Path to the annotations for the development set.
    path_lf_votes_dev : str, default="~/wedsak/data/datasets/note_nlp_dev"
        Path to the LF votes for the development set.
    path_metrics_label_model : str, default="~/wedsak/data/label_model/"
        Path to save label model metrics.
    negative_value : int, default=1
        Value representing the negative class in LF votes.
    positive_value : int, default=2
        Value representing the positive class in LF votes.
    abstention_value : int, default=0
        Value representing abstention in LF votes.
    multi_class_lfs : list, default=["lf_ccam", "lf_icd10", "lf_imaging_md", "lf_note_md"]
        List of multi-class LFs that should not be treated as binary.
    avoid_all_abstentions: bool, default=True
        Whether to avoid documents with all abstentions in the label model training.
    rescale_probs : bool, default=True
        Whether to rescale probabilities if the maximum value is less than 0.5.

    Returns
    -------
    edsnlp.core.stream.Stream
        Stream containing the exported dataset for training.

    """
    ## Config
    config = config_meta["resolved_config"].serialize()

    ## Setup logger
    logger = setup_logger(log_level, script_name=f"label_model_{task_id}")
    logger.info("Label Model started")

    # ## Label modelling
    # Read LF votes for training set
    path_lf_votes_train = Path(path_lf_votes_train).expanduser()
    lf_votes_train = get_votes(path_lf_votes_train)

    # Read LF reference (only "Implemented=='ok'" LFs)
    ref_lfs, lf_names = get_lf_reference(path_lf_reference)

    # Read tasks
    tasks = get_tasks(path_tasks)
    # tasks

    # Get task name
    group_id = tasks.query("task_id==@task_id").Group.iloc[0]  # screening group
    task = tasks.query("task_id==@task_id").task.iloc[0]
    print("##########################")
    logger.info("--------------------------")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Task name: {task}")

    # Get LF for this task
    lf_names_for_task = list(ref_lfs.query("`Task Number`==@task_id").ref_name.unique())
    logger.warning(f"LF names for task: {lf_names_for_task}")

    # Specify all not multiclass LF as to consider None as Negative
    if fillnone_as_negative:
        lf_names_fillnone_as_negative = list(
            set(lf_names_for_task).difference(set(multi_class_lfs))
        )
    else:
        lf_names_fillnone_as_negative = None
    print("LF names (fill none as negative):", lf_names_fillnone_as_negative)

    # Read LF votes for development set and ground truth
    lf_votes_dev = get_votes_with_annotations(
        path_annotations=path_annotations_dev,
        path_votes=path_lf_votes_dev,
        task_id=task_id,
        positive_value=positive_value,
        negative_value=negative_value,
    )

    # Compute class balance for the development set
    class_balance_dev = lf_votes_dev.ground_truth_label.value_counts(
        normalize=True
    ).values
    logger.info(f"Class balance (dev): {class_balance_dev}")

    support_dev = int(sum(lf_votes_dev.ground_truth_label == positive_value))
    logger.info(f"Support dev: {support_dev}")
    if support_dev >= min_support_dev:
        class_balance_train = class_balance_dev

        if hasattr(label_model, "class_balance_train"):
            logger.warning("Class balance changed")
            label_model.class_balance_train = class_balance_train
            logger.info(
                f"New Class balance for training: {label_model.class_balance_train}"
            )
        else:
            logger.info("class_balance not set, no class balance for this label model")

    # ### Preprocess LF Votes & Label Matrix

    # Initialize the processor to conver LF votes to a Label Matrix
    lf_to_matrix = PreprocessLFVotesToLabelMatrix(
        task=task,
        lf_names_for_task=lf_names_for_task,
        lf_names_fillnone_as_negative=lf_names_fillnone_as_negative,
        multi_class_lfs=multi_class_lfs,
        negative_value=negative_value,
        positive_value=positive_value,
        abstention_value=abstention_value,
        avoid_all_abstentions=avoid_all_abstentions,
    )

    # Get L matrix (train)
    L_train = lf_to_matrix(lf_votes_train)

    # Get L matrix (dev)
    lf_to_matrix.avoid_all_abstentions = False
    L_dev = lf_to_matrix(lf_votes_dev)

    # Get Y (ground truth for development set)
    Y_dev = np.asarray(lf_votes_dev.ground_truth_label.values, dtype=int)

    # Estimate performance of LFs using the dev set
    summary_dev_set = lf_summary(
        L_dev,
        Y_dev,
        lf_names=lf_names_for_task,
        **{"pos_label": positive_value, "abstention_value": abstention_value},
    )
    summary_dev_set["Polarity"] = summary_dev_set["Polarity"].apply(
        lambda p: [x.item() for x in p] if isinstance(p, list) else p.item()
    )  # FIXME metal bug
    summary_dev_set = summary_dev_set.replace({np.nan: None})
    logger.info("LF Summary (dev):")
    print(L_dev.shape[0], "instances in dev set")
    print(summary_dev_set)

    # Estimate the coverage and overlaps
    summary_train = lf_summary(
        L_train, lf_names=lf_names_for_task, **{"pos_label": positive_value}
    )
    logger.info("LF Summary (train):")
    print(summary_train)
    # Apply the label model
    label_model.fit(L_train=L_train, **kwargs)
    _, preds_dev = label_model.predict_label(L_train=L_train, L_dev=L_dev)

    # Evaluate LM  model
    logger.debug("Unique values in Y_dev: %s", np.unique(Y_dev, return_counts=True))
    p_sklearn, r_sklearn, f1_sklearn, _ = precision_recall_fscore_support(
        y_true=Y_dev,
        y_pred=preds_dev,
        pos_label=positive_value,
        average="binary",
    )

    metrics_lm = {
        "support": support_dev,
        "fscore": f1_sklearn,
        "precision": p_sklearn,
        "recall": r_sklearn,
        "instances_dev": L_dev.shape[0],
    }
    logger.info(f"Metrics (dev): {metrics_lm}")

    # Get normalized task name
    normalized_task_name = normalize_task_name(task)
    path_metrics_label_model_file = Path(
        path_metrics_label_model, normalized_task_name + ".json"
    ).expanduser()
    if not path_metrics_label_model_file.parent.exists():
        path_metrics_label_model_file.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Creating directory: {path_metrics_label_model_file.parent}")
    logger.info(f"Path to metrics label model file: {path_metrics_label_model_file}")

    # Save metrics
    json.dump(metrics_lm, open(path_metrics_label_model_file, "w"))

    # ### Predict using LM model
    # Read LF votes for training set
    lf_votes_train = get_votes(path_lf_votes_train)
    # Get L matrix (train)
    lf_to_matrix.avoid_all_abstentions = False
    L_train = lf_to_matrix(lf_votes_train)

    # Get column names
    class0 = f"{normalized_task_name}_0"
    class1 = f"{normalized_task_name}_1"

    # Get probabilities using the LM
    probs_train, probs_dev = label_model.predict_probs(L_train=L_train, L_dev=L_dev)
    logger.info("### Probability distribution (dev): ###")

    probs_tmp, counts_tmp = np.unique(probs_dev[:, 1], return_counts=True)
    for p, c in zip(probs_tmp, counts_tmp):
        print(f"Probability: {p:.5f}, Count: {c}")

    # Check if probabilities are not in the right range
    if rescale_probs:
        max_value = probs_train[:, 1].max()
        if max_value < 0.5:
            logger.info(f"Probs max value is : {max_value}")
            logger.info(
                f"Old probs: {np.unique(probs_train[:, 1], return_counts=True)}"
            )
            logger.warning("## Rescaling probabilities")
            probs_train[:, 1] = probs_train[:, 1] / max_value
            probs_train[:, 0] = 1 - probs_train[:, 1]

    # Make dictionnaries of probs
    probs_train_list_dicts = [{"0": i, "1": j} for i, j in probs_train]

    # Assign to dataframe
    lf_votes_train[[class0, class1]] = probs_train
    lf_votes_train[normalized_task_name] = probs_train_list_dicts
    logger.info("### Probability distribution (train): ###")
    for p, c in lf_votes_train[class1].value_counts().sort_index().items():
        print(f"Probability: {p:.5f}, Count: {c}")

    count_positives = sum(lf_votes_train[class1] >= 0.5)
    count_negatives = sum(lf_votes_train[class1] < 0.5)
    logger.info(
        f"Number of positive labels (train) with threshold 0.5: {count_positives} / {lf_votes_train.shape[0]}"
    )
    COLS = [
        "person_id",
        "instance_id",
        "note_id",
        "lexical_variant",
        "label",
        "start",
        "end",
        "context",
        class0,
        class1,
        normalized_task_name,  # TODO add span attribute with hard label
    ]
    lf_votes_train = lf_votes_train[COLS]

    path_save_labels_train = Path(path_save_labels_train).expanduser()
    path_save_labels_train.parent.mkdir(parents=True, exist_ok=True)
    lf_votes_train.to_pickle(path_save_labels_train)
    logger.info(f"Path to saved labels (train): {path_save_labels_train}")

    # Save config & hash dataset
    metadata = {
        "config": config,
        "task_id": task_id,
        "task_name": task,
        "metrics_label_model": metrics_lm,
        "metrics_by_lf_dev_set": summary_dev_set.to_dict(),
        "count_positives_train": int(count_positives),
        "count_negatives_train": int(count_negatives),
        "train_votes_hash": hash_file_or_directory(path_lf_votes_train),
        "train_labels_hash": hash_file_or_directory(path_save_labels_train),
    }

    save_json(
        metadata,
        sub_folder=f"label_model/{normalized_task_name}",
    )

    return lf_votes_train
