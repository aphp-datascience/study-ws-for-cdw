import copy
from typing import List

import pandas as pd
from confit import Cli

from wedsak.label_model.base import LabelModelProtocol
from wedsak.label_model.pipeline import label_model_pipeline
from wedsak.registry import registry
from wedsak.misc.constants import PATH_LF_REFERENCE, PATH_TASKS

pd.set_option("future.no_silent_downcasting", True)


app = Cli()


@app.command(name="label_model", registry=registry)
def label_model(
    task_ids: List[int],
    # Path
    path_lf_votes_train: str,
    path_save_labels_train: str,
    path_lf_votes_dev: str,
    path_docs_dev: str,
    path_annotations_dev: str,
    # LM
    label_model_protocol: LabelModelProtocol,
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
):
    for task_id in task_ids:
        label_model_pipeline(
            task_id=task_id,
            path_lf_votes_train=path_lf_votes_train,
            label_model=copy.deepcopy(label_model_protocol),
            path_lf_reference=path_lf_reference,
            path_tasks=path_tasks,
            path_docs_dev=path_docs_dev,
            path_annotations_dev=path_annotations_dev,
            path_lf_votes_dev=path_lf_votes_dev,
            path_save_labels_train=path_save_labels_train.format(task_id=task_id),
            path_metrics_label_model=path_metrics_label_model,
            negative_value=negative_value,
            positive_value=positive_value,
            abstention_value=abstention_value,
            multi_class_lfs=multi_class_lfs,
            fillnone_as_negative=fillnone_as_negative,
            min_support_dev=min_support_dev,
            avoid_all_abstentions=avoid_all_abstentions,
            rescale_probs=rescale_probs,
            log_level=log_level,
            config_meta=config_meta,
            **kwargs,
        )


if __name__ == "__main__":
    app()
