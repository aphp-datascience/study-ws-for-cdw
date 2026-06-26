from typing import List, Optional, Tuple

import pandas as pd
from confit import Cli

from wedsak.processing.export_dataset import (
    data_selection_pipeline,
    export_dataset_multiple_tasks,
)
from wedsak.misc.constants import PATH_DATA_SCRATCH

pd.set_option("future.no_silent_downcasting", True)


app = Cli()


@app.command(name="data_selection")
def data_selection(
    path_train_labels: str,
    task_ids: List[int],
    path_docs_train: str,
    path_train_dataset: str = f"{PATH_DATA_SCRATCH}/wedsak_datasets/train/",
    max_number_of_positive_examples: int = 1000,
    prob_thresholds: Tuple[float, float] = (0.5, 0.5),
    seed: int = 123,
    sampling_params: dict = {},
    negative_on_positive_ratio: Optional[float] = 1.0,
    strategy: Optional[str] = "subsampling",
    log_level: str = "INFO",
    config_meta=None,
    **kwargs,
):
    if strategy == "one_dataset_for_all_tasks":
        export_dataset_multiple_tasks(
            path_train_labels=path_train_labels,
            task_ids=task_ids,
            path_docs_train=path_docs_train,
            path_train_dataset=path_train_dataset,
            strategy=strategy,
            log_level=log_level,
            config_meta=config_meta,
            **kwargs,
        )
    else:
        for task_id in task_ids:
            data_selection_pipeline(
                path_train_labels=path_train_labels.format(task_id=task_id),
                task_id=task_id,
                path_docs_train=path_docs_train,
                path_train_dataset=path_train_dataset,
                max_number_of_positive_examples=max_number_of_positive_examples,
                prob_thresholds=prob_thresholds,
                seed=seed,
                sampling_params=sampling_params,
                negative_on_positive_ratio=negative_on_positive_ratio,
                strategy=strategy,
                log_level=log_level,
                config_meta=config_meta,
                **kwargs,
            )


if __name__ == "__main__":
    app()
