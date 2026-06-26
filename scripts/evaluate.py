import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import edsnlp
import edsnlp.pipes as eds
import torch
from confit import Cli

from wedsak.misc.constants import PATH_EVALUATION_DATASET
from wedsak.misc.getters import get_tasks
from wedsak.misc.logger_utils import setup_logger
from wedsak.misc.metrics import evaluate_predictions
from wedsak.misc.utils import hash_file_or_directory, make_unique_path
from wedsak.registry import registry

app = Cli()


@app.command(name="evaluate", registry=registry)
def evaluate(
    model_path: str,
    dataset_path: str = PATH_EVALUATION_DATASET,
    output_json_path: Optional[str] = None,
    overwrite_output: bool = False,
    log_level: str = "INFO",
    comment: Optional[str] = None,
    config_meta=None,
):
    """
    Evaluate a trained model on a dataset and write metrics/metadata to JSON.

    Parameters
    ----------
    model_path:
        Path to a model folder previously saved with nlp.to_disk(...).
    dataset_path:
        Path to parquet dataset in OMOP format.
    output_json_path:
        Optional target JSON path. If not provided, a file is created in
        ~/wedsak/logs/evaluation/.
    overwrite_output:
        If False and output path already exists, a unique path is generated.
    """

    logger = setup_logger(log_level, script_name="evaluate")
    start_time = datetime.datetime.now(tz=ZoneInfo("Europe/Paris"))

    script_config = None
    if config_meta is not None and "resolved_config" in config_meta:
        script_config = config_meta["resolved_config"].serialize()

    path_model = Path(model_path).expanduser()
    path_dataset = Path(dataset_path).expanduser()

    if not path_model.exists():
        raise FileNotFoundError(f"Model path does not exist: {path_model}")
    if not path_dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path_dataset}")

    logger.info("Loading model from %s", path_model)
    nlp = edsnlp.load(path_model)
    nlp.add_pipe(eds.dates(), before="span_classifier")

    # Move the model to the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp.to(device)

    model_name = path_model.name
    model_hash = hash_file_or_directory(path_model)
    dataset_hash = hash_file_or_directory(path_dataset)

    # Infer tasks from the model's span classifier attributes when available.
    task_names: List[str] = []
    if "span_classifier" in nlp.pipe_names:
        span_classifier = nlp.get_pipe("span_classifier")
        span_classifier = span_classifier.eval()
        attributes = getattr(span_classifier, "attributes", {})
        task_names = [
            attr[2:] if isinstance(attr, str) and attr.startswith("_.") else str(attr)
            for attr in attributes.keys()
        ]
        task_names = list(dict.fromkeys(task_names))

    if len(task_names) == 0:
        logger.warning(
            "Could not infer task names from model. Falling back to all tasks from get_tasks()."
        )
        tasks = get_tasks()
        task_names = tasks.normalized_task_name.dropna().tolist()

    logger.info("Tasks requested for evaluation: %s", task_names)

    logger.info("Loading evaluation dataset from %s", path_dataset)
    ref_docs = edsnlp.data.read_parquet(
        path_dataset,
        converter="omop",
        span_setter=["dates"],
    )

    predicted_docs = nlp.pipe(ref_docs)

    gold = edsnlp.data.to_pandas(
        ref_docs,
        converter="ents",
        span_getter="dates",
        span_attributes=task_names,
    )
    pred = edsnlp.data.to_pandas(
        predicted_docs,
        converter="ents",
        span_getter="dates",
        span_attributes=task_names,
    )

    evaluated_task_names: List[str] = []
    skipped_task_names: List[str] = []
    metrics_tracking: Dict[str, Dict[str, Any]] = {}

    for task_name in task_names:
        if task_name in gold.columns and task_name in pred.columns:
            metrics = evaluate_predictions(gold=gold, pred=pred, task_name=task_name)
            metrics_tracking[task_name] = metrics
            evaluated_task_names.append(task_name)
            logger.info("%s: %s", task_name, metrics)
        else:
            skipped_task_names.append(task_name)
            logger.warning(
                "Skipping task %s because it is missing from gold or pred columns.",
                task_name,
            )

    if output_json_path is None:
        timestamp = str(start_time).replace(" ", "_").replace(":", "-")
        output_path = Path(
            "~/wedsak/logs/evaluation",
            f"{model_name}_{timestamp}.json",
        ).expanduser()
    else:
        output_path = Path(output_json_path).expanduser()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite_output:
        output_path = make_unique_path(output_path)

    metadata = {
        "timestamp_start": str(start_time),
        "timestamp_end": str(datetime.datetime.now(tz=ZoneInfo("Europe/Paris"))),
        "comment": comment,
        "model_name": model_name,
        "path_model": str(path_model),
        "hash_model": model_hash,
        "path_evaluation_dataset": str(path_dataset),
        "evaluation_dataset_hash": dataset_hash,
        "tasks_requested": task_names,
        "tasks_evaluated": evaluated_task_names,
        "tasks_skipped_missing_columns": skipped_task_names,
        "metrics_on_dataset": metrics_tracking,
        "config": script_config,
        "model_config": nlp.config.to_str(),
        "output_json_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=True, default=str)

    logger.info("Evaluation metadata saved to %s", output_path)
    return str(output_path)


if __name__ == "__main__":
    app()
