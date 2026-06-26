from wedsak.registry import registry
from pathlib import Path
from typing import Callable, Optional, List, Any, Dict

from confit import Cli
from wedsak.misc.constants import PATH_DATA_SCRATCH, PATH_EVALUATION_DATASET
from wedsak.misc.getters import (
    get_tasks,
)
import edsnlp
from wedsak.misc.getters import get_dataset, task_id_to_task_name, normalize_task_name
from wedsak.misc.logger_utils import setup_logger
import datetime
import edsnlp.pipes as eds
import torch
from edsnlp.utils.batching import stat_batchify
import numpy as np
from tqdm import tqdm
from edsnlp.training import ScheduledOptimizer
from wedsak.misc.metrics import evaluate_predictions
from wedsak.misc.utils import save_json, hash_file_or_directory
from zoneinfo import ZoneInfo
from edsnlp.data import read_parquet
from wedsak.misc.utils import make_unique_path

POOLING_MODE = "mean"
WINDOW = 512
STRIDE = 256
app = Cli()


@app.command(name="train", registry=registry)
def train(
    task_ids: List[int],
    batch_size: int = 32,
    model_name: Optional[str] = None,
    path_to_training_datasets: List[str] = [
        f"{PATH_DATA_SCRATCH}/wedsak_datasets/train/conf_train_500/all_tasks/"
    ],
    path_evaluation_dataset: str = PATH_EVALUATION_DATASET,
    path_dir_save_model: Optional[str] = None,
    save_corrected_targets: bool = False,
    log_level: str = "INFO",
    path_embedding: str = "almanach/camembertav2-base",
    context_getter: str = "words[-75:75]",
    exist_soft_labels: bool = True,
    embedding_params: dict = {},
    lr_transformer: float = 1e-5,
    start_value_transformer: float = 0,
    warmup_rate: float = 0,
    lr_span_classifier: float = 5e-5,
    dropout_span_classifier: float = 0.1,
    start_value_span_classifier: float = 0,
    max_steps: int = 500,
    log_every_n_steps: int = 30,
    loss_fn: Optional[Callable] = None,
    use_lrt: bool = False,
    initial_delta_lrt: float = 3.2,
    last_delta_lrt: float = 4.0,
    step_min_lrt: int = 100,
    step_max_lrt: int = 200,
    log_name_extension: str = "",
    shuffle: Optional[str] = "dataset",  # dataset? or fragment?
    label_weights: Optional[Dict[str, Dict[Any, float]]] = None,
    comment: Optional[str] = None,
    config_meta=None,
):
    """
    label_weights: Dict[str, Dict[Any, float]]
        The weight of each label for each attribute. The keys are the attribute names
        and the values are dictionaries with the labels as keys and the weights as
        values. For instance, `{"_.negation": {True: 1, False: 2}}` will give a weight
        of 1 to the `True` value of the `negation` attribute and 2 to the `False` value.
    """
    # TODO add span attribute with hard label
    # TODO add case for hard labels in span classifier

    ## Config
    config = config_meta["resolved_config"].serialize()

    # Setup logger
    logger = setup_logger(log_level, script_name="train")
    start_time = datetime.datetime.now()

    # Get task names
    task_names = []
    for task_id in task_ids:
        task_name = task_id_to_task_name(task_id)
        normalized_task_name = normalize_task_name(task_name)
        task_names.append(normalized_task_name)

    n_tasks = len(task_names)
    tasks = get_tasks()

    # Paths
    path_evaluation_dataset = Path(path_evaluation_dataset).expanduser()

    # Dataset hashes
    evaluation_dataset_hash = hash_file_or_directory(path_evaluation_dataset)
    path_to_training_datasets = [
        Path(path).expanduser() for path in path_to_training_datasets
    ]
    training_datasets_hash = {
        i: hash_file_or_directory(path)
        for i, path in enumerate(path_to_training_datasets)
    }
    # path_to_training_datasets = Path(path_to_training_datasets).expanduser()

    # Metadata file name
    timestamp = (
        str(datetime.datetime.now(tz=ZoneInfo("Europe/Paris")))
        .replace(" ", "_")
        .replace(":", "-")
    )
    file_name_metadata = f"{log_name_extension}{timestamp}.json"

    # Log config
    logger.info(f"Number of tasks: {n_tasks}")
    logger.info(f"Tasks: {task_names}")
    logger.info(f"Path to training datasets: {path_to_training_datasets}")
    logger.info(f"Path to evaluation dataset: {path_evaluation_dataset}")
    logger.info(f"Path to model directory: {path_dir_save_model}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"lr_transformer: {lr_transformer}")
    logger.info(f"start_value_transformer: {start_value_transformer}")
    logger.info(f"warmup_rate: {warmup_rate}")
    logger.info(f"lr_span_classifier: {lr_span_classifier}")
    logger.info(f"log_every_n_steps: {log_every_n_steps}")
    logger.info(f"Use LRT: {use_lrt}")
    logger.info("Embedding params: %s", embedding_params)
    logger.info("Shuffle: %s", shuffle)

    if model_name is None:
        if n_tasks == 1:
            model_name = (
                f"wedsak-{task_names[0]}-{start_time.strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            task_ids_str = "-".join(str(i) for i in task_ids)
            model_name = f"wedsak-{task_ids_str}-{start_time.strftime('%Y%m%d_%H%M%S')}"

    # Initialize NLP pipeline
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.sentences())
    _ = nlp.add_pipe(
        eds.span_classifier(
            embedding=eds.span_pooler(
                pooling_mode=embedding_params.get("pooling_mode", POOLING_MODE),
                embedding=eds.transformer(
                    model=path_embedding,
                    window=embedding_params.get("window", WINDOW),
                    stride=embedding_params.get("stride", STRIDE),
                ),
            ),
            span_getter=["dates", "train_date"],
            attributes={"_." + t: ["date", "train_date"] for t in task_names},
            context_getter=context_getter,
            loss_fn=loss_fn,
            dropout_prob=dropout_span_classifier,
            label_weights=label_weights,
        ),
        name="span_classifier",
    )

    # Training datasets
    training_datasets = []
    for path in path_to_training_datasets:
        parquet = read_parquet(
            path,
            converter="omop",
            span_setter=["train_date"],
            span_attributes=task_names + ["instance_id"],
            tokenizer=nlp.tokenizer,
        )
        training_datasets.append(parquet)

    # Post init
    bindings = []
    values = ["0", "1"]
    span_classifier = nlp.get_pipe("span_classifier")
    for attr, labels in span_classifier.attributes.items():
        b = (attr, labels, values)
        bindings.append(b)

    span_classifier.exist_soft_labels = exist_soft_labels
    span_classifier.update_bindings(bindings)

    # Log CUDA availability
    logger.info("Torch CUDA available: %s", torch.cuda.is_available())

    # Making the stream of mini-batches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_loaders = []

    logger.info("Preparing data loaders")
    for dataset in training_datasets:
        batches = dataset.loop().map(eds.explode(span_getter=["train_date"]))
        if shuffle is not None:
            batches = batches.shuffle(shuffle)
        batches = (
            batches.map(nlp.preprocess, kwargs={"supervision": True})
            .batchify(batch_size=batch_size, batch_by=stat_batchify("spans"))
            .map(nlp.collate, kwargs={"device": device})
        )
        batch_loaders.append(batches)

    logger.info("Creating k iterators")
    iterators = {}
    for i, batch_loader in enumerate(batch_loaders):
        iterators[i] = iter(batch_loader)

    logger.info("Iteratiors: %s", list(iterators.keys()))

    # Nombre de steps réel : max_steps x n_accumulation_steps
    n_accumulation_steps = len(iterators)
    logger.info("Number of accumulation steps: %d", n_accumulation_steps)

    # Move the model to the GPU
    nlp.to(device)

    # Initialize optimizer
    optimizer = ScheduledOptimizer(
        optim=torch.optim.AdamW,
        module=nlp,
        total_steps=max_steps,
        groups={
            "transformer": {
                "lr": {
                    "@schedules": "linear",
                    "warmup_rate": warmup_rate,
                    "start_value": start_value_transformer,
                    "max_value": lr_transformer,
                },
            },
            "span_classifier": {
                "lr": {
                    "@schedules": "linear",
                    "warmup_rate": warmup_rate,
                    "start_value": start_value_span_classifier,
                    "max_value": lr_span_classifier,
                },
            },
        },
    )
    logger.debug("Optimizer initialized")

    # Track loss and metrics
    loss_tracking = []
    metrics_tracking = {task: [] for task in task_names}

    # Get reference docs and gold labels for evaluation
    path_ref_docs = Path(path_evaluation_dataset).expanduser()
    ref_docs = edsnlp.data.read_parquet(
        path_ref_docs,
        converter="omop",
        span_setter=["dates"],
    )
    logger.debug("Reference documents loaded")
    # Apply pipeline before training
    predicted_docs = ref_docs.map_pipeline(nlp)
    predicted_docs.set_processing(
        backend="multiprocessing", num_gpu_workers=1, num_cpu_workers=4
    )
    logger.debug("Reference documents processed through the pipeline")

    gold = edsnlp.data.to_pandas(
        ref_docs,
        converter="ents",
        span_getter="dates",
        span_attributes=tasks.normalized_task_name.to_list(),
    )
    logger.debug("Gold labels for reference documents extracted")  # OK

    pred = edsnlp.data.to_pandas(
        predicted_docs,
        converter="ents",
        span_getter="dates",
        span_attributes=tasks.normalized_task_name.to_list(),
    )
    logger.debug("Initial predictions for reference documents extracted")
    for task_name in task_names:
        logger.info("Evaluating task: %s", task_name)
        logger.info(
            "%s", evaluate_predictions(gold=gold, pred=pred, task_name=task_name)
        )

    # Training loop
    logger.info("Starting training")
    for step in tqdm(range(max_steps), "Training model", leave=True):
        for i, iterator in iterators.items():
            nlp.train(True)
            batch = next(iterator)

            loss = torch.zeros((), device=device)
            with nlp.cache():
                # loss = torch.zeros((), device=device)
                for name, component in nlp.torch_components():
                    if not use_lrt:
                        output = component(
                            batch[name],
                            # weights={"_.biopsy_procedure": [1.0, 10.0]},
                        )
                    else:  # LRT is used
                        if step < step_min_lrt:
                            output = component(
                                batch[name],
                                apply_lrt=False,
                                use_corrected_targets=False,
                            )
                        elif step_min_lrt <= step <= step_max_lrt:
                            delta = initial_delta_lrt + (
                                (last_delta_lrt - initial_delta_lrt)
                                * (step - step_min_lrt)
                                / (step_max_lrt - step_min_lrt)
                            )
                            output = component(
                                batch[name],
                                apply_lrt=True,
                                use_corrected_targets=True,
                                lrt_parameters={"delta": delta},
                            )
                        else:  # step > step_max_lrt
                            output = component(
                                batch[name],
                                apply_lrt=False,
                                use_corrected_targets=True,
                            )
                    if "loss" in output:
                        loss = output["loss"] / n_accumulation_steps / n_tasks  # FIXME
                        loss.backward()

            loss_tracking.append(loss.item())
            if ((step % log_every_n_steps) == 0) or (step == max_steps - 1):
                loss_rolling_mean = np.mean(loss_tracking[-log_every_n_steps:]).item()

                logger.info("Step: %s", step)
                logger.info(
                    "Rolling Mean Loss of %d previous steps: %s",
                    log_every_n_steps,
                    loss_rolling_mean,
                )
                current_lr_transformer = optimizer.optim.param_groups[0]["lr"]
                current_lr_span_classifier = optimizer.optim.param_groups[1]["lr"]
                logger.info(
                    "Current LR - Transformer: %s - Span Classifier: %s",
                    current_lr_transformer,
                    current_lr_span_classifier,
                )

                # Evaluate on the reference set
                span_classifier = nlp.get_pipe("span_classifier")
                _ = span_classifier.eval()
                predicted_docs = nlp.pipe(ref_docs)
                pred = edsnlp.data.to_pandas(
                    predicted_docs,
                    converter="ents",
                    span_getter="dates",
                    span_attributes=tasks.normalized_task_name.to_list(),
                )

                for task_name in task_names:
                    logger.info("Evaluating task: %s", task_name)
                    metrics = evaluate_predictions(
                        gold=gold, pred=pred, task_name=task_name
                    )
                    metrics["step"] = step
                    metrics["lr_transformer"] = current_lr_transformer
                    metrics["lr_span_classifier"] = current_lr_span_classifier
                    logger.info("%s: %s", task_name, metrics)
                    metrics_tracking[task_name].append(metrics)

                # Save config & hash dataset
                metadata = {
                    "config": config,
                    "metrics_on_devset": metrics_tracking,
                    "loss": loss_tracking,
                    "evaluation_dataset_hash": evaluation_dataset_hash,
                    "training_datasets_hash": training_datasets_hash,
                }
                save_json(metadata, sub_folder="train", file_name=file_name_metadata)
                nlp.train(True)

        optimizer.step()
        optimizer.zero_grad()
    end_time = datetime.datetime.now()
    logger.info("Training started at: %s", start_time)
    logger.info("Training ended at: %s", end_time)
    logger.info("Total training time: %s", end_time - start_time)

    # Save the trained model
    hash_model = None
    if path_dir_save_model is not None:  # FIXME model_name cannot be none
        path_dir_save_model = Path(path_dir_save_model, model_name).expanduser()
        path_dir_save_model = make_unique_path(path_dir_save_model)
        logger.info("Saving model to %s", path_dir_save_model)
        nlp.to_disk(path_dir_save_model)
        hash_model = hash_file_or_directory(path_dir_save_model)
        logger.info("Hash of the saved model: %s", hash_model)
    if save_corrected_targets:
        logger.info(
            "Saving corrected targets to subfolder 'logs/corrected_targets'",
        )
        span_classifier = nlp.get_pipe("span_classifier")
        corrected_targets = span_classifier.corrected_targets
        save_json(
            corrected_targets,
            sub_folder="corrected_targets",
            file_name=file_name_metadata,
        )

    # Save config & hash dataset
    metadata = {
        "config": config,
        "comment": comment,
        "metrics_on_devset": metrics_tracking,
        "loss": loss_tracking,
        "evaluation_dataset_hash": evaluation_dataset_hash,
        "training_datasets_hash": training_datasets_hash,
        "path_to_saved_model": str(path_dir_save_model)
        if path_dir_save_model is not None
        else None,
    }
    metadata["hash_model"] = hash_model

    save_json(metadata, sub_folder="train", file_name=file_name_metadata)


if __name__ == "__main__":
    app()
