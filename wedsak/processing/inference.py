import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

import edsnlp
import edsnlp.pipes as eds
import pandas as pd
import torch
from pydantic import BaseModel

from wedsak.misc.data_wrangling import SpanToRowConverter
from wedsak.misc.getters import get_tasks
from wedsak.misc.logger_utils import setup_logger
from wedsak.processing.fhir.converter import SpantoFHIRConverter
from wedsak.processing.fhir.data_models import Procedure


def _resolve_model(
    model: Optional[object],
    model_path: Optional[Union[str, Path]],
    logger: logging.Logger,
):
    if model is not None:
        return model
    if model_path is None:
        raise ValueError("Provide either model or model_path.")
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    logger.info("Loading model from %s", path)
    return edsnlp.load(path)


def _ensure_dates_pipe(nlp, logger: logging.Logger) -> None:
    if "dates" in nlp.pipe_names:
        return
    if "span_classifier" in nlp.pipe_names:
        nlp.add_pipe(eds.dates(), before="span_classifier")
        logger.info("Added dates pipe before span_classifier.")
    else:
        nlp.add_pipe(eds.dates())
        logger.info("Added dates pipe to pipeline.")


def _infer_task_names(nlp) -> List[str]:
    if "span_classifier" not in nlp.pipe_names:
        return []
    span_classifier = nlp.get_pipe("span_classifier").eval()
    attributes = getattr(span_classifier, "attributes", {})
    task_names = [
        attr[2:] if isinstance(attr, str) and attr.startswith("_.") else str(attr)
        for attr in attributes.keys()
    ]
    return list(dict.fromkeys(task_names))


def get_prob(prob_dict, name, class_label="1"):
    if not isinstance(prob_dict, dict):
        return None
    task_probs = prob_dict.get(f"_.{name}")
    if not isinstance(task_probs, dict):
        return None
    return task_probs.get(class_label)


def _load_docs(
    data: Union[pd.DataFrame, str, Path],
    tokenizer,
    logger: logging.Logger,
    doc_attributes=["person_id", "note_datetime"],
):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if "note_id" in df.columns:
            df["note_id"] = df["note_id"].astype(str)
        return edsnlp.data.from_pandas(
            df,
            converter="omop",
            tokenizer=tokenizer,
            doc_attributes=doc_attributes,
        )

    path = Path(data).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input data path does not exist: {path}")

    if path.is_dir() or path.suffix == ".parquet":
        logger.info("Loading dataset from %s", path)
        return edsnlp.data.read_parquet(
            path,
            converter="omop",
            tokenizer=tokenizer,
            doc_attributes=doc_attributes,
        )
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "note_id" in df.columns:
            df["note_id"] = df["note_id"].astype(str)
        return edsnlp.data.from_pandas(
            df, converter="omop", tokenizer=tokenizer, doc_attributes=doc_attributes
        )

    raise ValueError("Supported input formats are pandas DataFrame, .parquet, or .csv")


def row_to_fhir(row):
    procedure = Procedure()
    pass


def map_to_fhir(pred):
    tasks = get_tasks()
    pred = pred.rename(columns={"task_name": "normalized_task_name"}, inplace=False)
    pred = pred.merge(
        tasks[["normalized_task_name", "task", "SNOMED CT - SCTID"]],
        on="normalized_task_name",
        validate="many_to_one",
    )

    pass


def inference(
    data: Union[pd.DataFrame, str, Path],
    model: Optional[object] = None,
    model_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    task_names: Optional[List[str]] = None,
    device: str = "auto",
    log_level: str = "INFO",
    convert_to_long_format: bool = False,
    include_context: bool = True,
    context_window: int = 25,
    context_getter: Optional[Callable[[object, object], str]] = None,
    export_to_fhir: bool = True,
    fhir_ressource: Optional[BaseModel] = Procedure,
):
    """
    Run inference on a dataset using a trained model.

    Parameters
    ----------
    data:
            A pandas DataFrame or a path to a .parquet/.csv dataset in OMOP format.
    model:
            An edsnlp pipeline instance (optional if model_path is provided).
    model_path:
            Path to a model folder saved with nlp.to_disk(...).
    output_path:
            Optional path to save predictions as parquet. If None, returns a DataFrame.
            The output includes a `prob` column with span probabilities when available.
        include_context:
            Whether to add a context column around each span.
        context_window:
            Number of characters to include on both sides of the span when using the
            default context getter.
    """
    logger = setup_logger(log_level, script_name="inference")
    nlp = _resolve_model(model, model_path, logger)
    _ensure_dates_pipe(nlp, logger)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    nlp.to(device)

    if task_names is None:
        task_names = _infer_task_names(nlp)
        if len(task_names) == 0:
            logger.warning(
                "Could not infer task names from model. Falling back to all tasks from get_tasks()."
            )
            tasks = get_tasks()
            task_names = tasks.normalized_task_name.dropna().tolist()

    logger.info("Tasks requested for inference: %s", task_names)

    docs = _load_docs(data, nlp.tokenizer, logger)
    num_docs = 0

    def _count_docs(doc):
        nonlocal num_docs
        num_docs += 1
        return doc

    docs = docs.map(_count_docs)
    start_time = time.perf_counter()
    predicted_docs = docs.map_pipeline(nlp, batch_size)

    if export_to_fhir:
        logger.info("Converting predictions to FHIR format...")
        converter = SpantoFHIRConverter(
            span_task_attributes=task_names,
            span_getter=["dates"],
            fhir_ressource=fhir_ressource,
            k=context_window,
            include_context=include_context,
            context_getter=context_getter,
        )

    else:
        logger.info("Converting predictions to entity format...")
        span_attributes = list(dict.fromkeys(task_names + ["prob", "date.datetime"]))
        converter = SpanToRowConverter(
            span_attributes=span_attributes,
            span_getter=["dates"],
            doc_attributes=[],
            k=context_window,
            include_context=include_context,
            context_getter=context_getter,
        )
    pred = edsnlp.data.to_pandas(predicted_docs, converter=converter)
    inference_time = time.perf_counter() - start_time
    logger.info("Inference completed in %.2fs for %d docs.", inference_time, num_docs)

    if (convert_to_long_format) and (not export_to_fhir):
        id_vars = [
            "note_id",
            "start",
            "end",
            "label",
            "lexical_variant",
            "prob",
            "date.datetime",
        ]
        if include_context:
            id_vars.append("context")
        pred = pred.melt(
            id_vars=id_vars,
            value_vars=task_names,
            var_name="task_name",
        ).query("value=='1'")
        pred["score"] = pred.apply(
            lambda x: get_prob(x["prob"], x["task_name"]), axis=1
        )
        pred = pred.drop(columns=["value", "prob"])

    if export_to_fhir:
        if model_path is not None:
            model_name = Path(model_path).expanduser().name
        else:
            model_name = "unknown_model"
        pred["meta__source"] = "nlp_model_" + model_name
        pred.drop(
            columns=["id", "encounter", "extension_nlp___0__related_span"],
            inplace=True,
            errors="ignore",
        )
        pred.reset_index(drop=False, inplace=True, names="id")

    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(path, index=False)
        logger.info("Predictions saved to %s", path)
        metadata = {
            "num_docs": num_docs,
            "inference_time_sec": inference_time,
            "task_names": task_names,
        }
        metadata_path = path.with_suffix(path.suffix + ".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=True, default=str)
        logger.info("Inference metadata saved to %s", metadata_path)
        return str(path)

    return pred
