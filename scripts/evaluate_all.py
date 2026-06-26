import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import edsnlp
import edsnlp.pipes as eds
import pandas as pd
import torch
from confit import Cli
from edsnlp.pipes.qualifiers.contextual.contextual import ContextualQualifier

from wedsak.misc.data_wrangling import normalize_task_name
from wedsak.misc.getters import get_lf_reference, get_tasks
from wedsak.misc.logger_utils import setup_logger
from wedsak.misc.metrics import evaluate_predictions
from wedsak.misc.utils import hash_file_or_directory, make_unique_path


app = Cli()
GENERAL_COLS = ["note_id", "start", "end", "label"]


def _resolve_model_path(
    path_metrics_train: Optional[str],
    path_model: Optional[str],
    logger: logging.Logger,
) -> Path:
    if path_metrics_train is not None and pd.notna(path_metrics_train):
        metrics_path = Path(str(path_metrics_train)).expanduser()
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                path_from_metrics = metrics_data.get("path_to_saved_model")
                if path_from_metrics:
                    return Path(path_from_metrics).expanduser()
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Could not read metrics file %s (%s). Falling back to path_model.",
                    metrics_path,
                    exc,
                )
        else:
            logger.warning(
                "Metrics file does not exist at %s. Falling back to path_model.",
                metrics_path,
            )

    if path_model is None or not pd.notna(path_model):
        raise ValueError("Missing both path_metrics_train and path_model in row.")

    return Path(str(path_model)).expanduser()


def _infer_task_names(nlp) -> List[str]:
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
    return task_names


def _evaluate_model_on_dataset(
    nlp,
    task_names: List[str],
    path_dataset: Path,
    logger: logging.Logger,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
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

    return metrics_tracking, evaluated_task_names, skipped_task_names


def _evaluate_from_frames(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    task_names: List[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
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

    return metrics_tracking, evaluated_task_names, skipped_task_names


def _suffix_candidates(suffix_task_name: Optional[str]) -> List[str]:
    if suffix_task_name is None or pd.isna(suffix_task_name):
        return []
    suffix = str(suffix_task_name)
    candidates = []
    if suffix.startswith("_"):
        candidates.append(suffix)
        candidates.append(suffix.lstrip("_"))
    else:
        candidates.append(f"_{suffix}")
        candidates.append(suffix)
    return list(dict.fromkeys([c for c in candidates if c]))


def _prepare_llm_pred_frame(
    pred: pd.DataFrame,
    task_names: List[str],
    suffix_task_name: Optional[str],
    gold_note_ids: Optional[set],
    logger: logging.Logger,
) -> pd.DataFrame:
    missing_cols = [col for col in GENERAL_COLS if col not in pred.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in LLM predictions: {', '.join(missing_cols)}"
        )

    pred = pred.copy()
    pred["note_id"] = pred["note_id"].astype(str)
    if gold_note_ids is not None:
        pred = pred[pred["note_id"].isin(gold_note_ids)].copy()

    suffixes = _suffix_candidates(suffix_task_name)
    for task_name in task_names:
        pred_col = None
        for suffix in suffixes:
            candidate = f"{task_name}{suffix}"
            if candidate in pred.columns:
                pred_col = candidate
                break
        if pred_col is None and task_name in pred.columns:
            pred_col = task_name
        if pred_col is None:
            logger.warning(
                "Prediction column for task %s not found in LLM results.", task_name
            )
            continue
        pred[task_name] = pred[pred_col]
        pred[task_name] = pred[task_name].replace({True: "1", False: "0"}).fillna("0")

    keep_cols = GENERAL_COLS + [name for name in task_names if name in pred.columns]
    return pred[keep_cols]


def _load_context_patterns(
    path_patterns: Path,
    logger: logging.Logger,
    path_lf_reference: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]]]:
    if path_lf_reference is None:
        ref_lfs, _ = get_lf_reference()
        tasks = get_tasks()
    else:
        ref_lfs, _ = get_lf_reference(path=path_lf_reference)
        tasks = get_tasks(path=path_lf_reference)

    with open(path_patterns, "r", encoding="utf-8") as f:
        patterns = json.load(f)

    context_patterns: Dict[str, Dict[str, Any]] = {}
    task_map: Dict[str, Dict[str, str]] = {}

    for fsn, pattern in patterns.items():
        lf_name = pattern.get("lf_name")
        if not lf_name:
            logger.warning("Skipping pattern without lf_name: %s", fsn)
            continue
        if lf_name not in ref_lfs.ref_name.unique():
            logger.warning("LF name %s not found in reference.", lf_name)
            continue

        normalized_task = (
            ref_lfs.loc[ref_lfs.ref_name == lf_name, "normalized_task_name"]
            .dropna()
            .iloc[0]
        )
        task_display = tasks.loc[tasks.normalized_task_name == normalized_task, "task"]
        task_display = task_display.iloc[0] if not task_display.empty else fsn

        pattern_config = dict(pattern)
        pattern_config.pop("lf_name", None)
        context_patterns[lf_name] = {fsn: pattern_config}
        if normalized_task in task_map:
            logger.warning(
                "Duplicate task mapping for %s (existing LF %s, new LF %s).",
                normalized_task,
                task_map[normalized_task]["lf_name"],
                lf_name,
            )
        task_map[normalized_task] = {
            "lf_name": lf_name,
            "task_name": task_display,
        }

    return context_patterns, task_map


def _match_pred_value(value: Any, expected: str) -> bool:
    if isinstance(value, list):
        return expected in value
    if isinstance(value, str):
        return value == expected
    if isinstance(value, bool):
        return value
    return False


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if pd.isna(value):
        return None
    return value


@app.command(name="evaluate_all")
def evaluate_all(
    path_model_refs: str,
    path_dataset_A: Optional[str] = None,
    path_dataset_B: Optional[str] = None,
    types_to_evaluate: Optional[List[str]] = ["WS"],
    name_code_filter: Optional[List[str]] = None,
    only_active: bool = True,
    path_base_metrics: str = "~/wedsak/logs/evaluation/",
    log_level: str = "INFO",
    overwrite_metrics: bool = True,
):
    logger = setup_logger(log_level, script_name="evaluate_all")

    path_model_refs = Path(path_model_refs).expanduser()
    if not path_model_refs.exists():
        raise FileNotFoundError(f"Model refs file does not exist: {path_model_refs}")

    df_refs = pd.read_excel(path_model_refs)

    if only_active:
        df_refs = df_refs[df_refs["active"] == 1]
    if name_code_filter is not None:
        df_refs = df_refs[df_refs["name_code"].isin(name_code_filter)]
    if types_to_evaluate is not None:
        df_refs = df_refs[df_refs["type"].isin(types_to_evaluate)]

    if df_refs.empty:
        logger.warning("No models to evaluate after filtering.")
        return []

    datasets: List[Tuple[str, Optional[str]]] = [
        ("A", path_dataset_A),
        ("B", path_dataset_B),
    ]
    base_metrics_dir = Path(path_base_metrics).expanduser()
    base_metrics_dir.mkdir(parents=True, exist_ok=True)

    output_paths: List[str] = []

    for _, row in df_refs.iterrows():
        model_id = row.get("id")
        model_name_code = row.get("name_code")
        model_type = str(row.get("type")).lower() if pd.notna(row.get("type")) else ""

        if model_type in {"llm", "rule_based"}:
            path_model = Path(str(row.get("path_model"))).expanduser()
        else:
            path_model = _resolve_model_path(
                row.get("path_metrics_train"),
                row.get("path_model"),
                logger,
            )

        if not path_model.exists():
            logger.warning(f"Model path does not exist: {path_model}")
            continue

        model_name = path_model.name
        model_hash = hash_file_or_directory(path_model)

        if model_type == "llm":
            pred_llm = pd.read_pickle(path_model)
            suffix_task_name = model_name_code
            task_names = []
            if pd.notna(row.get("task_name")):
                task_names = [normalize_task_name(str(row.get("task_name")))]
            if not task_names:
                suffixes = _suffix_candidates(suffix_task_name)
                for col in pred_llm.columns:
                    for suffix in suffixes:
                        if col.endswith(suffix):
                            base = col[: -len(suffix)]
                            if not suffix.startswith("_"):
                                base = base.rstrip("_")
                            if base:
                                task_names.append(base)
                            break
                task_names = list(dict.fromkeys(task_names))

            logger.info("Tasks requested for evaluation: %s", task_names)

        elif model_type == "rule_based":
            context_patterns, task_map = _load_context_patterns(path_model, logger)
            task_names = []
            if pd.notna(row.get("task_name")):
                normalized_task = normalize_task_name(str(row.get("task_name")))
                task_names = [normalized_task]
            if not task_names:
                task_names = list(task_map.keys())

            logger.info("Tasks requested for evaluation: %s", task_names)

        else:
            logger.info("Loading model from %s", path_model)
            nlp = edsnlp.load(path_model)
            nlp.add_pipe(eds.dates(), before="span_classifier")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            nlp.to(device)

            task_names = _infer_task_names(nlp)
            if len(task_names) == 0:
                logger.warning(
                    "Could not infer task names from model. Falling back to all tasks from get_tasks()."
                )
                tasks = get_tasks()
                task_names = tasks.normalized_task_name.dropna().tolist()

            logger.info("Tasks requested for evaluation: %s", task_names)

        for dataset_label, dataset_path in datasets:
            if dataset_path is None:
                continue

            path_dataset = Path(dataset_path).expanduser()
            if not path_dataset.exists():
                raise FileNotFoundError(
                    f"Dataset {dataset_label} path does not exist: {path_dataset}"
                )

            start_time = datetime.datetime.now(tz=ZoneInfo("Europe/Paris"))
            if model_type == "llm":
                nlp = edsnlp.blank("eds")
                ref_docs = edsnlp.data.read_parquet(
                    path_dataset,
                    converter="omop",
                    span_setter=["dates"],
                    tokenizer=nlp.tokenizer,
                )
                gold = edsnlp.data.to_pandas(
                    ref_docs,
                    converter="ents",
                    span_getter="dates",
                    span_attributes=task_names,
                )
                gold["note_id"] = gold["note_id"].astype(str)
                gold_note_ids = set(gold["note_id"].unique())
                pred = _prepare_llm_pred_frame(
                    pred_llm, task_names, model_name_code, gold_note_ids, logger
                )
                metrics_tracking, evaluated_task_names, skipped_task_names = (
                    _evaluate_from_frames(gold, pred, task_names, logger)
                )
            elif model_type == "rule_based":
                nlp = edsnlp.blank("eds")
                ref_docs = edsnlp.data.read_parquet(
                    path_dataset,
                    converter="omop",
                    span_setter=["dates"],
                    tokenizer=nlp.tokenizer,
                )
                gold = edsnlp.data.to_pandas(
                    ref_docs,
                    converter="ents",
                    span_getter="dates",
                    span_attributes=task_names,
                )

                nlp = edsnlp.blank("eds")
                nlp.add_pipe(eds.normalizer())
                nlp.add_pipe(
                    eds.sentences(check_capitalized=False, min_newline_count=1)
                )
                nlp.add_pipe(eds.dates())
                nlp.add_pipe(
                    ContextualQualifier(span_getter="dates", patterns=context_patterns)
                )

                predicted_docs = nlp.pipe(ref_docs)
                lf_names = [
                    task_map[task]["lf_name"] for task in task_names if task in task_map
                ]
                pred_lf = edsnlp.data.to_pandas(
                    predicted_docs,
                    converter="ents",
                    span_getter="dates",
                    span_attributes=lf_names,
                )

                pred = pred_lf[GENERAL_COLS].copy()
                for task in task_names:
                    mapping = task_map.get(task)
                    if mapping is None:
                        logger.warning("No rule-based mapping found for task %s", task)
                        continue
                    lf_name = mapping["lf_name"]
                    fsn = mapping["task_name"]
                    if lf_name not in pred_lf.columns:
                        logger.warning(
                            "LF %s missing in predictions for task %s", lf_name, task
                        )
                        continue
                    pred[task] = pred_lf[lf_name].apply(
                        lambda x, expected=fsn: _match_pred_value(x, expected)
                    )
                    pred[task] = pred[task].replace({True: "1", False: "0"}).fillna("0")

                metrics_tracking, evaluated_task_names, skipped_task_names = (
                    _evaluate_from_frames(gold, pred, task_names, logger)
                )
            else:
                metrics_tracking, evaluated_task_names, skipped_task_names = (
                    _evaluate_model_on_dataset(nlp, task_names, path_dataset, logger)
                )
            end_time = datetime.datetime.now(tz=ZoneInfo("Europe/Paris"))

            dataset_hash = hash_file_or_directory(path_dataset)
            output_file = (
                base_metrics_dir / f"{model_id}_{model_name_code}_{dataset_label}.json"
            )
            if not overwrite_metrics:
                output_file = make_unique_path(output_file)

            metadata = {
                "timestamp_start": str(start_time),
                "timestamp_end": str(end_time),
                "model_id": model_id,
                "name_code": model_name_code,
                "name_display": row.get("name_display"),
                "type": row.get("type"),
                "comment": row.get("comment"),
                "task_name": row.get("task_name"),
                "model_name": model_name,
                "path_model": str(path_model),
                "hash_model": model_hash,
                "path_evaluation_dataset": str(path_dataset),
                "evaluation_dataset_hash": dataset_hash,
                "dataset_label": dataset_label,
                "tasks_requested": task_names,
                "tasks_evaluated": evaluated_task_names,
                "tasks_skipped_missing_columns": skipped_task_names,
                "metrics_on_dataset": metrics_tracking,
                "path_metrics_train": row.get("path_metrics_train"),
                "path_model_reference": row.get("path_model"),
                "path_label_model": row.get("path_label_model"),
                "output_json_path": str(output_file),
            }

            metadata = _sanitize_for_json(metadata)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=True, default=str)

            logger.info("Evaluation metadata saved to %s", output_file)
            output_paths.append(str(output_file))

    return output_paths


if __name__ == "__main__":
    app()
