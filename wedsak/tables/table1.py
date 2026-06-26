import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from wedsak.misc.data_wrangling import normalize_task_name
from wedsak.misc.getters import get_tasks
from wedsak.misc.logger_utils import setup_logger


DATASET_LABELS = ["A", "B"]


def _find_metrics_file(
    base_dir: Path,
    model_id: str,
    name_code: str,
    dataset_label: str,
    logger,
) -> Optional[Path]:
    pattern = f"{model_id}_{name_code}_{dataset_label}*.json"
    matches = sorted(
        base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not matches:
        logger.warning(
            "No metrics file found for id=%s name_code=%s dataset=%s in %s",
            model_id,
            name_code,
            dataset_label,
            base_dir,
        )
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple metrics files found for id=%s name_code=%s dataset=%s. Using %s",
            model_id,
            name_code,
            dataset_label,
            matches[0],
        )
    return matches[0]


def _column_key(
    name_display: str, dataset_label: str, metric: Optional[str]
) -> Tuple[str, str] | Tuple[str, str, str]:
    if metric is None:
        return (name_display, dataset_label)
    return (name_display, dataset_label, metric)


def metrics_table(
    path_model_refs: str,
    path_base_metrics: str = "~/wedsak/logs/evaluation/",
    metric: str = "fscore",
    metrics: Optional[List[str]] = None,
    output_csv_path: Optional[str] = None,
    log_level: str = "INFO",
    use_only_active: bool = True,
) -> pd.DataFrame:
    """
    Build a metrics table from evaluate_all outputs.

    Parameters
    ----------
    path_model_refs:
        Path to the Excel file with model references.
    path_base_metrics:
        Base directory where evaluation JSON files are stored.
    metric:
        Single metric to report when metrics list is not provided.
    metrics:
        Optional list of metrics to report (e.g., ["fscore", "precision"]).
    output_csv_path:
        Optional path to save the resulting CSV.
    """
    logger = setup_logger(
        log_level, script_name="metrics_table", route_stdout_stderr_to_logger=False
    )

    path_model_refs = Path(path_model_refs).expanduser()
    if not path_model_refs.exists():
        raise FileNotFoundError(f"Model refs file does not exist: {path_model_refs}")

    df_refs = pd.read_excel(path_model_refs)
    if use_only_active:
        df_refs = df_refs[df_refs["active"].isin([1, 1.0, True, "1"])]
    if df_refs.empty:
        logger.warning("No models found in refs file.")
        return pd.DataFrame()

    tasks = get_tasks()
    tasks = tasks.dropna(subset=["normalized_task_name"]).sort_values("task_id")
    task_index = tasks["normalized_task_name"].tolist()

    metrics_list = metrics or [metric]
    use_multi_metric = len(metrics_list) > 1

    base_dir = Path(path_base_metrics).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Metrics base directory does not exist: {base_dir}")

    data: Dict[Tuple[str, ...], pd.Series] = {}
    support_map: Dict[str, pd.Series] = {
        label: pd.Series(index=task_index, dtype=float) for label in DATASET_LABELS
    }
    seen_names: Dict[str, str] = {}

    for _, row in df_refs.iterrows():
        name_display = row.get("name_display")
        model_id = row.get("id")
        name_code = row.get("name_code")

        if pd.isna(name_display) or pd.isna(model_id) or pd.isna(name_code):
            logger.warning("Skipping row with missing id/name_display/name_code.")
            continue

        name_display = str(name_display)
        model_id = str(model_id)
        name_code = str(name_code)

        is_task_specific = name_display.strip().lower() == "task specific"
        if not is_task_specific:
            if name_display in seen_names and seen_names[name_display] != model_id:
                raise ValueError(
                    f"Duplicate name_display found: {name_display}. "
                    "Please ensure name_display is unique."
                )
            seen_names[name_display] = model_id

        for dataset_label in DATASET_LABELS:
            metrics_path = _find_metrics_file(
                base_dir, model_id, name_code, dataset_label, logger
            )
            if metrics_path is None:
                continue

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            metrics_on_dataset = metrics_data.get("metrics_on_dataset", {})
            if not isinstance(metrics_on_dataset, dict):
                logger.warning("Invalid metrics structure in %s", metrics_path)
                continue

            row_task_name = row.get("task_name")
            normalized_row_task = (
                normalize_task_name(str(row_task_name))
                if is_task_specific and pd.notna(row_task_name)
                else None
            )

            for task_name, task_metrics in metrics_on_dataset.items():
                if normalized_row_task is not None and task_name != normalized_row_task:
                    continue
                if task_name not in task_index:
                    logger.warning("Unknown task in metrics: %s", task_name)
                    continue
                if not isinstance(task_metrics, dict):
                    continue

                support_value = task_metrics.get("1", {}).get("support")
                if support_value is not None:
                    existing_support = support_map[dataset_label].get(task_name)
                    if pd.notna(existing_support) and existing_support != support_value:
                        raise ValueError(
                            f"Support mismatch for task {task_name} dataset {dataset_label}: "
                            f"{existing_support} vs {support_value}"
                        )
                    support_map[dataset_label].loc[task_name] = support_value

                for metric_name in metrics_list:
                    value = task_metrics.get("1", {}).get(metric_name)
                    if value is None:
                        continue
                    column = _column_key(
                        name_display,
                        dataset_label,
                        metric_name if use_multi_metric else None,
                    )
                    if column not in data:
                        data[column] = pd.Series(index=task_index, dtype=float)
                    data[column].loc[task_name] = value

    columns: List[Tuple[str, ...]] = []
    for label in DATASET_LABELS:
        support_col = _column_key(
            "Support", label, "support" if use_multi_metric else None
        )
        data[support_col] = support_map[label]

    columns = list(data.keys())
    result = pd.DataFrame(data, index=task_index)
    if use_multi_metric:
        result.columns = pd.MultiIndex.from_tuples(
            columns, names=["name_display", "dataset", "metric"]
        )
    else:
        result.columns = pd.MultiIndex.from_tuples(
            columns, names=["name_display", "dataset"]
        )

    task_lookup = tasks.set_index("normalized_task_name")
    task_ids = task_lookup.loc[result.index, "task_id"].astype("Int64")
    task_labels = task_lookup.loc[result.index, "snomed_short_name"]

    if use_multi_metric:
        result[("Task id", "", "")] = task_ids
        result[("Task (Procedure)", "", "")] = task_labels
    else:
        result[("Task id", "")] = task_ids
        result[("Task (Procedure)", "")] = task_labels

    for col in result.columns:
        if col[0] == "Support":
            result[col] = result[col].astype("Int64")

    for dataset_label in DATASET_LABELS:
        if use_multi_metric:
            for metric_name in metrics_list:
                col_a = ("Medgemma-27b", dataset_label, metric_name)
                col_b = ("Qwen3-8b", dataset_label, metric_name)
                if col_a in result.columns and col_b in result.columns:
                    result[("LLM Average", dataset_label, metric_name)] = (
                        result[col_a] + result[col_b]
                    ) / 2
        else:
            col_a = ("Medgemma-27b", dataset_label)
            col_b = ("Qwen3-8b", dataset_label)
            if col_a in result.columns and col_b in result.columns:
                result[("LLM Average", dataset_label)] = (
                    result[col_a] + result[col_b]
                ) / 2

    desired_order = [
        "Task id",
        "Task (Procedure)",
        "Support",
        "Rule based",
        "Medgemma-27b",
        "Qwen3-8b",
        "LLM Average",
        "Fully supervised",
        "All tasks",
        "All tasks on Medgemma vote",
        "Task specific",
    ]

    ordered_columns: List[Tuple[str, ...]] = []
    if use_multi_metric:
        for name in desired_order:
            if name == "Task id":
                col = (name, "", "")
                if col in result.columns:
                    ordered_columns.append(col)
                continue
            if name == "Task (Procedure)":
                col = (name, "", "")
                if col in result.columns:
                    ordered_columns.append(col)
                continue
            if name == "Support":
                for dataset_label in DATASET_LABELS:
                    col = (name, dataset_label, "support")
                    if col in result.columns:
                        ordered_columns.append(col)
                continue
            for dataset_label in DATASET_LABELS:
                for metric_name in metrics_list:
                    col = (name, dataset_label, metric_name)
                    if col in result.columns:
                        ordered_columns.append(col)
    else:
        for name in desired_order:
            if name in {"Task id", "Task (Procedure)"}:
                col = (name, "")
                if col in result.columns:
                    ordered_columns.append(col)
                continue
            if name == "Support":
                for dataset_label in DATASET_LABELS:
                    col = (name, dataset_label)
                    if col in result.columns:
                        ordered_columns.append(col)
                continue
            for dataset_label in DATASET_LABELS:
                col = (name, dataset_label)
                if col in result.columns:
                    ordered_columns.append(col)

    for col in result.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)

    result = result.loc[:, ordered_columns]

    if output_csv_path is not None:
        output_path = Path(output_csv_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path)
        logger.info("Saved metrics table to %s", output_path)

    return result
