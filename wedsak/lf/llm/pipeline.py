import os
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import edsnlp
import edsnlp.pipes as eds
import pandas as pd

from wedsak.lf.llm.llm_utils import initialize_llm_lf_module
from wedsak.misc.data_wrangling import SpanToRowConverter
from wedsak.misc.logger_utils import setup_logger
from wedsak.misc.getters import get_lf_reference

USER = os.getenv("USER")


def llm_lf_pipeline(
    task_ids: List[Union[int, str]],
    prompt_ids: Dict[Union[int, str], Union[int, str]],
    path_notes: str,
    path_prompts: str,
    path_save_votes: Optional[str] = None,
    log_level: str = "INFO",
    api_key: str = "EMPTY_API_KEY",
    model_names: Dict[str, str] = {},
    temperature: float = 0,
    batch_size: Union[int, float, str] = "24 docs",
    max_concurrent_requests: int = 200,
    timeout: Optional[int] = None,
    context_getter: str = "words[-75:75]",
    max_tokens: int = 4000,
    extra_body: dict = {},
    span_getter: str = "dates",
    path_lf_reference: str = f"/export/home/{USER}/wedsak/data/LF_definition.xlsx",
    debug: int = 0,
    default_prompt_id: Union[int, str] = "14",
    schema_type: str = "dict",
) -> pd.DataFrame:
    """
    Apply Labelling Functions (LF) to notes.
    The LF are based on:
    - LLM span qualifier

    Parameters
    ----------
    task_ids : List[Union[int, str]]
        List of task IDs to process.
    prompt_ids : Dict[Union[int, str], Union[int, str]]
        Mapping from task ID to prompt ID.
    path_notes : str
        Path to the notes file (pickle format).
    path_prompts : str
        Path to the prompts directory.
    path_save_votes : Optional[str], optional
        Path to save the votes (pickle format), by default None.
    log_level : str, optional
        Logging level, by default "INFO".
    api_key : str, optional
        API key for LLM access, by default "EMPTY_API_KEY".
    model_names : Dict[str, str], optional
        Mapping from model name to API URL, by default {}.
    temperature : float, optional
        Temperature for LLM generation, by default 0.
    batch_size : int, optional
        Batch size for processing, by default 24.
    max_concurrent_requests : int, optional
        Maximum number of concurrent requests to the LLM API, by default 200.
    timeout : Optional[int], optional
        Timeout for LLM API requests, by default None.
    context_getter : str, optional
        Getter for context around the span, by default "words[-75:75]".
    max_tokens : int, optional
        Maximum number of tokens for LLM response, by default 4000.
    extra_body : dict, optional
        Extra body parameters for LLM API requests, by default {}.
    span_getter : str, optional
        Getter for spans to classify, by default "dates".
    path_lf_reference : str, optional
        Path to the LF reference Excel file, by default "/export/home/{USER}/wedsak/data/LF_definition.xlsx".
    debug : int, optional
        If > 0, process only the first `debug` notes, by default 0.
    default_prompt_id : Union[int, str], optional
        Default prompt ID to use if not specified for a task, by default "12".
    schema_type : str, optional
        Type of output schema ("boolean" or other), by default "dict".
    Returns
    -------
    pd.DataFrame
        DataFrame containing the votes for each span.
    """
    # Setup logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = setup_logger(log_level, script_name="llm_lf")

    # Warning if path_save_votes is None
    if path_save_votes is None:
        logger.warning(
            "path_save_votes is None. The results will not be saved to disk."
        )

    # Read notes
    path_notes = Path(path_notes).expanduser()
    if path_notes.suffix == ".csv":
        df = pd.read_csv(path_notes)
    else:
        df = pd.read_pickle(
            path_notes,
        )

    if debug > 0:
        df = df.iloc[:debug]
        logger.warning("Debug mode: processing only %d notes", len(df))

    # Read LF reference
    ref_lfs, _ = get_lf_reference(path_lf_reference)
    ##############################################################################

    logger.info(f"Models: {model_names.keys()}")
    logger.info("Path to prompts: %s", path_prompts)
    logger.info("Temperature: %s", temperature)
    logger.info("Max tokens: %s", max_tokens)
    logger.info("Context getter: %s", context_getter)
    if path_save_votes is None:
        logger.warning("The results will not be saved to disk.")
    ##############################################################################

    # Set API key
    os.environ["OPENAI_API_KEY"] = api_key

    ##############################################################################
    # Define NLP pipeline
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe(eds.dates())
    lf_names = []
    snomed_short_names = []
    for task_id in task_ids:
        for model_name, api_url in model_names.items():
            prompt_id = prompt_ids.get(task_id, default_prompt_id)
            logger.info(
                f"-------------------------------------------------------\n"
                f"Initializing LLM LF for task {task_id} with prompt {prompt_id} using model {model_name} at {api_url}"
            )

            llm_lf = initialize_llm_lf_module(
                task_id=int(task_id),
                prompt_id=prompt_id,
                path_prompts=path_prompts,
                model_name=model_name,
                api_url=api_url,
                verbose=False,
                timeout=timeout,
                context_getter=context_getter,
                span_getter=span_getter,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrent_requests=max_concurrent_requests,
                extra_body=extra_body,
                schema_type=schema_type,
            )

            snomed_short_name = llm_lf.get("snomed_short_name")
            lf_name = llm_lf.get("lf_name")
            span_classifier = llm_lf.get("span_classifier")

            assert lf_name in ref_lfs.ref_name.unique(), (
                f"LF name {lf_name} not found in reference (Excel LF definition)."
            )

            lf_names.append(lf_name)
            snomed_short_names.append(snomed_short_name)
            nlp.add_pipe(span_classifier)
            logger.info(
                f"Added LLM LF: {lf_name} using model {model_name} point to {api_url}"
            )

    ##############################################################################
    ## Process docs
    doc_iterator = edsnlp.data.from_pandas(
        df,
        converter="omop",
        doc_attributes=["person_id"],
    )

    docs = doc_iterator.map_pipeline(nlp, batch_size)
    # docs = docs.set_processing(
    #     backend="multiprocessing",
    #     deterministic=False,
    #     num_cpu_workers=-1,
    #     show_progress=True,
    # )

    print("## LF names ##")
    for i, name in enumerate(lf_names, 1):
        print(i, name)

    # Convert each doc to a list of dicts (one by entity)
    converter = SpanToRowConverter(
        span_attributes=lf_names,
        span_getter=["dates"],
        k=25,
    )
    # and store the result in a pandas DataFrame
    t0 = datetime.datetime.now()
    note_nlp = docs.to_pandas(
        converter=converter,
    )

    t1 = datetime.datetime.now()
    logger.info("Time to process: %s", t1 - t0)
    logger.info("Number of entities (dates) found: %d", len(note_nlp))

    # Parse the dictionary results to extract votes (hypothesis: one task per LF)
    if schema_type == "dict":
        for lf_name, snomed_short_name in zip(lf_names, snomed_short_names):
            note_nlp[lf_name] = note_nlp[lf_name].map(
                lambda x: x.get(snomed_short_name) if isinstance(x, dict) else None
            )

    # Write results
    if path_save_votes is not None:
        path_save_votes = Path(path_save_votes).expanduser()
        path_save_votes.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving results to %s", path_save_votes)
        note_nlp.to_pickle(path_save_votes)
    return note_nlp
