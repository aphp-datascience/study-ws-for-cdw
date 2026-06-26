from pydantic import Field, create_model
from spacy.tokens import Doc
from wedsak.misc.data_wrangling import SafeDict
from wedsak.misc.getters import get_tasks
import logging
from pathlib import Path
from typing import Optional, TypedDict
from wedsak.misc.data_wrangling import normalize_task_name
import edsnlp.pipes as eds


def format_examples(raw_examples, prefix_prompt, suffix_prompt):
    examples = []

    for date, context, answer in raw_examples:
        prompt = (
            prefix_prompt.format(span=date) + context + suffix_prompt.format(span=date)
        )
        examples.append((prompt, answer))

    return examples


class ContextFormatter:
    def __init__(self, prefix: str, suffix: str):
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, context: Doc) -> str:
        span = context.ents[0].text if context.ents else ""
        prefix = self.prefix.format(span=span)
        suffix = self.suffix.format(span=span)
        return f"{prefix}{context.text}{suffix}"


def format_prompt(
    prompt: str,
    attribute: str,
    snomed_short_name: str,
) -> str:
    if prompt:
        return prompt.format_map(
            SafeDict(attribute=attribute, snomed_short_name=snomed_short_name)
        )
    else:
        return prompt


def initialize_llm_lf_module(
    task_id: int,
    prompt_id: str,
    path_prompts: str,
    model_name: str,
    api_url: str,
    verbose: bool = False,
    timeout: Optional[int] = None,
    context_getter: str = "words[-75:75]",
    span_getter: str = "dates",
    temperature: float = 0,
    max_tokens: int = 4000,
    max_concurrent_requests: int = 200,
    extra_body: dict = {},
    schema_type: str = "dict",
) -> dict:
    """
    Initialize LLM-based labelling function module.
    Returns a dictionary with LF name, span classifier, and snomed short name.

    -----------
    Parameters:
    - task_id: ID of the task.
    - prompt_id: ID of the prompt to use.
    - path_prompts: Path to the prompts file.
    - model_name: Name of the LLM model.
    - api_url: API URL of the LLM model.
    - verbose: Whether to print verbose output.
    - timeout: Timeout for API requests.
    - context_getter: How to get context around the span.
    - span_getter: How to get the span from the document.
    - temperature: Temperature setting for the LLM.
    - max_tokens: Maximum tokens for the LLM response.
    - max_concurrent_requests: Maximum concurrent requests to the LLM API.
    - extra_body: Extra parameters to pass to the LLM API.
    - schema_type: Type of output schema ("boolean" or other).

    -----------
    Returns:
    - dict with keys 'lf_name', 'span_classifier', 'snomed_short_name'.
    """
    # Setup logger
    logger = logging.getLogger()
    logger.info("Prompt ID: %s", prompt_id)

    ##############################################################################
    # Task

    tasks = get_tasks()
    task_row = tasks.loc[tasks.task_id == task_id].iloc[0]
    group_id = task_row["Group"]
    snomed_short_name = task_row["snomed_short_name"]
    normalized_task_name = task_row["normalized_task_name"]
    logger.info(f"Task ID: {task_id}, Normalized task name: {normalized_task_name}")
    logger.info(f"Snomed short name: {snomed_short_name}, Group ID: {group_id}")
    ##############################################################################
    # Read prompts

    import json

    path_prompts = Path(path_prompts).expanduser()
    if path_prompts.exists():
        with open(path_prompts, "r") as f:
            task_prompts = json.load(f)
    else:
        task_prompts = {}
    raw_examples = task_prompts.get(prompt_id).get("examples", [])
    prefix_prompt = task_prompts.get(prompt_id).get("prefix_prompt")
    system_prompt = task_prompts.get(prompt_id).get("system_prompt")
    suffix_prompt = task_prompts.get(prompt_id).get("suffix_prompt")

    prefix_prompt = format_prompt(
        prefix_prompt,
        attribute=normalized_task_name,
        snomed_short_name=snomed_short_name,
    )

    system_prompt = format_prompt(
        system_prompt,
        attribute=normalized_task_name,
        snomed_short_name=snomed_short_name,
    )

    suffix_prompt = format_prompt(
        suffix_prompt,
        attribute=normalized_task_name,
        snomed_short_name=snomed_short_name,
    )

    examples = format_examples(
        raw_examples, prefix_prompt, suffix_prompt
    )  # TODO modify to new API #FIXME
    if len(examples) == 0:
        examples = None

    # Define output schema
    if schema_type == "boolean":
        # Example:
        # ent._.biopsy_procedure → False
        task_schema = create_model(
            "Schema",
            **{
                snomed_short_name: (
                    bool,
                    Field(..., description=f"Is the span a {snomed_short_name} or not"),
                )
            },
        )
    else:
        # Example:
        # ent._.biopsy_procedure → {'biopsy_procedure': False}
        task_schema = TypedDict("Schema", {snomed_short_name: bool})

    # Set up context formatter
    context_formatter = ContextFormatter(prefix=prefix_prompt, suffix=suffix_prompt)

    # Define LF name
    lf_name = normalized_task_name + "_" + normalize_task_name(Path(model_name).name)

    ##############################################################################
    if verbose:
        print("########## system prompt ##########\n")
        print(system_prompt)

        print("########## examples ##########\n")
        print(examples)

        print("########## prefix prompt ##########\n")
        print(prefix_prompt)

        print("########## suffix prompt ##########\n")
        print(suffix_prompt)

    ##############################################################################
    llm_span_classifier = eds.llm_span_qualifier(
        api_url=api_url,
        model=model_name,
        prompt=system_prompt,
        span_getter=span_getter,
        context_getter=context_getter,
        context_formatter=context_formatter,
        attributes={f"_.{lf_name}": True},  # [normalized_task_name],
        output_schema=task_schema,
        examples=examples,
        max_concurrent_requests=max_concurrent_requests,
        seed=0,
        use_retriever=False,
        on_error="warn",
        timeout=timeout,
        api_kwargs=dict(
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        ),
    )
    return {
        "lf_name": lf_name,
        "span_classifier": llm_span_classifier,
        "snomed_short_name": snomed_short_name,
    }
