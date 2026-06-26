import datetime
from pathlib import Path
from typing import Annotated, Optional

import edsnlp
import edsnlp.pipes as eds
import pandas as pd
import typer
from edsnlp.pipes.qualifiers.contextual.contextual import (
    ContextualQualifier,
)
from edsnlp.pipes.qualifiers.external_information.external_information import (
    ExternalInformation,
    ExternalInformationQualifier,
)
from wedsak.lf.rule_based.logical import LF_based_on_LF
from wedsak.misc.data_wrangling import SpanToRowConverter
from wedsak.misc.logger_utils import setup_logger
from confit import Cli
from wedsak.misc.getters import get_lf_reference
from wedsak.misc.constants import PATH_LF_REFERENCE

app = Cli()


@app.command(name="lf_no_llm")
def apply_lf(
    path_notes: Annotated[str, typer.Option()],
    path_external_knowledge: Annotated[str, typer.Option()],
    path_patterns_contextual_qualifier: Annotated[str, typer.Option()],
    path_save_votes: Annotated[Optional[str], typer.Option()],
    path_logical_relation_qualifier: Annotated[Optional[str], typer.Option()] = None,
    batch_size: Annotated[int, typer.Option()] = 24,
    log_level: Annotated[str, typer.Option()] = "INFO",
    path_lf_reference: Annotated[str, typer.Option()] = PATH_LF_REFERENCE,
):
    """
    Apply Labelling Functions (LF) to notes.
    The LF are based on:
    - External Knowledge (EK)
    - Contextual patterns
    - Logical relations between LFs

    # Il y a k LFs (labelling functions) et p Tasks.
    #
    # Pour chaque entité (span de date) nous avons :
    #
    # |     |T1|T2|...|Tp|
    # |---- |--|--|---|--|
    # |LF1  |  |x |   |  |
    # |LF2  |  |x |x  |  |
    # |...  |  |  |   |  |
    # |LFk  |  |  |   |x |

    """
    ## Setup logger
    logger = setup_logger(log_level, script_name="no_llm_lf")
    logger.info("LF annotation started")
    path_save_votes = Path(path_save_votes).expanduser()
    path_patterns_contextual_qualifier = Path(
        path_patterns_contextual_qualifier
    ).expanduser()

    path_external_knowledge = Path(path_external_knowledge).expanduser()
    df_ek = pd.read_pickle(path_external_knowledge)

    path_notes = Path(path_notes).expanduser()
    if path_notes.suffix == ".csv":
        df = pd.read_csv(path_notes)
    else:
        df = pd.read_pickle(
            path_notes,
        )

    df = df.merge(df_ek, on="person_id", how="inner")

    logger.info(f"Number of lines in merged DataFrame: {len(df)}")

    # Read LF reference
    ref_lfs, _ = get_lf_reference(path_lf_reference)

    ##############################################################################
    # Define NLP pipeline
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.normalizer())
    nlp.add_pipe(eds.sentences(check_capitalized=False, min_newline_count=1))
    nlp.add_pipe(eds.dates())

    ##############################################################################
    ## Add ExternalInformation pipe
    lf_names_external_knowledge = df_ek.columns

    external_information = {
        lf: ExternalInformation(
            doc_attr=f"_.{lf}",
            span_attribute="_.date.to_datetime()",
            threshold=datetime.timedelta(days=0),
        )
        for lf in lf_names_external_knowledge
    }
    external_information

    nlp.add_pipe(
        ExternalInformationQualifier(
            nlp=nlp, span_getter="dates", external_information=external_information
        )
    )
    ##############################################################################
    ## Add  ContextualQualifier pipe

    # Opening JSON file
    with open(path_patterns_contextual_qualifier, "r") as openfile:
        import json

        # Reading from json file
        patterns = json.load(openfile)

    context_patterns = {}
    for i, fsn in enumerate(patterns, 1):
        lf_name = patterns.get(fsn).pop("lf_name")
        assert lf_name in ref_lfs.ref_name.unique(), (
            f"LF name {lf_name} not found in reference (Excel LF definition)."
        )
        context_patterns[lf_name] = {fsn: patterns.get(fsn)}

    for key, value in context_patterns.items():
        print(key, value.keys())

    nlp.add_pipe(ContextualQualifier(span_getter="dates", patterns=context_patterns))

    ##############################################################################
    ## Add logical LFs
    if path_logical_relation_qualifier is not None:
        path_logical_relation_qualifier = Path(
            path_logical_relation_qualifier
        ).expanduser()
        # Opening JSON file
        with open(path_logical_relation_qualifier, "r") as openfile:
            import json

            # Reading from json file
            logical_relations = json.load(openfile)

        logical_relations_lf_names = []

        for lr in logical_relations:
            lf_name = lr.get("name")
            assert lf_name in ref_lfs.ref_name.unique(), (
                f"LF name {lf_name} not found in reference (Excel LF definition)."
            )
            logger.info(f"Adding logical relation LF: {lf_name}")
            nlp.add_pipe(
                LF_based_on_LF(
                    span_getter="dates",
                    lf_in=lr.get("lf_in"),
                    attribute_out=lf_name,
                    value_out=lr.get("value_out"),
                    relation=lr.get("relation"),
                )
            )
            logical_relations_lf_names.append(lf_name)
    ##############################################################################
    ## Process docs
    doc_iterator = edsnlp.data.from_pandas(
        df,
        converter="omop",
        doc_attributes=lf_names_external_knowledge.to_list()  # External Knowledge information
        + ["person_id"],
    )

    docs = doc_iterator.map_pipeline(nlp, batch_size)
    docs = docs.set_processing(
        backend="multiprocessing",
        deterministic=False,
        num_cpu_workers=-1,
        show_progress=True,
    )

    lf_names = (
        list(lf_names_external_knowledge)
        + list(context_patterns.keys())
        + logical_relations_lf_names
    )

    # Check that all LF names are in reference
    logger.info("## LF names ##")
    for i, lf_name in enumerate(lf_names, 1):
        logger.info("i=%s lf_name=%s", i, lf_name)
        assert lf_name in ref_lfs.ref_name.unique(), (
            f"LF name {lf_name} not found in reference (Excel LF definition)."
        )

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

    path_save_votes.parent.mkdir(parents=True, exist_ok=True)
    note_nlp.to_pickle(path_save_votes)


if __name__ == "__main__":
    # typer.run(apply_lf)
    app()
