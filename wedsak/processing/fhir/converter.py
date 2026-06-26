from wedsak.misc.getters import get_task_mapping_to_fhir_coding
from pydantic import BaseModel
from typing import Callable, List, Optional, Union
from wedsak.processing.fhir.data_models import Procedure
from wedsak.processing.fhir.utils import FlatRessource
from edsnlp.utils.span_getters import get_spans, validate_span_getter


class SpantoFHIRConverter:
    def __init__(
        self,
        span_task_attributes: List[str],
        span_getter: List[str],
        task_mapping: dict = get_task_mapping_to_fhir_coding(),
        fhir_ressource: BaseModel = Procedure,
        k: int = 25,
        include_context: bool = True,
        context_getter: Optional[Callable[[object, object], str]] = None,
        prob_threshold: float = 0.5,
    ):
        self.span_task_attributes = span_task_attributes
        self.span_getter = span_getter
        self.task_mapping = task_mapping
        self.k = k
        self.include_context = include_context
        self.context_getter = context_getter
        self.prob_threshold = prob_threshold
        self.fhir_ressource = fhir_ressource
        self.flatter = FlatRessource(fhir_ressource)

    def __call__(
        self,
        doc,
    ):
        span_getter = validate_span_getter(self.span_getter)
        fhir_instances_list = []
        for span in get_spans(doc, span_getter):
            for normalized_task_name in self.span_task_attributes:
                model_confidence = span._.prob.get("_." + normalized_task_name, {}).get(
                    "1"
                )
                if model_confidence is None or model_confidence < self.prob_threshold:
                    continue

                if self.include_context:
                    if self.context_getter is None:
                        text_evidence = doc[
                            span.start - self.k : span.end + self.k
                        ].text
                    else:
                        text_evidence = self.context_getter(doc, span)
                else:
                    text_evidence = None

                fhir_instance = self.fhir_ressource.from_span(
                    doc=doc,
                    span=span,
                    normalized_task_name=normalized_task_name,
                    task_mapping=self.task_mapping,
                    model_confidence=model_confidence,
                    text_evidence=text_evidence,
                )

                fhir_instance_flat = self.flatter(fhir_instance)
                fhir_instances_list.append(fhir_instance_flat)
        return fhir_instances_list
