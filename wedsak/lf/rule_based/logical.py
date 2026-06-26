from edsnlp.utils.extensions import rgetattr
from edsnlp.pipes.base import (
    BaseSpanAttributeClassifierComponent,
    SpanGetterArg,
)
from typing import Union, Dict
from spacy.tokens import Doc, Span
from edsnlp.core.pipeline import Pipeline
from edsnlp.utils.span_getters import get_spans, validate_span_getter


class LF_based_on_LF(BaseSpanAttributeClassifierComponent):
    def __init__(
        self,
        nlp: Pipeline,
        name: str = "LF_based_on_LF",
        *,
        span_getter: SpanGetterArg,
        lf_in: Dict[str, Union[str, bool]],
        attribute_out: str,
        value_out: Union[str, bool],
        relation: str,
    ):
        """
        A weak supervision labeling function (LF) that assigns a label to spans based on the values of other LF-assigned attributes.

        # Parameters
        nlp : Pipeline
            The edsnlp NLP pipeline.
        name : str, optional
            The name of the component, by default "LF_based_on_LF".
        span_getter : SpanGetterArg
            The span getter argument.
        lf_in : Dict[str, Union[str, bool]]
            A dictionary where keys are attribute names and values are the expected values for those attributes.
        lf_out : Tuple[str, Union[str, bool]]
            A tuple where the first element is the output attribute name and the second element is the expected value for that attribute.
        relation : str
            A string representing the logical relation between the input attributes.
            Supported operators are: and, or, not.

        # Example
        ```python
        nlp.add_pipe(
            LF_based_on_LF(
                span_getter="dates",
                lf_in={
                    "lf_6": "Fiberoptic bronchoscopy (procedure)",
                    "lf_18": "Anatomic pathology procedure (procedure)",
                },
                lf_out=("logical_lf", "Biopsy (procedure)"),
                relation="lf_6 and lf_18",
            )
        )
            ```

        """

        self.attribute_out = attribute_out
        self.value_out = value_out
        self.lf_in = lf_in
        self.relation = relation
        self.set_extensions()
        super().__init__(
            nlp=nlp,
            name=name,
            span_getter=validate_span_getter(span_getter),
        )

    def set_extensions(self) -> None:
        """
        Sets custom extensions on the Span object.
        """
        if not Span.has_extension(self.attribute_out):
            Span.set_extension(self.attribute_out, default=None)

    def __call__(self, doc: Doc):
        safe_globals = {"__builtins__": None}  # disable builtins
        for span in get_spans(doc, self.span_getter):
            span_attribute_results = {}
            for attribute, expected_value in self.lf_in.items():
                span_value = rgetattr(span, f"_.{attribute}")
                if isinstance(span_value, list):
                    result_attribute = expected_value in span_value
                elif isinstance(span_value, (str, bool)):
                    result_attribute = expected_value == span_value
                elif span_value is None:
                    result_attribute = expected_value is None
                else:
                    raise NotImplementedError
                span_attribute_results[attribute] = result_attribute
            try:
                result = bool(eval(self.relation, safe_globals, span_attribute_results))
            except Exception as e:
                raise ValueError(f"Error evaluating relation '{self.relation}': {e}")
            if result:
                span._.set(self.attribute_out, self.value_out)

        return doc
