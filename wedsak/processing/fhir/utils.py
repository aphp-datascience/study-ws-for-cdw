from typing import Any, Union

from pydantic import BaseModel


def populate_reference_from_id(cls, data: Any, prefix: str) -> Any:
    if not isinstance(data, dict):
        return data

    reference = data.get("reference")
    reference_id = data.get("reference_id")
    if (reference is None or reference == "") and reference_id is not None:
        data["reference"] = f"{prefix}/{reference_id}"
    return data


def flatten_value(value: Any, prefix: str = "") -> dict:
    if value is None:
        return {prefix: None} if prefix else {}

    if isinstance(value, BaseModel):
        return flatten_value(value.model_dump(exclude_none=False), prefix)

    if isinstance(value, dict):
        flat: dict = {}
        for key, item in value.items():
            next_prefix = f"{prefix}__{key}" if prefix else key
            flat.update(flatten_value(item, next_prefix))
        return flat

    if isinstance(value, list):
        flat: dict = {}
        if not value and prefix:
            flat[prefix] = []
            return flat

        for index, item in enumerate(value):
            next_prefix = f"{prefix}___{index}" if prefix else str(index)
            flat.update(flatten_value(item, next_prefix))
        return flat

    return {prefix: value} if prefix else {"value": value}


class FlatRessource:
    """
    Class to flatten a FHIR resource (e.g., Procedure) into a flat dictionary for 2D table representation.

    Example usage
    -------------
    ```python

    from datetime import date
    from wedsak.processing.fhir.data_models import Procedure, FlatRessource

    procedure = Procedure(
        status="completed",
        code={
            "coding": [
                {
                    "system": "https://www.snomed.org/",
                    "code": "113091000",
                    "display": "Magnetic resonance imaging (procedure)",
                }
            ]
        },
        subject={"reference_id": 123},
        performedDateTime=date(2026, 2, 27),
        meta={
            "security": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue",
                    "code": "AIAST",
                    "display": "Artificial Intelligence asserted",
                }
            ],
        },
        extension_nlp=[
            {
                "document_reference": {"reference_id": 456},
                "model_confidence": 0.95,
                "text_evidence": "27 février 2026",
                "span_start": 350,
                "span_end": 365,
                "attributes": {"negation": False},
            }
        ],
    )
    procedure_dumped = procedure.model_dump()

    flatter = FlatRessource(Procedure)
    flatter(procedure_dumped)
    ```
    """

    def __init__(self, resource_definition: Union[BaseModel, type]):
        self.resource_definition = resource_definition

    def __call__(self, object) -> dict:
        if isinstance(object, BaseModel):
            data = object.model_dump(exclude_none=False)
        else:
            data = object

        if isinstance(self.resource_definition, type) and issubclass(
            self.resource_definition, BaseModel
        ):
            resource_type = self.resource_definition.__name__
        elif isinstance(self.resource_definition, BaseModel):
            resource_type = self.resource_definition.__class__.__name__
        else:
            resource_type = None

        flattened = flatten_value(data)
        if resource_type is not None and "resourceType" not in flattened:
            flattened = {"resourceType": resource_type, **flattened}
        return flattened
