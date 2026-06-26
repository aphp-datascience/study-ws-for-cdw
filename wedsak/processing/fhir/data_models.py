"""Pydantic models for representing a FHIR Observation payload."""

from datetime import date
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from wedsak.processing.fhir.utils import populate_reference_from_id


class Coding(BaseModel):
    system: Optional[str] = None
    code: Optional[Union[int, str]] = None
    display: Optional[str] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


class Meta(BaseModel):
    security: Optional[List[Coding]] = None
    source: Optional[str] = None


class Code(BaseModel):
    coding: List[Coding]

    class Config:
        populate_by_name = True
        extra = "ignore"


class SubjectReference(BaseModel):
    reference: Optional[str] = None
    reference_id: Optional[Union[int, str]] = Field(None, alias="_subjectReferenceId")

    @model_validator(mode="before")
    @classmethod
    def populate_reference_from_id(cls, data: Any) -> Any:
        return populate_reference_from_id(cls, data, "Patient")

    class Config:
        populate_by_name = True
        extra = "ignore"


class DocumentReference(BaseModel):
    reference: str
    reference_id: Optional[Union[int, str]] = Field(
        None,
        alias="_documentReferenceId",
        description="An optional ID for the document reference,"
        "used to populate the 'reference' field if it's missing. "
        "The 'reference' field will be populated as 'DocumentReference/{reference_id}' "
        "if 'reference_id' is provided and 'reference' is missing or empty.",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_reference_from_id(cls, data: Any) -> Any:
        return populate_reference_from_id(cls, data, "DocumentReference")

    class Config:
        populate_by_name = True
        extra = "ignore"


class SpanRelationship(BaseModel):
    reference_span_id: Optional[Union[int, str]] = Field(
        None,
        description="An optional ID referencing another span that is related to this NLP extraction result.",
    )
    relation_type: Optional[str] = Field(
        None,
        description="The type of relationship between the spans (e.g., 'partOf', 'temporal', 'causal','before', etc.).",
    )


class ExtensionSpanNLP(BaseModel):
    id: Optional[Union[int, str]] = Field(
        None, alias="id", description="An optional ID for the NLP extraction result."
    )
    related_span: Optional[SpanRelationship] = Field(
        None,
        alias="related_span",
        description="An optional ID referencing another span that is related to this NLP extraction result.",
    )
    document_reference: Optional[DocumentReference] = Field(
        None,
        description="Reference to the document from which the NLP extraction was made.",
    )
    model_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score from the NLP model, between 0 and 1.",
    )
    text_evidence: Optional[str] = Field(
        default=None,
        description="A context around the span that served as evidence for the NLP extraction.",
    )
    span_text: Optional[str] = Field(
        default=None,
        description="The actual text span extracted by the NLP model.",
    )
    span_start: Optional[int] = Field(
        default=None,
        description="The starting index of the extracted span in the text.",
    )
    span_end: Optional[int] = Field(
        default=None,
        description="The ending index of the extracted span in the text.",
    )
    attributes: Optional[dict] = Field(
        default=None,
        description="Additional attributes related to the NLP extraction.",
    )
    span_label: Optional[str] = Field(
        default=None,
        description="An optional label for the span, which can be used to categorize the type"
        " of information extracted (e.g., 'date', 'drug', 'measurement', etc.).",
    )


class MemberReference(BaseModel):
    reference: str
    reference_id: Optional[Union[int, str]] = Field(None, alias="_observationId")

    class Config:
        populate_by_name = True
        extra = "ignore"


class EncounterReference(BaseModel):
    reference: str
    reference_id: Optional[Union[int, str]] = Field(None, alias="_encounterId")

    class Config:
        populate_by_name = True
        extra = "ignore"


class Procedure(BaseModel):
    """Model representing a FHIR Procedure resource.

    Parameters
    -----------
    - status: The status of the procedure (e.g., "completed", "not-done", "unknown", etc).
    - code: The code representing the procedure, using a coding system like SNOMED CT or CCAM.
    - subject: Reference to the patient on whom the procedure was performed.
    - performedDateTime: The date and time when the procedure was performed.
    - meta: Metadata about the procedure, including security labels.
        - source: Optional string indicating the source of the procedure information.
            It could be the ID of the NLP model or extraction pipeline that generated this procedure record.
        - security: A list of security labels to indicate the provenance and trustworthiness of the procedure information.
            Notably, it can include a label with system "http://terminology.hl7.org/CodeSystem/v3-ObservationValue" and code "AIAST"
    - encounter: Reference to the encounter during which the procedure was performed.
    - extension_nlp: A list of NLP extraction results related to the procedure,
        including text evidence, model confidence, and span information.

    Cardinalities
    -------------
    - status: 1..1
    - code: 1..1
    - subject: 1..1
    - performedDateTime: 0..1
    - meta: 0..1
    - encounter: 0..1
    - extension_nlp: 0..*

    Example usage
    -------------
    ```python
    from datetime import date
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
            meta={"security": [
                            {"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue",
                            "code": "AIAST",
                            "display": "Artificial Intelligence asserted"}
                                ],
                },
            extension_nlp=[
                {
                    "document_reference": {
                        "reference_id": 456
                    },
                    "model_confidence": 0.95,
                    "text_evidence": "27 février 2026",
                    "span_start": 350,
                    "span_end": 365,
                    "attributes": {"negation": False}
                }]
    )
    """

    resourceType: str = Field(
        default="Procedure", description="The FHIR resource type, fixed to 'Procedure'"
    )
    id: Optional[str] = None
    status: str = Field(
        default="completed",
        description="The status of the procedure (e.g., 'completed', 'not-done', 'unknown', etc).",
    )
    code: Code = Field(
        description="The code representing the procedure, using a coding system like SNOMED CT or CCAM."
    )
    subject: SubjectReference = Field(
        description="Reference to the patient on whom the procedure was performed."
    )
    performedDateTime: Optional[date] = Field(
        default=None, description="The date and time when the procedure was performed."
    )
    meta: Optional[Meta] = Field(
        default=None,
        description="Metadata about the procedure, including security labels.",
    )
    encounter: Optional[EncounterReference] = Field(
        default=None,
        description="Reference to the encounter during which the procedure was performed.",
    )
    extension_nlp: Optional[List[ExtensionSpanNLP]] = Field(
        default=None,
        description="A list of NLP extraction results related to the procedure.",
    )

    @classmethod
    def from_span(
        cls,
        *,
        doc: Any,
        span: Any,
        normalized_task_name: str,
        task_mapping: dict,
        model_confidence: Optional[float] = None,
        text_evidence: Optional[str] = None,
        status: str = "completed",
        default_day: int = 15,
    ) -> "Procedure":
        """Build a Procedure instance from a document span and task mapping.
        The task mapping is used to populate the code field of the Procedure resource based on the normalized task name."""

        task_info = task_mapping.get(normalized_task_name, {})
        procedure_dict = dict(
            status=status,
            extension_nlp=[
                dict(
                    id=(
                        f"{doc._.note_id}-{span.start_char}-{span.end_char}-{span.label_}-{normalized_task_name}"
                    ),
                    document_reference=dict(reference_id=doc._.note_id),
                    model_confidence=model_confidence,
                    text_evidence=text_evidence,
                    span_start=span.start_char,
                    span_end=span.end_char,
                    span_label=span.label_,
                    attributes=dict(mode=str(span._.date.mode)),
                    span_text=span.text,
                )
            ],
            code=dict(
                coding=[
                    dict(
                        code=task_info.get("code__coding___code"),
                        display=task_info.get("code__coding___display"),
                        system=task_info.get("code__coding___system"),
                    )
                ]
            ),
            subject=dict(reference_id=doc._.person_id),
            performedDateTime=span._.date.to_datetime(
                infer_from_context=True,
                note_datetime=doc._.note_datetime,
                default_day=default_day,
            ).date(),
            meta=dict(
                security=[
                    dict(
                        system="http://terminology.hl7.org/CodeSystem/v3-ObservationValue",
                        code="AIAST",
                        display="Artificial Intelligence asserted",
                    )
                ]
            ),
        )
        return cls(**procedure_dict)

    class Config:
        populate_by_name = True
        extra = "ignore"
