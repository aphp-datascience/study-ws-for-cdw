from spacy.tokens import Doc
from typing import List, Dict, Optional
from edsnlp.utils.collections import get_deep_attr
import edsnlp
from pret_joy import Box, Divider, Stack

from metanno.recipes.data_widget_factory import DataWidgetFactory, infer_fields
from pret.hooks import use_ref
from pret.react import div
from pret_simple_dock import Layout, Panel


def convert_ents_to_metanno_ents(
    doc: Doc, attributes: Optional[List[str]] = None
) -> Dict:
    ents = []
    if doc.has_extension("note_id"):
        note_id = doc._.note_id
    else:
        note_id = None
    for ent in doc.ents:
        if attributes:
            collected_attributes = [
                {"label": a, "value": get_deep_attr(ent, f"_.{a}")} for a in attributes
            ]
        else:
            collected_attributes = []

        ent_label = ent.label_
        begin = ent.start_char
        end = ent.end_char

        dict_ent = {
            "id": "automatic"
            + "-{label}-{start}-{end}".format(label=ent_label, start=begin, end=end),
            "text": ent.text,
            "label": ent_label,
            "begin": begin,
            "end": end,
            # "attributes": collected_attributes,
        }

        ents.append(dict_ent)
    dict_doc = {
        "note_id": str(note_id),
        "person_id": str(doc._.person_id),
        "note_date": str(doc._.note_date),
        "visit_start_datetime": str(doc._.visit_start_datetime),
        "entities": ents,
        "note_text": doc.text,
        "annotation_order": doc._.annotation_order,
        "seen": False,
    }

    return dict_doc


def build_data(df, nlp):
    # Apply pipeline
    docs = edsnlp.data.from_pandas(
        df,
        converter="omop",
        doc_attributes=[
            "birth_datetime",
            "visit_start_datetime",
            "visit_occurrence_id",
            "person_id",
            "note_id",
            "note_date",
            "note_class_source_value",
            "note_title",
            "note_text",
            "screening_group",
            "annotation_order",
        ],
    )
    docs = docs.map_pipeline(nlp)

    # And assemble your data in collections of dicts
    notes = []
    for idx, doc in enumerate(docs):
        notes.append(convert_ents_to_metanno_ents(doc))
    return {"notes": notes}


class CreateInitialData:
    def __init__(self, df, nlp=None):
        if nlp is None:
            self.nlp = self.std_nlp()
        else:
            self.nlp = nlp
        self.df = df

    def std_nlp(self):
        nlp = edsnlp.blank("eds")
        nlp.add_pipe("sentencizer")
        nlp.add_pipe(
            edsnlp.pipes.dates(
                span_setter={
                    "dates": ["date"],
                    "durations": ["duration"],
                    "periods": ["period"],
                    "ents": ["date", "duration", "period"],
                }
            )
        )
        return nlp

    def run(self):
        return build_data(self.df, self.nlp)


def get_label_config(tasks_names, add_shortcuts=True):
    label_config = {}

    label_config = {task_name: {"color": c} for task_name, c in tasks_names.items()}
    label_config["date"] = {"name": "Date", "color": "lightblue", "shortcut": "d"}
    shortcuts = set(["d"])
    if add_shortcuts:
        for lab in tasks_names.keys():
            letter = next((c for c in lab.lower() if c not in shortcuts), None)
            if letter:
                shortcuts.add(letter)
                label_config[lab]["shortcut"] = letter

    return label_config


def get_annotation_ui(initial_data_generator, sync_path=None, label_config=None):
    factory = DataWidgetFactory(
        data=initial_data_generator,
        sync=sync_path,
    )

    note_text_view, ent_view = factory.create_text_widget(
        store_text_key="notes",
        store_spans_key="notes.entities",
        text_key="note_text",
        text_primary_key="note_id",
        spans_primary_key="id",
        fields=infer_fields(
            [e for n in factory.data["notes"] for e in n["entities"]],
            visible_keys=["label"],
            editable_keys=["label"],
            categorical_keys=["label"],
        ),
        labels=label_config,
    )

    note_form_handle = use_ref()
    note_form_view = factory.create_form_widget(
        store_key="notes",
        primary_key="note_id",
        fields=[
            {"key": "note_id", "kind": "text"},
            {"key": "seen", "kind": "boolean", "editable": True},
        ],
        add_navigation_buttons=True,
        handle=note_form_handle,
    )

    notes_table_handle = use_ref()
    notes_view = factory.create_table_widget(
        store_key="notes",
        primary_key="note_id",
        # Instead of using infer_fields, we can also define the
        # fields manually which can actually be simpler
        fields=[  # type: ignore
            {
                "key": "annotation_order",
                "name": "annotation_order",
                "kind": "text",
                "filterable": True,
            },
            {"key": "note_id", "name": "note_id", "kind": "text", "filterable": True},
            {
                "key": "seen",
                "name": "seen",
                "kind": "boolean",
                "editable": True,
                "filterable": True,
            },  # noqa: E501
        ],
        style={"--min-notebook-height": "300px"},
        handle=notes_table_handle,
    )

    info_view = Stack(
        Box(note_form_view, sx={"m": "10px"}),
        Divider(),
        Box(ent_view, sx={"m": "10px"}),
    )

    note_header = factory.create_selected_field_view(
        store_key="notes",
        shown_key="note_id",
        fallback="Note",
    )

    layout = div(
        Layout(
            Panel(note_text_view, key="Note Text", header=note_header),
            Panel(info_view, key="Info"),
            Panel(notes_view, key="NoteID"),
            default_config={
                "kind": "row",
                "children": [
                    {"tabs": ["Note Text"], "size": 50},
                    {
                        "kind": "column",
                        "size": 50,
                        "children": [
                            {"tabs": ["Info"], "size": 75},
                            {"tabs": ["NoteID"], "size": 25},
                        ],
                    },
                ],
            },
        ),
        # factory.create_connection_status_bar(),
        style={
            "background": "var(--joy-palette-background-level2, #f0f0f0)",
            "width": "100%",
            "height": "97vh",
            "minHeight": "600px",
            "--sd-background-color": "transparent",
        },
    )
    return layout
