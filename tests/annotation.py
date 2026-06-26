from wedsak.processing.export_dataset import merge_highlight_annotations_with_dates
import pandas as pd


def test_merge_annotations():
    DATES_SPAN = [
        {"begin": 0, "end": 5, "label": None},
        {"begin": 34, "end": 37, "label": None},
        {"begin": 49, "end": 52, "label": None},
    ]
    EXPECTED_RESULT = [
        # {"begin": 0, "end": 5, "label": None},
        {"begin": 34, "end": 37, "label": "Biopsie"},
        # {"begin": 49, "end": 52, "label": None},
    ]

    examples = [
        {
            "note_id": 1,
            "text": "10/11 toto tata. Biopsie faite le 3/6. lele lulu 4/8",
            "dates_span": DATES_SPAN,
            "highlight_span": [
                {"begin": 17, "end": 39, "label": "Biopsie"},
                {"begin": 7, "end": 10, "label": "mistake"},
            ],
            "expected_result": EXPECTED_RESULT,
        },
        {
            "note_id": 2,
            "text": "10/11 toto tata. Biopsie faite le 3/6. lele lulu 4/8",
            "dates_span": DATES_SPAN,
            "highlight_span": [
                {"begin": 17, "end": 36, "label": "Biopsie"},
                {"begin": 7, "end": 10, "label": "mistake"},
            ],
            "expected_result": EXPECTED_RESULT,
        },
        {
            "note_id": 3,
            "text": "10/11 toto tata. Biopsie faite le 3/6. lele lulu 4/8",
            "dates_span": DATES_SPAN,
            "highlight_span": [
                {"begin": 35, "end": 45, "label": "Biopsie"},
                {"begin": 7, "end": 10, "label": "mistake"},
            ],
            "expected_result": EXPECTED_RESULT,
        },
        {
            "note_id": 4,
            "text": "10/11 toto tata. Biopsie faite le 3/6. lele lulu 4/8",
            "dates_span": DATES_SPAN,
            "highlight_span": [
                {"begin": 34, "end": 45, "label": "Biopsie"},
                {"begin": 7, "end": 10, "label": "mistake"},
            ],
            "expected_result": EXPECTED_RESULT,
        },
        {
            "note_id": 5,
            "text": "10/11 toto tata. Biopsie faite le 3/6. lele lulu 4/8",
            "dates_span": DATES_SPAN,
            "highlight_span": [
                {"begin": 30, "end": 45, "label": "Biopsie"},
                {"begin": 7, "end": 10, "label": "mistake"},
            ],
            "expected_result": EXPECTED_RESULT,
        },
    ]

    for example in examples:
        dates = pd.DataFrame(example.get("dates_span"))
        dates["note_id"] = example.get("note_id")

        highlights = pd.DataFrame(example.get("highlight_span"))
        highlights["note_id"] = example.get("note_id")

        expected_result = pd.DataFrame(example.get("expected_result"))
        expected_result["note_id"] = example.get("note_id")

        result = merge_highlight_annotations_with_dates(highlights, dates)
        result.drop(columns=["uid"], inplace=True, errors="ignore")
        assert (result == expected_result).all().all()


test_merge_annotations()
