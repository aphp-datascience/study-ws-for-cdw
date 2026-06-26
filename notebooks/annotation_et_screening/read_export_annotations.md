---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: wedsak_python
    language: python
    name: wedsak_python
---

```python
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("future.no_silent_downcasting", True)
```

```python
from wedsak.misc.annotation import (
    CreateInitialData,
    get_label_config,
    get_annotation_ui,
)
```

```python
from pret.store import load_store_snapshot
```

```python
from wedsak.processing.export_dataset import process_annotations_to_doc
from wedsak.misc.getters import get_tasks
```

```python
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score
```

## Paths

```python
PATH_TASKS = "/export/home/cse250022/wedsak/data/LF_definition.xlsx"
# PATH_TEXTS = "~/wedsak/data/annotation/dev/sample_annotated_group_1.pickle"
```

```python
subset = "test_ek_group_A_0_99_B_50_100"
```

```python
paths = dict(
    dev=dict(
        PATH_ANNOTATIONS="~/wedsak/data/annotation/dev/dev_set_group_1",
        PATH_EXPORT="/export/home/cse250022/wedsak/data/annotation/dev/docs",
        PATH_EXPORT_DATES="/export/home/cse250022/wedsak/data/annotation/dev/annotated_processed_dates.pickle",
    ),
    test_ac_group_B_0_49=dict(
        PATH_ANNOTATIONS="~/wedsak/data/annotation/test/test_set_group_B_0_49_AC",
        PATH_EXPORT="/export/home/cse250022/wedsak/data/annotation/test/docs_group_B_0_49_AC",
        PATH_EXPORT_DATES="/export/home/cse250022/wedsak/data/annotation/test/annotated_processed_dates_group_B_0_49_AC.pickle",
        PATH_RAW_TEXT="/export/home/cse250022/wedsak/data/annotation/test/test_set_B_0-49.csv",
    ),
    test_ek_group_B_0_49=dict(
        PATH_ANNOTATIONS="~/wedsak/data/annotation/test/test_set_group_B_0_49_EK",
        PATH_EXPORT="/export/home/cse250022/wedsak/data/annotation/test/docs_group_B_0_49_EK",
        PATH_EXPORT_DATES="/export/home/cse250022/wedsak/data/annotation/test/annotated_processed_dates_group_B_0_49_EK.pickle",
    ),
    test_ac_group_A_0_99_B_50_100=dict(
        PATH_ANNOTATIONS="~/wedsak/data/annotation/test/test_set_group_A_0_99_B_50_100_AC",
        PATH_EXPORT="/export/home/cse250022/wedsak/data/annotation/test/docs_group_A_0_99_B_50_100_AC",
        PATH_EXPORT_DATES="/export/home/cse250022/wedsak/data/annotation/test/annotated_processed_dates_group_A_0_99_B_50_100_AC.pickle",
        PATH_RAW_TEXT="/export/home/cse250022/wedsak/data/annotation/test/test_set_group_A_0_99_B_50_100_AC.csv",
    ),
    test_ek_group_A_0_99_B_50_100=dict(
        PATH_ANNOTATIONS="~/wedsak/data/annotation/test/test_set_group_A_0_99_B_50_100_EK",
        PATH_EXPORT="/export/home/cse250022/wedsak/data/annotation/test/docs_group_A_0_99_B_50_100_EK",
        PATH_EXPORT_DATES="/export/home/cse250022/wedsak/data/annotation/test/annotated_processed_dates_group_A_0_99_B_50_100_EK.pickle",
        PATH_RAW_TEXT="/export/home/cse250022/wedsak/data/annotation/test/test_set_group_A_0_99_B_50_100_EK.csv",
    ),
)
```

```python
with open("/export/home/cse250022/wedsak/config/annotation_replications.json") as f:
    replications = json.load(f).get("replications")
    print(len(replications))
```

```python
tasks = get_tasks()
task_names = list(tasks.normalized_task_name)
```

# Read annotations

```python
from pret.store import load_store_snapshot

existing_data = load_store_snapshot(
    paths.get(subset).get("PATH_ANNOTATIONS"),
)
```

```python
type(existing_data)
```

```python
i = 23
```

```python
existing_data["notes"][i].get("seen")
```

```python
existing_data["notes"][i].keys()
```

```python
existing_data["notes"][i]["note_id"]
```

```python
existing_data.keys()
```

```python
# existing_data["notes"][i]["entities"]
```

# Organize annotations into A & B sets

```python
final_set_name = "test_set"
```

```python
# PATH_EXPORT = (
#     "/export/home/cse250022/wedsak/data/annotation/test/docs_group_A_0_99_B_50_100_AC",
# )
# PATH_EXPORT_DATES = (
#     "/export/home/cse250022/wedsak/data/annotation/test/annotated_processed_dates_group_group_A_0_99_B_50_100_AC.pickle",
# )
```

```python
final_sets = {
    "test_set": {
        "sessions": [
            "test_ac_group_A_0_99_B_50_100",
            "test_ac_group_B_0_49",
            "test_ek_group_A_0_99_B_50_100",
        ],
        "hospital_group": {
            "A": {
                "path_export_docs": "/export/home/cse250022/wedsak/data/annotation/test/docs_group_A",
                "path_export_dates": "/export/home/cse250022/wedsak/data/annotation/test/annotated_dates_group_A.pickle",
            },
            "B": {
                "path_export_docs": "/export/home/cse250022/wedsak/data/annotation/test/docs_group_B",
                "path_export_dates": "/export/home/cse250022/wedsak/data/annotation/test/annotated_dates_group_B.pickle",
            },
        },
    }
}
```

```python
dates_with_multilabels_ac
```

```python
raw_texts = []
path_annotations = []
for session in final_sets.get(final_set_name).get("sessions"):
    p = paths.get(session).get("PATH_RAW_TEXT")
    raw_text = pd.read_csv(p)
    annotator = "EK" if "ek" in session else "AC"
    raw_text["annotator"] = annotator
    raw_texts.append(raw_text)
    couple_path_annotator = {
        "path": paths.get(session).get("PATH_ANNOTATIONS"),
        "annotator": annotator,
    }
    path_annotations.append(couple_path_annotator)
raw_texts = pd.concat(raw_texts)
path_annotations
```

```python
raw_texts.Groupe.value_counts()
```

```python
raw_texts.note_id.nunique()
```

```python
len(raw_texts)
```

```python
raw_texts[["annotator", "Groupe"]].value_counts()
```

```python
note_id_by_hospital_group = {
    "A": set(raw_texts.loc[raw_texts.Groupe == "A"].note_id.astype(str)),
    "B": set(raw_texts.loc[raw_texts.Groupe == "B"].note_id.astype(str)),
}
```

# Check / correct discordances

```python
docs_ek, dates_with_multilabels_ek = process_annotations_to_doc(
    path_annotation=paths.get("test_ek_group_A_0_99_B_50_100").get("PATH_ANNOTATIONS"),
    replications=replications,
    path_export=None,
    subset_notes_id=None,
    context_window=75,
    # **{"output_format": "json"}
)
```

```python
docs_ac, dates_with_multilabels_ac = process_annotations_to_doc(
    path_annotation=paths.get("test_ac_group_A_0_99_B_50_100").get("PATH_ANNOTATIONS"),
    replications=replications,
    path_export=None,
    subset_notes_id=None,
    context_window=75,
    # **{"output_format": "json"}
)
```

```python
common_note_id = set(dates_with_multilabels_ek.note_id).intersection(
    set(dates_with_multilabels_ac.note_id)
)
len(common_note_id)
```

```python
dates_with_multilabels_ek = dates_with_multilabels_ek.loc[
    dates_with_multilabels_ek.note_id.isin(common_note_id)
]
dates_with_multilabels_ac = dates_with_multilabels_ac.loc[
    dates_with_multilabels_ac.note_id.isin(common_note_id)
]
```

```python
combo = dates_with_multilabels_ek.merge(
    dates_with_multilabels_ac,
    on=["note_id", "start_char", "end_char", "text", "label", "id"],
    suffixes=("_ek", "_ac"),
    how="outer",
)
```

```python
combo.fillna("0", inplace=True)
```

```python
list_of_tmps = []
BASE_COLS = ["id", "start_char", "end_char", "label", "text", "note_id", "context_ek"]
for task in task_names:
    kappa = cohen_kappa_score(combo[task + "_ac"], combo[task + "_ek"]).item()
    f1 = f1_score(combo[task + "_ac"], combo[task + "_ek"], pos_label="1")
    tmp = combo.query(f"{task}_ac !={task}_ek ")[
        BASE_COLS + [f"{task}_ac", f"{task}_ek"]
    ]
    n = len(tmp)
    tmp["task"] = task
    tmp["label_ek"] = tmp[f"{task}_ek"]
    tmp["label_ac"] = tmp[f"{task}_ac"]

    list_of_tmps.append(tmp[BASE_COLS + ["task", "label_ek", "label_ac"]])
    n_positive_ek = len(combo.query(f"{task}_ek == '1'"))
    n_positive_ac = len(combo.query(f"{task}_ac == '1'"))

    print(
        f"# {task}\n\t{n} discordances / {n_positive_ek} positives EK and {n_positive_ac} positives AC.\n\tKappa: {kappa:.2} - F1: {f1:.2}"
    )
```

```python
discordance = pd.concat(list_of_tmps)
discordance.rename(columns={"context_ek": "context"}, inplace=True)
discordance.sort_values(["label_ac", "note_id", "start_char"], inplace=True)
discordance.reset_index(inplace=True, drop=True)
```

```python
# discordance[["context"]].style.set_properties(**{"white-space": "pre-wrap"})
```

```python
# print(discordance.drop(columns=["context"]).to_csv(sep=";", index=False))
```

```python
discordance
```

# Process & export

```python
hospital_group = "B"
path_export_docs = (
    final_sets.get(final_set_name)
    .get("hospital_group")
    .get(hospital_group)
    .get("path_export_docs")
)
path_export_dates = (
    final_sets.get(final_set_name)
    .get("hospital_group")
    .get(hospital_group)
    .get("path_export_dates")
)
```

```python
# TODO : add dedup notes when combining with EK notes
```

```python
docs, dates_with_multilabels = process_annotations_to_doc(
    path_annotation=path_annotations,
    replications=replications,
    path_export=None,
    subset_notes_id=note_id_by_hospital_group.get(hospital_group),
    # **{"output_format": "json"}
)
print("hospital group", hospital_group)
```

```python
dates_with_multilabels.annotator.value_counts()
```

```python
docs, dates_with_multilabels = process_annotations_to_doc(
    path_annotation=path_annotations,
    replications=replications,
    path_export=path_export_docs,
    subset_notes_id=note_id_by_hospital_group.get(hospital_group),
    # **{"output_format": "json"}
)
dates_with_multilabels.to_pickle(path_export_dates)
```

```python
dates_with_multilabels.head()
```

```python
dates_with_multilabels.magnetic_resonance_imaging_procedure.value_counts()
```

# Description

```python
dates_with_multilabels.query(
    "nuclear_medicine_imaging_procedure_procedure=='1' & positron_emission_tomography_procedure=='0'	"
)
```

```python
# dates_with_multilabels.query(
#     "enrollment_in_clinical_trial_procedure=='1'"
# ).context.iloc[0]
```

```python
# dates_with_multilabels.query("enrollment_in_clinical_trial_procedure=='1'").iloc[
#     0
# ].note_id
```

```python
stat_list = []
for i, task in enumerate(task_names, start=1):
    # print("##################\n", task)
    task_stat = dict()

    task_stat["task_id"] = i
    task_stat["task"] = task

    for a, b in dates_with_multilabels[task].value_counts(dropna=False).items():

        task_stat[a] = b
    stat_list.append(task_stat)
stats = pd.DataFrame(stat_list)
stats.fillna(0, inplace=True)
stats["1"] = stats["1"].astype(int)
# print(dates_with_multilabels[task].value_counts())
```

```python
print(stats.to_markdown(index=False))
```

```python
# stats.to_pickle("~/wedsak/data/annotation/dev/stats_dev_set.pickle")
stats.to_csv(
    f"~/wedsak/data/annotation/test/stats_{final_set_name}_{hospital_group}.csv"
)
```

```python

```
