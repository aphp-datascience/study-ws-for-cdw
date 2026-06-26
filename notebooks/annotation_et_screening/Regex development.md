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
    display_name: wedsak_kernel
    language: python
    name: wedsak_kernel
---

```python
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
import pandas as pd
```

```python
import datetime

import pandas as pd

import edsnlp
from edsnlp.pipes.qualifiers.contextual.contextual import (
    ClassPatternsContext,
    ContextualQualifier,
)
import edsnlp.pipes as eds
from edsnlp.utils.collections import get_deep_attr
import json
from pathlib import Path
```

```python
import spacy
from spacy import displacy
```

```python
ents = pd.read_csv("/export/home/acohen/wedsak/data/annotation/dev/spans_screening.csv")
```

```python
ents
```

```python
ents.note_id.nunique()
```

```python
tasks = pd.read_excel(
    "/export/home/acohen/wedsak/data/LF_definition.xlsx", sheet_name="Tasks"
)
tasks.head()
tasks_names = list(tasks["Task Name (date de)"].unique())
tasks_names
```

```python
tasks.head()
```

```python
iterable_tasks = iter(tasks_names)
```

## Iterate

```python
task = next(iterable_tasks)
task
```

```python
task_en = tasks.loc[
    tasks["Task Name (date de)"] == task, "SNOMED CT - Fully Specified Name (FSN)"
].iloc[0]

task_en
```

```python
patterns_path = Path(Path.home(),"wedsak/config/patterns.json")
```

```python
# Opening JSON file
with open(patterns_path, "r") as openfile:

    # Reading from json file
    patterns = json.load(openfile)
```

```python
patterns[task_en]
```

```python
task_en
```

```python
patterns_lf1 = {
    task_en: {
        "terms": {
            "rcp": [
                "RCP",
                "Concertation Pluridisciplinaire",
                "concertation pluridisciplinaire",
            ]
        },
        # "regex": {
        #     "biopsie_diag": [
        #         r"(?i)biopsie .{1,150} ((ad.nocarcinome)|(adk)|(tumorale)|(histologie)|(histologique)|(histo\b))",
        #     ],
        #     "type_cancer": ["carcinome", "sarcome"],
        # },
        # "context_words": (8, 10),
        "context_sents": 1,
        "attr": "TEXT",
    },
}

# patterns_lf1 = {task_en: patterns[task_en]}

###
nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.dates())
nlp.add_pipe(
    ContextualQualifier(
        span_getter="dates",
        patterns={"lf1": patterns_lf1},
    )
)

docs = list(nlp.pipe(ents.loc[ents.label == task].span_text.unique()))

for doc in docs:
    dates = doc.spans["dates"]
    dates = doc.spans["dates"]
    tagged_spans = []
    for date in dates:
        for attr in [
            "lf1",
        ]:
            value = get_deep_attr(date, "_." + attr)

            if value:
                # print(date.start, date.end, date, attr, value)
                date.label_ = "tag"
            tagged_spans.append(date)
    doc.ents = tagged_spans

colors = {"date": "#ff5733", "tag": "#14ad02"}
options = {"colors": colors}
for doc in docs:
    print("###")
    displacy.render(doc, style="ent", options=options)
```

```python
patterns_lf1
```

```python
import json

# Opening JSON file
with open(patterns_path, "r") as openfile:

    # Reading from json file
    patterns = json.load(openfile)


# Data to be written
patterns.update(patterns_lf1)
print(patterns)

# Serializing json
json_object = json.dumps(patterns)

# Writing to sample.json
with open(patterns_path, "w") as outfile:
    outfile.write(json_object)
```

## Test specific

```python
text = "03/07/2023  La biopsie à l’aiguille fine"
# text = "10/10/2010 : RCP"

doc = nlp(text)

dates = doc.spans["dates"]
tagged_spans = []
for date in dates:
    for attr in [
        "lf1",
    ]:
        value = get_deep_attr(date, "_." + attr)
        if value:
            print(date.start, date.end, date, attr, value)
            date.label_ = "tag"
#         tagged_spans.append(date)
# doc.ents = tagged_spans
```

```python
dates
```

```python
list(doc.sents)
```

```python
for token in doc:
    print(token.norm_)
```

```python
# text = ""
# for i, span in enumerate(ents.loc[ents.label == task].span_text.unique()):
#     # print("####")
#     # print(span)
#     text = text + "\n####\n" + f"{str(i)} " + span + ".\n"

# print(text)
```

```python

```

```python

```
