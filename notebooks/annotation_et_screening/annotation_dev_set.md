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

# Guidelines
- "Date de diagnostic": j'ai marqué Biopsie + Anapath
- Il n'est pas obligatoire de marquer la classe supérieure dans le cas d'une hiérarchie. Par exemple, si l'on marque « IRM », il n'est pas obligatoire de marquer « Imagerie médicale ». Les annotations seront propagées ultérieurement de toute façon. 

# Retex
- Moyenne 5.5 min / doc

```python
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
import pandas as pd
```

```python
from wedsak.misc.annotation import (
    CreateInitialData,
    get_label_config,
    get_annotation_ui,
)
```

```python
df = pd.read_pickle("~/wedsak/data/annotation/dev/sample_annotated_group_1.pickle")
print("Number of docs:", len(df))
```

```python
df = df.reset_index(drop=True).reset_index(drop=False, names="annotation_order")
```

```python
tasks = pd.read_excel(
    "/export/home/cse250022/wedsak/data/LF_definition.xlsx", sheet_name="Tasks"
)
print("Number of tasks", len(tasks))

tasks_names = {t: c for t, c in tasks[["Task Name (date de)", "color"]].values}
```

```python
data = CreateInitialData(df)
```

```python
label_config = get_label_config(tasks_names, add_shortcuts=True)
```

```python
PATH_ANNOTATIONS = "~/wedsak/data/annotation/dev/dev_set_group_1"
```

```python
get_annotation_ui(
    initial_data_generator=data.run,
    label_config=label_config,
    sync_path=PATH_ANNOTATIONS,
)
```

# Read

```python
import pandas as pd
```

```python
from pret.store import load_store_snapshot

existing_data = load_store_snapshot(
    PATH_ANNOTATIONS,
)
```

```python
i = 0
```

```python
existing_data["notes"][i].keys()
```

```python
existing_data["notes"][i]["seen"]
```

```python
existing_data["notes"][i]["entities"]
```
