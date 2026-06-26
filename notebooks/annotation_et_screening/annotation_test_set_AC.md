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
- Surlignage chevauchant la date en question. Il n'est pas nécessaire de prendre l'élément déclencheur.
- "Date de diagnostic": j'ai marqué uniquement Anapath
- Il n'est pas obligatoire de marquer la classe supérieure dans le cas d'une hiérarchie. Par exemple, si l'on marque « IRM », il n'est pas obligatoire de marquer « Imagerie médicale ». Les annotations seront propagées ultérieurement de toute façon. 
- "Fiche RCP validée le.." date de RCP ?

# Retex
- Moyenne 5.5 min / doc

```python
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
import pandas as pd

from wedsak.misc.annotation import (
    CreateInitialData,
    get_label_config,
    get_annotation_ui,
)
from pathlib import Path
```

```python
PATH_FOLDER = Path("~/wedsak/data/annotation/test/")
##########################

PATH_DOCS = Path(PATH_FOLDER, "test_set_group_A_0_99_B_50_100_AC.csv")
PATH_ANNOTATIONS = "~/wedsak/data/annotation/test/test_set_group_A_0_99_B_50_100_AC"

#########################
# PATH_DOCS = Path(PATH_FOLDER, "test_set_B_0-49.csv")
# PATH_ANNOTATIONS = "~/wedsak/data/annotation/test/test_set_group_B_0_49_AC"
```

```python
# Read data
if PATH_DOCS.suffix == ".csv":
    df = pd.read_csv(PATH_DOCS)
else:
    df = pd.read_pickle(
        PATH_DOCS,
    )

print("Number of docs:", len(df))
```

```python
# Set a column 'annotation order'
df = df.reset_index(drop=True).reset_index(drop=False, names="annotation_order")

# Get tasks names
tasks = pd.read_excel(
    "/export/home/cse250022/wedsak/data/LF_definition.xlsx", sheet_name="Tasks"
)
print("Number of tasks", len(tasks))
tasks_names = {t: c for t, c in tasks[["Task Name (date de)", "color"]].values}
```

```python
# Create data for annotation
data = CreateInitialData(df)

# Set up labels
label_config = get_label_config(tasks_names, add_shortcuts=True)
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
len(existing_data["notes"])
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

```python

```
