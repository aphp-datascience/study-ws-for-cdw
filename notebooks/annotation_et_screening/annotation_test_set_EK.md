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
- "Date de diagnostic": j'ai marqué Biopsie + Anapath
- Il n'est pas obligatoire de marquer la classe supérieure dans le cas d'une hiérarchie. Par exemple, si l'on marque « IRM », il n'est pas obligatoire de marquer « Imagerie médicale ». Les annotations seront propagées ultérieurement de toute façon. 
- Il est important de marquer sur "seen" une fois l'annotation de chaque document terminée.



```python
import pandas as pd

from wedsak.misc.annotation import (
    CreateInitialData,
    get_label_config,
    get_annotation_ui,
)
from pathlib import Path
##########################

PATH_FOLDER = Path("~/wedsak/data/annotation/test/")
PATH_DOCS = Path(PATH_FOLDER,"test_set_group_A_0_99_B_50_100_EK.csv")
PATH_ANNOTATIONS = "~/wedsak/data/annotation/test/test_set_group_A_0_99_B_50_100_EK"
first_execution = False
#########################


# Get tasks names
tasks = pd.read_excel(
    "/export/home/cse250022/wedsak/data/LF_definition.xlsx", sheet_name="Tasks"
)
print("Number of tasks", len(tasks))
tasks_names = {t: c for t, c in tasks[["Task Name (date de)", "color"]].values}


# Set up labels
label_config = get_label_config(tasks_names, add_shortcuts=True)

#########################
# if first_execution:
#     # Read data
#     if PATH_DOCS.suffix == ".csv":
#             df = pd.read_csv(PATH_DOCS)
#     else:
#         df = pd.read_pickle(
#             PATH_DOCS,
#         )

#     print("Number of docs:", len(df))

#     # Set a column 'annotation order'
#     df = df.reset_index(drop=True).reset_index(drop=False, names="annotation_order")
    
#     # Create data for annotation
#     data = CreateInitialData(df)
    
#     get_annotation_ui(
#     initial_data_generator=data.run,
#     label_config=label_config,
#     sync_path=PATH_ANNOTATIONS,
#     )
    
# else:
get_annotation_ui(
    initial_data_generator=lambda x:x,
    label_config=label_config,
    sync_path=PATH_ANNOTATIONS,
)
```

```python
# import pandas as pd

# from pret.store import load_store_snapshot

# existing_data = load_store_snapshot(
#     PATH_ANNOTATIONS,
# )

# i = 0

# len(existing_data["notes"])

# existing_data["notes"][i].keys()

# existing_data["notes"][i]["seen"]

# existing_data["notes"][i]["entities"]


```

```python

```
