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
    display_name: wedsak_client
    language: python
    name: wedsak_client
---

```python
%config Completer.use_jedi = False
%load_ext jupyter_black
```

```python
from pyspark.sql import functions as F

import pandas as pd

from wedsak.misc.utils import read_table, get_spark_sql

from wedsak.misc.data_wrangling import keep_one
from loguru import logger
from pathlib import Path
```

```python
import os

user = os.getenv("USER")
```

```python
extra_conf = {
    "spark.memory.offHeap.enabled": "true",
    "spark.memory.offHeap.size": "1g",
    "spark.driver.memory": "3g",
    "spark.sql.session.timeZone": "Europe/Paris",
}
spark, sql = get_spark_sql(extra_conf)
```

```python
spark.sparkContext.getConf().getAll()
```

```python
cohort = read_table("cohort")
```

```python
cohort.count()  # 83683
```

```python
db = "cse250022_20260206_143905937275"
```

```python
sql("show databases like '*{}*'".format("cse250022")).show(truncate=False)
# sql("use {}".format(db))
```

```python
# # hospital groups
hospital_split = pd.read_excel(
    f"/export/home/{user}/wedsak/data/hospitals.xlsx",
    usecols=["care_site_id", "care_site_name", "care_site_short_name", "Groupe"],
)
hospital_split = hospital_split.query("Groupe.isin(['A','B'])")
hospital_split = spark.createDataFrame(hospital_split)

## join
cohort = cohort.join(
    hospital_split.select(["care_site_id", "Groupe"]), on="care_site_id", how="inner"
)
```

```python
cohort.cache().count()
```

```python
cohort_A = cohort.filter(F.col("Groupe") == "A")
```

```python
cohort_A.cache().count()
```

```python
cohort_B = cohort.filter(F.col("Groupe") == "B")
```

```python
cohort_B.count()
```

```python
cohort_A.groupby("care_site_name").count().orderBy(F.col("count").desc()).show(
    truncate=False
)
```

```python
cohort_B.groupby("care_site_name").count().orderBy(F.col("count").desc()).show(
    truncate=False
)
```

```python
# Get notes
columns = [
    "person_id",
    "note_id",
    "visit_occurrence_id",
    "care_site_id",
    "note_date",
    "note_datetime",
    "note_class_source_value",
    "note_class_source_concept_id",
    "note_title",
    "note_text",
    "row_status_source_value",
]
```

```python
note = read_table("note", db=db).select(columns)
```

```python
note = note.filter(F.col("note_text").isNotNull())
note = note.filter(F.col("row_status_source_value") == "Actif")
note = note.drop("row_status_source_value")
```

```python
# Filter only RCP
note = note.filter(F.array_contains(F.col("note_class_source_value"), "CR-RCP"))
```

```python
vo = read_table("visit_occurrence", db=db).select(
    ["visit_occurrence_id", "care_site_id"]
)
```

```python
vo = vo.withColumnRenamed("care_site_id", "care_site_id_hopital")
```

```python
note = note.join(vo, on="visit_occurrence_id", how="inner")
```

```python
# Filter notes by cohort A
note_A = note.join(
    cohort_A.select(["person_id", "organ", "localisation_first_date"]).drop_duplicates(
        subset=["person_id"]
    ),
    on="person_id",
    how="inner",
)
```

```python
note_A.count()
```

```python
# Avoid notes from other hospitals (not group A)
note_AA = note_A.join(
    hospital_split.select(["care_site_id", "Groupe", "care_site_name"]),
    on=hospital_split.care_site_id == note_A.care_site_id_hopital,
    how="inner",
)
```

```python
note_AA = note_AA.drop(
    *[
        "care_site_id_hopital",
        "care_site_id",
        "care_site_id",
        # "note_datetime",
        "note_class_source_concept_id",
    ]
)
```

```python
note_AA.groupby("Groupe").count().show()
```

```python
note_AA = note_AA.filter(F.col("Groupe") == "A")
```

```python
note_AA = note_AA.filter(F.length("note_text") > 30)
```

```python
note_AA_one = keep_one(note_AA, how="last", sort_column="note_date")
```

```python
note_AA_one.count()  # > 27k
```

### Dev set 

```python
PATH_DEV_SET = "../../data/annotation/dev/sample.csv"
PATH_DEV_SET = Path(PATH_DEV_SET)
```

```python
PATH_DEV_SET.exists()
```

```python
if not PATH_DEV_SET.exists():

    sample_size = 501

    sample = note_AA_one.orderBy(F.rand()).limit(sample_size)

    samplepd = sample.toPandas()

    # Split df in 3 groups (it's already randomly sorted)
    n_groups = 3
    i = 0
    j = int(sample_size / n_groups)
    samplepd["screening_group"] = None
    for k in range(1, n_groups + 1):
        samplepd.loc[i:j, "screening_group"] = k
        i = j
        j = int((k + 1) * j)

    samplepd.screening_group.value_counts()

    samplepd.head(1)

    try:
        samplepd.to_csv(PATH_DEV_SET, index=False, mode="x")
    except FileExistsError:
        logger.error(f"File Exists in {PATH_DEV_SET}, the dataframe was not overwrited")
```

```python
samplepd = pd.read_csv(PATH_DEV_SET)
```

```python
samplepd.head(1)
```

### Test set

```python
from wedsak.misc.data_wrangling import filter_by_person_set
```

```python
dev_person_id = list(samplepd.person_id.unique())
```

```python
note_AA_one.select("person_id").distinct().count()
```

```python
note_AA_one = filter_by_person_set(note_AA_one, dev_person_id, method="leftanti")
```

```python
note_AA_one.select("person_id").distinct().count()
```

```python
sample_size_test_set_A = 250
```

```python
test_set_A = note_AA_one.orderBy(F.rand()).limit(sample_size_test_set_A)
```

```python
test_set_Apd = test_set_A.toPandas()
```

```python
PATH_TEST_SET_A = Path("../../data/annotation/test/test_set_A.csv")
```

```python
PATH_TEST_SET_A.exists()
```

```python
try:
    test_set_Apd.to_csv(PATH_TEST_SET_A, index=False, mode="x")
except FileExistsError:
    logger.error(f"File Exists in {PATH_TEST_SET_A}, the dataframe was not overwrited")
```

```python
test_set_Apd = pd.read_csv(PATH_TEST_SET_A)
```

```python
test_set_Apd.head(1)
```

### Test set B

```python
# Filter notes by cohort A
note_B = note.join(
    cohort_B.select(["person_id", "organ", "localisation_first_date"]).drop_duplicates(
        subset=["person_id"]
    ),
    on="person_id",
    how="inner",
)
```

```python
note_B.count()
```

```python
# Avoid notes from other hospitals (not group A)
note_BB = note_B.join(
    hospital_split.select(["care_site_id", "Groupe", "care_site_name"]),
    on=hospital_split.care_site_id == note_B.care_site_id_hopital,
    how="inner",
)
```

```python
note_BB = note_BB.drop(
    *[
        "care_site_id_hopital",
        "care_site_id",
        "care_site_id",
        # "note_datetime",
        "note_class_source_concept_id",
    ]
)
```

```python
note_BB.groupby("Groupe").count().show()
```

```python
note_BB = note_BB.filter(F.col("Groupe") == "B")
```

```python
note_BB = note_BB.filter(F.length("note_text") > 30)
```

```python
note_BB_one = keep_one(note_BB, how="last", sort_column="note_date")
```

```python
note_BB_one.count()
```

```python
sample_size_test_set_B = 250
```

```python
test_set_B = note_BB_one.orderBy(F.rand()).limit(sample_size_test_set_B)
```

```python
test_set_Bpd = test_set_B.toPandas()
```

```python
PATH_TEST_SET_B = Path("../../data/annotation/test/test_set_B.csv")
```

```python
PATH_TEST_SET_B.exists()
```

```python
try:
    test_set_Bpd.to_csv(PATH_TEST_SET_B, index=False, mode="x")
except FileExistsError:
    logger.error(f"File Exists in {PATH_TEST_SET_B}, the dataframe was not overwrited")
```

```python
test_set_Bpd = pd.read_csv(PATH_TEST_SET_B)
```

```python
test_set_Bpd.head()
```

### Train set

```python
note_AA_one = filter_by_person_set(note_AA_one, test_set_A, method="leftanti")
```

```python
note_AA_one.select("person_id").distinct().count()
```

```python
note_AA_one = filter_by_person_set(note_AA_one, test_set_B, method="leftanti")
```

```python
note_AA_one.select("person_id").distinct().count()
```

```python
note_AA_one.select(["note_id", "person_id"]).write.mode("overwrite").parquet(
    "wedsak_note_AA_one"
)
```

```python
sample_size_train = 1000
```

```python
train_sample = note_AA_one.orderBy(F.rand()).limit(sample_size_train)
```

```python
train_notes = train_sample.toPandas()
```

```python
PATH_TRAIN_SET = Path("../../data/annotation/train/train_notes.pickle")
```

```python
try:
    PATH_TRAIN_SET = Path(PATH_TRAIN_SET)
    PATH_TRAIN_SET.parent.mkdir(parents=True, exist_ok=True)
    train_notes.to_pickle(
        PATH_TRAIN_SET,
    )
except FileExistsError:
    logger.error(f"File Exists in {PATH_TRAIN_SET}, the dataframe was not overwrited")
```

```python
train_notes = pd.read_pickle(PATH_TRAIN_SET)
```

```python
train_notes
```

```python
len(train_notes)
```

```python

```
