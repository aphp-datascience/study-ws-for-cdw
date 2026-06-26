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
import pandas as pd

%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
%load_ext jupyter_black
pd.set_option("display.max_columns", None)
```

```python
from wedsak.events.cancer_stays_inca import INCaCancerStays
```

```python
from pyspark.sql import functions as F
```

# Sur la base totale

```python
# db = "cse250022_20250818_093922142129"
db = None
```

```python jupyter={"outputs_hidden": true}
stays = INCaCancerStays(db=db)()
```

```python
stays = stays.withColumn("year", F.year("visit_start_datetime"))
```

```python jupyter={"outputs_hidden": true}
(stays.write.mode("overwrite").saveAsTable(f"stays_INCa"))
```

```python
from wedsak.misc.utils import read_table, get_spark_sql
```

```python
spark, sql = get_spark_sql(
    conf_dict={
        "spark.sql.parquet.datetimeRebaseModeInWrite": "LEGACY",
    }
)
```

```python
df = spark.table("stays_INCa")
```

```python
df.count()
```

```python
df.select(["person_id"]).distinct().count()
```

```python
df_year = df.filter(F.col("year") == 2023)
df_year.count()
```

```python
df_year = df_year.filter((F.col("step_1")) | (F.col("step_2")) | (F.col("step_4")))
```

```python
df_year.count()
```

```python
df_year.select(["person_id"]).distinct().count()
```

```python
df_year.groupby("system").count().orderBy(F.col("count").desc()).show()
```

```python
df_year.drop_duplicates(subset=["system", "person_id"]).groupby(
    "system"
).count().orderBy(F.col("count").desc()).show()
```

# Selection de la cohorte

```python
from wedsak.cohort.cancer_inca import CancerCohortSelector
```

```python
df = spark.table("stays_INCa")
```

```python
cohort = CancerCohortSelector.aggregate_stays(df)
```

```python
cohort.write.mode("overwrite").saveAsTable(f"cohort_cancer")
```

```python
cohort = spark.table("cohort_cancer")
```

```python
cohort.count()
```

```python
organs = [
    "Sein",
    "Trachée, Bronches, Poumon",
    "Prostate",
    "Colon-Rectum-Anus",
    "Peau",
]
```

```python
subcohort = cohort.filter(F.col("organ").isin(organs))
```

```python
subcohort.count()
```

```python
subcohort = subcohort.filter(F.col("localisation_first_date") >= "2018-01-01")
```

```python
subcohort.count()
```

```python
subcohort.groupby("organ").count().orderBy(F.col("count").desc()).show()
```

```python
subcohort
```

```python
# db = "wedsak"
# sql(
#     f"CREATE DATABASE IF NOT EXISTS {db} COMMENT 'Cancer patients' LOCATION 'hdfs://bbsedsi/user/acohen/warehouse/{db}.db' "
# )
```

```python
# sql("DROP DATABASE wedsak CASCADE")
```

```python
# sql("DESCRIBE DATABASE EXTENDED wedsak ").show(truncate=False)
```

```python
# db_out = "wedsak"
# table = "cohort"
```

```python
# subcohort.write.mode("overwrite").saveAsTable(f"{db_out}.{table}")
```

```python
table = "cohort"
```

```python
subcohort.write.mode("overwrite").saveAsTable(f"{table}")
```

```python
spark.table(table).show()
```

```python
spark.table("cohort").count()
```

```python
subcohort
```

```python
subcohort.write.mode("overwrite").parquet("/data/hdd/cse250022/wedsak/cohort/cohort/")
```

```python
subcohort = spark.read.parquet("/data/hdd/cse250022/wedsak/cohort/cohort/")
subcohort
```

```python
subcohort
```

```python
note = spark.read.parquet("/data/hdd/cse250022/wedsak/cohort/note/")
note.count()
```

```python
note_reduced = note.join(subcohort.select("person_id").distinct(), on="person_id")
```

```python
note_reduced.count()
```

```python
spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "LEGACY")
```

```python
note_reduced.write.mode("overwrite").parquet(
    "/data/hdd/cse250022/wedsak/cohort/note_r/"
)
```

```python

```
