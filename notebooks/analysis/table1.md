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
from wedsak.tables.table1 import metrics_table
```

```python
%config Completer.use_jedi = False
%load_ext jupyter_black

%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd

pd.set_option("display.max_columns", None)
```

```python
path_model_refs = "/export/home/cse250022/wedsak/data/model_refs.xlsx"
metric = "fscore"  # "precision" # "recall" # "fscore"
```

```python
table = metrics_table(
    path_model_refs=path_model_refs,
    metric=metric,
)
```

```python
import numpy as np


def replace_no_support(df, k=5):
    PLACEHOLDER = np.nan  # or "-"

    # Boolean mask per row: is support < k for A or B?
    mask_A = df[("Support", "A")] < k
    mask_B = df[("Support", "B")] < k

    # Grab all columns belonging to each stratum (excluding Support itself)
    cols_A = [col for col in df.columns if col[1] == "A" and col[0] != "Support"]
    cols_B = [col for col in df.columns if col[1] == "B" and col[0] != "Support"]

    # Replace in-place where support is below threshold
    df.loc[mask_A, cols_A] = PLACEHOLDER
    df.loc[mask_B, cols_B] = PLACEHOLDER
    return df
```

```python
replace_no_support(table)
```

```python
mask = table.columns.get_level_values(0).isin(
    [
        "All tasks (HyperLabel Model)",
        "All tasks",
        "All tasks (Majority Vote Model)",
        "Task id",
        "Task (Procedure)",
        "Support",
    ]
)
subset = table.loc[:, mask]
```

```python
subset
```

```python
table = table.drop(
    columns=["All tasks (HyperLabel Model)", "All tasks (Majority Vote Model)"],
    level=0,
)
```

```python
std = table.describe().loc["std"]
```

```python
mean = table.describe().loc["mean"]
```

```python
cv = std / mean
cv
```

```python
table.iloc[:, 4:].median().round(2)
```

```python
table.iloc[:, 4:].mean().round(2)
```

```python
table.iloc[:, 4:].median(axis=1).round(2)
```

```python
table.iloc[:, 4:].describe().round(2)
```

```python
# WS
table.iloc[1:, 14:].describe().round(2).mean(axis=1)
```

```python
# WS (CV)
table.iloc[1:, 14:].std().mean() / table.iloc[1:, 6:10].mean().mean()
```

```python
cv.loc[["All tasks", "All tasks on Medgemma vote", "Task specific"]].mean()
```

```python
# LLM (CV)
table.iloc[1:, 6:10].std().mean() / table.iloc[1:, 6:10].mean().mean()
```

```python
cv.loc[["Qwen3-8b", "Medgemma-27b"]].mean()
```

```python
# Rule based
table.iloc[1:, 4:6].describe().round(2).mean(axis=1)
```

```python
# Rule based (CV)
table.iloc[1:, 4:6].std().mean() / table.iloc[1:, 4:6].mean().mean()
```

```python
cv.loc[["Rule based"]].mean()
```

```python
# Fully supervised
table.iloc[1:, 12:14].describe().round(2).mean(axis=1)
```

```python
# Fully supervised (CV)
table.iloc[1:, 12:14].std().mean() / table.iloc[1:, 12:14].mean().mean()
```

```python
cv.loc[["Fully supervised"]].mean()
```

```python
table_rounded = table.round(2).reset_index(drop=True)
table_rounded
```

```python
table_rounded.drop(
    columns=[
        ("Task (Procedure)", ""),
        ("Medgemma-27b", "A"),
        ("Medgemma-27b", "B"),
        ("Qwen3-8b", "A"),
        ("Qwen3-8b", "B"),
    ]
).reset_index(drop=True)
```

```python
table.loc[:, ("Task (Procedure)", "")]
```

```python
import textwrap
import pandas as pd


def wrap_with_makecell(val, width=20, align="l"):
    if not isinstance(val, str) or len(val) <= width:
        return str(val)
    lines = textwrap.wrap(val, width)
    inner = r" \\ ".join(lines)
    return rf"\makecell[{align}]{{{inner}}}"


def df_to_latex_makecell(df, wrap_width=20, **to_latex_kwargs):
    table_latex = (
        df.drop(
            columns=[
                ("Medgemma-27b", "A"),
                ("Medgemma-27b", "B"),
                ("Qwen3-8b", "A"),
                ("Qwen3-8b", "B"),
            ]
        )
        .reset_index(drop=True)
        .rename(columns={"A": "I", "B": "E"}, level=1)
    )

    table_latex.loc[:, ("Task (Procedure)", "")] = table_latex.loc[
        :, ("Task (Procedure)", "")
    ].map(lambda x: wrap_with_makecell(x, width=wrap_width))

    to_latex_kwargs.setdefault("escape", False)

    old_cols = table_latex.columns.to_list()
    new_cols = [
        (wrap_with_makecell(t[0], width=10, align="c"), *t[1:]) for t in old_cols
    ]
    table_latex.columns = pd.MultiIndex.from_tuples(
        new_cols, names=table_latex.columns.names
    )

    # Build column_format with a | after every 2 columns
    n_cols = len(table_latex.columns)
    fmt_chars = []
    for i in range(n_cols):
        fmt_chars.append("l" if i in [0, 1] else "c")
        if (i + 1) % 2 == 0 and i < n_cols - 1:
            fmt_chars.append("|")
    col_format = "".join(fmt_chars)
    to_latex_kwargs.setdefault("column_format", col_format)

    return table_latex.to_latex(
        float_format="%.2f", index=False, na_rep="-", **to_latex_kwargs
    )
```

```python
print(df_to_latex_makecell(table))
```

```python
median = table.iloc[:, 4:].describe().round(2).loc[["25%", "50%", "75%"], :]
median
```

```python
print(
    median.drop(
        columns=[
            # ("Task (Procedure)", ""),
            ("Medgemma-27b", "A"),
            ("Medgemma-27b", "B"),
            ("Qwen3-8b", "A"),
            ("Qwen3-8b", "B"),
        ]
    ).to_latex(
        float_format="%.2f",
        index=False,
        na_rep="-",
    )
)
```

```python
table
```

```python
subset
```

```python
table.columns
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# --- Filter rows where Support (A or B) < k ---
# Keep rows where BOTH A and B support meet the threshold (adjust to your needs)
def box_plot(table, k=5, y_label="F1-score", base_name="boxplot"):
    mask = (table[("Support", "A")] >= k) & (table[("Support", "B")] >= k)
    df_filtered = table[mask]

    # --- Reshape ---
    df_long = (
        df_filtered.drop(
            columns=[
                ("Task id", ""),
                # ("Task (Procedure)", ""),
                ("Support", "A"),
                ("Support", "B"),
                ("LLM Average", "A"),
                ("LLM Average", "B"),
            ],
            errors="ignore",
        )
        .set_index("Task (Procedure)")
        .stack(level="dataset")
        .reset_index()
        .melt(
            id_vars=[
                # "level_0",
                "Task (Procedure)",
                "dataset",
            ],
            var_name="Method",
            value_name="Score",
        )
        .rename(columns={"level_0": "row"})
    )

    df_long.dataset = df_long.dataset.replace({"A": "Internal", "B": "External"})
    df_long.Method = df_long.Method.replace(
        {
            "All tasks on Medgemma vote": "All tasks on\nMedgemma Vote",
        }
    )
    # print(df_long.Method.unique())

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.boxplot(
        data=df_long,
        x="Method",
        y="Score",
        hue="dataset",
        palette={"Internal": "#6EA36D", "External": "#267AAD"},
        width=0.6,
        linewidth=1.2,
        ax=ax,
        fill=True,
        dodge=True,
    )

    # ax.set_title(f"{y_label} by method and dataset (Support ≥ {k})", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis="x", rotation=30, labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    ax.legend(title="Dataset", loc="lower right", fontsize=15, title_fontsize=15)
    ax.grid(visible=True, which="major", axis="y")
    sns.despine()

    plt.tight_layout()
    plt.savefig(f"../../figures/{base_name}_{y_label}.png", dpi=150)
    plt.show()
```

```python
box_plot(table)
```

```python
box_plot(subset, base_name="otherLM")
```
