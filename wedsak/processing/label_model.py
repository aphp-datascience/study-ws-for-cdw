from typing import List, Optional
import pandas as pd
from scipy.sparse import csr_array

toy_data = pd.DataFrame.from_dict(
    {
        "lf_1": [None, "A", None, "A"],
        "lf_2": [None, None, None, "A"],
        "lf_3": [None, ["B", "A"], ["B"], ["A"]],
        "lf_4": [None, None, "B", None],
    }
)


class PreprocessLFVotesToLabelMatrix:
    """PreprocessLFVotesToLabelMatrix preprocesses label function (LF) votes to create a label matrix suitable for weak supervision tasks.

    This class provides methods to:
    - Convert multi-class LF outputs to binary values based on the presence of the target task.
    - Replace task-specific values with a positive label.
    - Replace values corresponding to other tasks with an abstention value.
    - Handle missing values (None) by filling them with either abstention or negative values, depending on configuration.

    ## Parameters:
        task (str): The name of the target task for which the label matrix is being constructed.
        lf_names_for_task (List[str]): List of LF column names relevant to the current task.
        negative_value (int, optional): Value to use for negative labels. Defaults to 1.
        positive_value (int, optional): Value to use for positive labels. Defaults to 2.
        abstention_value (int, optional): Value to use for abstentions (missing or irrelevant labels). Defaults to 0.
        lf_names_fillnone_as_negative (Optional[List[str]], optional): List of LF names for which missing values should be filled as negative. If None, all missing values are filled as abstention.
        multi_class_lfs (List[str], optional): List of LF names that output multi-class labels (lists of tasks). Defaults to ["lf_ccam", "lf_icd10", "lf_imaging_md", "lf_note_md"].
        avoid_all_abstentions (bool, optional): Whether to avoid documents with all abstentions. Defaults to False.

    ## Methods:
        replace_values_of_multiclass(df: pd.DataFrame) -> pd.DataFrame:
            Converts multi-class LF outputs to binary values based on the presence of the target task.

        replace_with_positive_value(df: pd.DataFrame) -> pd.DataFrame:
            Replaces occurrences of the target task in LF columns with the positive label value.

        replace_other_tasks_with_abstention_value(df: pd.DataFrame) -> pd.DataFrame:
            Replaces values corresponding to tasks other than the target with the abstention value.

        replace_none(df: pd.DataFrame) -> pd.DataFrame:
            Fills missing values in LF columns with either the abstention or negative value, depending on configuration.

        __call__(df: pd.DataFrame) -> csr_array:
            Applies all preprocessing steps to the input DataFrame and returns a sparse label matrix (csr_array).

    ## Example:
    ```python
    import pandas as pd
    from scipy.sparse import csr_array
    from wedsak.processing.label_model import PreprocessLFVotesToLabelMatrix

    # Example DataFrame with LF votes
    toy_data = pd.DataFrame.from_dict(
    {
        "lf_1": [None, "A", None, "A"],
        "lf_2": [None, None, None, "A"],
        "lf_3": [None, ["B", "A"], ["B"], ["A"]],
        "lf_4": [None, None, "B", None],
    }
    )

    # Parameters
    task = "A"
    lf_names_fillnone_as_negative = ["lf_2"]
    multi_class_lfs = ["lf_3"]
    lf_names_for_task = ["lf_1", "lf_2", "lf_3"]
    negative_value = -1
    positive_value = 1

    # Create an instance of the PreprocessLFVotesToLabelMatrix class
    # and preprocess the DataFrame

    p = PreprocessLFVotesToLabelMatrix(
    task=task,
    lf_names_for_task=lf_names_for_task,
    lf_names_fillnone_as_negative=lf_names_fillnone_as_negative,
    multi_class_lfs=multi_class_lfs,
    negative_value=negative_value,
    positive_value=positive_value,
    )
    L = p(toy_data)
    L.toarray()

    >>> array( [[ 0, -1,  0],
                [ 1, -1,  1],
                [ 0, -1,  0],
                [ 1,  1,  1]])
    ```
    """

    def __init__(
        self,
        task: str,
        lf_names_for_task: List[str],
        negative_value: int = 1,
        positive_value: int = 2,
        abstention_value: int = 0,
        lf_names_fillnone_as_negative: Optional[List[str]] = None,
        multi_class_lfs: List[str] = [
            "lf_ccam",
            "lf_icd10",
            "lf_imaging_md",
            "lf_note_md",
        ],
        avoid_all_abstentions: bool = False,
    ):
        self.task = task
        self.lf_names_for_task = lf_names_for_task
        self.negative_value = negative_value
        self.positive_value = positive_value
        self.abstention_value = abstention_value
        self.lf_names_fillnone_as_negative = lf_names_fillnone_as_negative
        self.multi_class_lfs = multi_class_lfs
        self.avoid_all_abstentions = avoid_all_abstentions

    def replace_values_of_multiclass(
        self,
        df: pd.DataFrame,
    ):
        for col in self.multi_class_lfs:
            if col in df.columns:
                # Check if task is in the column that have lists of tasks
                df[f"{col}_has_task"] = df[col].apply(
                    lambda x: self.task in x if x is not None else False
                )
                idx = df[f"{col}_has_task"]
                df.loc[idx, col] = self.positive_value
                df.loc[~idx, col] = None
                df.drop(columns=[f"{col}_has_task"], inplace=True)

        return df

    def replace_with_positive_value(self, df: pd.DataFrame):
        df[self.lf_names_for_task] = df[self.lf_names_for_task].replace(
            {self.task: self.positive_value, True: self.positive_value}
        )
        return df

    def replace_with_negative_value(self, df: pd.DataFrame):
        df[self.lf_names_for_task] = df[self.lf_names_for_task].replace(
            {False: self.negative_value}
        )
        return df

    def replace_other_tasks_with_abstention_value(
        self, df: pd.DataFrame
    ):  # Discuss with the team # TODO >> plutot with None ?
        df[self.lf_names_for_task] = df[self.lf_names_for_task].replace(
            to_replace=f"^(?!{self.task}$).*", value=self.abstention_value, regex=True
        )
        return df

    def replace_none(self, df: pd.DataFrame):
        if self.lf_names_fillnone_as_negative is None:
            df[self.lf_names_for_task] = (
                df[self.lf_names_for_task].fillna(self.abstention_value).astype(int)
            )

        else:
            abstention_cols = list(
                set(self.lf_names_for_task).difference(
                    set(self.lf_names_fillnone_as_negative)
                )
            )

            df.loc[:, self.lf_names_fillnone_as_negative] = (
                df.loc[:, self.lf_names_fillnone_as_negative]
                .fillna(self.negative_value)
                .astype("int")
            )
            df.loc[:, abstention_cols] = (
                df.loc[:, abstention_cols].fillna(self.abstention_value).astype("int")
            )

        return df

    def avoid_docs_with_all_abstentions(self, df: pd.DataFrame):
        if self.avoid_all_abstentions:
            df["all_abstentions_ent"] = df[self.lf_names_for_task].isna().all(axis=1)
            all_abstentions_doc = df.groupby("note_id", as_index=False).agg(
                all_abstentions_doc=("all_abstentions_ent", "all")
            )
            df = df.merge(all_abstentions_doc, on="note_id", how="left")
            df = df.query("~all_abstentions_doc")
        return df

    def __call__(
        self,
        df: pd.DataFrame,
    ):
        df = df.copy()
        df = self.replace_values_of_multiclass(df)
        df = self.avoid_docs_with_all_abstentions(df)
        df = self.replace_with_positive_value(df)
        df = self.replace_with_negative_value(df)
        df = self.replace_other_tasks_with_abstention_value(df)
        df = self.replace_none(df)

        # Label matrix
        L = csr_array(df[self.lf_names_for_task].astype("int").values)

        return L
