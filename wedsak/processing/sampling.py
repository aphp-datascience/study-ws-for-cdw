from typing import Optional, Tuple, Union
from scipy.stats import norm
import pandas as pd

DEFAULT_SCALE = 500
ONLY_NOTES_WITH_POSITIVE_EXAMPLES = True
USE_WEIGHTED_DISTANCE = True
epsilon = 1e-5


def weight(df: pd.DataFrame, scale: Union[int, float] = DEFAULT_SCALE) -> pd.DataFrame:
    """Weight the dataframe based on the distance from the positive examples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a 'start' column.
    scale : Union[int, float]
        Scale for the normal distribution.

    Returns
    -------
    pd.DataFrame
        Dataframe with an additional 'p' column containing the weights.
    """
    locs = df.query("binary_label_class1").start.values
    df["p"] = df.start.apply(lambda x: norm.pdf(x, loc=locs, scale=scale).sum())
    df["p"] = df["p"] + epsilon
    return df


def get_negative_examples(
    df: pd.DataFrame,
    positive_class_col_name: str,
    n: Optional[int] = None,
    random_state: int = 0,
    scale: Union[int, float] = DEFAULT_SCALE,
    use_weight: bool = USE_WEIGHTED_DISTANCE,
    use_only_notes_with_positive_examples: bool = ONLY_NOTES_WITH_POSITIVE_EXAMPLES,
    prob_thresholds: Tuple[float, float] = (0.5, 0.5),
):
    """Get negative examples for training.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'note_id', 'start', and the positive class column.
    positive_class_col_name : str
        Name of the column containing the positive class probabilities.
    n : int, optional
        Number of negative examples to sample. If None, it will sample the same number as positive examples.
    random_state : int, default=0
        Random state for reproducibility.
    scale : Union[int, float], default=500
        Scale for the normal distribution used in weighting.
    use_weight : bool, default=True
        Whether to use weights based on distance from positive examples.
    use_only_notes_with_positive_examples : bool, default=True
        Whether to filter out notes without positive examples.

    Returns
    -------
    pd.DataFrame
        DataFrame containing negative examples.
    """
    # Binarize positive class probability
    df = df.copy()
    df["binary_label_class1"] = df[positive_class_col_name] > prob_thresholds[1]
    df["binary_label_class0"] = df[positive_class_col_name] <= prob_thresholds[0]

    # Filter out docs without positive values
    if use_only_notes_with_positive_examples:
        has_positive = df.groupby("note_id").agg(
            has_positive=("binary_label_class1", "max")
        )
        df = df.merge(has_positive, on="note_id", how="inner")

        df = df.query("has_positive==1")

    if use_weight:
        # Inverse distance weight
        df = df.groupby("note_id").apply(
            weight, include_groups=False, **{"scale": scale}
        )
        df.reset_index(inplace=True, drop=False, level=0)

    # If n is not setted sample all negative examples
    if n is None:
        n = len(df.query("binary_label_class0"))

    # Weighted sample among negative values
    tmp = df.query("binary_label_class0")
    negatives = tmp.sample(
        n=n, random_state=random_state, weights=tmp.p if use_weight else None
    )
    return negatives
