from typing import List, Optional
import pandas as pd
from pathlib import Path
from confit import Cli

app = Cli()


@app.command(name="merge_lf_results")
def merge_lf_results(
    lf_path_results: List[str], path_save_votes: Optional[str]
) -> pd.DataFrame:
    """Merge the results of multiple LF applications into a single DataFrame."""
    lf_path_results = [Path(p).expanduser() for p in lf_path_results]
    df = pd.read_pickle(lf_path_results[0])
    for path in lf_path_results[1:]:
        df_tmp = pd.read_pickle(path)
        df_tmp.drop(
            columns=["lexical_variant", "label", "context", "person_id"],
            errors="ignore",
            inplace=True,
        )
        df = df.merge(
            df_tmp, on=["note_id", "start", "end"], how="left", validate="one_to_one"
        )

    if path_save_votes is not None:
        path_save_votes = Path(path_save_votes).expanduser()
        path_save_votes.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path_save_votes)
        print("Merged results saved at:", path_save_votes)
    return df


if __name__ == "__main__":
    app()
