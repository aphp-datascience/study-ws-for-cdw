import os
from typing import List, Optional
import matplotlib.pyplot as plt
from pandas import DataFrame
from pathlib import Path


def _save_plot(
    fig,
    filename: str,
    conf_name: str,
    legend=None,
    tables: Optional[List[DataFrame]] = None,
    formats=["pdf", "png"],
    project_name: str = "wedsak",
    path_dir: str = "~/wedsak/figures",
    save_index: bool = False,
    **kwargs,
):
    """
    Auxiliary function to save plot.
    """
    filenameimgs = [filename + "_" + conf_name + "." + format for format in formats]

    path_files = [
        Path(path_dir, filenameimg).expanduser() for filenameimg in filenameimgs
    ]

    path_dir_extended = path_files[0].parent
    if not path_dir_extended.is_dir():
        path_dir_extended.mkdir(parents=True)

    for path_file, format in zip(path_files, formats):
        if legend is not None:
            fig.savefig(
                path_file,
                bbox_extra_artists=tuple(legend),
                bbox_inches="tight",
                format=format,
            )
        else:
            fig.savefig(
                path_file,
                format=format,
                bbox_inches="tight",
            )

        print("Saved at:", path_file)

    plt.cla()
    plt.close("all")

    if tables:
        for i, table in enumerate(tables, start=1):
            filenametable = filename + "_" + str(i) + "_" + conf_name + ".csv"
            path_table = os.path.join(path_dir, filenametable)
            table.to_csv(path_table, index=save_index)

    print("Done -", filename)


def show_or_save(
    fig,
    filename: Optional[str] = None,
    conf_name: Optional[str] = None,
    legend=None,
    tables: Optional[List[DataFrame]] = None,
    **kwargs,
):
    # Show or save plot
    if (conf_name is not None) & (filename is not None):
        _save_plot(
            fig,
            filename=filename,
            conf_name=conf_name,
            legend=legend,
            tables=tables,
            **kwargs,
        )

    else:
        plt.show()
        plt.cla()
        plt.close("all")
