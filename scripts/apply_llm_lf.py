from confit import Cli
from wedsak.lf.llm.pipeline import llm_lf_pipeline

app = Cli()
llm_lf_command = app.command(name="llm_lf")(llm_lf_pipeline)


if __name__ == "__main__":
    app()
