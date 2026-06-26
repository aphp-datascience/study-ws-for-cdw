from confit import Cli

from wedsak.tables.table1 import metrics_table

app = Cli()
metric_command = app.command(name="metrics_table")(metrics_table)

if __name__ == "__main__":
    app()
