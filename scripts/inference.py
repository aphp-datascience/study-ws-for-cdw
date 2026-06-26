from typing import List, Optional

from confit import Cli
from pydantic import BaseModel

from wedsak.processing.fhir.data_models import Procedure
from wedsak.processing.inference import inference

app = Cli()


@app.command(name="inference")
def inference_cli(
    model_path: str,
    data_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 32,
    task_names: Optional[List[str]] = None,
    device: str = "auto",
    log_level: str = "INFO",
    convert_to_long_format: bool = False,
    include_context: bool = True,
    context_window: int = 25,
    export_to_fhir: bool = True,
    fhir_ressource: Optional[BaseModel] = Procedure,
):
    return inference(
        data=data_path,
        model_path=model_path,
        output_path=output_path,
        batch_size=batch_size,
        task_names=task_names,
        device=device,
        log_level=log_level,
        convert_to_long_format=convert_to_long_format,
        include_context=include_context,
        context_window=context_window,
        export_to_fhir=export_to_fhir,
        fhir_ressource=fhir_ressource,
    )


if __name__ == "__main__":
    app()
