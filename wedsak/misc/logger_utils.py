import logging
import sys
import os
from datetime import datetime

# import traceback
USER = os.getenv("USER")


class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        if not message:
            return

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""


def setup_logger(
    level: int = logging.DEBUG,
    log_dir: str = f"/export/home/{USER}/wedsak/logs/running_logs/",
    script_name: str = "",
    route_stdout_stderr_to_logger: bool = True,
):
    """
    Initializes a logger that sends INFO and DEBUG to stdout,
    and WARNING and higher levels to stderr.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.propagate = False  # Avoid duplicated logs if root logger is configured

    original_stdout = sys.__stdout__
    original_stderr = sys.__stderr__

    # Clear existing handlers if any
    root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handler for stdout (DEBUG and INFO)
    stdout_handler = logging.StreamHandler(original_stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)

    # Handler for stderr (WARNING and above)
    stderr_handler = logging.StreamHandler(original_stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    # Add handlers to logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Create file handler for logging to file
    os.makedirs(log_dir, exist_ok=True)

    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(file_handler)

    # Route warnings.warn() through logging
    logging.captureWarnings(True)

    if route_stdout_stderr_to_logger:
        # Route plain stdout/stderr writes, including print() and
        # uncaught tracebacks, through the logging system so they are also persisted.
        sys.stdout = _StreamToLogger(root_logger, logging.INFO)
        sys.stderr = _StreamToLogger(root_logger, logging.ERROR)

    # # Log uncaught exceptions
    # def _log_excepthook(exc_type, exc_value, exc_tb):
    #     tbe = traceback.TracebackException.from_exception(
    #         exc_value, capture_locals=True
    #     )
    #     formatted = "".join(tbe.format())
    #     root_logger.critical("Uncaught exception:\n%s", formatted)

    # sys.excepthook = _log_excepthook
    return root_logger
