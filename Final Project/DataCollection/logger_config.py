import logging
from pathlib import Path

def setup_logging():
    log_dir = Path(__file__).resolve().parents[1] / "Logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "api.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        encoding="utf-8",
    )