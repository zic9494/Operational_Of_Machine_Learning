from datetime import datetime
from pathlib import Path
import csv
import logging

from CWA import CWA
from TDX import TDX
from Holiday import Holiday
from logger_config import setup_logging


DATASET_DIR = Path(__file__).resolve().parents[1] / "Datasets"
CSV_PATH = DATASET_DIR / "ParkingRemainDataset.csv"

FIELDNAMES = [
    "CollectedAt",
    "IsHoliday",
    "ParkId",
    "TotalSpaces",
    "AvailableSpaces",
    "FullStatus",
    "ParkDataCollectTime",
    "RainNow",
    "RainPast10Min",
    "RainPast1hr",
    "RainPast3hr",
    "RainPast12hr",
    "RainPast24hr",
]

logger = logging.getLogger(__name__)

def append_rows(rows):
    file_is_empty = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if file_is_empty:
            writer.writeheader()

        writer.writerows(rows)


def hourly():
    collected_at = datetime.now()
    holiday_data = Holiday().isHoliday(collected_at)

    cwa = CWA()
    cwa.get_rain()
    cwa_data = cwa.to_dict()

    tdx = TDX()
    tdx.getParkSpace()
    tdx_data = tdx.to_dict()

    rows = []
    for park in tdx_data:
        rows.append({
            "CollectedAt": collected_at.isoformat(timespec="seconds"),
            "IsHoliday": holiday_data,
            "ParkId": park["ParkId"],
            "TotalSpaces": park["TotalSpaces"],
            "AvailableSpaces": park["AvailableSpaces"],
            "FullStatus": park["FullStatus"],
            "ParkDataCollectTime": park["time"],
            "RainNow": cwa_data["Now"],
            "RainPast10Min": cwa_data["Past10Min"],
            "RainPast1hr": cwa_data["Past1hr"],
            "RainPast3hr": cwa_data["Past3hr"],
            "RainPast12hr": cwa_data["Past12hr"],
            "RainPast24hr": cwa_data["Past24hr"],
        })

    append_rows(rows)
    print(f"Inserted {len(rows)} rows into {CSV_PATH}")

setup_logging()
DATASET_DIR.mkdir(exist_ok=True)
hourly()
