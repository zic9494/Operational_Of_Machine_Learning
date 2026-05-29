import subprocess
import sys
from pathlib import Path

file_path = Path(__file__).parent / "main.py"

subprocess.run([
    "schtasks",
    "/create",
    "/tn", "ParkingCollector",
    "/tr", f'"{sys.executable}" "{file_path}"',
    "/sc", "hourly",
    "/ed", "2026-06-30",
    "/st", "06:00",
    "/et", "23:00",
    "/f"
], check=True)