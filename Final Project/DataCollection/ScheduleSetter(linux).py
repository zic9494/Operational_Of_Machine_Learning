import subprocess
import sys
from pathlib import Path

file_path = Path(__file__).parent / "main.py"
python_path = sys.executable

cron_job = f'0 * * * * "{python_path}" "{file_path}"\n'

result = subprocess.run(
    ["crontab", "-l"],
    capture_output=True,
    text=True
)

current_crontab = result.stdout if result.returncode == 0 else ""

if cron_job not in current_crontab:
    subprocess.run(
        ["crontab", "-"],
        input=current_crontab + cron_job,
        text=True,
        check=True
    )