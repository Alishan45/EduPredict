from pathlib import Path
import io
import zipfile
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Official UCI Student Performance dataset archive
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def _extract_archive(archive_path: Path, target_dir: Path):
    """Extract CSV and metadata files from the UCI archive, handling nested zips."""
    with zipfile.ZipFile(archive_path) as zf:
        namelist = zf.namelist()
        # Case 1: Nested student.zip archive inside the downloaded zip (UCI current structure)
        if "student.zip" in namelist:
            with zipfile.ZipFile(io.BytesIO(zf.read("student.zip"))) as inner_zf:
                for member in inner_zf.namelist():
                    (target_dir / Path(member).name).write_bytes(inner_zf.read(member))
            return

        # Case 2: Files placed directly or in subdirectories
        for member in namelist:
            basename = Path(member).name
            if basename in {"student-mat.csv", "student-por.csv", "student.txt", "student-merge.R"}:
                (target_dir / basename).write_bytes(zf.read(member))


def load_student_data(subject="mat", data_dir: Path = None):
    """Download and load the UCI Mathematics ('mat') or Portuguese ('por') dataset."""
    if subject not in {"mat", "por"}:
        raise ValueError("subject must be 'mat' or 'por'")

    target_dir = Path(data_dir) if data_dir else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"student-{subject}.csv"

    if not csv_path.exists():
        archive_path = target_dir / "student-performance.zip"
        if not archive_path.exists():
            response = requests.get(UCI_URL, timeout=30)
            response.raise_for_status()
            archive_path.write_bytes(response.content)

        _extract_archive(archive_path, target_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Failed to find or extract {csv_path.name} in {target_dir}")

    return pd.read_csv(csv_path, sep=";")
