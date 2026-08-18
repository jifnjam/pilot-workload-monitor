import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
GUI_PATH = PROJECT_ROOT / "gui" / "app.py"


def check_prerequisites():
    """Ensuring that notebooks are run, full_features.csv is made,
    and model is created."""

    epo_files = list(PROCESSED_DIR.glob("*-epo.fif"))
    checks = [
        (
            len(epo_files) > 0,
            "notebooks/02_preprocessing.ipynb",
            "no processed *-epo.fif files found in data/processed/",
        ),
        (
            (PROCESSED_DIR / "full_features.csv").exists(),
            "notebooks/05_model_training.ipynb",
            "models/baseline_random_forest.joblib not found",
        ),
    ]

    missing = [
        (notebook, message) for exists, notebook, message in checks if not exists
    ]

    if missing:
        print(
            "Prerequisites for pipeline are missing | make sure to run these notebooks first:\n"
        )
        for notebook, message in missing:
            print(f"{notebook}")
            print(f"({message})")
        print("\nNow re-rerun this script.")
        sys.exit(1)

    print(
        f"All prerequisities found: {len(epo_files)} processed segments, features, and trained model."
    )


def launch_gui():
    """Launches the GUI and reacts to launch errors accordingly"""
    if not GUI_PATH.exists():
        print(f"GUI not found at {GUI_PATH}")
        sys.exit(1)
    print(f"Launching GUI: {GUI_PATH}")
    subprocess.run([sys.executable, str(GUI_PATH)], check=True)


if __name__ == "__main__":
    check_prerequisites()
    launch_gui()
