from pathlib import Path

ACES_DIR = Path(__file__).resolve().parent.parent   # .../src/ACES-2026
DATA_DIR = ACES_DIR / "Data"
PLOTS_DIR = ACES_DIR / "plots"
PARAMETERS_FILE = ACES_DIR / "parameters.yaml"