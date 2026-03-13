"""Central configuration for the reentrancy ML project."""
from pathlib import Path
import shutil, sys

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # reentrancy_ml_project/
WORKSPACE    = PROJECT_ROOT.parent                             # ml_reentrancy/
DATASET_ROOT = WORKSPACE / "manually-verified-reentrancy-dataset" / "dataset"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ── Dataset layout ─────────────────────────────────────────────────────────
VERSION_FOLDERS = ["0_4", "0_5", "0_8"]
PRAGMA_MAP = {"0_4": "0.4.24", "0_5": "0.5.0", "0_8": "0.8.0"}
SAFE_FOLDER = "always-safe"                # label 0
REENTRANT_FOLDERS = ["cross-contract", "cross-function", "single-function"]  # label 1

# ── Slither reentrancy detectors ──────────────────────────────────────────
REENTRANCY_DETECTORS = [
    "reentrancy-eth",
    "reentrancy-no-eth",
    "reentrancy-benign",
    "reentrancy-events",
    "reentrancy-unlimited-gas",
]

# ── Feature extraction ────────────────────────────────────────────────────
FEATURE_CSV   = OUTPUTS_DIR / "features.csv"
BASELINE_CSV  = OUTPUTS_DIR / "baseline_slither.csv"

# ── Model names ───────────────────────────────────────────────────────────
MODEL_NAMES = ["xgboost", "random_forest", "logistic_regression", "svm_rbf"]

# ── Python / solc helpers ─────────────────────────────────────────────────
PYTHON_EXE = sys.executable
VENV_BIN   = Path(PYTHON_EXE).parent

def slither_executable() -> str:
    """Return the path to the slither executable."""
    path = shutil.which("slither")
    if path:
        return path
    candidate = VENV_BIN / ("slither.exe" if sys.platform == "win32" else "slither")
    if candidate.exists():
        return str(candidate)
    return "slither"

def solc_executable() -> str:
    """Return the path to the solc executable (solc-select shim)."""
    path = shutil.which("solc")
    if path:
        return path
    candidate = VENV_BIN / ("solc.exe" if sys.platform == "win32" else "solc")
    if candidate.exists():
        return str(candidate)
    return "solc"
