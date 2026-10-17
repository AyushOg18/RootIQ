"""Configuration module for RootIQ project."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
RAW_DATA_PATH = DATA_RAW_DIR / "ai4i2020.csv"
UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

RANDOM_SEED = 42

# Column designations
TARGET_COL = "Machine failure"
IDENTIFIER_COLS = ["UDI", "Product ID"]
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

NUMERICAL_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

CATEGORICAL_COLS = ["Type"]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# Realistic operational domain constraints for physical validation & perturbation bounds
FEATURE_CONSTRAINTS = {
    "Air temperature [K]": {"min": 290.0, "max": 310.0, "default": 300.0, "unit": "K"},
    "Process temperature [K]": {"min": 300.0, "max": 320.0, "default": 310.0, "unit": "K"},
    "Rotational speed [rpm]": {"min": 1100.0, "max": 2900.0, "default": 1500.0, "unit": "rpm"},
    "Torque [Nm]": {"min": 3.0, "max": 80.0, "default": 40.0, "unit": "Nm"},
    "Tool wear [min]": {"min": 0.0, "max": 260.0, "default": 100.0, "unit": "min"},
    "Type": {"allowed": ["L", "M", "H"], "default": "M"}
}

# Train/Val/Test Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Perturbation experiment settings
PERTURBATION_EPSILONS = [0.01, 0.05, 0.10, 0.20]
STABILITY_SAMPLE_SIZE = 40  # Held-out test observations across positive and negative classes
LIME_SEED_REPEATS = 5
LIME_NUM_PERTURBED_SAMPLES = 2000
