import os
from pathlib import Path

# Project Root (this script's directory)
# This assumes paths.py is in the root of your project 'code/'
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Input Data Directories ---
# Base directory for external data, configurable via environment variable
# Users should set OCO_DATA_BASE in their environment.
# Default path if OCO_DATA_BASE is not set (e.g., a 'data' folder alongside the 'code' folder)
DEFAULT_DATA_PATH = PROJECT_ROOT.parent / "data"
BASE_DATA_DIR = Path(os.getenv("OCO_DATA_BASE", str(DEFAULT_DATA_PATH)))

# Specific input data locations (scripts will search within these)
OCO_LITE_FILES_DIR = BASE_DATA_DIR / "OCO_Lite_Files"
TCCON_FILES_DIR = BASE_DATA_DIR / "TCCON_Files"
MODEL_INPUT_DATA_DIR = BASE_DATA_DIR # Or, e.g., BASE_DATA_DIR / "geos_chem_output"

# Directory for 3D Cloud Metrics Data (used in 05_integrate_cloud_data.py)
CLOUD_DATA_DIR = BASE_DATA_DIR / 'Cloud_Data'

# Directory for Emission Data (e.g., EDGAR, used in 07_filter_strong_emission_sources.py)
EMISSION_DATA_DIR = BASE_DATA_DIR / 'Emission_Data'

# --- Output & Working Directories (within the project) ---
# These are typically relative to PROJECT_ROOT

# Main export directory for processed data (matches existing 'export/' directory)
EXPORT_DIR = PROJECT_ROOT / "export"

# Subdirectories for outputs
PAR_DIR = EXPORT_DIR / "parquet_files"
PRELOAD_DIR = PAR_DIR

# Directory for saving trained bias correction models
MODEL_SAVE_DIR = PROJECT_ROOT / "bias_corr_models"

# Directory for filter models
FILTER_DIR = PROJECT_ROOT / "filter_models"

# Directory for figures/plots
FIGURE_DIR = PROJECT_ROOT / "visualization_scripts" / "figures" # Suggesting a subfolder under visualization_scripts

# --- Pipeline Script Directories ---
BIAS_CORRECTION_DIR = PROJECT_ROOT / "bias_correction"
VISUALIZATION_SCRIPTS_DIR = PROJECT_ROOT / "visualization_scripts"
# DATA_QUALITY_FILTERING_DIR = PROJECT_ROOT / "data_quality_filtering" # For future use

# --- Utility file ---
UTIL_FILE = PROJECT_ROOT / "util.py"

# --- Function to ensure directories exist ---
def ensure_dir_exists(path_to_check: Path):
    """Checks if a directory exists, and creates it if it doesn't."""
    if not path_to_check.exists():
        print(f"Directory {path_to_check} does not exist. Creating it.")
        path_to_check.mkdir(parents=True, exist_ok=True)
    elif not path_to_check.is_dir():
        raise NotADirectoryError(f"{path_to_check} exists but is not a directory.")

# --- Specific Model Paths (B11.2 example) ---
# Bias Correction Models
TC_LND_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_xco2_TCCON_biasLndNDGL_lnd_RF0'
TC_OCN_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_xco2_TCCON_biasSeaGL_sea_RF0'
SA_LND_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_prec_xco2raw_SA_biasLndNDGL_lnd_RF0'
SA_OCN_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_prec_xco2raw_SA_biasSeaGL_sea_RF0'

# Filter Models
TC_LND_FILTER_MODEL = FILTER_DIR / 'V1_11.2_M_tc_lnd.joblib'
TC_OCN_FILTER_MODEL = FILTER_DIR / 'V1_11.2_M_tc_ocn.joblib'
SA_LND_FILTER_MODEL = FILTER_DIR / 'V1_11.2_M_sa_lnd.joblib'
SA_OCN_FILTER_MODEL = FILTER_DIR / 'V1_11.2_M_sa_ocn.joblib'

# --- Function to construct specific preload file paths ---
def get_preload_filepath(mode: str, qf: int | None, year: int) -> Path:
    """
    Constructs the full path to a preloaded data file.

    Args:
        mode (str): The mode (e.g., 'LndNDGL', 'SeaGL').
        qf (int | None): The quality flag. Appears as 'None' in filename if None.
        year (int): The year of the data.

    Returns:
        Path: The full Path object to the preloaded .parquet file.
    """
    qf_str = str(qf) if qf is not None else "None"
    filename = f'PreLoadB112v2_balanced_5M_{mode}_{year}.parquet'
    return PRELOAD_DIR / filename

def check_and_create_dirs():
    """Checks if necessary directories exist and creates them if they don't."""
    ensure_dir_exists(PAR_DIR)
    ensure_dir_exists(PRELOAD_DIR)
    ensure_dir_exists(FIGURE_DIR)
    ensure_dir_exists(MODEL_SAVE_DIR)
    ensure_dir_exists(FILTER_DIR)
    print("Checked/created PAR_DIR, PRELOAD_DIR, FIGURE_DIR, MODEL_SAVE_DIR, and FILTER_DIR.")

if __name__ == "__main__":
    print("--- Defined Paths (from paths.py) ---")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Using BASE_DATA_DIR: {BASE_DATA_DIR}")

    if "OCO_DATA_BASE" not in os.environ:
        print(f"  (Note: OCO_DATA_BASE environment variable not set, using default: {DEFAULT_DATA_PATH})")
        print(f"  (Create this directory or set OCO_DATA_BASE in your environment if your data is elsewhere.)")
    else:
        print(f"  (OCO_DATA_BASE environment variable is set to: {os.getenv('OCO_DATA_BASE')})")


    print(f"\nInput data is expected in (or under):")
    print(f"  OCO Lite Files Dir: {OCO_LITE_FILES_DIR}")
    print(f"  TCCON Files Dir: {TCCON_FILES_DIR}")
    print(f"  Model Input Data Dir: {MODEL_INPUT_DATA_DIR}")
    print(f"  Cloud Data Dir: {CLOUD_DATA_DIR}")
    print(f"  Emission Data Dir: {EMISSION_DATA_DIR}")

    print(f"\nOutput directories (will be created if they don't exist when ensure_dir_exists is called):")
    print(f"  Export Directory: {EXPORT_DIR}")
    print(f"  PAR Files Directory: {PAR_DIR}")
    print(f"  Preloaded Data Directory: {PRELOAD_DIR}")
    print(f"  Trained Models Directory: {MODEL_SAVE_DIR}")
    print(f"  Output Figures Directory: {FIGURE_DIR}")
    print(f"  Filter Models Directory: {FILTER_DIR}")

    print(f"\nSpecific B11.2 Models:")
    print(f"  TC LND CORR: {TC_LND_CORR_MODEL}")
    print(f"  TC OCN CORR: {TC_OCN_CORR_MODEL}")
    print(f"  SA LND CORR: {SA_LND_CORR_MODEL}")
    print(f"  SA OCN CORR: {SA_OCN_CORR_MODEL}")
    print(f"  TC LND FILTER: {TC_LND_FILTER_MODEL}")
    print(f"  TC OCN FILTER: {TC_OCN_FILTER_MODEL}")
    print(f"  SA LND FILTER: {SA_LND_FILTER_MODEL}")
    print(f"  SA OCN FILTER: {SA_OCN_FILTER_MODEL}")

    print(f"\nScript directories:")
    print(f"  Bias Correction Scripts: {BIAS_CORRECTION_DIR}")
    print(f"  Visualization Scripts: {VISUALIZATION_SCRIPTS_DIR}")
    print(f"  Utility File: {UTIL_FILE}")

    print("\n--- Example: Ensuring output directories exist ---")
    check_and_create_dirs() 