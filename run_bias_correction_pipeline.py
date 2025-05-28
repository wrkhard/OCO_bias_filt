import subprocess
import sys
from pathlib import Path
import os
import argparse


# Define the sequence of scripts to run
# These paths are relative to the 'code/' directory where this script is intended to be run.
# If a script is in a subdirectory (like bias_correction), reflect that.
BIAS_CORRECTION_DIR = Path('bias_correction')
SCRIPTS_TO_RUN = [
    BIAS_CORRECTION_DIR / '01_process_lite_to_parquet.py',
    BIAS_CORRECTION_DIR / '02_create_small_areas.py',
    BIAS_CORRECTION_DIR / '03_flag_coastal_soundings.py',
    BIAS_CORRECTION_DIR / '04_integrate_tccon_data.py',
    BIAS_CORRECTION_DIR / '05_integrate_cloud_data.py',
    BIAS_CORRECTION_DIR / '06_integrate_flux_model_data.py',
    BIAS_CORRECTION_DIR / '07_filter_strong_emission_sources.py',
    BIAS_CORRECTION_DIR / '08_remove_outliers.py',
    BIAS_CORRECTION_DIR / '09_prepare_model_input_data.py',
    BIAS_CORRECTION_DIR / '10_feature_selection.py',
]

STATUS_FILE = Path('pipeline_status.txt')

def get_last_completed_script():
    """Reads the status file and returns the name of the last successfully completed script."""
    if STATUS_FILE.exists():
        try:
            return Path(STATUS_FILE.read_text().strip())
        except Exception as e:
            print(f"Warning: Could not read status file {STATUS_FILE}: {e}")
            return None
    return None

def update_status(script_path: Path):
    """Updates the status file with the name of the successfully completed script."""
    try:
        STATUS_FILE.write_text(str(script_path))
        print(f"Updated status: {script_path.name} completed.")
    except Exception as e:
        print(f"Error: Could not write to status file {STATUS_FILE}: {e}")
        print("Please check permissions or remove the status file and restart.")
        sys.exit(1)


def run_script(script_path: Path):
    """Runs a single python script using subprocess."""
    print(f"--- Running script: {script_path} ---")
    try:
        # Add the 'code' directory to PYTHONPATH so scripts in subdirectories can import modules from 'code'.
        code_dir = Path(__file__).parent.resolve()
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(code_dir) + os.pathsep + env.get("PYTHONPATH", "")

        process = subprocess.run(
            [sys.executable, script_path.resolve()],
            check=True,
            capture_output=False, # Set to True if you need to capture stdout/stderr
            text=True,
            env=env
        )
        print(f"--- Script {script_path.name} completed successfully. ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Script {script_path.name} failed with exit code {e.returncode}. !!!")
        print("Output (if any):")
        if e.stdout:
            print("STDOUT:\n", e.stdout)
        if e.stderr:
            print("STDERR:\n", e.stderr)
        print(f"--- Pipeline halted due to error in {script_path.name}. ---")
        print("Please fix the error in the script and then re-run this pipeline manager.")
        return False
    except FileNotFoundError:
        print(f"!!! ERROR: Script not found: {script_path}. Please check the path. !!!")
        print(f"--- Pipeline halted. ---")
        return False


def main():
    parser = argparse.ArgumentParser(description="Manage and run the bias correction data processing pipeline.")
    parser.add_argument(
        "--clean-status",
        action="store_true",
        help="If set, deletes the pipeline status file and exits."
    )
    args = parser.parse_args()

    if args.clean_status:
        if STATUS_FILE.exists():
            try:
                STATUS_FILE.unlink()
                print(f"Successfully deleted status file: {STATUS_FILE}")
            except Exception as e:
                print(f"Error deleting status file {STATUS_FILE}: {e}")
        else:
            print(f"Status file {STATUS_FILE} does not exist. Nothing to clean.")
        sys.exit(0) # Exit after cleaning

    print("Starting the bias correction data processing pipeline...")

    last_completed = get_last_completed_script()
    start_index = 0

    if last_completed:
        print(f"Last successfully completed script: {last_completed.name}")
        try:
            resolved_last_completed = last_completed.resolve()
            
            found = False
            for i, script_path_obj in enumerate(SCRIPTS_TO_RUN):
                if script_path_obj.resolve() == resolved_last_completed:
                    start_index = i + 1
                    found = True
                    break
            if not found:
                print(f"Warning: Last completed script '{last_completed}' not found in the defined pipeline. Starting from the beginning.")
                start_index = 0
            elif start_index >= len(SCRIPTS_TO_RUN):
                print("All scripts in the pipeline have already been completed successfully!")
                return
        except Exception as e:
            print(f"Error processing status file or script list: {e}. Starting from the beginning.")
            start_index = 0

    if start_index == 0:
        print("Starting pipeline from the beginning.")
    else:
        print(f"Resuming pipeline from script: {SCRIPTS_TO_RUN[start_index].name}")

    for i in range(start_index, len(SCRIPTS_TO_RUN)):
        current_script_path = SCRIPTS_TO_RUN[i]
        if not run_script(current_script_path):
            return 
        update_status(current_script_path)

    print("--- All scripts in the pipeline completed successfully! ---")
    # Optionally, clean up the status file after full completion
    if STATUS_FILE.exists():
        try:
            STATUS_FILE.unlink()
            print(f"Pipeline finished. Status file {STATUS_FILE} removed.")
        except Exception as e:
            print(f"Pipeline finished. Could not remove status file {STATUS_FILE}: {e}")


if __name__ == "__main__":
    # This script should be run from the 'code/' directory for paths to resolve correctly.
    main() 