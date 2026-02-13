# clear_output_dir.py

"""
With this script you can clear the output directory of all .xlsx files.
"""

# --- imports ---
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Constant variable names ---
OUTPUT_DIR = Path("output")

# --- main ---
def main():
    # --- remove only .xlsx files ---
    xlsx_files = list(OUTPUT_DIR.glob("*.xlsx"))
    for file in xlsx_files:
        try:
            file.unlink()  # deletes the file
        except PermissionError:
            logger.warning(f"Could not delete {file.name}, file is in use")

    logger.info("All .xlsx files cleared from output directory")

if __name__ == "__main__":
    main()
