# load_data_yaml.py
import os
import yaml
import pandas as pd
from pathlib import Path
from joblib import Memory
from dotenv import load_dotenv
import logging
from typing import Optional, Dict

# --- Environment and logging setup ---
load_dotenv()

VERBOSE = os.getenv("VERBOSE", "0") == "1"

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Disk cache ---
CACHE_DIR = "cache_dir"
memory = Memory(CACHE_DIR, verbose=0)

# --- Load configuration ---
ROOT_PATH = os.getenv("ROOT_PATH")
OKTOBER_PATH = os.getenv("OKTOBER_PATH")

if not ROOT_PATH or not OKTOBER_PATH:
    raise ValueError("Environment variables ROOT_PATH and OKTOBER_PATH must be set.")

CONFIG_FILE = Path("configuration.yaml")
with CONFIG_FILE.open("r") as f:
    _config = yaml.safe_load(f)


def _replace_env_vars(path: str) -> str:
    return path.replace("${root_path}", ROOT_PATH).replace("${oktober_path}", OKTOBER_PATH)


# --- Paths from YAML ---
_paths: Dict[str, Dict[str, str]] = {
    section_name: {k: _replace_env_vars(v) for k, v in section.items()}
    for section_name, section in _config.get("paths", {}).items()
}

_other_paths: Dict[str, str] = {
    k: _replace_env_vars(v) for k, v in _config.get("other_paths", {}).items()
}


# --- Helper loader ---
def _load_file(path: str, file_type: str = "csv", **kwargs) -> Optional[pd.DataFrame]:
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"File not found: {path}")
        return None

    logger.debug(f"Loading file: {path}")
    if file_type == "csv":
        return pd.read_csv(path_obj, **kwargs)
    elif file_type == "excel":
        return pd.read_excel(path_obj, **kwargs)
    return None


# --- Caching loaders ---
@memory.cache
def load_cumulative() -> Optional[pd.DataFrame]:
    '''
    Instead of downloading the full cumulative dataset every week, this function appends the new weekly file ("Vooraanmeldingen.csv") 
    to the existing cumulative file ("vooraanmeldingen_cumulatief.csv") automatically, keeping the dataset up to date.
    '''

    path_cum = _paths['input']['path_cumulative']
    path_new = _paths['input']['path_cumulative_new']

    logger.debug("Loading cumulative dataset...")
    data = _load_file(path_cum, file_type="csv", sep=";", skiprows=[1], low_memory=True)

    if Path(path_new).exists():
        logger.info(f"New cumulative file found: {path_new}, merging...")
        data_new = _load_file(path_new, file_type="csv", sep=";", skiprows=[1])

        if data_new is not None:
            columns_to_cast = [
                "Ongewogen vooraanmelders",
                "Gewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding",
                "Inschrijvingen"
            ]
            
            for col in columns_to_cast:
                data_new = _cast_string_to_float(data_new, col)

        data = pd.concat([data, data_new], ignore_index=True).drop_duplicates(keep="last") if data is not None else data_new

        data.to_csv(path_cum, sep=";", index=False)
        logger.info(f"Cumulative file updated: {path_cum}. Removed new file: {path_new}")
        os.remove(path_new)

    return data


@memory.cache
def load_individual() -> Optional[pd.DataFrame]:
    logger.debug("Loading individual dataset...")
    return _load_file(_paths['input']['path_individual'], file_type="csv", sep=";", skiprows=[1])


@memory.cache
def load_distances() -> Optional[pd.DataFrame]:
    logger.debug("Loading distances dataset...")
    return _load_file(_paths['input']['path_distances'], file_type="excel")


@memory.cache
def load_latest() -> Optional[pd.DataFrame]:
    logger.debug("Loading latest dataset...")
    return _load_file(_paths['input']['path_latest'], file_type="excel")


@memory.cache
def load_lookup_higher_years() -> Optional[pd.DataFrame]:
    logger.debug("Loading lookup higher years dataset...")
    return _load_file(_paths['input']['path_lookup_higher_years'], file_type="excel")


@memory.cache
def load_student_numbers_first_years() -> Optional[pd.DataFrame]:
    logger.debug("Loading first-year student numbers dataset...")
    return _load_file(_paths['input']['path_student_count_first_years'], file_type="excel")

@memory.cache
def load_oktober_file() -> Optional[pd.DataFrame]:
    logger.debug("Loading oktober file...")
    return _load_file(_other_paths['path_october'], file_type="excel")


# --- Public API ---
def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load all main datasets using cached loaders.
    Returns a dictionary with keys:
    'cumulative', 'individual', 'distances', 'latest',
    'lookup_higher_years', 'weighted_ensemble', 'student_numbers_first_years'
    """
    return {
        "cumulative": load_cumulative(),
        "individual": load_individual(),
        "distances": load_distances(),
        "latest": load_latest(),
        "lookup_higher_years": load_lookup_higher_years(),
        "student_numbers_first_years": load_student_numbers_first_years(),
    }


# --- Helpers to access paths ---
def get_paths() -> Dict[str, Dict[str, str]]:
    return _paths.copy()


def get_other_paths() -> Dict[str, str]:
    return _other_paths.copy()

# --- Helper to cast string to float ---
def _cast_string_to_float(data, key):
    if not pd.api.types.is_string_dtype(data[key]):
        return data
    
    data[key] = (
        data[key]
        .str.replace(r"\.", "", regex=True)
        .str.replace(",", ".", regex=False)
    )

    data[key] = pd.to_numeric(data[key], errors="coerce")

    return data

# --- Clear cache ---
def main():
    memory.clear(warn=False)

if __name__ == "__main__":
    main()
