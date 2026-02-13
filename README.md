# Student Forecasting Model

This Python project predicts the influx of students at Radboud University for the current year and week. It allows for specifying particular years and weeks for prediction.

## Installation

### Prerequisites
- **uv**: An extremely fast Python package installer and resolver.

### Installing UV
If you haven't installed `uv` yet, you can do so with the following commands:

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setting Up the Environment
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd studentprognose
    ```

2.  **Install dependencies:**
    `uv` will automatically handle the virtual environment and dependencies when you run commands. However, you can explicitly sync the environment:
    ```bash
    uv sync
    ```

## Configuration

### Environment Variables
Create a `.env` file in the root directory to specify the locations of your data files. You can use the provided `.env.example` (if available) or create one with the following content:

```env
ROOT_PATH="C:\\Path\\To\\Your\\Project\\Root"
OKTOBER_PATH="C:\\Path\\To\\Student\\Counts"
```
*   `ROOT_PATH`: The root folder containing your project files and data.
*   `OKTOBER_PATH`: The folder containing specific student count files (e.g., October census data).

### Configuration File
The `configuration.yaml` file controls various aspects of the model, including:
*   **Filtering**: Run models for specific programmes, origins (herkomst), or exam types.
*   **Paths**: specific input and output file paths.
*   **Numerus Fixus**: Settings for fixed-quota programmes.
*   **Faculties**: Mappings for faculty names.

Example filtering in `configuration.yaml`:
```yaml
filtering:
  programme:
   - B Sociologie
  herkomst:
   - NL
  examentype:
   - Bachelor
```

## Usage

You can run the scripts using `uv run`. This ensures the script runs in the project's virtual environment with the correct dependencies.

### Main Prediction Script
The `main` script is the entry point for predicting student influx.

**Run for the current week and year:**
```bash
uv run main
```

**Specify weeks and years:**
```bash
# Predict for week 6 of 2024
uv run main -w 6 -y 2024

# Predict for weeks 1, 2, 3 of 2024
uv run main -w 1 2 3 -y 2024

# Predict for a range of weeks (10 to 20) in 2023
uv run main -w 10:20 -y 2023
```

**Command Line Arguments:**
*   `-w`, `--weeks`: One or more week numbers or ranges (e.g., `5 6 7`, `10:15`, `39:38`).
*   `-y`, `--years`: One or more academic years or ranges (e.g., `2023 2024`, `2022:2025`).
*   `-wf`, `--write-file`: Write predictions to the total file.
*   `-p`, `--print`: Print programme output.
*   `-ev`, `--evaluate`: Evaluate the predictions.
*   `-v`, `--verbose`: Prints the model output.

### Other Scripts

**Calculate Student Counts:**
Generates files with aggregated student counts.
```bash
uv run scripts/standalone/calculate_student_count.py
```

**Calculate Ensemble Weights:**
Computes weights for the ensemble prediction model based on past performance.
```bash
uv run scripts/standalone/calculate_ensemble_weights.py
```

**Append Student Count and Compute Errors:**
Merges actual student counts with predictions and calculates errors (useful for historical validation).
```bash
uv run scripts/standalone/append_studentcount_and_compute_errors.py
```

**Predict Higher-Year Students:**
Runs predictions for higher-year students using an XGBoost model.
```bash
uv run scripts/higher_years/higher_years.py
```

## Data Description

The model relies on several datasets defined in `configuration.yaml`:
*   **cumulative**: Applications per programme/origin/year/week.
*   **individual**: Data on individual (pre-)applications.
*   **student_count_first-years**: Actual first-year student counts.
*   **student_count_higher-years**: Actual higher-year student counts.
*   **distances**: Distance data for NL students (used in XGBoost).

## Project Structure
*   `scripts/`: Contains the source code for models and utilities.
    *   `models/`: Core prediction models (cumulative, individual, ensemble, etc.).
    *   `standalone/`: Independent scripts for data processing and analysis.
    *   `utils/`: Helper functions.
*   `configuration.yaml`: Configuration settings.
*   `pyproject.toml`: Project metadata and dependencies.

## Development

### Pre-commit Hooks
This project uses pre-commit hooks to ensure code quality (formatting, syntax checks).
To install the hooks:
```bash
uv pip install pre-commit
pre-commit install
```
These hooks run automatically on commit.

### Running Tests
(Add instructions for running tests if available, e.g., `uv run pytest`)
