"""Range-validatie van gevraagde year/week tegen de geladen trainingsdata.

Draait *ná* het laden van de DataFrames — anders dan ``data/validation.py``, dat
ruwe inputbestanden *vóór* de ETL op datakwaliteit controleert. Dit module checkt
of de gevraagde ``year``/``week``-combinatie binnen de beschikbare
``Collegejaar``/``Weeknummer``-range valt.

Pure laag: detecteren en tekst formatteren, géén side effects. De CLI
(``sys.exit``) en de API (``raise ValueError``) in ``main.py`` beslissen zelf
hoe ze op een mismatch reageren, bovenop hetzelfde detectieresultaat.
"""

from typing import NamedTuple, Optional

import pandas as pd


class DataRangeMismatch(NamedTuple):
    """Resultaat van de range-detectie: welke jaren/weken ontbreken en wat wél beschikbaar is."""

    year_range: str  # "2018-2025" of "2025"
    week_range: str  # "1-52", "10", of "n.v.t." (individueel: geen Weeknummer-kolom)
    missing_years: list[int]
    missing_weeks: list[int]


def detect_data_range_mismatch(
    datasets: tuple[Optional[pd.DataFrame], ...],
    years: list[int],
    weeks: list[int],
) -> Optional[DataRangeMismatch]:
    """Detecteer of gevraagde jaren/weken buiten de beschikbare trainingsdata vallen.

    Pure functie: schrijft niets, beëindigt niets, gooit niets. Geeft ``None`` terug
    wanneer alles binnen bereik valt (of er geen bruikbare data is), anders een
    ``DataRangeMismatch`` met de beschikbare range en de ontbrekende waarden. Zowel het
    CLI-pad (``main._check_data_range``) als het API-pad
    (``main.run_pipeline_from_dataframes``) bouwen hun eigen melding bovenop dit resultaat.

    Args:
        datasets: Tuple waarvan de eerste twee elementen ``data_individual`` en
            ``data_cumulative`` zijn (langere tuples worden via ``*_`` genegeerd).
        years: Gevraagde jaren.
        weeks: Gevraagde weken.
    """
    data_individual, data_cumulative, *_ = datasets

    # Pick whichever dataset is loaded based on the chosen mode
    data = data_cumulative if data_cumulative is not None else data_individual
    if data is None:
        return None

    available_years = sorted(int(y) for y in data["Collegejaar"].dropna().unique())
    if not available_years:
        return None

    missing_years = [y for y in years if y not in available_years]

    # Individual dataset has no Weeknummer column before preprocessing
    if "Weeknummer" in data.columns:
        available_weeks = sorted(int(w) for w in data["Weeknummer"].dropna().unique())
        missing_weeks = [w for w in weeks if w not in available_weeks]
    else:
        available_weeks = []
        missing_weeks = []

    if not (missing_years or missing_weeks):
        return None

    year_range = _format_range(available_years)
    # "n.v.t." voorkomt een IndexError op available_weeks[0] wanneer er (nog) geen
    # weken bekend zijn (individuele dataset vóór preprocessing).
    week_range = _format_range(available_weeks) if available_weeks else "n.v.t."

    return DataRangeMismatch(year_range, week_range, missing_years, missing_weeks)


def _format_range(values: list[int]) -> str:
    """Formatteer een gesorteerde lijst als ``"min-max"`` (of ``"x"`` bij één waarde)."""
    return f"{values[0]}-{values[-1]}" if len(values) > 1 else str(values[0])


def _format_available(mismatch: DataRangeMismatch) -> str:
    """Beschrijf de beschikbare data; laat weken weg als die niet bepaald kunnen worden."""
    if mismatch.week_range == "n.v.t.":
        return f"jaren {mismatch.year_range}"
    return f"jaren {mismatch.year_range}, weken {mismatch.week_range}"


def format_cli_range_warning(mismatch: DataRangeMismatch) -> str:
    """Bouw de meerregelige CLI-waarschuwing uit een ``DataRangeMismatch``."""
    if mismatch.week_range == "n.v.t.":
        adjust_line = f"  Pas je flags aan tussen -y {mismatch.year_range},"
    else:
        adjust_line = (
            f"  Pas je flags aan tussen -y {mismatch.year_range} "
            f"en -w {mismatch.week_range},"
        )

    return "\n".join(
        [
            "\nWaarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data.",
            f"  Beschikbare data: {_format_available(mismatch)}.",
            adjust_line,
            "  of voeg nieuwe trainingsdata toe in data/input_raw/ om je gewenste tijdstip te voorspellen.",
        ]
    )


def format_api_range_error(year: int, week: int, mismatch: DataRangeMismatch) -> str:
    """Bouw de ValueError-tekst voor het API-pad uit een ``DataRangeMismatch``."""
    if mismatch.missing_years and mismatch.missing_weeks:
        subject = f"year={year} en week={week} vallen buiten de beschikbare trainingsdata."
    elif mismatch.missing_years:
        subject = f"year={year} valt buiten de beschikbare trainingsdata."
    else:
        subject = f"week={week} valt buiten de beschikbare trainingsdata."

    return (
        f"{subject}\n"
        f"  Beschikbare data: {_format_available(mismatch)}.\n"
        "  Pas year/week aan binnen deze range, of voeg trainingsdata toe."
    )
