import warnings

import pandas as pd


def apply_excluded_data_points(
    df: pd.DataFrame,
    rules: list[dict],
    predict_year: int,
    programme_col: str = "Croho groepeernaam",
    herkomst_col: str = "Herkomst",
    examentype_col: str = "Examentype",
    year_col: str = "Collegejaar",
) -> pd.DataFrame:
    """Remove training rows that match any exclusion rule in configuration.

    predict_year is always protected — rules never remove it, regardless of
    what the rule specifies. This prevents accidentally excluding the year
    being predicted.

    Each rule is a dict; all keys within one rule are AND-ed together.
    Multiple rules are OR-ed. Supported keys:

      year        – int, exact year to exclude
      year_before – int, excludes years strictly less than this value
      year_after  – int, excludes years strictly greater than this value
      herkomst    – str or list[str]
      examentype  – str or list[str]
      opleiding   – str or list[str], matched against programme_col
    """
    if not rules:
        return df

    # Vroeg valideren welke kolommen de regels vereisen. Dit geeft een duidelijke
    # foutmelding in plaats van een KeyError diep in de loop, en voorkomt stille
    # partial-matching als een configuratiefout een kolomnaam verkeerd spelt.
    _required = set()
    for rule in rules:
        if "herkomst" in rule:
            _required.add(herkomst_col)
        if "examentype" in rule:
            _required.add(examentype_col)
        if "opleiding" in rule:
            _required.add(programme_col)
    _missing = _required - set(df.columns)
    if _missing:
        raise ValueError(
            f"excluded_data_points: regel verwijst naar kolom(men) die niet in het "
            f"dataframe bestaan: {sorted(_missing)}. "
            f"Beschikbare kolommen: {sorted(df.columns)}."
        )

    # Elke regel beschrijft één uitsluitingsgebeurtenis (bv. COVID-jaar voor
    # specifieke opleiding). Sleutels binnen één regel worden ge-AND-ed om exact
    # die combinatie te treffen; meerdere regels worden ge-OR-ed zodat
    # onafhankelijke gebeurtenissen apart beschreven kunnen worden.
    exclude_mask = pd.Series(False, index=df.index)

    for i, rule in enumerate(rules):
        mask = pd.Series(True, index=df.index)

        if "year" in rule:
            mask &= df[year_col] == rule["year"]
        if "year_before" in rule:
            mask &= df[year_col] < rule["year_before"]
        if "year_after" in rule:
            mask &= df[year_col] > rule["year_after"]

        if "herkomst" in rule:
            values = rule["herkomst"] if isinstance(rule["herkomst"], list) else [rule["herkomst"]]
            mask &= df[herkomst_col].isin(values)
        if "examentype" in rule:
            values = rule["examentype"] if isinstance(rule["examentype"], list) else [rule["examentype"]]
            mask &= df[examentype_col].isin(values)
        if "opleiding" in rule:
            values = rule["opleiding"] if isinstance(rule["opleiding"], list) else [rule["opleiding"]]
            mask &= df[programme_col].isin(values)

        if not mask.any():
            warnings.warn(
                f"excluded_data_points: regel {i} ({rule}) matcht geen enkele rij in de "
                f"trainingsdata. Controleer of de waarden exact overeenkomen met de data "
                f"(hoofdlettergevoelig).",
                UserWarning,
                stacklevel=2,
            )

        exclude_mask |= mask

    # Een brede regel als year_before:2025 met predict_year=2024 zou anders de
    # rijen verwijderen waarop het model juist moet voorspellen. Dit is een
    # vangnet tegen regels die per ongeluk te ruim geformuleerd zijn.
    exclude_mask &= df[year_col] != predict_year

    return df[~exclude_mask].reset_index(drop=True)
