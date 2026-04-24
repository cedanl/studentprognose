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

    exclude_mask = pd.Series(False, index=df.index)

    for rule in rules:
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

        exclude_mask |= mask

    # Never exclude the prediction year — it contains the data to predict on.
    exclude_mask &= df[year_col] != predict_year

    return df[~exclude_mask].reset_index(drop=True)
