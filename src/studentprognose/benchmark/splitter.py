import pandas as pd


def time_series_split(
    data: pd.DataFrame,
    min_training_year: int = 2016,
    min_train_years: int = 3,
    year_col: str = "Collegejaar",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Genereer (train, test) splits met strikte temporale scheiding.

    Train op jaren [min_training_year, N-1], test op jaar N.
    Garandeert minimaal min_train_years trainingsjaren per fold.

    Args:
        data: DataFrame met een jaar-kolom.
        min_training_year: Vroegste jaar dat in training gebruikt mag worden.
        min_train_years: Minimaal aantal trainingsjaren per fold.
        year_col: Naam van de jaar-kolom (configureerbaar via
            ``config["column_roles"]["academic_year"]``).

    Returns:
        Lijst van (train_df, test_df) tuples.
    """
    all_years = sorted(data[year_col].unique())
    eligible_years = [y for y in all_years if y >= min_training_year]

    if len(eligible_years) <= min_train_years:
        return []

    splits = []
    for i in range(min_train_years, len(eligible_years)):
        test_year = eligible_years[i]
        train_years = eligible_years[:i]

        train_df = data[data[year_col].isin(train_years)]
        test_df = data[data[year_col] == test_year]

        if train_df.empty or test_df.empty:
            continue

        assert train_df[year_col].max() < test_year, (
            f"Leakage: traindata bevat jaar {train_df[year_col].max()} "
            f"maar testjaar is {test_year}"
        )

        splits.append((train_df, test_df))

    return splits
