import random

CI_SEED = 42
CI_MIN_YEAR = 2022


def apply_ci_test_subset(
    n_programs,
    data_individual,
    data_cumulative,
    data_student_numbers_first_years,
    data_latest,
    data_weighted_ensemble,
):
    """
    Create a deterministic in-memory subset of all loaded datasets.

    Filters to Collegejaar >= 2022, then selects N random programs
    using a hardcoded seed so all colleagues get identical results.
    """

    all_programs = set()

    if data_individual is not None and "Croho groepeernaam" in data_individual.columns:
        recent = data_individual[data_individual["Collegejaar"] >= CI_MIN_YEAR]
        all_programs.update(recent["Croho groepeernaam"].unique())

    if data_cumulative is not None and "Groepeernaam Croho" in data_cumulative.columns:
        recent = data_cumulative[data_cumulative["Collegejaar"] >= CI_MIN_YEAR]
        all_programs.update(recent["Groepeernaam Croho"].unique())

    if len(all_programs) == 0:
        print("WARNING: No programs found for CI test subset. Skipping subsetting.")
        return (
            data_individual,
            data_cumulative,
            data_student_numbers_first_years,
            data_latest,
            data_weighted_ensemble,
        )

    sorted_programs = sorted(all_programs)
    max_programs = len(sorted_programs)

    if n_programs < 1 or n_programs > max_programs:
        raise ValueError(
            f"Invalid number of programs: {n_programs}. "
            f"Use a value between 1 and {max_programs}."
        )

    n = n_programs

    rng = random.Random(CI_SEED)
    selected_programs = rng.sample(sorted_programs, n)
    selected_set = set(selected_programs)

    print(f"CI test mode: selected {n} programs (seed={CI_SEED}, year>={CI_MIN_YEAR}):")
    for prog in sorted(selected_programs):
        print(f"  - {prog}")

    if data_individual is not None:
        data_individual = data_individual[
            (data_individual["Collegejaar"] >= CI_MIN_YEAR)
            & (data_individual["Croho groepeernaam"].isin(selected_set))
        ]

    if data_cumulative is not None:
        data_cumulative = data_cumulative[
            (data_cumulative["Collegejaar"] >= CI_MIN_YEAR)
            & (data_cumulative["Groepeernaam Croho"].isin(selected_set))
        ]

    if data_student_numbers_first_years is not None:
        if "Croho groepeernaam" in data_student_numbers_first_years.columns:
            mask = data_student_numbers_first_years["Croho groepeernaam"].isin(selected_set)
            if "Collegejaar" in data_student_numbers_first_years.columns:
                mask = mask & (data_student_numbers_first_years["Collegejaar"] >= CI_MIN_YEAR)
            data_student_numbers_first_years = data_student_numbers_first_years[mask]

    if data_latest is not None:
        if (
            "Collegejaar" in data_latest.columns
            and "Croho groepeernaam" in data_latest.columns
        ):
            data_latest = data_latest[
                (data_latest["Collegejaar"] >= CI_MIN_YEAR)
                & (data_latest["Croho groepeernaam"].isin(selected_set))
            ]

    if data_weighted_ensemble is not None:
        if (
            "Collegejaar" in data_weighted_ensemble.columns
            and "Programme" in data_weighted_ensemble.columns
        ):
            data_weighted_ensemble = data_weighted_ensemble[
                (data_weighted_ensemble["Collegejaar"] >= CI_MIN_YEAR)
                & (data_weighted_ensemble["Programme"].isin(selected_set))
            ]

    return (
        data_individual,
        data_cumulative,
        data_student_numbers_first_years,
        data_latest,
        data_weighted_ensemble,
    )
