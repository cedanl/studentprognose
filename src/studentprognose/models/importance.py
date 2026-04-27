def extract_grouped_importance(
    model, preprocessor, numeric_cols: list[str], categorical_cols: list[str],
) -> dict[str, float]:
    """Extract feature importances, grouping OneHotEncoded features back to original columns."""
    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    cats_by_length = sorted(categorical_cols, key=len, reverse=True)

    grouped = {}
    for name, imp in zip(feature_names, importances):
        if name.startswith("numeric__"):
            original = name.removeprefix("numeric__")
        elif name.startswith("categorical__"):
            remainder = name.removeprefix("categorical__")
            original = remainder
            for cat_col in cats_by_length:
                if remainder.startswith(cat_col + "_") or remainder == cat_col:
                    original = cat_col
                    break
        else:
            original = name

        if original.isdigit():
            original = f"Week {original}"

        grouped[original] = grouped.get(original, 0.0) + imp

    return grouped
