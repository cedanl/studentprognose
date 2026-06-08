import warnings

import pandas as pd


def apply_institution_filter(
    data: pd.DataFrame,
    institution_filter: list[str],
    institution_column: str = "Korte naam instelling",
) -> pd.DataFrame:
    """Beperk de teldata tot één of meer instellingen.

    De teldata (telbestanden / Studielink-teldata) kan rijen van meerdere
    instellingen bevatten. Deze functie wordt toegepast **vóór** de cumulatieve
    aggregatie, want die groepeert per opleiding zónder de instellingskolom: zou
    je niet filteren, dan worden cijfers van dezelfde opleiding (bv. "B
    Psychologie") van verschillende instellingen bij elkaar opgeteld tot een
    betekenisloos totaal.

    Gedrag:

    - ``institution_filter`` leeg: er wordt niet gefilterd. Bevat de data méér
      dan één instelling, dan volgt een waarschuwing — de aggregatie telt de
      instellingen anders samen.
    - ``institution_filter`` gezet, maar ``institution_column`` ontbreekt in de
      data: ``ValueError`` (config/data-mismatch).
    - ``institution_filter`` gezet, maar geen enkele rij matcht: ``ValueError``
      met de wél beschikbare instellingen, zodat de gebruiker de code direct kan
      corrigeren in plaats van een stille lege uitvoer te krijgen.

    Args:
        data: Cumulatieve teldata vóór aggregatie.
        institution_filter: Lijst met instellingscodes (bv. BRIN ``["21PB"]``).
            Een lege lijst betekent "alle instellingen".
        institution_column: Naam van de kolom die de instelling identificeert.
            Standaard ``"Korte naam instelling"``; bij Studielink-teldata
            doorgaans ``"Brincode"`` (instelbaar via ``column_roles.institution``).

    Returns:
        De (eventueel) gefilterde data. Het oorspronkelijke DataFrame wordt niet
        gewijzigd.
    """
    if not institution_filter:
        if institution_column in data.columns:
            n_institutions = data[institution_column].nunique(dropna=True)
            if n_institutions > 1:
                warnings.warn(
                    f"De teldata bevat {n_institutions} instellingen, maar "
                    f"'institution_filter' is leeg. Hun cijfers worden per opleiding "
                    f"bij elkaar opgeteld, wat zelden bedoeld is. Zet "
                    f"'institution_filter' in configuration.json op je eigen "
                    f"instellingscode(s) om alleen die instelling te modelleren.",
                    UserWarning,
                    stacklevel=2,
                )
        return data

    if institution_column not in data.columns:
        raise ValueError(
            f"'institution_filter' is gezet, maar de instellingskolom "
            f"{institution_column!r} ontbreekt in de teldata. Beschikbare kolommen: "
            f"{sorted(data.columns)}. Controleer 'column_roles.institution' in je "
            f"configuratie."
        )

    available = sorted(data[institution_column].dropna().unique())
    filtered = data[data[institution_column].isin(institution_filter)]

    if filtered.empty:
        raise ValueError(
            f"'institution_filter' {institution_filter} matcht geen enkele rij in de "
            f"teldata. Beschikbare instellingen: {available}. Controleer of de codes "
            f"exact overeenkomen (hoofdlettergevoelig)."
        )

    return filtered
