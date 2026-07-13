import warnings

import pandas as pd


def apply_institution_filter(
    df: pd.DataFrame | None,
    institutions,
    institution_col: str = "Korte naam instelling",
) -> pd.DataFrame | None:
    """Beperk een dataset tot de opgegeven instelling(en).

    De landelijke Studielink-teldata bevat rijen van álle instellingen (elke
    Brincode staat in de kolom ``Korte naam instelling``). Deze filter scoopt de
    data tot de instelling(en) die de gebruiker wil voorspellen, zodat de
    modellen niet op andermans instroom trainen.

    Toegepast op load-tijd (vóór preprocessing en CI-subset), zodat zowel de
    training als de predictie op de gekozen instelling(en) draaien.

    Args:
        df: De te filteren dataset, of ``None``. ``None`` wordt ongewijzigd
            teruggegeven (spoor niet geladen).
        institutions: Lijst van instellingscodes/-namen om te behouden. Een lege
            lijst (of ``None``) betekent: geen filter — alle instellingen
            behouden (het default-gedrag).
        institution_col: Naam van de instellingskolom. Config-driven via
            ``column_roles.institution``; valt terug op ``"Korte naam instelling"``.

    Returns:
        Het gefilterde DataFrame (index gereset), of het originele object wanneer
        er niet gefilterd wordt.

    Raises:
        ValueError: Wanneer een niet-lege filter op een aanwezige instellingskolom
            geen enkele rij overhoudt. Stil doorgaan met een lege trainingsset zou
            de pipeline verderop met een cryptische fout laten crashen; dit maakt
            de oorzaak (verkeerde code) direct zichtbaar.
    """
    if df is None or not institutions:
        return df

    # Het spoor draagt geen instellingsdimensie (bijv. het individuele spoor is
    # de eigen aanmeldexport van één instelling). Niets te filteren — geen fout.
    if institution_col not in df.columns:
        return df

    # Vergelijk als string: Brincodes zijn tekstcodes ("00IC"), maar een gebruiker
    # kan een numerieke code als int in de config zetten. Casten voorkomt een
    # stille no-match op int-vs-str.
    wanted = [str(i) for i in institutions]
    col_as_str = df[institution_col].astype(str)

    present = set(col_as_str.unique())
    missing = [w for w in wanted if w not in present]
    if missing:
        warnings.warn(
            f"institution_filter: instelling(en) {sorted(missing)} komen niet voor in "
            f"kolom '{institution_col}'. Beschikbaar: {sorted(present)}. "
            f"Controleer of de code exact overeenkomt (hoofdlettergevoelig).",
            UserWarning,
            stacklevel=2,
        )

    filtered = df[col_as_str.isin(wanted)]

    if filtered.empty:
        raise ValueError(
            f"institution_filter: geen enkele rij komt overeen met de gevraagde "
            f"instelling(en) {wanted} in kolom '{institution_col}'. "
            f"Beschikbare instellingen: {sorted(present)}."
        )

    return filtered.reset_index(drop=True)
