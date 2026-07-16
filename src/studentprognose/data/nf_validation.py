"""Guard op de ``numerus_fixus``-sleutels (issue #258).

De numerus-fixus-behandeling (aparte regressor, ratio-cap, aparte
WAPE-rapportage) grijpt alleen aan als een geconfigureerde ``numerus_fixus``-key
*exact* matcht met een waarde in de programmakolom (``Croho groepeernaam``).
Een niet-matchende key faalt **stil**: ``.isin``/``==`` levert simpelweg
``False`` op — geen error, geen waarschuwing. De opleiding wordt dan als een
gewone opleiding behandeld, belandt in de gedeelde regressor-pool en wordt niet
afgetopt.

De sleutelruimtes van de twee sporen overlappen bovendien niet: het cumulatieve
spoor keyt op numerieke Isatcodes, het individuele spoor op leesbare namen
(#238). Eén key kan daardoor per definitie maar één spoor matchen.

Deze module dwingt daarom af:

- **hard error** — een key matcht *geen enkel* geladen spoor (typefout of
  verkeerd dtype/formaat). De pipeline stopt.
- **waarschuwing** — een key matcht *wel* een spoor maar niet een ander geladen
  spoor (de bekende namen-vs-Isatcodes-mismatch, #238). De NF-behandeling grijpt
  dan niet aan in dat spoor; de gebruiker wordt gewaarschuwd i.p.v. stil
  genegeerd.
"""

from __future__ import annotations

import pandas as pd

from studentprognose.data.validation import ValidationResult


class NumerusFixusConfigError(ValueError):
    """Een numerus_fixus-sleutel matcht geen enkele geladen programmakolom.

    Subklasse van ``ValueError`` zodat het in-memory API-pad (dat excepties
    doorgeeft i.p.v. ``sys.exit``) de fout normaal kan afvangen, terwijl de CLI
    hem opvangt en netjes met exitcode 1 afsluit.
    """


def validate_numerus_fixus_keys(
    numerus_fixus_list: dict | None,
    programme_series_by_track: dict[str, pd.Series | None] | None,
    result: ValidationResult | None = None,
) -> ValidationResult:
    """Controleer of elke ``numerus_fixus``-key een programmakolom matcht.

    Args:
        numerus_fixus_list: De (reeds dtype-genormaliseerde) numerus-fixus-dict
            ``{sleutel: cap}``. Leeg of ``None`` → no-op.
        programme_series_by_track: Mapping van spoornaam
            (bijv. ``"cumulatief"``, ``"individueel"``) naar de programmakolom
            van dat spoor. Alleen de daadwerkelijk geladen sporen hoeven erin te
            zitten (afhankelijk van ``-d``).
        result: Optioneel bestaand :class:`ValidationResult` om aan toe te
            voegen; anders wordt een nieuwe aangemaakt.

    Returns:
        Het :class:`ValidationResult` met eventuele ``hard_errors`` (key matcht
        nergens) en ``warnings`` (key matcht niet elk geladen spoor).
    """
    result = result or ValidationResult()

    if not isinstance(numerus_fixus_list, dict) or not numerus_fixus_list:
        return result
    if not programme_series_by_track:
        return result

    # Bouw per spoor een set van unieke programmawaarden. .dropna() haalt
    # pandas-NA weg; .tolist() geeft native Python-typen zodat een int-key
    # (Isatcode) een Int64-kolom matcht en een str-key een namen-kolom.
    track_values: dict[str, set] = {}
    for track, series in programme_series_by_track.items():
        if series is None:
            continue
        track_values[track] = set(pd.Series(series).dropna().tolist())

    if not track_values:
        return result

    for key in numerus_fixus_list:
        matched = [track for track, values in track_values.items() if key in values]

        if not matched:
            result.hard_errors.append(
                f"numerus_fixus-sleutel {key!r} komt in geen enkele geladen "
                f"programmakolom voor (gecontroleerde sporen: "
                f"{', '.join(sorted(track_values))}). Gebruik de exacte waarde en "
                f"het exacte dtype uit de programmakolom — voor het cumulatieve "
                f"spoor de numerieke Isatcode, voor het individuele spoor de "
                f"leesbare opleidingsnaam (zie #258/#238)."
            )
            continue

        for track in sorted(track_values):
            if track not in matched:
                result.warnings.append(
                    f"numerus_fixus-sleutel {key!r} matcht wel het spoor "
                    f"'{matched[0]}' maar niet '{track}'. De numerus-fixus-"
                    f"behandeling grijpt daardoor niet aan in '{track}' (bekende "
                    f"sleutel-mismatch tussen namen en Isatcodes, #238)."
                )

    return result


def enforce_numerus_fixus_keys(
    numerus_fixus_list: dict | None,
    programme_series_by_track: dict[str, pd.Series | None] | None,
) -> None:
    """Draai de guard en handel de bevindingen af.

    Waarschuwingen worden geprint (pipeline loopt door); een niet-matchende
    sleutel gooit :class:`NumerusFixusConfigError`. Deze exception werkt in beide
    routes: het API-pad geeft hem door aan de aanroeper, de CLI vangt hem en sluit
    af met exitcode 1.

    Raises:
        NumerusFixusConfigError: Als minstens één sleutel geen enkel geladen
            spoor matcht.
    """
    result = validate_numerus_fixus_keys(numerus_fixus_list, programme_series_by_track)

    for warning in result.warnings:
        print(f"  [WAARSCHUWING] {warning}")

    if result.hard_errors:
        raise NumerusFixusConfigError(
            "Ongeldige numerus_fixus-configuratie:\n"
            + "\n".join(f"  - {e}" for e in result.hard_errors)
        )
