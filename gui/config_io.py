"""Lezen, valideren en schrijven van ``configuration.json``.

Pure logica (geen NiceGUI) zodat de validatie getest kan worden. De editor-UI
(:mod:`gui.pages.config_page`) bouwt hierop.
"""

from __future__ import annotations

import json

#: Keuzelijsten voor de modelkeuzes (spiegelen de opties die de pipeline kent).
TIMESERIES_CHOICES = ["sarima", "ets", "theta", "auto_arima"]
REGRESSOR_CHOICES = [
    "xgboost",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "extra_trees",
]
CLASSIFIER_CHOICES = [
    "xgboost",
    "random_forest",
    "logistic_regression",
    "gradient_boosting",
    "extra_trees",
]

#: De vier verwachte ensemble-gewichtgroepen.
ENSEMBLE_GROUPS = ["master_week_17_23", "week_30_34", "week_35_37", "default"]

#: Tolerantie waarbinnen een gewichtsgroep als "telt op tot 1.0" geldt.
_WEIGHT_TOLERANCE = 1e-6


def load_config(path: str) -> dict:
    """Laad een configuratie-JSON.

    Raises:
        FileNotFoundError: Als het bestand niet bestaat.
        json.JSONDecodeError: Als het geen geldige JSON is.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_config(path: str, config: dict) -> None:
    """Schrijf de configuratie als nette, ingesprongen JSON (UTF-8, 4 spaties)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        f.write("\n")


def validate_ensemble_weights(ensemble_weights: dict) -> list[str]:
    """Controleer dat elke gewichtsgroep optelt tot 1.0.

    Args:
        ensemble_weights: De ``ensemble_weights``-dict uit de configuratie.

    Returns:
        Een lijst met foutmeldingen (leeg = geldig). Elke groep met een som die
        niet (binnen tolerantie) 1.0 is levert één melding op.
    """
    errors: list[str] = []
    for group, weights in ensemble_weights.items():
        if not isinstance(weights, dict):
            errors.append(f"Groep '{group}' heeft geen geldige gewichten.")
            continue
        individual = weights.get("individual")
        cumulative = weights.get("cumulative")
        if individual is None or cumulative is None:
            errors.append(f"Groep '{group}' mist 'individual' of 'cumulative'.")
            continue
        total = individual + cumulative
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            errors.append(
                f"Groep '{group}': individual + cumulative = {total:g} (moet 1.0 zijn)."
            )
    return errors


def validate_config(config: dict) -> list[str]:
    """Voer alle validaties uit en geef een gecombineerde lijst met fouten terug."""
    errors: list[str] = []
    errors.extend(validate_ensemble_weights(config.get("ensemble_weights", {})))
    return errors


def parse_json(text: str) -> dict:
    """Parse een JSON-string naar een dict.

    Raises:
        json.JSONDecodeError: Bij ongeldige JSON.
        ValueError: Als de top-level geen object is.
    """
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("De configuratie moet een JSON-object zijn.")
    return parsed
