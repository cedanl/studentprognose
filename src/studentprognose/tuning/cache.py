import json
import os
from datetime import datetime, timezone

CACHE_PATH = os.path.join("data", "output", "tuning_cache.json")


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def load_cached_params(model_type: str) -> dict | None:
    """Load cached best parameters for a model type, or None if absent."""
    cache = _load_cache()
    return cache.get(model_type)


def save_cached_params(
    model_type: str, params: dict, cv_score: float, n_folds: int, n_combinations: int
) -> None:
    """Save best parameters to the tuning cache."""
    cache = _load_cache()
    cache[model_type] = {
        "params": params,
        "cv_score": cv_score,
        "n_folds": n_folds,
        "n_combinations": n_combinations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_cache(cache)


def load_sarima_order(
    programme: str, herkomst: str, examentype: str, strategy_type: str
) -> dict | None:
    """Load a cached SARIMA order for a specific combination."""
    cache = _load_cache()
    sarima_orders = cache.get("sarima_orders", {})
    key = f"{programme}|{herkomst}|{examentype}|{strategy_type}"
    return sarima_orders.get(key)


def save_sarima_orders(orders: dict[str, dict]) -> None:
    """Save all SARIMA orders at once. Each key is 'programme|herkomst|examentype|strategy'."""
    cache = _load_cache()
    cache["sarima_orders"] = orders
    cache["sarima_orders_timestamp"] = datetime.now(timezone.utc).isoformat()
    _save_cache(cache)
