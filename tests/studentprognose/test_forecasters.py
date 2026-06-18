import numpy as np
import pandas as pd
import pytest

from studentprognose.models.base import BaseForecaster
from studentprognose.models.sarima import (
    SARIMAForecaster,
    predict_with_sarima_cumulative,
    shrink_season_length_to_period,
)
from studentprognose.models.forecasters import (
    ETSForecaster,
    ThetaForecaster,
    AutoARIMAForecaster,
)
from studentprognose.utils.constants import SARIMA_ORDER, SARIMA_SEASONAL_ORDER
from studentprognose.utils.weeks import (
    get_all_weeks_ordered,
    compute_pred_len,
    academic_start_week,
)


def _seasonal_series(n_years: int = 5, season_length: int = 52) -> np.ndarray:
    """Genereer synthetische seizoensgebonden tijdreeks."""
    n = n_years * season_length
    trend = np.linspace(100, 200, n)
    season = 20 * np.sin(2 * np.pi * np.arange(n) / season_length)
    noise = np.random.default_rng(42).normal(0, 2, n)
    return (trend + season + noise).astype(np.float64)


FORECASTER_CLASSES = [
    SARIMAForecaster,
    ETSForecaster,
    ThetaForecaster,
    AutoARIMAForecaster,
]


@pytest.mark.parametrize("cls", FORECASTER_CLASSES, ids=lambda c: c.__name__)
def test_forecaster_implements_base(cls):
    assert issubclass(cls, BaseForecaster)


@pytest.mark.parametrize("cls", FORECASTER_CLASSES, ids=lambda c: c.__name__)
def test_fit_forecast_returns_array(cls):
    ts = _seasonal_series()
    steps = 10
    model = cls()
    model.fit(ts)
    result = model.forecast(steps=steps)

    assert isinstance(result, np.ndarray)
    assert len(result) == steps
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("cls", [ETSForecaster, ThetaForecaster])
def test_exog_ignored_gracefully(cls):
    """ETS en Theta ondersteunen geen exogene variabelen — moeten exog negeren."""
    ts = _seasonal_series()
    model = cls()
    exog = np.random.default_rng(42).normal(0, 1, len(ts))
    model.fit(ts, exog=exog)
    result = model.forecast(steps=5)
    assert len(result) == 5


def test_auto_arima_supports_exog():
    """AutoARIMA moet exogene variabelen accepteren."""
    ts = _seasonal_series()
    n = len(ts)
    exog = np.random.default_rng(42).normal(0, 1, n)

    model = AutoARIMAForecaster()
    model.fit(ts, exog=exog)

    exog_future = np.random.default_rng(99).normal(0, 1, 5)
    result = model.forecast(steps=5, exog=exog_future)
    assert len(result) == 5


def test_forecaster_predictions_reasonable():
    """Voorspellingen moeten in dezelfde orde van grootte liggen als de trainingsdata."""
    ts = _seasonal_series()
    model = ETSForecaster()
    model.fit(ts)
    pred = model.forecast(steps=52)

    assert np.mean(pred) > 50
    assert np.mean(pred) < 500


def test_sarima_short_series_falls_back_to_nonseasonal():
    """Regressie (#231): de native statsforecast-backend crasht het PROCES
    (SIGSEGV/SIGABRT door heap-corruptie) bij ARIMA(1,1,1)x52 op een reeks korter
    dan twee volledige seizoenen. SARIMAForecaster valt dan terug op een niet-
    seizoensmodel. Zonder die guard haalt deze test het pytest-proces onderuit
    (exitcode 134)."""
    rng = np.random.default_rng(0)
    n = 93  # < 2 * 52: precies in het empirische crash-bereik
    ts = (np.cumsum(np.abs(rng.standard_normal(n))) + np.arange(n)).astype(np.float64)

    model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    model.fit(ts)

    assert model._model.season_length == 1  # seizoenscomponent uitgeschakeld
    pred = model.forecast(steps=10)
    assert len(pred) == 10
    assert np.all(np.isfinite(pred))


def test_sarima_long_series_keeps_seasonal_model():
    """Lange reeksen (>= 2 seizoenen) houden het volledige seizoensmodel intact."""
    ts = _seasonal_series(n_years=5)  # 260 >= 2 * 52

    model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    model.fit(ts)

    assert model._model.season_length == SARIMA_SEASONAL_ORDER[3]  # 52 behouden


class _RecordingForecaster(BaseForecaster):
    """Legt vast met welke ``season_length`` er gefit wordt (start als SARIMA)."""

    def __init__(self):
        self.season_length = SARIMA_SEASONAL_ORDER[3]  # = 52, zoals SARIMA
        self.fitted_season_length = None

    def fit(self, ts_data, exog=None):
        self.fitted_season_length = self.season_length
        return self

    def forecast(self, steps, exog=None):
        return np.zeros(steps)


def _cumulative_df(weeks, years=(2020, 2021, 2022, 2023, 2024)):
    """Long-format cumulatieve data voor één opleiding/herkomst.

    ``weeks`` is een lijst (zelfde weken voor elk jaar) of een dict
    ``{jaar: [weken]}`` voor uiteenlopende weekdekking per jaar.
    """
    rows = []
    yrs = sorted(weeks) if isinstance(weeks, dict) else years
    for yr in yrs:
        wks = weeks[yr] if isinstance(weeks, dict) else weeks
        for i, wk in enumerate(wks):
            rows.append({
                "Collegejaar": yr, "Faculteit": "F", "Herkomst": "NL",
                "Examentype": "Bachelor", "Croho groepeernaam": "TEST",
                "ts": 100.0 + 10.0 * i, "Weeknummer": int(wk),
            })
    return pd.DataFrame(rows)


def _funnel_cumulative_df(weeks, years, final_week=36, peak_week=23):
    """Long-format cumulatieve data met een funnel-vorm per jaar: stijgt naar de
    piek op ``peak_week`` en daalt daarna naar ~0 (zoals openstaande
    vooraanmeldingen die inschrijvingen worden)."""
    start = academic_start_week(final_week)

    def apos(w):
        return (w - start) % 52

    ppos = apos(peak_week)
    maxpos = max(apos(w) for w in weeks)
    rows = []
    for yr in years:
        for wk in weeks:
            p = apos(wk)
            val = 100.0 + 60.0 * (p / ppos) if p <= ppos else 160.0 * max(
                0.0, (maxpos - p) / (maxpos - ppos)
            )
            rows.append({
                "Collegejaar": yr, "Faculteit": "F", "Herkomst": "NL",
                "Examentype": "Bachelor", "Croho groepeernaam": "TEST",
                "ts": float(val), "Weeknummer": int(wk),
            })
    return pd.DataFrame(rows)


_ROW = {"Croho groepeernaam": "TEST", "Herkomst": "NL", "Examentype": "Bachelor"}


def _capturing_factory():
    """Factory die _RecordingForecasters bijhoudt zodat de fit-season_length
    inspecteerbaar is."""
    captured = []

    def factory():
        f = _RecordingForecaster()
        captured.append(f)
        return f

    return factory, captured


def test_shrink_season_length_helper():
    """Unit-test op de gedeelde helper: krimpt naar de periode, laat een volledige
    cyclus met rust, raakt modellen zonder season_length niet, en crasht niet op
    een alleen-lezen season_length."""
    cols = [str(w) for w in (list(range(45, 53)) + list(range(1, 31)) + [37])]

    m = SARIMAForecaster()
    shrink_season_length_to_period(m, cols, final_week=36)
    assert m.season_length == 39  # 8 (wk45-52) + 30 (wk1-30) + reset-week 37

    full = [str(w) for w in get_all_weeks_ordered(36)]
    m2 = SARIMAForecaster()
    shrink_season_length_to_period(m2, full, final_week=36)
    assert m2.season_length == 52  # volledige cyclus blijft ongemoeid

    class _NoSeason:
        pass

    obj = _NoSeason()
    shrink_season_length_to_period(obj, cols, final_week=36)  # no-op, geen crash
    assert not hasattr(obj, "season_length")

    class _ReadOnly:
        @property
        def season_length(self):
            return 52

    ro = _ReadOnly()
    shrink_season_length_to_period(ro, cols, final_week=36)  # mag niet crashen
    assert ro.season_length == 52


def test_cumulative_sarima_season_length_shrinks_to_data_period():
    """Regressie (#231/funnel): als een academisch jaar minder dan 52 gevulde
    weken heeft (lege start-/eindweken door de ≥21-drempel), moet de SARIMA-
    seizoenslengte naar de WERKELIJKE periode krimpen. Met de hardgecodeerde 52
    zit de seizoenslag uit fase en kan de piek-en-daling niet gevolgd worden."""
    weeks = list(range(45, 53)) + list(range(1, 31))  # 38 weken, geen volledige cyclus
    df = _cumulative_df(weeks)
    factory, captured = _capturing_factory()

    pred_len = compute_pred_len(20, 36)
    predict_with_sarima_cumulative(
        df, _ROW, 2024, 20, pred_len,
        already_printed=True, forecaster_factory=factory, final_week=36,
    )

    # 8 (wk45-52) + 30 (wk1-30) + 1 geïnjecteerde reset-week (37) = 39
    assert captured and captured[0].fitted_season_length == 39


def test_cumulative_sarima_period_is_union_across_years():
    """De periode is de UNION van gevulde weken over alle jaren, niet de dekking
    van één enkel jaar — verschilt de weekdekking per jaar, dan telt de ruimste."""
    weeks_by_year = {
        2020: list(range(45, 53)) + list(range(1, 21)),  # t/m wk20
        2021: list(range(45, 53)) + list(range(1, 21)),
        2022: list(range(45, 53)) + list(range(1, 31)),  # t/m wk30 (ruimst)
        2023: list(range(45, 53)) + list(range(1, 31)),
        2024: list(range(45, 53)) + list(range(1, 26)),  # t/m wk25
    }
    df = _cumulative_df(weeks_by_year)
    factory, captured = _capturing_factory()

    pred_len = compute_pred_len(20, 36)
    predict_with_sarima_cumulative(
        df, _ROW, 2024, 20, pred_len,
        already_printed=True, forecaster_factory=factory, final_week=36,
    )

    # union = wk45-52 + wk1-30 (van 2022/2023) + reset-week 37 = 39,
    # NIET de wk1-20 / wk1-25 van de smallere jaren.
    assert captured and captured[0].fitted_season_length == 39


def test_cumulative_sarima_period_is_union_across_programmes():
    """Productie pivot de HELE dataset en filtert dán op opleiding, dus de
    seizoenslengte is de cross-OPLEIDING-union van gevulde weken — niet de dekking
    van de gevraagde opleiding alleen. De benchmark spiegelt dit (evaluate_ts.py
    pivot ook eerst de hele dataset)."""
    narrow = list(range(45, 53)) + list(range(1, 21))   # opleiding NARROW: t/m wk20
    wider = list(range(45, 53)) + list(range(1, 31))    # opleiding WIDE:   t/m wk30
    rows = []
    for prog, wks in (("NARROW", narrow), ("WIDE", wider)):
        for yr in (2020, 2021, 2022, 2023, 2024):
            for i, wk in enumerate(wks):
                rows.append({
                    "Collegejaar": yr, "Faculteit": "F", "Herkomst": "NL",
                    "Examentype": "Bachelor", "Croho groepeernaam": prog,
                    "ts": 100.0 + 10.0 * i, "Weeknummer": int(wk),
                })
    df = pd.DataFrame(rows)
    factory, captured = _capturing_factory()

    pred_len = compute_pred_len(20, 36)
    # Voorspel voor de SMALLE opleiding; season_length moet tóch de cross-opleiding-
    # union (t/m wk30, van WIDE) zijn, niet NARROW's eigen dekking (t/m wk20).
    row = {"Croho groepeernaam": "NARROW", "Herkomst": "NL", "Examentype": "Bachelor"}
    predict_with_sarima_cumulative(
        df, row, 2024, 20, pred_len,
        already_printed=True, forecaster_factory=factory, final_week=36,
    )

    # union = wk45-52 (8) + wk1-30 (30) + reset-week 37 = 39, NIET 8 + 20 + 1 = 29.
    assert captured and captured[0].fitted_season_length == 39


def test_cumulative_sarima_keeps_full_season_when_all_weeks_present():
    """Een volledig academisch jaar (52 gevulde weken) blijft season_length=52:
    de fix verkleint alleen, hij verandert een correcte volledige cyclus niet."""
    weeks = [int(w) for w in get_all_weeks_ordered(36)]  # alle 52 weken
    df = _cumulative_df(weeks)
    factory, captured = _capturing_factory()

    pred_len = compute_pred_len(20, 36)
    predict_with_sarima_cumulative(
        df, _ROW, 2024, 20, pred_len,
        already_printed=True, forecaster_factory=factory, final_week=36,
    )

    assert captured and captured[0].fitted_season_length == 52


def test_cumulative_sarima_legacy_week38_no_crash():
    """De gedeelde cumulatieve route blijft werken voor de legacy-grens (week 38):
    geen crash en een zinnige (verkleinde) seizoenslengte."""
    weeks = list(range(40, 53)) + list(range(1, 31))
    df = _cumulative_df(weeks)
    factory, captured = _capturing_factory()

    pred_len = compute_pred_len(20, 38)
    predict_with_sarima_cumulative(
        df, _ROW, 2024, 20, pred_len,
        already_printed=True, forecaster_factory=factory, final_week=38,
    )

    assert captured and 1 < captured[0].fitted_season_length <= 52


def test_cumulative_sarima_funnel_forecast_declines():
    """Gedragstest met de ECHTE SARIMA-backend: op een funnel-reeks (piek week 23,
    daarna daling) volgt de prognose ná de piek de seizoensdaling i.p.v. omhoog te
    drijven. Borgt de kernclaim die de mechanische season_length-tests zelf niet
    kunnen aantonen (een fake-forecaster maskeert de richting)."""
    weeks = list(range(45, 53)) + list(range(1, 35))  # 42 weken -> periode 43
    years = (2019, 2020, 2021, 2022, 2023, 2024)
    df = _funnel_cumulative_df(weeks, years, final_week=36, peak_week=23)

    pred_len = compute_pred_len(23, 36)
    pred = np.asarray(predict_with_sarima_cumulative(
        df, _ROW, 2024, 23, pred_len, already_printed=True, final_week=36,
    ))

    assert len(pred) == pred_len
    assert np.all(np.isfinite(pred))
    assert pred[-1] < pred[0]  # daalt netto na de piek
    assert pred[: len(pred) // 2].mean() > pred[len(pred) // 2:].mean()
