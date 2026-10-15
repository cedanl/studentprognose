import numpy as np
from numpy import linalg as LA
from statsforecast.models import ARIMA

from studentprognose.models.base import BaseForecaster
from studentprognose.data.transforms import ensure_start_week
from studentprognose.utils.weeks import (
    get_all_weeks_valid, get_all_weeks_ordered, get_forecast_weeks, compute_pred_len,
)
from studentprognose.utils.constants import (
    FINAL_ACADEMIC_WEEK,
    SARIMA_ORDER, SARIMA_ORDER_INDIVIDUAL, SARIMA_SEASONAL_ORDER, SARIMA_SEASONAL_ORDER_ALT,
    SARIMA_BACHELOR_DEADLINE_WEEKS,
)


class SARIMAForecaster(BaseForecaster):
    """Unified SARIMA forecaster used by both individual and cumulative strategies.

    Uses statsforecast ARIMA (CSS-ML estimation) as backend.
    """

    def __init__(self, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER):
        # Coerce naar tuples zodat ordes uit JSON-config (lijsten, bijv. getunede
        # waarden in model_config.forecaster_params) net zo werken als de
        # tuple-constanten — de ARIMA-backend verwacht tuples.
        self.order = tuple(order)
        seasonal_order = tuple(seasonal_order)
        self.seasonal_order = seasonal_order[:3]
        self.season_length = seasonal_order[3]
        self._model = None

    def fit(self, ts_data, exog=None):
        season_length = self.season_length
        seasonal_order = self.seasonal_order

        # Seizoens-ARIMA op te weinig data: de native statsforecast/numba-backend
        # crasht (SIGSEGV/SIGABRT door heap-corruptie) zodra een seizoens-
        # differentiatie/-AR/-MA wordt geschat op een reeks die korter is dan
        # twee volledige seizoenen. Empirisch precies bij ``len < 2*season_length``
        # (bijv. <104 weken bij season_length=52). Een seizoenscomponent met
        # periode ``season_length`` is uit <2 seizoenen ook statistisch niet te
        # schatten, dus val terug op een niet-seizoensmodel. Dit raakt korte
        # reeksen (bijv. nieuwe/kleine opleidingen, of fijnmazige CI-test-subsets);
        # lange reeksen (legacy-historie) houden het volledige seizoensmodel.
        if season_length > 1 and any(seasonal_order) and len(ts_data) < 2 * season_length:
            season_length = 1
            seasonal_order = (0, 0, 0)

        self._model = ARIMA(
            order=self.order,
            season_length=season_length,
            seasonal_order=seasonal_order,
        )
        X = exog.reshape(-1, 1) if exog is not None else None
        self._model.fit(y=ts_data.astype(np.float64), X=X)
        return self

    def forecast(self, steps, exog=None):
        X = exog.reshape(-1, 1) if exog is not None else None
        result = self._model.predict(h=steps, X=X)
        return result["mean"]


def split_time_series(
    data, predict_year: int, predict_week: int, final_week: int = FINAL_ACADEMIC_WEEK
) -> tuple[np.ndarray, list[str]]:
    """Bouw de trainingsreeks tot en met ``predict_week`` en bepaal de horizon.

    De jaren (rijen van ``data``, wide format) worden achter elkaar geplakt over
    de weekkolommen die bestaan (``get_all_weeks_valid``). De horizon is het
    aantal van díe kolommen ná ``predict_week`` — niet ``compute_pred_len``, dat
    alle kalenderweken telt. Zo eindigt de trainingsreeks altijd precies op
    ``predict_week``, ook als er weekkolommen ontbreken (#301).

    Args:
        data: Wide-format data voor één opleiding/herkomst/examentype, één rij
            per collegejaar; de laatste rij is het voorspeljaar.
        predict_year: Het collegejaar waarin ``predict_week`` valt.
        predict_week: Laatste bekende week.
        final_week: Laatste week van het academisch jaar.

    Returns:
        (trainingsreeks, doelweken): de doelweken zijn de weekkolommen
        waarvoor de forecast geldt, in volgorde.

    Raises:
        ValueError: Als het voorspeljaar niet de laatste rij is, of als
            ``predict_week`` geen weekkolom is. Afkappen zou dan in een ander
            jaar of op een andere week vallen.
    """
    data = data.sort_values("Collegejaar", kind="stable")
    if data.empty or int(data["Collegejaar"].iloc[-1]) != int(predict_year):
        raise ValueError(
            f"Geen rij voor collegejaar {predict_year}; de tijdreeks kan niet op "
            f"week {predict_week} van dat jaar worden afgekapt."
        )
    target_weeks = get_forecast_weeks(data.columns, predict_week, final_week)
    ts_data = data.loc[:, get_all_weeks_valid(data.columns, final_week)].values.flatten()
    if target_weeks:
        ts_data = ts_data[: -len(target_weeks)]
    return np.asarray(ts_data, dtype=np.float64), target_weeks


def align_forecast_to_calendar(
    forecast, target_weeks: list[str], predict_week: int, final_week: int = FINAL_ACADEMIC_WEEK
) -> np.ndarray:
    """Zet een forecast over ``target_weeks`` op de kalenderweken ná ``predict_week``.

    Aanroepers verwachten één waarde per kalenderweek (``predict_week`` + 1 t/m
    ``final_week``, lengte ``compute_pred_len``). Weken zonder weekkolom in de
    data krijgen ``NaN``: daarvoor bestaat geen waarneming en dus geen forecast.
    """
    ordered = get_all_weeks_ordered(final_week)
    calendar = ordered[ordered.index(str(int(predict_week))) + 1:]
    aligned = np.full(compute_pred_len(int(predict_week), final_week), np.nan)
    position = {week: i for i, week in enumerate(calendar)}
    forecast = np.asarray(forecast, dtype=np.float64)
    for week, value in zip(target_weeks, forecast):
        aligned[position[week]] = value
    return aligned


def _default_forecaster_factory() -> BaseForecaster:
    return SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)


def shrink_season_length_to_period(model, columns, final_week: int = FINAL_ACADEMIC_WEEK):
    """Stem de seizoenslengte van ``model`` af op de werkelijke jaar-periode.

    ``create_time_series`` plakt de trainingsjaren achter elkaar over precies de
    weekkolommen uit ``get_all_weeks_valid`` (de gevulde-week-kolommen van de
    gepivote data — de kruisjaarse/kruisopleiding-union — plus de geïnjecteerde
    reset-week). Dát aantal, niet de nominale 52, is de jaarblok-stride van de
    afgevlakte reeks. De seizoenslag moet daaraan gelijk zijn; staat hij vast op
    52 terwijl de stride korter is, dan wijst de lag naar de verkeerde
    week-in-het-jaar en staat de jaarcyclus uit fase. Het model kan de
    seizoensvorm dan niet reproduceren — op de UvA-funneldata uit zich dat als
    een prognose die ná de piek omhoog drijft i.p.v. de daling te volgen.

    Verkleint alleen: een volledig gevuld jaar (``periode == season_length``) en
    modellen zonder ``season_length`` blijven ongemoeid. Geldt voor elk model met
    een ``season_length`` (SARIMA, ETS, Theta, AutoARIMA). De toewijzing is
    defensief: een forecaster met een alleen-lezen ``season_length`` wordt
    overgeslagen i.p.v. te crashen. Wordt op exact dezelfde wijze toegepast in de
    benchmark (``evaluate_ts.py``) zodat die hetzelfde model meet als productie.
    """
    period = len(get_all_weeks_valid(columns, final_week))
    if hasattr(model, "season_length") and 1 < period < model.season_length:
        try:
            model.season_length = period
        except AttributeError:
            pass
    return model


def predict_with_sarima_cumulative(
    data_cumulative,
    row,
    predict_year,
    predict_week,
    pred_len: int,
    skip_years=0,
    already_printed=False,
    min_training_year: int = 2016,
    forecaster_factory: "callable | None" = None,
    final_week: int = FINAL_ACADEMIC_WEEK,
) -> list:
    """Voorspelt vooraanmeldingen per programme/herkomst/week voor cumulatieve data.

    Args:
        pred_len: Aantal kalenderweken ná ``predict_week`` t/m ``final_week``
            (``compute_pred_len``); de lengte van de geretourneerde reeks. De
            forecast-horizon zelf volgt uit de weekkolommen (zie
            ``split_time_series``).
        forecaster_factory: Callable die een vers BaseForecaster-object retourneert.
            Wordt per aanroep gecalld zodat joblib-parallellisatie veilig werkt.
            Default: SARIMAForecaster met standaard ordes.
        final_week: Laatste week van het academisch jaar (default 38; UvA 36).
            Bepaalt de seizoensvolgorde en de reset-week-injectie.

    Returns:
        Eén voorspelling per kalenderweek ná ``predict_week`` t/m ``final_week``
        (lengte ``pred_len``); ``NaN`` voor weken die als weekkolom in de data
        ontbreken. Lege lijst bij fout.
    """
    programme = row["Croho groepeernaam"]
    herkomst = row["Herkomst"]
    examentype = row["Examentype"]

    if not already_printed:
        print(
            f"Prediction for {programme}, {herkomst}, year: {predict_year}, week: {predict_week}"
        )

    data_cumulative = data_cumulative.astype(
        {"Weeknummer": "int32", "Collegejaar": "int32"}
    )
    data = _get_transformed_data(data_cumulative.copy(deep=True), min_training_year, final_week)

    data = data[
        (data["Herkomst"] == herkomst)
        & (data["Collegejaar"] <= predict_year - skip_years)
        & (data["Croho groepeernaam"] == programme)
        & (data["Examentype"] == examentype)
    ]

    data = ensure_start_week(data, final_week)

    try:
        ts_data, target_weeks = split_time_series(
            data, predict_year - skip_years, predict_week, final_week
        )
        if not target_weeks:
            return align_forecast_to_calendar([], [], predict_week, final_week)
        factory = forecaster_factory or _default_forecaster_factory
        # Seizoenslengte afstemmen op de werkelijke jaar-periode (gevulde-week-
        # kolommen), niet de vaste 52 — zie shrink_season_length_to_period.
        model = shrink_season_length_to_period(factory(), data.columns, final_week)
        model.fit(ts_data)
        pred = model.forecast(steps=len(target_weeks))
        return align_forecast_to_calendar(pred, target_weeks, predict_week, final_week)

    except (LA.LinAlgError, IndexError, ValueError) as error:
        print(f"Cumulative sarima error on: {programme}, {herkomst}")
        print(error)
        return []


def predict_with_sarima_individual(data_individual, row, predict_year, predict_week, max_year, numerus_fixus_list, data_exog=None, already_printed=False) -> list:
    """
    Predicts nr of students with SARIMA per programme/origin/week for individual data.

    Returns:
        list: predictions for each future week, or empty list on error.
    """
    from studentprognose.data.transforms import transform_data

    data = data_individual.copy()
    programme = row["Croho groepeernaam"]
    herkomst = row["Herkomst"]
    examentype = row["Examentype"]

    if not already_printed:
        print(
            f"Prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}"
        )

    def filter_data(data, programme, herkomst, examentype, jaar, max_year):
        data = data[data["Herkomst"] == herkomst]
        if jaar != max_year:
            data = data[data["Collegejaar"] <= jaar]
        data = data[data["Croho groepeernaam"] == programme]
        data = data[data["Examentype"] == examentype]
        return data

    if data_exog is not None:
        data_exog = filter_data(
            data_exog, programme, herkomst, examentype, predict_year, max_year
        )
    data = filter_data(data, programme, herkomst, examentype, predict_year, max_year)

    def deadline_week(weeknummer, croho, examentype):
        if (
            weeknummer in [16, 17]
            and examentype == "Bachelor"
            and croho not in numerus_fixus_list
        ):
            return 1
        elif (
            weeknummer in [1, 2]
            and examentype == "Bachelor"
            and croho in numerus_fixus_list
        ):
            return 1
        else:
            return 0

    if data_exog is not None:
        data_exog["Deadline"] = data_exog.apply(
            lambda x: deadline_week(x["Weeknummer"], x["Croho groepeernaam"], x["Examentype"]),
            axis=1,
        )

    try:
        if data_exog is not None:
            data_exog = transform_data(data_exog, "Deadline")

        if predict_week == FINAL_ACADEMIC_WEEK:
            ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            try:
                return [ts_data[-1]]
            except IndexError:
                return []

        ts_data, target_weeks = split_time_series(data, predict_year, predict_week)

        if data_exog is not None:
            exogenous_train_1, exog_weeks = split_time_series(data_exog, predict_year, predict_week)
            if exog_weeks != target_weeks:
                raise ValueError("Exogene reeks heeft andere weekkolommen dan de tijdreeks.")
            exg_data_test_1 = (
                data_exog.sort_values("Collegejaar", kind="stable")
                .iloc[-1][target_weeks].to_numpy(dtype=np.float64)
            )
        else:
            exogenous_train_1 = None

        if ts_data.size == 0:
            return []

        try:
            if programme.startswith("B") and predict_week in SARIMA_BACHELOR_DEADLINE_WEEKS:
                model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
            else:
                model = SARIMAForecaster(order=SARIMA_ORDER_INDIVIDUAL, seasonal_order=SARIMA_SEASONAL_ORDER_ALT)

            model.fit(ts_data, exog=exogenous_train_1)

            if data_exog is not None:
                pred = model.forecast(steps=len(target_weeks), exog=exg_data_test_1)
            else:
                pred = model.forecast(steps=len(target_weeks))

            return align_forecast_to_calendar(pred, target_weeks, predict_week)
        except (LA.LinAlgError, IndexError, ValueError) as error:
            print(f"Individual error on: {programme}, {herkomst}")
            print(error)
            return []
    except (KeyError, ValueError) as error:
        print(f"Individual error on: {programme}, {herkomst}")
        print(error)
        return []


def _get_transformed_data(data, min_training_year: int = 2016, final_week: int = FINAL_ACADEMIC_WEEK):
    """Helper to transform cumulative data for SARIMA.

    Args:
        data: Cumulative pre-application data.
        min_training_year: Earliest academic year included in training. Should be
            read from ``model_config.min_training_year`` in the caller's configuration.
        final_week: Laatste week van het academisch jaar (default 38; UvA 36).

    Een week die binnen een jaar ontbreekt tussen twee waarnemingen in, wordt
    lineair geïnterpoleerd in plaats van op 0 gezet: in een cumulatieve reeks is
    een val naar nul midden in het jaar een artefact (bijv. een ontbrekend
    telbestand), geen echte waarneming (#301).
    """
    from studentprognose.data.transforms import transform_data

    data = data.drop_duplicates()
    data = data[data["Collegejaar"] >= min_training_year]
    data = transform_data(data, "ts", final_week, interpolate_interior_gaps=True)
    return data
