import numpy as np
import pandas as pd
import joblib
import math

import warnings

from studentprognose.config import get_cpu_count
from studentprognose.utils.parallel import run_parallel_with_fallback
from studentprognose.utils.programme_key import normalize_programme_series
from studentprognose.strategies.base import PredictionStrategy
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
from studentprognose.utils.weeks import (
    compute_pred_len, get_all_weeks_ordered, week_sort_key,
)
from studentprognose.models import create_forecaster, create_regressor
from studentprognose.models.sarima import predict_with_sarima_cumulative, _get_transformed_data
from studentprognose.models.xgboost_regressor import predict_with_xgboost
from studentprognose.data.transforms import (
    TRANSFORM_GROUP_COLS, drop_weeks_after_predict_week, ensure_start_week,
)
from studentprognose.data.preprocessing.excluded_data_points import apply_excluded_data_points

_EPS = 1e-8
_LAGS = [2, 5]
ENGINEERED_FEATURE_COLS = (
    [f"Gewogen_t-{lag}" for lag in _LAGS]
    + ["Gewogen_acceleration", "exclusivity_ratio"]
)


def _academic_weeks_until(predict_week: int, final_week: int = FINAL_ACADEMIC_WEEK) -> list[int]:
    """De weken van het academisch jaar t/m ``predict_week``, in seizoensvolgorde.

    ISO-week 53 telt mee als ``predict_week`` 53 is; hij valt tussen 52 en 1.
    """
    weeks = {int(w) for w in get_all_weeks_ordered(final_week)} | {int(predict_week)}
    limit = week_sort_key(int(predict_week), final_week)
    return sorted(
        (w for w in weeks if week_sort_key(w, final_week) <= limit),
        key=lambda w: week_sort_key(w, final_week),
    )


def _lag_reference_week(
    predict_week: int, lag: int, final_week: int = FINAL_ACADEMIC_WEEK
) -> int:
    """De week ``lag`` weken vóór ``predict_week``, in academische volgorde.

    Telt terug binnen hetzelfde academisch jaar en stopt bij de startweek. Een
    kale ``predict_week - lag`` wijst aan het begin van het seizoen (bv. week 37
    bij ``final_week`` 36) naar het éínde van hetzelfde collegejaar: in een
    afgerond jaar is dat de toekomst, in een live run bestaat die week nog niet
    (#318).
    """
    past = _academic_weeks_until(predict_week, final_week)
    return past[max(len(past) - 1 - lag, 0)]


def _add_engineered_features(
    full_data: pd.DataFrame,
    data_cumulative: pd.DataFrame,
    predict_week: int,
    final_week: int = FINAL_ACADEMIC_WEEK,
) -> pd.DataFrame:
    """Add lagged features, Gewogen_acceleration en exclusivity_ratio to wide-format data.

    Computes lags (defined by _LAGS) relative to predict_week from the long-format source
    data and merges them into the pivoted XGBoost feature matrix. Lags count back in
    academic week order (see ``_lag_reference_week``), so a feature never uses a week
    after ``predict_week``. Missing reference weeks fall back to week 1 when week 1 lies
    at or before ``predict_week``; otherwise, or if week 1 is also absent, the value is
    filled with 0.

    Args:
        full_data: Wide-format feature matrix (one row per group and year).
        data_cumulative: Long-format source data.
        predict_week: Week from which is predicted.
        final_week: Last week of the academic year; determines the week order.

    Returns:
        ``full_data`` with exactly the columns in ENGINEERED_FEATURE_COLS added.
    """
    src = data_cumulative
    grp = TRANSFORM_GROUP_COLS
    week1_known = week_sort_key(1, final_week) <= week_sort_key(int(predict_week), final_week)

    for lag in _LAGS:
        ref_week = _lag_reference_week(int(predict_week), lag, final_week)
        col_name = f"Gewogen_t-{lag}"

        ref_values = (
            src[src["Weeknummer"] == ref_week]
            .groupby(grp)["Gewogen vooraanmelders"]
            .mean()
            .rename(col_name)
            .reset_index()
        )
        full_data = full_data.merge(ref_values, on=grp, how="left")

        missing_mask = full_data[col_name].isna()
        if missing_mask.any():
            # Week 1 als terugval alleen als die op het voorspelmoment al bekend was.
            fallback_src = src[src["Weeknummer"] == 1] if week1_known else src.iloc[0:0]
            fallback = (
                fallback_src
                .groupby(grp)["Gewogen vooraanmelders"]
                .mean()
                .rename("_fallback")
                .reset_index()
            )
            fill_series = (
                full_data.loc[missing_mask, grp]
                .merge(fallback, on=grp, how="left")["_fallback"]
            )
            n_zero = fill_series.isna().sum()
            if n_zero:
                warnings.warn(
                    f"{n_zero} rijen missen week-1 fallback voor lag={lag}. Gevuld met 0.",
                    UserWarning,
                    stacklevel=2,
                )
            full_data.loc[missing_mask, col_name] = fill_series.fillna(0).values

    # Compute acceleration and exclusivity ratio from a single predict_week slice.
    curr_src = src[src["Weeknummer"] == predict_week]

    current_gewogen = (
        curr_src
        .groupby(grp)["Gewogen vooraanmelders"]
        .mean()
        .rename("_gewogen_curr")
        .reset_index()
    )
    full_data = full_data.merge(current_gewogen, on=grp, how="left")
    full_data["Gewogen_acceleration"] = (
        (full_data["_gewogen_curr"].fillna(0) - full_data["Gewogen_t-2"])
        - (full_data["Gewogen_t-2"] - full_data["Gewogen_t-5"])
    )
    full_data = full_data.drop(columns=["_gewogen_curr"])

    excl_src = (
        curr_src
        .groupby(grp)[["Aantal aanmelders met 1 aanmelding", "Ongewogen vooraanmelders"]]
        .sum()
        .reset_index()
    )
    full_data = full_data.merge(excl_src, on=grp, how="left")
    full_data["exclusivity_ratio"] = (
        full_data["Aantal aanmelders met 1 aanmelding"].fillna(0)
        / (full_data["Ongewogen vooraanmelders"].fillna(0) + _EPS)
    )
    full_data = full_data.drop(
        columns=["Aantal aanmelders met 1 aanmelding", "Ongewogen vooraanmelders"]
    )

    return full_data


def _cast_string_to_float(data: pd.DataFrame, key: str) -> pd.DataFrame:
    if pd.api.types.is_string_dtype(data[key].dtype):
        data[key] = data[key].str.replace(".", "")
        data[key] = data[key].str.replace(",", ".")
    data[key] = pd.to_numeric(data[key], errors="coerce")
    data[key] = data[key].astype("float64")
    return data


def prepare_cumulative_data(data: pd.DataFrame) -> pd.DataFrame:
    """Breng de ruwe cumulatieve vooraanmelddata naar het formaat van de modellen.

    Gedeeld door het cumulatieve spoor en het CWF-model: castet de numerieke
    kolommen, hernoemt de bronkolommen, normaliseert de programmesleutel, houdt
    alleen eerstejaars (``Hogerejaars == "Nee"``, Pre-master altijd) over en
    aggregeert naar opleiding × herkomst × examentype × faculteit × jaar × week.

    Args:
        data: Cumulatieve data zoals ``load_data`` hem oplevert.

    Returns:
        Een DataFrame met de kolommen ``Weeknummer``, ``Collegejaar``,
        ``Faculteit``, ``Examentype``, ``Herkomst``, ``Croho groepeernaam`` en de
        vier aanmeldkolommen.
    """
    data = _cast_string_to_float(data, "Ongewogen vooraanmelders")
    data = _cast_string_to_float(data, "Gewogen vooraanmelders")
    data = _cast_string_to_float(data, "Aantal aanmelders met 1 aanmelding")
    data = _cast_string_to_float(data, "Inschrijvingen")

    data = data.rename(
        columns={
            "Type hoger onderwijs": "Examentype",
            "Groepeernaam Croho": "Croho groepeernaam",
        }
    )

    # Croho groepeernaam is in het legacy-formaat een leesbare string ("B
    # Psychologie"); bij het UvA SQL-formaat is het (voorlopig, #232) de
    # numerieke Isatcode. Normaliseer naar één canoniek dtype (Int64 voor
    # isatcodes, string voor namen) zodat downstream-merges op deze sleutel
    # (ratio-model, student_count, audittrail) en .isin-filters niet op
    # gemengde types struikelen. Zie utils.programme_key.
    data["Croho groepeernaam"] = normalize_programme_series(data["Croho groepeernaam"])

    data.loc[(data["Examentype"] == "Pre-master"), "Hogerejaars"] = "Nee"
    data = data[data["Hogerejaars"] == "Nee"]

    data = (
        data.groupby([
            "Collegejaar", "Croho groepeernaam", "Faculteit",
            "Examentype", "Herkomst", "Weeknummer",
        ])
        .sum(numeric_only=False)
        .reset_index()
    )
    return data[[
        "Weeknummer", "Collegejaar", "Faculteit", "Examentype",
        "Herkomst", "Croho groepeernaam", "Gewogen vooraanmelders",
        "Ongewogen vooraanmelders", "Aantal aanmelders met 1 aanmelding",
        "Inschrijvingen",
    ]]


class CumulativeStrategy(PredictionStrategy):
    def __init__(self, data_cumulative, data_studentcount, configuration,
                 data_latest, ensemble_weights, cwd, data_option, ci_test_n):
        super().__init__(configuration, data_latest, ensemble_weights,
                         data_studentcount, cwd, data_option, ci_test_n)

        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.skip_years = 0
        self.xgboost_importance = None
        self._importance_dicts: list[dict] = []

        self._forecaster_factory = lambda: create_forecaster(configuration)
        self._regressor = create_regressor(configuration)

        # Opt-in tuning (default uit). De API kiest welke trap(pen) van het
        # cumulatieve spoor getuned worden via ``model_config.tune_targets``
        # (target → zoekruimte/None): "regressor" (stap 2, hyperparameters) en/of
        # "sarima" (stap 1, ARIMA-ordes). Tuning gebeurt één keer — bij de eerste
        # predict, zodra de feature-matrix bestaat — via tijd-bewuste CV; daarna
        # gebruiken alle (jaar × week)-combinaties de gevonden waarden. Zie
        # models/tuning.py.
        model_config = configuration.get("model_config", {})
        tune_targets = dict(model_config.get("tune_targets") or {})
        # Backward-compat: de oude losse vlaggen mappen op de regressor-target.
        if model_config.get("tune_hyperparameters") and "regressor" not in tune_targets:
            tune_targets["regressor"] = model_config.get("tuning_grid")
        self._tune = "regressor" in tune_targets
        self._tuning_grid = tune_targets.get("regressor")
        self._tune_sarima = "sarima" in tune_targets
        self._sarima_tuning_grid = tune_targets.get("sarima")
        self._tuned = False
        # Eerste voorspeljaar bij het tunen; tuning gebruikte alleen jaren daarvóór.
        self._tuned_before_year: int | None = None

    def get_dashboard_data(self) -> dict:
        return {
            "data_cumulative": self.data_cumulative,
            "xgboost_curve": None,
        }

    def get_programme_columns_by_track(self) -> dict:
        col = self._programme_column_name()
        if self.data_cumulative is None or col not in self.data_cumulative.columns:
            return {}
        return {"cumulatief": self.data_cumulative[col]}

    def preprocess(self):
        self.data_cumulative = prepare_cumulative_data(self.data_cumulative)
        self.data_cumulative_backup = self.data_cumulative

        return self.data_cumulative

    def _cast_string_to_float(self, data, key):
        return _cast_string_to_float(data, key)

    def predict_nr_of_students(self, predict_year, predict_week, skip_years=0):
        self.data_cumulative = self.data_cumulative_backup.copy(deep=True)
        self.set_year_week(predict_year, predict_week, self.data_cumulative)
        if self.excluded_data_points:
            self.data_cumulative = apply_excluded_data_points(
                self.data_cumulative, self.excluded_data_points, self.predict_year
            )
        self._prepare_data()
        self.data_cumulative = self.data_cumulative.astype(
            {"Weeknummer": "int32", "Collegejaar": "int32"}
        )

        target_year = int(self.predict_year) - int(skip_years)
        model_data, full_data = self._build_model_input(target_year)

        if not self._tuned and (self._tune or self._tune_sarima):
            # Tuning ziet alleen jaren vóór het voorspeljaar: met het voorspeljaar
            # (of later) als testfold kiest hij parameters die de uitkomst al kennen.
            # Getuned wordt één keer, bij het eerste voorspelde jaar.
            # Elke trap wordt apart getuned en apart gelogd (eigen ✓-tabel).
            if self._tune_sarima:
                self._tune_sarima_orders(
                    self.data_cumulative[self.data_cumulative["Collegejaar"] < target_year]
                )
            if self._tune:
                self._tune_regressor(full_data[full_data["Collegejaar"] < target_year])
            self._tuned = True
            self._tuned_before_year = target_year
        elif self._tuned_before_year is not None and target_year < self._tuned_before_year:
            warnings.warn(
                f"De tuning is gedaan op alle jaren vóór {self._tuned_before_year}, dus ook op "
                f"{target_year}. De voorspelling voor {target_year} is daardoor geen eerlijke "
                "terugblik. Geef de jaren in oplopende volgorde op.",
                UserWarning,
                stacklevel=2,
            )

        self.skip_years = skip_years

        data_to_predict = model_data[
            (model_data["Collegejaar"] == self.predict_year)
            & (model_data["Weeknummer"] == self.predict_week)
        ]
        if self.programme_filtering != []:
            data_to_predict = data_to_predict[
                data_to_predict["Croho groepeernaam"].isin(self.programme_filtering)
            ]
        if self.herkomst_filtering != []:
            data_to_predict = data_to_predict[
                data_to_predict["Herkomst"].isin(self.herkomst_filtering)
            ]
        if self.examentype_filtering != []:
            data_to_predict = data_to_predict[
                data_to_predict["Examentype"].isin(self.examentype_filtering)
            ]

        if len(data_to_predict) == 0:
            return None

        data_to_predict = data_to_predict.copy()

        nr_CPU_cores = get_cpu_count(self.configuration)
        chunk_size = math.ceil(len(data_to_predict) / nr_CPU_cores)
        chunks = [
            data_to_predict[i : i + chunk_size] for i in range(0, len(data_to_predict), chunk_size)
        ]

        print("Start parallel predicting...")
        self.predicted_data = run_parallel_with_fallback(
            (
                joblib.delayed(self._predict_sarima)(row)
                for chunk in chunks
                for _, row in chunk.iterrows()
            ),
            n_jobs=nr_CPU_cores,
        )

        data_to_predict["SARIMA_individual"] = np.nan
        data_to_predict["Voorspelde vooraanmelders"] = np.nan
        if skip_years > 0:
            data_to_predict["Skip_prediction"] = np.nan

        data_to_predict = self.postprocessor.add_predicted_preregistrations(
            data_to_predict, [x[: self.pred_len] for x in self.predicted_data]
        )

        data_to_predict = self._predict_students_with_preapplicants(
            full_data, self.predicted_data, data_to_predict
        )

        return data_to_predict

    def _build_model_input(self, target_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Bouw de modelinput: alleen wat op het voorspelmoment bekend was.

        Van ``target_year`` (en later) blijven de weken t/m de voorspelweek over;
        de trainingsjaren blijven volledig (#318). ``self.data_cumulative`` wordt
        niet aangepast, zodat het dashboard de echte curve blijft tonen. Gedeeld
        met de both-modus (``CombinedStrategy``).

        Args:
            target_year: Het collegejaar waarin de voorspelweek valt
                (``predict_year - skip_years``).

        Returns:
            ``(model_data, full_data)``: de afgekapte data in lang formaat en de
            wide-format feature-matrix met engineered features.
        """
        model_data = drop_weeks_after_predict_week(
            self.data_cumulative, target_year, int(self.predict_week), self.final_academic_week
        )
        full_data = _get_transformed_data(
            model_data.copy(deep=True), self.min_training_year, self.final_academic_week
        )
        full_data = ensure_start_week(full_data, self.final_academic_week)
        full_data = _add_engineered_features(
            full_data, model_data, int(self.predict_week), self.final_academic_week
        )
        return model_data, full_data

    def _tune_regressor(self, full_data):
        """Tune de cumulatieve regressor op de feature-matrix en vervang ``self._regressor``.

        Draait een tijd-bewuste grid search (zie ``models/tuning.py``). Bij
        succes worden de beste parameters teruggekoppeld in
        ``configuration["model_config"]["regressor_params"]`` — zodat ze
        zichtbaar/te-bevriezen zijn — en wordt de actieve regressor ermee
        herbouwd. Bij te weinig trainingsjaren (geen tijdreeks-split mogelijk)
        wordt een waarschuwing gegeven en blijven de standaardparameters staan.
        """
        from studentprognose.models import REGRESSOR_REGISTRY
        from studentprognose.models.tuning import format_tuning_results, tune_regressor

        name = self.configuration.get("model_config", {}).get("cumulative_regressor", "xgboost")
        result = tune_regressor(
            full_data,
            self.data_studentcount,
            self.configuration,
            regressor_name=name,
            grid=self._tuning_grid,
            min_training_year=self.min_training_year,
            numerus_fixus_list=list(self.numerus_fixus_list),
        )

        # Print het volledige kandidatenoverzicht (zelfde tabel + snippet als het
        # `tune`-CLI-commando), zodat tune=True op de API hetzelfde inzicht geeft.
        print(format_tuning_results(result))

        if result["best_params"] is None:
            warnings.warn(
                "Hyperparameter tuning overgeslagen: te weinig trainingsjaren voor een "
                "tijdreeks-split. De standaardparameters van de regressor worden gebruikt.",
                UserWarning,
                stacklevel=2,
            )
            return

        best = result["best_params"]
        model_config = self.configuration.setdefault("model_config", {})
        model_config.setdefault("regressor_params", {})[name] = best
        self._regressor = REGRESSOR_REGISTRY[name](**best)

    def _tune_sarima_orders(self, data_cumulative: pd.DataFrame | None = None):
        """Tune de SARIMA-ordes (stap 1) en vervang ``self._forecaster_factory``.

        Draait een tijd-bewuste grid search over ARIMA-ordes (zie
        ``models/tuning.py``), met dezelfde curve-MAPE als de benchmark. Bij
        succes worden de beste ordes teruggekoppeld in
        ``configuration["model_config"]["forecaster_params"]["sarima"]`` — zodat
        ze zichtbaar/te-bevriezen zijn — en wordt de tijdreeks-factory ermee
        herbouwd. Bij te weinig trainingsjaren blijft het standaardmodel staan en
        volgt een waarschuwing.

        Args:
            data_cumulative: Lang-formaat data om op te tunen. ``None`` gebruikt
                ``self.data_cumulative``; de pipeline geeft alleen jaren vóór het
                voorspeljaar mee.
        """
        from studentprognose.models.sarima import SARIMAForecaster
        from studentprognose.models.tuning import format_tuning_results, tune_sarima

        result = tune_sarima(
            self.data_cumulative if data_cumulative is None else data_cumulative,
            int(self.predict_week),
            self.configuration,
            grid=self._sarima_tuning_grid,
            min_training_year=self.min_training_year,
        )

        # Eigen kandidatenoverzicht (eigen ✓-tabel), los van de regressor-tabel.
        print(format_tuning_results(result))

        if result["best_params"] is None:
            warnings.warn(
                "SARIMA-tuning overgeslagen: te weinig trainingsjaren voor een "
                "tijdreeks-split. De standaard SARIMA-ordes worden gebruikt.",
                UserWarning,
                stacklevel=2,
            )
            return

        best = result["best_params"]
        model_config = self.configuration.setdefault("model_config", {})
        model_config.setdefault("forecaster_params", {})["sarima"] = best
        # Default-arg bindt de gevonden ordes; elke parallelle SARIMA-fit krijgt
        # zo een verse forecaster met dezelfde getunede ordes.
        self._forecaster_factory = lambda b=best: SARIMAForecaster(**b)

    def _prepare_data(self):
        if self.data_studentcount is not None:
            self.data_cumulative = self.data_cumulative.merge(
                self.data_studentcount,
                on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
                how="left",
            )
        self.data_cumulative["ts"] = (
            self.data_cumulative["Gewogen vooraanmelders"] + self.data_cumulative["Inschrijvingen"]
        )
        self.data_cumulative = self.data_cumulative.drop_duplicates()

        self.pred_len = compute_pred_len(int(self.predict_week), self.final_academic_week)

    def _predict_sarima(self, row, already_printed=False):
        return predict_with_sarima_cumulative(
            self.data_cumulative, row, self.predict_year, self.predict_week,
            self.pred_len, self.skip_years, already_printed, self.min_training_year,
            forecaster_factory=self._forecaster_factory,
            final_week=self.final_academic_week,
        )

    def _fill_future_weeks(
        self, data: pd.DataFrame, mask: pd.Series, forecast: list[float] | np.ndarray
    ) -> bool:
        """Vul de weekkolommen ná de voorspelweek met de SARIMA-forecast.

        ``forecast`` heeft één waarde per kalenderweek ná de voorspelweek (zie
        ``align_forecast_to_calendar``). Een week zonder (eindige) forecast krijgt
        de laatst ingevulde waarde, te beginnen bij de voorspelweek zelf
        (persistentie). Dat gebeurt ook als SARIMA helemaal niets oplevert
        (lege ``forecast``). De regressor rekent dan met "er komt niets meer
        bij" in plaats van met nullen; de data van het voorspeljaar is vooraf al
        afgekapt, dus echte latere waarden zijn er niet (#318).

        Alleen bestaande weekkolommen worden gevuld: bij een dalende reeks valt
        de laatste weekkolom soms weg (combinaties zakken onder de
        openbaarmakingsdrempel).

        Args:
            data: Wide-format feature-matrix; wordt in place aangepast.
            mask: Rijen van deze combinatie in het voorspeljaar.
            forecast: Forecast per kalenderweek ná de voorspelweek, of leeg.

        Returns:
            True als elke gevulde week uit de forecast kwam, False als
            persistentie nodig was.
        """
        ordered = get_all_weeks_ordered(self.final_academic_week)
        limit = week_sort_key(int(self.predict_week), self.final_academic_week)
        calendar = [w for w in ordered if week_sort_key(int(w), self.final_academic_week) > limit]
        future = [w for w in calendar if w in data.columns]
        if not future or not mask.any():
            return True

        forecast = np.asarray(forecast, dtype=np.float64)
        by_week = {
            wk: forecast[k]
            for k, wk in enumerate(calendar)
            if k < len(forecast) and np.isfinite(forecast[k])
        }

        predict_col = str(int(self.predict_week))
        n_rows = int(mask.sum())
        if predict_col in data.columns:
            last = data.loc[mask, predict_col].to_numpy(dtype=np.float64)
        else:
            last = np.zeros(n_rows)
        values = np.empty((n_rows, len(future)))
        complete = True
        for j, wk in enumerate(future):
            if wk in by_week:
                last = np.full(n_rows, by_week[wk])
            else:
                complete = False
            values[:, j] = last
        data.loc[mask, future] = values
        return complete

    def _predict_students_with_preapplicants(self, data, predictions, data_to_predict):
        persisted: list[str] = []
        i = 0
        for _, row in data_to_predict.iterrows():
            programme = row["Croho groepeernaam"]
            herkomst = row["Herkomst"]
            examentype = row["Examentype"]

            if i == len(predictions):
                break

            if self.predict_week != self.final_academic_week:
                mask = (
                    (data["Collegejaar"] == self.predict_year - self.skip_years)
                    & (data["Croho groepeernaam"] == programme)
                    & (data["Herkomst"] == herkomst)
                    & (data["Examentype"] == examentype)
                )
                if not self._fill_future_weeks(data, mask, predictions[i]):
                    persisted.append(f"{programme} | {herkomst} | {examentype}")
            i += 1

            if programme in self.numerus_fixus_list:
                train = data[
                    (data["Croho groepeernaam"] == programme) & (data["Examentype"] == examentype)
                ]
                test = data[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Herkomst"] == herkomst)
                    & (data["Examentype"] == examentype)
                ]
                data_to_predict = self._predict_with_xgboost_extra_year(
                    train, test, data_to_predict,
                    (data_to_predict["Croho groepeernaam"] == programme)
                    & (data_to_predict["Herkomst"] == herkomst)
                    & (data_to_predict["Examentype"] == examentype),
                )

        if persisted:
            shown = ", ".join(persisted[:10]) + (" …" if len(persisted) > 10 else "")
            warnings.warn(
                f"SARIMA gaf voor {len(persisted)} combinatie(s) geen (volledige) prognose van "
                f"de vooraanmeldingen: {shown}. De ontbrekende weken na week {self.predict_week} "
                "zijn gevuld met de laatst bekende waarde. De regressor rekent daarmee, dus "
                "de voorspelling voor deze combinaties kan te laag uitvallen.",
                UserWarning,
                stacklevel=2,
            )

        all_examtypes = data_to_predict["Examentype"].unique()
        for examtype in all_examtypes:
            train = data[
                (data["Examentype"] == examtype)
                & (~data["Croho groepeernaam"].isin(self.numerus_fixus_list))
            ]
            test = data[
                (data["Examentype"] == examtype)
                & (~data["Croho groepeernaam"].isin(self.numerus_fixus_list))
            ]
            data_to_predict = self._predict_with_xgboost_extra_year(
                train, test, data_to_predict,
                (data_to_predict["Examentype"] == examtype)
                & (~data_to_predict["Croho groepeernaam"].isin(self.numerus_fixus_list)),
            )

        if self._importance_dicts:
            all_keys = set()
            for d in self._importance_dicts:
                all_keys.update(d.keys())
            self.xgboost_importance = {
                k: sum(d.get(k, 0.0) for d in self._importance_dicts) / len(self._importance_dicts)
                for k in all_keys
            }

        return data_to_predict

    def _warn_no_xgboost_training_data(self):
        """Waarschuw als het XGBoost-instroommodel geen trainingsdata heeft.

        Vervangt het vroegere stille ``print``: gaat nu via de warning-machinery
        (zichtbaar in pipeline-/Fabric-runs, filterbaar) en benoemt de echte oorzaak.
        De upfront-guard ``_check_training_history`` vangt het volledige
        "geen historie"-geval al af; deze waarschuwing dekt de resterende per-groep
        gevallen (bijv. een te kleine ``--ci test``-subset of backtesting met
        ``--skipyears``).
        """
        warnings.warn(
            f"XGBoost-instroomvoorspelling overgeslagen: geen trainingsdata met "
            f"Collegejaar < {self.predict_year}. Oorzaak: de cumulatieve data bevat "
            f"geen (voldoende) historische collegejaren — voeg ze toe aan df_cum — of, "
            f"in --ci test-modus, is de subset te klein (verhoog N). 'SARIMA_cumulative' "
            f"blijft NaN voor deze groep.",
            UserWarning,
            stacklevel=3,
        )

    def _predict_with_xgboost_extra_year(self, train, test, data_to_predict, replace_mask):
        columns_to_match = [
            "Collegejaar", "Faculteit", "Examentype", "Herkomst", "Croho groepeernaam",
        ]

        if self.skip_years > 0:
            train2 = train[(train["Collegejaar"] < self.predict_year - (self.skip_years * 2))]
            train2["Collegejaar"] = train2["Collegejaar"] + self.skip_years
            test2 = test[(test["Collegejaar"] == self.predict_year - self.skip_years)]
            test2["Collegejaar"] = test2["Collegejaar"] + self.skip_years

            test2_merged = data_to_predict[
                (data_to_predict["Weeknummer"] == self.predict_week) & replace_mask
            ][columns_to_match].merge(test2, on=columns_to_match)
            if not test2_merged.empty:
                if train2.empty:
                    self._warn_no_xgboost_training_data()
                    return data_to_predict
                test2["Collegejaar"] = test2["Collegejaar"] - self.skip_years
                ahead_predictions, imp = predict_with_xgboost(train2, test2_merged, self.data_studentcount, ENGINEERED_FEATURE_COLS, regressor=self._regressor, config=self.configuration)
                if imp is not None:
                    self._importance_dicts.append(imp)
                test2["Collegejaar"] = test2["Collegejaar"] + self.skip_years

                mask = (
                    data_to_predict[columns_to_match]
                    .apply(tuple, axis=1)
                    .isin(test2_merged[columns_to_match].apply(tuple, axis=1))
                )
                full_mask = (
                    replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask
                )
                data_to_predict.loc[full_mask, "Skip_prediction"] = ahead_predictions[: full_mask.sum()]
        else:
            train = train[(train["Collegejaar"] < self.predict_year)]
            test = test[(test["Collegejaar"] == self.predict_year)]

            test_merged = data_to_predict[
                (data_to_predict["Weeknummer"] == self.predict_week) & replace_mask
            ][columns_to_match].merge(test, on=columns_to_match)

            if not test_merged.empty:
                if train.empty:
                    self._warn_no_xgboost_training_data()
                    return data_to_predict
                predictions, imp = predict_with_xgboost(train, test_merged, self.data_studentcount, ENGINEERED_FEATURE_COLS, regressor=self._regressor, config=self.configuration)
                if imp is not None:
                    self._importance_dicts.append(imp)

                mask = (
                    data_to_predict[columns_to_match]
                    .apply(tuple, axis=1)
                    .isin(test_merged[columns_to_match].apply(tuple, axis=1))
                )
                full_mask = (
                    replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask
                )
                data_to_predict.loc[full_mask, "SARIMA_cumulative"] = predictions[: full_mask.sum()]

        return data_to_predict
