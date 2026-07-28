"""Configuratie-editor: drie tabbladen — Basis, Geavanceerd, JSON.

Basis:       de 4 essentiële instellingen voor nieuwe gebruikers.
Geavanceerd: modelkeuze, ensemble-gewichten, numerus fixus, exclusieregels.
JSON:        directe bewerking voor experts.
"""

from __future__ import annotations

import csv
import datetime
import glob
import json
import os

from nicegui import ui

from gui import config_io, nav, theme
from gui.components.layout import page_shell
from gui.components.states import empty_state, error_banner, section_title
from gui.state import STATE


def _load_brincodes(project_dir: str) -> list[str]:
    """Lees unieke Brincodes uit de telbestanden van het project, gesorteerd."""
    tel_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    codes: set[str] = set()
    for path in glob.glob(os.path.join(tel_dir, "*.csv")):
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if "Brincode" not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    code = row.get("Brincode", "").strip()
                    if code:
                        codes.add(code)
        except Exception:
            continue
    return sorted(codes)

HELP = {
    "cumulative_timeseries": (
        "Tijdreeksmodel voor stap 1 van het cumulatieve spoor: extrapoleert de "
        "vooraanmelderscurve tot de laatste academische week. 'sarima' is de "
        "standaard; ets/theta/auto_arima zijn alternatieven (vergelijk ze via de "
        "Benchmark-tab)."
    ),
    "cumulative_regressor": (
        "Regressiemodel voor stap 2 van het cumulatieve spoor: vertaalt de "
        "vooraanmelderscijfers naar het verwachte aantal ingeschreven studenten."
    ),
    "individual_classifier": (
        "Classifier voor het individuele spoor: schat per aanmelding de kans op "
        "inschrijving. Standaard 'xgboost'."
    ),
    "min_training_year": (
        "Vroegste collegejaar dat als trainingsdata meetelt. Data van vóór dit "
        "jaar wordt genegeerd. Verlaag dit alleen als je betrouwbare historische "
        "data hebt die verder teruggaat."
    ),
    "final_academic_week": (
        "De laatste week van het academisch jaar in de Studielink-cyclus. Bepaalt "
        "de seizoensvolgorde van de voorspelhorizon en de reset-week in "
        "het cumulatieve spoor. Vaak 38; bij de UvA 36."
    ),
    "ensemble_weights": (
        "Gewicht van het individuele versus het cumulatieve spoor bij het "
        "combineren (modus 'beide'), per weeksegment. Elk paar moet optellen tot "
        "1,0. Wordt genegeerd in de losse sporen."
    ),
    "numerus_fixus": (
        "Opleidingen met een capaciteitslimiet (numerus fixus). Gebruik exact "
        "dezelfde programmasleutel als in je data; de voorspelling wordt op dit "
        "maximum afgetopt."
    ),
    "institution_filter": (
        "Beperk de teldata tot je eigen instelling(en) via Brincode of korte "
        "naam. Leeg = alle instellingen. De meeste gebruikers zetten hier hun "
        "eigen instelling."
    ),
    "excluded_data_points": (
        "Sluit bekende probleemjaren (bijv. een uitzonderlijk coronajaar) uit de "
        "trainingsdata. Het voorspeljaar zelf wordt altijd beschermd en nooit "
        "uitgesloten."
    ),
    "aggregate": (
        "Tel fijnmazige invoerrijen op naar de canonieke grain. Nodig voor o.a. de "
        "UvA-levering (bereken 'Gewogen' per rij, sommeer daarna); anders crasht "
        "de pivot op dubbele indexrijen."
    ),
    "drop_deleted": (
        "Filter rijen met de soft-delete-vlag (etl_is_deleted ≠ 0) uit de UvA "
        "SQL-levering weg."
    ),
    "cpu_count": (
        "Aantal CPU-cores voor de parallelle voorspelling. Leeg = automatisch "
        "(os.cpu_count()). Verlaag dit om het CPU-gebruik op een gedeelde machine "
        "te beperken."
    ),
    "weight_individual": "Gewicht van het individuele spoor in dit weeksegment.",
    "weight_cumulative": "Gewicht van het cumulatieve spoor in dit weeksegment.",
    "excl_year": "Exact collegejaar om uit te sluiten (bijv. 2020).",
    "excl_herkomst": "Optioneel: beperk de regel tot NL, EER of Niet-EER.",
    "excl_examentype": "Optioneel: beperk de regel tot Bachelor, Master of Pre-master.",
    "excl_opleiding": "Optioneel: beperk de regel tot één programmasleutel.",
}

_GROUP_LABELS = {
    "master_week_17_23": "Masters — begin jaar (wk 17–23)",
    "week_30_34": "Iedereen — midden jaar (wk 30–34)",
    "week_35_37": "Iedereen — vlak voor deadline (wk 35–37)",
    "default": "Alle overige situaties",
}

_EXCL_PLACEHOLDERS = {
    "year": "bijv. 2020",
    "herkomst": "NL / EER / Niet-EER",
    "examentype": "Bachelor / Master",
    "opleiding": "programmasleutel",
}


def create() -> None:
    """Registreer de route ``/config``."""
    nav.register_route("/config")

    @ui.page("/config")
    def config_page() -> None:
        with page_shell(active="/config", title="Configuratie"):
            section_title("Configuratie", "Pas de model- en pipelineparameters aan.")
            if not STATE.is_initialised:
                empty_state(
                    icon="folder_off",
                    title="Nog geen project",
                    message="Kies eerst een project voordat je de configuratie kunt bewerken.",
                    action_label="Project opzetten",
                    on_action=lambda: ui.navigate.to("/wizard"),
                )
                return
            _ConfigView(STATE.config_path)


class _ConfigView:
    """Drie-tabbladen configuratie-editor: Basis · Geavanceerd · JSON."""

    _GUARD_JS = """
        if (!window.__spUnloadGuard) {
            window.__spDirty = false;
            window.__spUnloadGuard = (e) => {
                if (window.__spDirty) { e.preventDefault(); e.returnValue = ''; }
            };
            window.addEventListener('beforeunload', window.__spUnloadGuard);
        }
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._dirty = False
        try:
            self._config = config_io.load_config(path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            error_banner(
                "De configuratie kon niet worden geladen.",
                f"Controleer het bestand {path}. Details: {exc}",
            )
            self._config = {}
            return
        self._nf_rows: list[dict] = [
            {"key": k, "value": v}
            for k, v in self._config.setdefault("numerus_fixus", {}).items()
        ]
        self._excl_rows: list[dict] = [
            dict(item) for item in self._config.setdefault("excluded_data_points", [])
        ]
        self._build()

    # ─── Hoofd-layout ────────────────────────────────────────────────────────

    def _build(self) -> None:
        with ui.row().classes("w-full items-center justify-between mb-3"):
            self._status = ui.label("").classes("text-sm font-medium")
            self._save_btn = ui.button(
                "Opslaan", icon="save", on_click=self._save
            ).props("unelevated")

        with ui.tabs().props("indicator-color=accent align=left").classes("w-full") as tabs:
            tab_basis = ui.tab("Basis", icon="rocket_launch")
            tab_adv = ui.tab("Geavanceerd", icon="tune")
            tab_json = ui.tab("JSON", icon="data_object")

        with ui.tab_panels(tabs, value=tab_basis).classes("w-full"):
            with ui.tab_panel(tab_basis):
                self._build_basis()
            with ui.tab_panel(tab_adv):
                self._build_advanced()
            with ui.tab_panel(tab_json):
                self._build_json()
                tab_json.on("click", self._refresh_json_text)

    # ─── Basis-tabblad ───────────────────────────────────────────────────────

    def _build_basis(self) -> None:
        # Intro-banner
        with ui.row().classes("items-center gap-3 mb-5 px-4 py-3 rounded-xl").style(
            f"background: {theme.ACCENT}0e; border: 1px solid {theme.ACCENT}30"
        ):
            ui.icon("rocket_launch").style(f"color: {theme.ACCENT}; font-size: 20px;")
            with ui.column().classes("gap-0"):
                ui.label("Begin hier").classes("font-medium text-sm")
                ui.label(
                    "Pas deze 3 instellingen aan — de rest werkt prima met de aanbevolen waarden."
                ).classes("text-sm opacity-60")

        self._institution_card()
        self._week_card()
        self._year_card()

        ui.button("Opslaan", icon="save", on_click=self._save).props("unelevated").classes("mt-3")

    def _institution_card(self) -> None:
        current = self._config.setdefault("institution_filter", [])
        available = _load_brincodes(STATE.project_dir or "")
        # Zorg dat codes die al in config staan maar niet in de telbestanden zitten
        # toch als optie beschikbaar zijn zodat de chip blijft staan.
        options = sorted(set(available) | set(current))
        has_data = bool(available)

        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("items-start gap-4 no-wrap"):
                with ui.element("div").classes(
                    "w-12 h-12 rounded-xl flex items-center justify-center flex-none mt-1"
                ).style(f"background: {theme.ACCENT}18"):
                    ui.icon("account_balance").style(f"color: {theme.ACCENT}; font-size: 22px;")
                with ui.column().classes("gap-1 grow"):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.label("Jouw instelling").classes("text-base font-medium")
                        ui.badge("Essentieel").props("color=orange-8").classes("text-xs px-2")
                    ui.label(
                        "Selecteer één of meer Brincodes uit jouw telbestanden. "
                        "Leeg = alle instellingen in de data."
                        if has_data else
                        "Typ een Brincode (bijv. 28DN) en druk op Enter. "
                        "Upload eerst telbestanden om de lijst automatisch te vullen."
                    ).classes("text-sm opacity-60 mt-1")

                    self._inst_select = (
                        ui.select(
                            options=options,
                            value=list(current),
                            multiple=True,
                            label=(
                                f"{len(available)} Brincodes gevonden in telbestanden"
                                if has_data else
                                "Instelling(en) — bijv. 28DN, 21PB"
                            ),
                        )
                        .props("use-chips new-value-mode=add-unique outlined use-input input-debounce=0")
                        .classes("w-full mt-2")
                    )
                    self._inst_select.on_value_change(self._on_institution_change)

                    if not current:
                        with ui.row().classes("items-center gap-1 mt-2"):
                            ui.icon("warning_amber").style(
                                f"color: {theme.WARNING}; font-size: 16px;"
                            )
                            ui.label(
                                "Nog niet ingesteld — de prognose gebruikt data van alle instellingen."
                            ).classes("text-xs").style(f"color: {theme.WARNING}")

    def _week_card(self) -> None:
        mc = self._config.setdefault("model_config", {})

        @ui.refreshable
        def _week_options() -> None:
            current = mc.get("final_academic_week", 38)
            with ui.grid(columns=2).classes("gap-3 mt-3 max-w-sm"):
                for week, subtitle, recommended in [
                    (36, "Typisch UvA", False),
                    (38, "Meeste instellingen", True),
                ]:
                    selected = current == week
                    border = theme.ACCENT if selected else "#e0e0e0"
                    bg = f"{theme.ACCENT}0e" if selected else "white"
                    col = theme.ACCENT if selected else "#333"

                    def _pick(w=week) -> None:
                        mc["final_academic_week"] = w
                        _week_options.refresh()
                        self._mark_dirty()

                    with ui.card().classes("cursor-pointer p-3 text-center").style(
                        f"border: 2px solid {border}; background: {bg};"
                    ).on("click", _pick):
                        with ui.column().classes("items-center gap-0.5"):
                            ui.label(f"Week {week}").classes("font-semibold text-base").style(
                                f"color: {col}"
                            )
                            ui.label(subtitle).classes("text-xs opacity-60")
                            if recommended:
                                ui.badge("★ Aanbevolen").props(
                                    "color=accent outline"
                                ).classes("text-xs mt-1")

        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("items-start gap-4 no-wrap"):
                with ui.element("div").classes(
                    "w-12 h-12 rounded-xl flex items-center justify-center flex-none mt-1"
                ).style(f"background: {theme.SECONDARY}18"):
                    ui.icon("event_note").style(
                        f"color: {theme.SECONDARY}; font-size: 22px;"
                    )
                with ui.column().classes("gap-1 grow"):
                    ui.label("Einde academisch jaar").classes("text-base font-medium")
                    ui.label(
                        "In welke Studielink-week eindigt het academisch jaar? "
                        "Dit bepaalt de seizoensvolgorde en de voorspelhorizon."
                    ).classes("text-sm opacity-60 mt-1")

                    _week_options()

                    ui.label(
                        "Andere waarde? Pas dit aan in het Geavanceerd-tabblad."
                    ).classes("text-xs opacity-40 mt-2")

    def _year_card(self) -> None:
        mc = self._config.setdefault("model_config", {})
        _REC_YEAR = 2022
        _THIS_YEAR = datetime.date.today().year
        _SLIDER_MAX = _THIS_YEAR - 1  # minstens 1 prognose-jaar nodig
        current_val = mc.get("min_training_year", _REC_YEAR)

        def years_back(yr: int) -> int:
            return max(0, _THIS_YEAR - yr)

        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("items-start gap-4 no-wrap"):
                with ui.element("div").classes(
                    "w-12 h-12 rounded-xl flex items-center justify-center flex-none mt-1"
                ).style(f"background: {theme.NPULS_GREEN}18"):
                    ui.icon("history").style(
                        f"color: {theme.NPULS_GREEN}; font-size: 22px;"
                    )
                with ui.column().classes("gap-1 grow"):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.label("Historische data").classes("text-base font-medium")
                        ui.badge(f"★ Aanbevolen: {_REC_YEAR}").props(
                            "color=accent outline"
                        ).classes("text-xs")
                    ui.label(
                        "Hoeveel jaar terugkijken voor de training? Meer jaren geeft een "
                        "stabielere trend, maar heel vroege data weerspiegelt de "
                        "huidige situatie minder goed."
                    ).classes("text-sm opacity-60 mt-1")

                    with ui.row().classes("items-center gap-4 mt-3 w-full"):
                        self._year_lbl = ui.label(
                            f"Vanaf {current_val} · {years_back(current_val)} jaar data"
                        ).classes("text-sm font-medium shrink-0 w-52")
                        yr_slider = ui.slider(
                            min=2010, max=_SLIDER_MAX, step=1, value=current_val
                        ).props("label-always").classes("grow")

                    with ui.row().classes("justify-between w-full -mt-1"):
                        ui.label(f"2010 ({years_back(2010)} jaar)").classes(
                            "text-xs opacity-40"
                        )
                        with ui.row().classes("items-center gap-1"):
                            ui.icon("star").style(
                                f"color: {theme.ACCENT}; font-size: 11px;"
                            )
                            ui.label(f"{_REC_YEAR} (aanbevolen)").classes(
                                "text-xs font-medium"
                            ).style(f"color: {theme.ACCENT}; opacity: 0.75")
                        ui.label(
                            f"{_SLIDER_MAX} ({years_back(_SLIDER_MAX)} jaar)"
                        ).classes("text-xs opacity-40")

                    def on_year(e, mc_=mc):
                        yr = int(e.value)
                        mc_["min_training_year"] = yr
                        self._year_lbl.set_text(
                            f"Vanaf {yr} · {years_back(yr)} jaar data"
                        )
                        self._mark_dirty()

                    yr_slider.on_value_change(on_year)

    def _excl_years_card(self) -> None:
        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("items-start gap-4 no-wrap"):
                with ui.element("div").classes(
                    "w-12 h-12 rounded-xl flex items-center justify-center flex-none mt-1"
                ).style(f"background: {theme.WARNING}18"):
                    ui.icon("block").style(f"color: {theme.WARNING}; font-size: 22px;")
                with ui.column().classes("gap-1 grow"):
                    ui.label("Uitzonderlijke jaren").classes("text-base font-medium")
                    ui.label(
                        "Sluit jaren met buitengewone instroom uit de trainingsdata. "
                        "Bijv. 2020–2021 (COVID). Het voorspeljaar zelf is altijd beschermd."
                    ).classes("text-sm opacity-60 mt-1")

                    self._excl_years_chips = ui.row().classes("gap-2 flex-wrap mt-3 min-h-8")
                    self._render_excl_year_chips()

                    with ui.row().classes("items-center gap-2 mt-3"):
                        self._new_year_input = ui.number(
                            label="Jaar toevoegen",
                            value=2020,
                            min=2010,
                            max=2030,
                            step=1,
                        ).props("dense outlined").classes("w-40")
                        ui.button(
                            "Toevoegen",
                            icon="add",
                            on_click=self._add_excl_year,
                        ).props("outline dense")

                    adv_count = sum(
                        1 for r in self._excl_rows if set(r.keys()) - {"year"}
                    )
                    if adv_count:
                        with ui.row().classes("items-center gap-1 mt-2"):
                            ui.icon("info").style(
                                f"color: {theme.INFO}; font-size: 14px;"
                            )
                            ui.label(
                                f"{adv_count} geavanceerde uitsluitingsregel(s) — "
                                "beheer ze in het Geavanceerd-tabblad."
                            ).classes("text-xs").style(f"color: {theme.INFO}")

    def _render_excl_year_chips(self) -> None:
        self._excl_years_chips.clear()
        years_seen: dict[str, list[int]] = {}
        for i, row in enumerate(self._excl_rows):
            yr = row.get("year")
            if yr is not None:
                years_seen.setdefault(str(yr), []).append(i)

        if not years_seen:
            with self._excl_years_chips:
                ui.label("Geen jaren uitgesloten.").classes("text-sm opacity-40 italic")
            return

        with self._excl_years_chips:
            for yr_str in sorted(years_seen.keys()):
                with ui.row().classes(
                    "items-center gap-1 px-3 py-1 rounded-full no-wrap"
                ).style(
                    f"background: {theme.WARNING}18; "
                    f"border: 1px solid {theme.WARNING}40;"
                ):
                    ui.label(yr_str).classes("text-sm font-medium font-mono").style(
                        f"color: {theme.WARNING}"
                    )
                    ui.button(
                        icon="close",
                        on_click=lambda _e, y=yr_str: self._remove_excl_year(y),
                    ).props("flat round dense").style(
                        f"color: {theme.WARNING}; width: 20px; height: 20px;"
                    )

    def _add_excl_year(self) -> None:
        yr = int(self._new_year_input.value or 2020)
        existing = {str(r.get("year", "")) for r in self._excl_rows}
        if str(yr) not in existing:
            self._excl_rows.append({"year": yr})
            self._config["excluded_data_points"] = self._excl_rows
            self._render_excl_year_chips()
            self._mark_dirty()
        else:
            ui.notify(f"Jaar {yr} is al uitgesloten.", type="warning")

    def _remove_excl_year(self, yr_str: str) -> None:
        self._excl_rows = [
            r for r in self._excl_rows if str(r.get("year", "")) != yr_str
        ]
        self._config["excluded_data_points"] = self._excl_rows
        self._render_excl_year_chips()
        self._mark_dirty()

    def _add_covid_years(self) -> None:
        existing = {str(r.get("year", "")) for r in self._excl_rows}
        added = []
        for yr in (2020, 2021):
            if str(yr) not in existing:
                self._excl_rows.append({"year": yr})
                added.append(yr)
        if added:
            self._config["excluded_data_points"] = self._excl_rows
            self._render_excl_year_chips()
            self._mark_dirty()
            ui.notify(
                f"COVID-jaren toegevoegd: {', '.join(str(y) for y in added)}.",
                type="positive",
            )
        else:
            ui.notify("2020 en 2021 zijn al uitgesloten.", type="info")

    # ─── Geavanceerd-tabblad ─────────────────────────────────────────────────

    def _build_advanced(self) -> None:
        with ui.row().classes("items-center gap-3 mb-5 px-4 py-3 rounded-xl").style(
            "background: #f5f5f5; border: 1px solid #e0e0e0;"
        ):
            ui.icon("tune").classes("text-xl opacity-50")
            with ui.column().classes("gap-0"):
                ui.label("Geavanceerde instellingen").classes("font-medium text-sm")
                ui.label(
                    "Modellen, ensemble-gewichten en uitsluitingsregels. "
                    "Wijzig alleen als je weet wat je doet."
                ).classes("text-sm opacity-50")

        self._model_section()
        self._ensemble_section()
        self._nf_section()
        self._excl_section()
        self._runtime_section()
        ui.button("Opslaan", icon="save", on_click=self._save).props("unelevated").classes("mt-3")

    def _model_section(self) -> None:
        mc = self._config.setdefault("model_config", {})
        with ui.expansion("Modelkeuze", icon="model_training", value=True).classes("w-full"):
            ui.label(
                "Welke algoritmen worden gebruikt voor tijdreeks, regressie en classificatie? "
                "Vergelijk alternatieven via de Benchmark-pagina."
            ).classes("text-sm opacity-60 mb-3")
            with ui.grid(columns=3).classes("w-full gap-4"):
                self._adv_select(
                    "Tijdreeksmodel",
                    config_io.TIMESERIES_CHOICES,
                    mc,
                    "cumulative_timeseries",
                    "sarima",
                    help=HELP["cumulative_timeseries"],
                    recommend="sarima",
                )
                self._adv_select(
                    "Regressor (cumulatief)",
                    config_io.REGRESSOR_CHOICES,
                    mc,
                    "cumulative_regressor",
                    "xgboost",
                    help=HELP["cumulative_regressor"],
                    recommend="xgboost",
                )
                self._adv_select(
                    "Classifier (individueel)",
                    config_io.CLASSIFIER_CHOICES,
                    mc,
                    "individual_classifier",
                    "xgboost",
                    help=HELP["individual_classifier"],
                    recommend="xgboost",
                )

    def _adv_select(
        self, label, choices, target, key, default, *, help=None, recommend=None
    ) -> None:
        value = target.get(key, default)
        options = list(choices)
        if value not in options:
            options.append(value)
        display = {o: f"{o}  ✓ aanbevolen" if o == recommend else o for o in options}
        sel = ui.select(display, value=value, label=label).classes("w-full")
        if help:
            sel.tooltip(help)

        def _on_change(e) -> None:
            target[key] = e.value
            self._mark_dirty()

        sel.on_value_change(_on_change)

    def _ensemble_section(self) -> None:
        weights = self._config.setdefault("ensemble_weights", {})
        with ui.expansion("Ensemble-gewichten", icon="balance", value=False).classes(
            "w-full"
        ):
            with ui.row().classes("items-start justify-between flex-wrap gap-2 mb-3"):
                ui.label(
                    "Per weeksegment: hoe zwaar telt het individuele spoor mee ten "
                    "opzichte van het cumulatieve spoor? De schuifregelaar bepaalt de "
                    "verdeling — beide sporen moeten optellen tot 100%."
                ).classes("text-sm opacity-60 grow")
                ui.badge("★ Aanbevolen: 50% / 50%").props(
                    "color=accent outline"
                ).classes("text-xs shrink-0 self-start")
            self._ensemble_error = ui.column().classes("w-full")

            for group in config_io.ENSEMBLE_GROUPS:
                grp = weights.setdefault(group, {"individual": 0.5, "cumulative": 0.5})
                ind_val = float(grp.get("individual", 0.5))

                with ui.card().classes("w-full mb-2").style(
                    "border: 1px solid #e8e8e8; box-shadow: none;"
                ):
                    with ui.row().classes("items-center gap-4 no-wrap w-full"):
                        with ui.column().classes("gap-0 shrink-0").style("min-width: 200px"):
                            ui.label(_GROUP_LABELS.get(group, group)).classes(
                                "text-sm font-medium"
                            )
                            ui.label(group).classes("text-xs font-mono opacity-35 mt-0.5")

                        with ui.column().classes("grow gap-2"):
                            with ui.row().classes("items-center gap-3 no-wrap w-full"):
                                with ui.row().classes("items-center gap-1 shrink-0").style(
                                    "min-width: 90px; justify-content: flex-end"
                                ):
                                    ui.icon("person").style(
                                        f"color: {theme.ACCENT}; font-size: 14px;"
                                    )
                                    ind_pct_lbl = ui.label(
                                        f"{ind_val:.0%}"
                                    ).classes("text-sm font-semibold").style(
                                        f"color: {theme.ACCENT}"
                                    )

                                sl = ui.slider(
                                    min=0, max=1, step=0.05, value=ind_val
                                ).classes("grow")

                                with ui.row().classes("items-center gap-1 shrink-0").style(
                                    "min-width: 90px"
                                ):
                                    cum_pct_lbl = ui.label(
                                        f"{1.0 - ind_val:.0%}"
                                    ).classes("text-sm opacity-50")
                                    ui.icon("stacked_line_chart").style(
                                        f"color: {theme.SECONDARY}; font-size: 14px;"
                                    )

                            with ui.row().classes("justify-between w-full -mt-1"):
                                ui.label("← meer individueel").classes("text-xs opacity-35")
                                ui.label("meer cumulatief →").classes("text-xs opacity-35")

                        def _make_handler(grp_d, i_lbl, c_lbl):
                            def _on_weight(e) -> None:
                                ind = round(float(e.value), 4)
                                cum = round(1.0 - ind, 4)
                                grp_d["individual"] = ind
                                grp_d["cumulative"] = cum
                                i_lbl.set_text(f"{ind:.0%}")
                                c_lbl.set_text(f"{cum:.0%}")
                                self._validate_ensemble()
                                self._mark_dirty()

                            return _on_weight

                        sl.on_value_change(_make_handler(grp, ind_pct_lbl, cum_pct_lbl))

            self._validate_ensemble()

    def _nf_section(self) -> None:
        with ui.expansion("Numerus fixus", icon="lock", value=False).classes("w-full"):
            ui.label(
                "Opleidingen met een capaciteitslimiet. Gebruik exact dezelfde "
                "programmasleutel als in je data — de voorspelling wordt op dit maximum afgetopt."
            ).classes("text-sm opacity-60 mb-2")
            self._nf_container = ui.column().classes("w-full gap-2")
            self._render_nf_rows()
            ui.button("Rij toevoegen", icon="add", on_click=self._add_nf_row).props("flat")

    def _excl_section(self) -> None:
        with ui.expansion("Uitsluitingsregels", icon="block", value=False).classes(
            "w-full"
        ):
            ui.label(
                "Sluit bekende probleemjaren (bijv. COVID) of specifieke deelpopulaties "
                "uit de trainingsdata. Het voorspeljaar zelf is altijd beschermd."
            ).classes("text-sm opacity-60 mb-3")

            # Aanbevolen: COVID-jaren snel toevoegen
            with ui.row().classes(
                "items-center gap-3 px-4 py-3 rounded-xl mb-4"
            ).style(f"background: {theme.ACCENT}09; border: 1px solid {theme.ACCENT}28"):
                ui.icon("star").style(f"color: {theme.ACCENT}; font-size: 18px;")
                with ui.column().classes("gap-0 grow"):
                    ui.label("Aanbevolen: COVID-jaren uitsluiten").classes(
                        "text-sm font-medium"
                    ).style(f"color: {theme.ACCENT}")
                    ui.label(
                        "2020 en 2021 veroorzaakten atypische aanmeldpatronen. "
                        "Uitsluiten verbetert de modelnauwkeurigheid voor de meeste instellingen."
                    ).classes("text-xs opacity-70")
                ui.button(
                    "Voeg 2020 & 2021 toe",
                    icon="add_circle",
                    on_click=self._add_covid_years,
                ).props("outline dense color=accent").classes("shrink-0")

            # Eenvoudige jaar-chips (quick add/remove)
            self._excl_years_chips = ui.row().classes("gap-2 flex-wrap mb-3 min-h-8")
            self._render_excl_year_chips()
            with ui.row().classes("items-center gap-2 mb-4"):
                self._new_year_input = ui.number(
                    label="Jaar toevoegen",
                    value=2020,
                    min=2010,
                    max=2030,
                    step=1,
                ).props("dense outlined").classes("w-40")
                ui.button(
                    "Toevoegen",
                    icon="add",
                    on_click=self._add_excl_year,
                ).props("outline dense")

            ui.separator().classes("mb-3")
            ui.label("Gedetailleerde regels (per jaar, herkomst, examentype of opleiding)").classes(
                "text-xs font-medium opacity-50 mb-1"
            )
            with ui.row().classes("w-full gap-2 no-wrap px-1 mb-1"):
                for col, w in [
                    ("Jaar", "grow"),
                    ("Herkomst", "grow"),
                    ("Examentype", "grow"),
                    ("Opleiding", "grow"),
                    ("", "w-8"),
                ]:
                    ui.label(col).classes(f"text-xs font-medium opacity-40 {w}")
            self._excl_container = ui.column().classes("w-full gap-1")
            self._render_excl_rows()
            ui.button("Rij toevoegen", icon="add", on_click=self._add_excl_row).props("flat")

    def _runtime_section(self) -> None:
        ci = self._config.setdefault("cumulative_input", {})
        runtime = self._config.setdefault("runtime", {})
        with ui.expansion("Runtime", icon="settings", value=False).classes("w-full"):
            ui.label("Verwerkingsopties voor de pipeline.").classes(
                "text-sm opacity-60 mb-3"
            )
            with ui.grid(columns=2).classes("w-full gap-x-8 gap-y-2"):
                self._adv_switch(
                    "Aggregeren (cumulatief)",
                    ci,
                    "aggregate",
                    True,
                    help=HELP["aggregate"],
                )
                self._adv_switch(
                    "Verwijderde rijen weglaten",
                    ci,
                    "drop_deleted",
                    True,
                    help=HELP["drop_deleted"],
                )
            self._adv_number(
                "CPU-cores (leeg = automatisch)",
                runtime,
                "cpu_count",
                None,
                allow_none=True,
                help=HELP["cpu_count"],
            )

    def _adv_switch(self, label, target, key, default, *, help=None) -> None:
        sw = ui.switch(label, value=target.get(key, default))
        if help:
            sw.tooltip(help)

        def _on_change(e) -> None:
            target[key] = e.value
            self._mark_dirty()

        sw.on_value_change(_on_change)

    def _adv_number(
        self, label, target, key, default, *, allow_none=False, help=None
    ) -> None:
        value = target.get(key, default)
        inp = ui.number(label=label, value=value).classes("w-full max-w-xs")
        if help:
            inp.tooltip(help)

        def _on_change(e) -> None:
            val = e.value
            if val is None and not allow_none:
                return
            target[key] = int(val) if val is not None else None
            self._mark_dirty()

        inp.on_value_change(_on_change)

    # ─── Dynamische tabellen ──────────────────────────────────────────────────

    def _render_nf_rows(self) -> None:
        self._nf_container.clear()
        with self._nf_container:
            if not self._nf_rows:
                ui.label("Nog geen opleidingen met numerus fixus ingesteld.").classes(
                    "text-sm opacity-40 italic py-1"
                )
                return
            for row in self._nf_rows:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    key_in = (
                        ui.input(value=row["key"], placeholder="Programmasleutel")
                        .props("dense outlined")
                        .classes("grow")
                    )
                    key_in.tooltip(HELP["numerus_fixus"])
                    key_in.on_value_change(
                        lambda e, r=row: (r.update(key=e.value), self._mark_dirty())
                    )
                    val_in = (
                        ui.number(value=row["value"], placeholder="Max. plaatsen")
                        .props("dense outlined")
                        .classes("w-36")
                    )
                    val_in.on_value_change(
                        lambda e, r=row: (r.update(value=e.value), self._mark_dirty())
                    )
                    ui.button(
                        icon="delete",
                        on_click=lambda r=row: self._remove_nf_row(r),
                    ).props("flat round color=negative")

    def _add_nf_row(self) -> None:
        self._nf_rows.append({"key": "", "value": 0})
        self._render_nf_rows()
        self._mark_dirty()

    def _remove_nf_row(self, row: dict) -> None:
        self._nf_rows.remove(row)
        self._render_nf_rows()
        self._mark_dirty()

    _EXCL_FIELDS = ["year", "herkomst", "examentype", "opleiding"]

    def _render_excl_rows(self) -> None:
        self._excl_container.clear()
        with self._excl_container:
            if not self._excl_rows:
                ui.label("Nog geen uitsluitingsregels.").classes(
                    "text-sm opacity-40 italic py-1"
                )
                return
            for row in self._excl_rows:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    for field in self._EXCL_FIELDS:
                        inp = (
                            ui.input(
                                value=str(row.get(field, "")),
                                placeholder=_EXCL_PLACEHOLDERS.get(field, field),
                            )
                            .props("dense outlined")
                            .classes("grow")
                        )
                        inp.tooltip(HELP.get(f"excl_{field}", field))
                        inp.on_value_change(
                            lambda e, r=row, f=field: (
                                r.update({f: e.value}),
                                self._mark_dirty(),
                            )
                        )
                    ui.button(
                        icon="delete",
                        on_click=lambda r=row: self._remove_excl_row(r),
                    ).props("flat round color=negative")

    def _add_excl_row(self) -> None:
        self._excl_rows.append({})
        self._render_excl_rows()
        self._mark_dirty()

    def _remove_excl_row(self, row: dict) -> None:
        self._excl_rows.remove(row)
        self._render_excl_rows()
        self._mark_dirty()

    # ─── JSON-tabblad ────────────────────────────────────────────────────────

    def _build_json(self) -> None:
        with ui.row().classes("items-center gap-3 mb-4 px-4 py-3 rounded-xl").style(
            "background: #1a1a1a08; border: 1px solid #1a1a1a18;"
        ):
            ui.icon("data_object").classes("text-xl opacity-40")
            with ui.column().classes("gap-0"):
                ui.label("Directe JSON-bewerking").classes("font-medium text-sm")
                ui.label(
                    "Voor gevorderde gebruikers. Elke sleutel is bewerkbaar. "
                    "Gebruik 'Toepassen' om wijzigingen naar de andere tabbladen te laden."
                ).classes("text-sm opacity-50")

        with ui.row().classes("items-center gap-2 mb-2"):
            ui.icon("folder_open").classes("text-sm opacity-30")
            ui.label(self._path).classes("text-xs font-mono opacity-50 break-all")

        with ui.card().classes("w-full").style("border: 1px solid #e0e0e0; box-shadow: none;"):
            self._json_area = (
                ui.textarea(
                    value=json.dumps(self._config, ensure_ascii=False, indent=4)
                )
                .props("outlined borderless")
                .classes("w-full font-mono")
                .style("min-height: 22rem; font-size: 12.5px; line-height: 1.5;")
            )
            self._json_error = ui.label("").classes("text-sm px-3 pb-1").style(
                f"color: {theme.NEGATIVE}"
            )
            with ui.row().classes("gap-2 px-3 pb-3 flex-wrap items-center").style(
                "border-top: 1px solid #f0f0f0; padding-top: 12px;"
            ):
                ui.button(
                    "Valideer",
                    icon="check_circle_outline",
                    on_click=self._validate_json_only,
                ).props("outline dense")
                ui.button(
                    "Opmaak",
                    icon="format_indent_increase",
                    on_click=self._format_json,
                ).props("flat dense")
                ui.separator().props("vertical").classes("mx-1").style("height: 28px;")
                ui.button(
                    "Toepassen",
                    icon="check",
                    on_click=self._apply_json,
                ).props("unelevated dense")
                ui.button("Opslaan", icon="save", on_click=self._save).props(
                    "unelevated dense color=positive"
                )

    def _refresh_json_text(self) -> None:
        self._json_area.set_value(
            json.dumps(self._config, ensure_ascii=False, indent=4)
        )

    def _validate_json_only(self) -> None:
        try:
            config_io.parse_json(self._json_area.value)
            self._json_error.set_text("")
            ui.notify("JSON is geldig.", type="positive", position="top-right")
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_error.set_text(f"Fout: {exc}")
            ui.notify("Ongeldige JSON — zie de foutmelding hieronder.", type="negative")

    def _format_json(self) -> None:
        try:
            parsed = config_io.parse_json(self._json_area.value)
            self._json_area.set_value(
                json.dumps(parsed, ensure_ascii=False, indent=4)
            )
            self._json_error.set_text("")
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_error.set_text(f"Kan niet opmaken: {exc}")

    def _apply_json(self) -> None:
        try:
            parsed = config_io.parse_json(self._json_area.value)
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_error.set_text(f"Ongeldige JSON: {exc}")
            ui.notify(f"Ongeldige JSON: {exc}", type="negative")
            return
        self._json_error.set_text("")
        self._config = parsed
        self._nf_rows = [
            {"key": k, "value": v}
            for k, v in self._config.get("numerus_fixus", {}).items()
        ]
        self._excl_rows = [
            dict(item) for item in self._config.get("excluded_data_points", [])
        ]
        self._mark_dirty()
        ui.notify(
            "JSON toegepast. Sla op om de wijzigingen te bewaren.",
            type="positive",
        )

    # ─── Validatie & opslaan ──────────────────────────────────────────────────

    def _validate_ensemble(self) -> bool:
        self._ensemble_error.clear()
        errors = config_io.validate_ensemble_weights(
            self._config.get("ensemble_weights", {})
        )
        if errors:
            with self._ensemble_error:
                for err in errors:
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("error").style(
                            f"color: {theme.NEGATIVE}; font-size: 16px;"
                        )
                        ui.label(err).classes("text-sm").style(
                            f"color: {theme.NEGATIVE}"
                        )
        return not errors

    def _sync_dynamic_into_config(self) -> None:
        self._config["numerus_fixus"] = {
            r["key"]: int(r["value"]) if r["value"] is not None else 0
            for r in self._nf_rows
            if r["key"]
        }
        cleaned = []
        for row in self._excl_rows:
            item = {k: v for k, v in row.items() if v not in ("", None)}
            if item:
                cleaned.append(item)
        self._config["excluded_data_points"] = cleaned

    def _save(self) -> None:
        self._sync_dynamic_into_config()
        errors = config_io.validate_config(self._config)
        if errors:
            self._validate_ensemble()
            ui.notify(
                "Kan niet opslaan: los eerst de validatiefouten op.",
                type="negative",
            )
            return
        try:
            config_io.save_config(self._path, self._config)
        except OSError as exc:
            ui.notify(f"Opslaan mislukt: {exc}", type="negative")
            return
        self._clear_dirty()
        ui.notify("Configuratie opgeslagen.", type="positive", position="top")

    def _on_institution_change(self, e) -> None:
        self._config["institution_filter"] = list(e.value or [])
        self._mark_dirty()

    # ─── Dirty-tracking ───────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            ui.run_javascript(self._GUARD_JS + "window.__spDirty = true;")
        self._status.set_text("● Niet-opgeslagen wijzigingen").style(
            f"color: {theme.WARNING}"
        )

    def _clear_dirty(self) -> None:
        self._dirty = False
        ui.run_javascript("window.__spDirty = false;")
        self._status.set_text("✓ Opgeslagen").style(f"color: {theme.POSITIVE}")
