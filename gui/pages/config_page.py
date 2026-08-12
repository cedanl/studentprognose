"""Configuratie-editor: drie tabbladen — Basis, Geavanceerd, JSON.

Basis:       de 4 essentiële instellingen voor nieuwe gebruikers.
Geavanceerd: modelkeuze, ensemble-gewichten, numerus fixus, exclusieregels.
JSON:        directe bewerking voor experts.
"""

from __future__ import annotations

import csv
import glob
import json
import os

from nicegui import ui

from gui import config_io, filtering_io, nav, theme
from gui.components.layout import page_shell
from gui.components.states import empty_state, error_banner, section_title
from gui.data_upload import scan_data_year_bounds, selectable_exclusion_years
from gui.state import STATE

#: Vertraging (s) waarmee een wijziging naar de auto-save wordt gedebounced, zodat
#: snel typen niet bij elke toetsaanslag valideert en naar schijf schrijft.
_AUTOSAVE_DELAY_S = 0.6


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
        except (OSError, csv.Error, UnicodeDecodeError):
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
        "Beperk de teldata tot je eigen instelling(en) via Brincode. "
        "Leeg = alle instellingen. De meeste gebruikers zetten hier hun "
        "eigen Brincode."
    ),
    "excluded_data_points": (
        "Sluit bekende probleemjaren (bijv. een uitzonderlijk coronajaar) uit de "
        "trainingsdata. Het voorspeljaar zelf wordt altijd beschermd en nooit "
        "uitgesloten."
    ),
    "ensemble_override_cumulative": (
        "Opleidingen waarvoor het ensemble altijd het cumulatieve SARIMA-model "
        "gebruikt, ongeacht weeknummer of examentype. Gebruik dit voor "
        "numerus-fixus-opleidingen of sterk afwijkende aanmeldpatronen waarbij "
        "het cumulatieve spoor aantoonbaar beter presteert."
    ),
    "exclude_from_combined": (
        "Opleidingen die worden overgeslagen in de combined-modus ('beide "
        "sporen'). Gebruik dit als de combined-voorspelling voor deze "
        "opleiding aantoonbaar slechter presteert dan het cumulatieve spoor "
        "alleen."
    ),
    "validation_separator": (
        "Scheidingsteken waarmee de validatie het ruwe telbestand inleest. "
        "';' voor het legacy Studielink-formaat, ',' voor het UvA SQL-formaat "
        "— moet gelijk zijn aan cumulative_input.separator."
    ),
    "validation_programme_column": (
        "Kolom waarop validatiefouten worden gegroepeerd. Meestal "
        "'Groepeernaam'; zet dit op 'Isatcode' als je databron die kolom niet "
        "levert (bijv. UvA)."
    ),
    "validation_herkomst_allowed": (
        "Toegestane waarden in de Herkomst-kolom van het telbestand. Een "
        "andere waarde levert een validatiefout op."
    ),
    "validation_required_columns": (
        "Kolommen die verplicht in het ruwe telbestand moeten staan. "
        "Ontbreekt er één, dan stopt de validatie met een foutmelding."
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

    def __init__(self, path: str) -> None:
        self._path = path
        #: Generatieteller voor de debounced auto-save (zie :meth:`_mark_dirty`).
        self._dirty_gen = 0
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
        #: Kiesbare uitsluitingsjaren = overlap tussen telbestand en oktober-bestand.
        try:
            project_dir = os.path.dirname(os.path.dirname(path))
            self._selectable_years = selectable_exclusion_years(
                scan_data_year_bounds(project_dir)
            )
        except Exception:
            self._selectable_years = []
        self._filtering_data = filtering_io.load_filtering(STATE.filtering_path)
        self._filtering = self._filtering_data["filtering"]
        self._student_df = self._load_student_count_df()
        #: {isatcode_str: label} voor de opleiding-dropdowns (filter + numerus fixus).
        self._programme_options = self._build_programme_options()
        self._build()

    # ─── Hoofd-layout ────────────────────────────────────────────────────────

    def _build(self) -> None:
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
                tab_json.on("click", self._run_json_editor_init)

        with ui.row().classes("w-full justify-end mt-6"):
            ui.button(
                "Volgende", icon="arrow_forward",
                on_click=self._on_next_click,
            ).props("unelevated color=accent")

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
                    "Pas deze 2 instellingen aan — de rest werkt prima met de aanbevolen waarden."
                ).classes("text-sm opacity-60")

        self._institution_card()
        self._week_card()

    def _institution_card(self) -> None:
        current = self._config.setdefault("institution_filter", [])
        current_code = current[0] if current else None

        try:
            project_dir = os.path.dirname(os.path.dirname(self._path))
            data_codes = _load_brincodes(project_dir)
        except Exception:
            data_codes = []

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
                        "Selecteer je Brincode. De prognose filtert de teldata automatisch op deze code."
                    ).classes("text-sm opacity-60 mt-1")

                    if data_codes:
                        options = data_codes if not (current_code and current_code not in data_codes) else [current_code, *data_codes]

                        self._inst_select = (
                            ui.select(
                                options=options,
                                value=current_code,
                                label="Selecteer Brincode",
                            )
                            .props("outlined use-input input-debounce=0 clearable")
                            .classes("w-full mt-2")
                        )
                        self._inst_select.on_value_change(self._on_institution_change)

                        with ui.row().classes("items-center gap-1.5 mt-2"):
                            ui.icon("folder_open").style(
                                f"color: {theme.INFO}; font-size: 14px;"
                            )
                            ui.label(
                                f"{len(data_codes)} instelling(en) gevonden in de telbestanden."
                            ).classes("text-xs").style(f"color: {theme.INFO}")
                    else:
                        self._inst_select = (
                            ui.input(
                                value=current_code or "",
                                label="Brincode (bijv. 25DW)",
                                placeholder="Voer de Brincode in",
                            )
                            .props("outlined clearable")
                            .classes("w-full mt-2")
                        )
                        self._inst_select.on_value_change(self._on_institution_change)

                        with ui.row().classes("items-center gap-1.5 mt-2"):
                            ui.icon("info").style("color: #aaa; font-size: 14px;")
                            ui.label(
                                "Upload telbestanden om de beschikbare Brincodes te detecteren."
                            ).classes("text-xs").style("color: #aaa;")

    def _week_card(self) -> None:
        mc = self._config.setdefault("model_config", {})
        _REC_WEEK = 38
        _PRESET_WEEKS = [36, 37, 38, 39, 40]
        _SUBTITLES = {38: "Meeste instellingen"}
        _MIN_WEEK, _MAX_WEEK = 1, 52

        def _current() -> int:
            return int(mc.get("final_academic_week", _REC_WEEK))

        def _start_week(final: int) -> int:
            """Eerste week van het academisch jaar (reset-week net na ``final``)."""
            return final + 1 if final < _MAX_WEEK else 1

        def _apply(week: int) -> None:
            """Zet de eindweek, houd de UI in sync en markeer als gewijzigd."""
            week = max(_MIN_WEEK, min(_MAX_WEEK, week))
            mc["final_academic_week"] = week
            self._mark_dirty()
            _week_tiles.refresh()
            _horizon_hint.refresh()

        @ui.refreshable
        def _week_tiles() -> None:
            A = theme.ACCENT
            current = _current()
            # Toon de vaste presets; valt de huidige waarde erbuiten, voeg hem toe
            # zodat een handmatig gekozen week ook als geselecteerde tegel oplicht.
            weeks = sorted(set(_PRESET_WEEKS) | {current})
            with ui.row().classes("gap-2 flex-wrap mt-3"):
                for wk in weeks:
                    is_sel = wk == current
                    is_rec = wk == _REC_WEEK
                    border = A if is_sel else ("#e0e0e0" if not is_rec else f"{A}55")
                    bg = f"{A}12" if is_sel else ("white" if not is_rec else f"{A}06")
                    label_col = A if is_sel else ("#555" if not is_rec else A)
                    subtitle = _SUBTITLES.get(wk, "")

                    tile = ui.element("div").style(
                        f"padding:10px 16px;border-radius:10px;"
                        f"border:2px solid {border};background:{bg};"
                        f"cursor:pointer;text-align:center;min-width:72px;"
                        f"transition:border-color 0.15s,background 0.15s,box-shadow 0.15s;"
                        + (f"box-shadow:0 0 0 3px {A}22;" if is_sel else "")
                    )
                    tile.on("click", lambda w=wk: _apply(w))
                    with tile:
                        ui.label(f"Week {wk}").style(
                            f"font-size:15px;font-weight:{'700' if is_sel else '600'};"
                            f"color:{label_col};line-height:1.2;"
                        )
                        if subtitle:
                            ui.label(subtitle).style(
                                f"font-size:10px;color:{'#aaa' if not is_sel else A};"
                                f"margin-top:2px;"
                            )
                        if is_rec:
                            ui.label("★ aanbevolen").style(
                                f"font-size:9px;color:{A};font-weight:600;"
                                f"margin-top:3px;letter-spacing:0.03em;"
                            )

        @ui.refreshable
        def _horizon_hint() -> None:
            current = _current()
            start = _start_week(current)
            with ui.row().classes("items-center gap-1.5 mt-3"):
                ui.icon("timeline").style(f"color:{theme.INFO};font-size:14px;")
                ui.label(
                    f"Het academisch jaar loopt dan van week {start} tot en met "
                    f"week {current} (het jaar erop)."
                ).classes("text-xs").style(f"color:{theme.INFO}")

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

                    _week_tiles()
                    _horizon_hint()

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

    def _available_year_options(self) -> list[int]:
        """Kiesbare jaren minus de reeds uitgesloten jaren (oplopend)."""
        excluded = {
            str(r.get("year")) for r in self._excl_rows if r.get("year") is not None
        }
        return [y for y in self._selectable_years if str(y) not in excluded]

    def _refresh_year_input(self) -> None:
        """Werk de select-opties bij nadat de uitsluitingslijst is veranderd."""
        inp = getattr(self, "_new_year_input", None)
        if inp is None:
            return
        opts = self._available_year_options()
        inp.set_options(opts, value=(opts[0] if opts else None))

    def _add_excl_year(self) -> None:
        if self._new_year_input is None:
            return
        raw = self._new_year_input.value
        if raw is None:
            ui.notify("Kies eerst een jaar.", type="warning")
            return
        try:
            yr = int(raw)
        except (TypeError, ValueError):
            ui.notify("Ongeldig jaar.", type="warning")
            return
        if yr not in self._selectable_years:
            rng = (
                f" ({self._selectable_years[0]}–{self._selectable_years[-1]})"
                if self._selectable_years
                else ""
            )
            ui.notify(
                f"Jaar {yr} valt buiten de beschikbare data{rng}.",
                type="warning",
            )
            return
        existing = {str(r.get("year", "")) for r in self._excl_rows}
        if str(yr) in existing:
            ui.notify(f"Jaar {yr} is al uitgesloten.", type="warning")
            return
        self._excl_rows.append({"year": yr})
        self._config["excluded_data_points"] = self._excl_rows
        self._render_excl_year_chips()
        self._refresh_year_input()
        self._mark_dirty()

    def _remove_excl_year(self, yr_str: str) -> None:
        self._excl_rows = [
            r for r in self._excl_rows if str(r.get("year", "")) != yr_str
        ]
        self._config["excluded_data_points"] = self._excl_rows
        self._render_excl_year_chips()
        self._refresh_year_input()
        self._mark_dirty()

    def _add_covid_years(self) -> None:
        # Alleen COVID-jaren die de data dekt (de knop verschijnt sowieso
        # alleen dan) en nog niet uitgesloten zijn.
        existing = {str(r.get("year", "")) for r in self._excl_rows}
        added = [
            yr
            for yr in (2020, 2021)
            if yr in self._selectable_years and str(yr) not in existing
        ]
        for yr in added:
            self._excl_rows.append({"year": yr})
        if added:
            self._config["excluded_data_points"] = self._excl_rows
            self._render_excl_year_chips()
            self._refresh_year_input()
            self._mark_dirty()
            ui.notify(
                f"COVID-jaren toegevoegd: {', '.join(str(y) for y in added)}.",
                type="positive",
            )
        else:
            ui.notify("De beschikbare COVID-jaren staan al in de lijst.", type="info")

    # ─── Geavanceerd-tabblad ─────────────────────────────────────────────────

    def _build_advanced(self) -> None:
        with ui.row().classes("items-center gap-3 mb-5 px-4 py-3 rounded-xl").style(
            f"background: linear-gradient(135deg, {theme.ACCENT}18, {theme.SECONDARY}0a); "
            f"border: 1px solid {theme.ACCENT}35;"
        ):
            with ui.element("div").classes(
                "w-10 h-10 rounded-xl flex items-center justify-center flex-none"
            ).style(f"background: {theme.ACCENT}38;"):
                ui.icon("tune").style(f"color: {theme.ACCENT}; font-size: 20px;")
            with ui.column().classes("gap-0.5 grow"):
                ui.label("Geavanceerde instellingen").classes("font-semibold text-sm")
                ui.label(
                    "Modellen, ensemble-gewichten en uitsluitingsregels. "
                    "Wijzig alleen als je weet wat je doet."
                ).classes("text-sm opacity-55")

        self._filtering_section()
        self._model_section()
        self._ensemble_section()
        self._programme_multiselect_section(
            config_key="ensemble_override_cumulative",
            title="Ensemble-uitzondering",
            icon="rule",
            color=theme.SECONDARY,
            help=HELP["ensemble_override_cumulative"],
            select_label="Opleidingen (altijd cumulatief spoor)",
        )
        self._programme_multiselect_section(
            config_key="exclude_from_combined",
            title="Uitgesloten van combined-modus",
            icon="block",
            color=theme.NEGATIVE,
            help=HELP["exclude_from_combined"],
            select_label="Opleidingen (overslaan in combined-modus)",
        )
        self._nf_section()
        self._excl_section()
        self._validation_section()
        self._runtime_section()

    def _programme_multiselect_section(
        self,
        *,
        config_key: str,
        title: str,
        icon: str,
        color: str,
        help: str,
        select_label: str,
    ) -> None:
        """Herbruikbare kaart voor een lijst van programmasleutels (isatcode of naam).

        Gebruikt dezelfde zoekbare opleiding-keuzelijst als *Filteren* en
        *Numerus fixus* (:attr:`_programme_options`), maar staat ook vrij
        getypte waarden toe — sommige secties (bijv. ``ensemble_override_cumulative``)
        keyen historisch op de leesbare opleidingsnaam in plaats van de isatcode.
        """
        values = self._config.setdefault(config_key, [])
        label = f"{title} — {len(values)} opleiding(en)" if values else f"{title} — geen"
        with ui.expansion(label, icon=icon, value=False).classes("w-full mb-3").style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {color}20; border-left: 4px solid {color}; "
            f"box-shadow: 0 1px 6px rgba(0,0,0,0.06);"
        ):
            ui.label(help).classes("text-sm opacity-60 mb-2")

            current = [filtering_io.isatcode_str(v) or str(v) for v in values]
            opts = dict(self._programme_options)
            for raw, key in zip(values, current):
                opts.setdefault(key, str(raw))

            sel = (
                ui.select(
                    options=opts,
                    value=current,
                    multiple=True,
                    with_input=True,
                    label=select_label,
                )
                .props("use-chips new-value-mode=add-unique outlined")
                .classes("w-full")
            )

            def _on_change(e, key=config_key) -> None:
                self._config[key] = list(e.value or [])
                self._mark_dirty()

            sel.on_value_change(_on_change)

    def _load_student_count_df(self):
        path = os.path.join(
            STATE.project_dir, "data", "input", "student_count_first-years.xlsx"
        )
        if not os.path.isfile(path):
            return None
        try:
            import pandas as pd
            return pd.read_excel(path)
        except (OSError, ValueError):
            return None

    def _programme_col(self) -> str:
        """Naam van de isatcode-/programmakolom (config-driven)."""
        return self._config.get("column_roles", {}).get("programme", "Croho groepeernaam")

    @staticmethod
    def _read_table(path: str, *, sep: str | None = None):
        """Lees een CSV/XLSX defensief; ``None`` bij ontbreken of leesfout.

        Voor CSV's zonder expliciete ``sep`` wordt de scheidingstekens
        automatisch bepaald: telbestanden komen zowel puntkomma-gescheiden
        (legacy Studielink) als komma-gescheiden (UvA SQL-export) voor. We
        proberen beide en kiezen de variant die meer dan één kolom oplevert.
        """
        if not os.path.isfile(path):
            return None
        try:
            import pandas as pd
            if path.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(path)
            if sep is not None:
                return pd.read_csv(path, sep=sep)
            best = None
            for candidate in (";", ","):
                try:
                    df = pd.read_csv(path, sep=candidate)
                except (OSError, ValueError):
                    continue
                if df.shape[1] > 1:
                    return df
                best = df if best is None else best
            return best
        except (OSError, ValueError):
            return None

    @staticmethod
    def _detect_csv_sep(path: str) -> str:
        """Bepaal ``;`` of ``,`` als scheidingsteken voor een telbestand-CSV.

        Leest alleen de kop van het bestand met elk kandidaat-scheidingsteken en
        kiest degene die de meeste kolommen oplevert. Valt terug op ``;`` (het
        legacy Studielink-formaat) als niets uitsluitsel geeft.
        """
        if not os.path.isfile(path):
            return ";"
        import pandas as pd
        best_sep, best_cols = ";", 0
        for candidate in (";", ","):
            try:
                head = pd.read_csv(path, sep=candidate, nrows=1)
            except (OSError, ValueError):
                continue
            if head.shape[1] > best_cols:
                best_sep, best_cols = candidate, head.shape[1]
        return best_sep

    def _build_programme_options(self) -> dict[str, str]:
        """{isatcode_str: label} voor de opleiding-dropdowns.

        Haalt de opleidingen (isatcodes, en de naam waar bekend) uit álle
        beschikbare bronnen, zodat de dropdown al gevuld is zodra de gebruiker
        data heeft geüpload — ook vóór de eerste pipeline-run. Bronnen, in
        prioriteitsvolgorde voor de naam:

        1. het ruwe **oktober-bestand** (``Isatcode`` + ``Groepeernaam Croho``);
        2. de ruwe **telbestanden** (``Isatcode`` + ``Groepeernaam``);
        3. het bewerkte **1cijferho**-bestand (isatcode + naam);
        4. het bewerkte **student_count**-bestand (alleen isatcodes);
        5. reeds geconfigureerde sleutels (filter + numerus fixus),

        zodat bestaande selecties zichtbaar blijven ook als de data ze (nog) niet
        bevat.
        """
        import glob

        prog_col = self._programme_col()
        project = STATE.project_dir or ""
        input_dir = os.path.join(project, "data", "input")
        raw_dir = os.path.join(project, "data", "input_raw")
        okt_cols = self._config.get("columns", {}).get("oktober", {})

        name_map: dict[str, str] = {}
        codes: list = []

        def _harvest(df, code_col: str, name_col: str | None) -> None:
            if df is None or code_col not in df.columns:
                return
            codes.extend(df[code_col].dropna().tolist())
            if name_col and name_col in df.columns:
                for key, name in filtering_io.programme_name_map(
                    df, code_col=code_col, name_col=name_col
                ).items():
                    # setdefault: eerste bron (hoogste prioriteit) wint de naam.
                    name_map.setdefault(key, name)

        # 1. Oktober-bestand — beschikbaar vóór de run, met naam.
        _harvest(
            self._read_table(os.path.join(raw_dir, "oktober_bestand.xlsx")),
            okt_cols.get("Isatcode", "Isatcode"),
            okt_cols.get("Groepeernaam Croho", "Groepeernaam Croho"),
        )
        # 2. Telbestanden — beschikbaar vóór de run; naam waar aanwezig.
        # Telbestanden komen puntkomma- (legacy Studielink) én komma-gescheiden
        # (UvA SQL-export, o.a. de demodataset) voor. Bepaal het scheidingsteken
        # één keer uit het eerste bestand en hergebruik dat, zodat we niet elk van
        # honderden bestanden twee keer hoeven te lezen.
        tel_paths = sorted(glob.glob(os.path.join(raw_dir, "telbestanden", "*.csv")))
        tel_sep = self._detect_csv_sep(tel_paths[0]) if tel_paths else ";"
        for path in tel_paths:
            _harvest(self._read_table(path, sep=tel_sep), "Isatcode", "Groepeernaam")
        # 3. 1cijferho — bewerkt, met naam.
        _harvest(
            self._read_table(
                os.path.join(input_dir, "1cijferho_student_count_first-years.csv")
            ),
            prog_col,
            "groepeernaam_croho",
        )
        # 4. student_count — bewerkt, alleen isatcodes.
        _harvest(self._student_df, prog_col, None)
        # 5. Reeds geconfigureerde sleutels.
        codes += list(self._filtering.get("programme", []))
        codes += [r.get("key") for r in self._nf_rows]

        return filtering_io.build_programme_options(codes, name_map)

    def _filtering_section(self) -> None:
        active_count = (
            len(self._filtering.get("programme", []))
            + len(self._filtering.get("herkomst", []))
            + len(self._filtering.get("examentype", []))
        )
        header_suffix = f" — {active_count} filter(s) actief" if active_count else " — geen filter (alle data)"

        with ui.expansion(
            f"Filteren{header_suffix}", icon="filter_alt", value=False
        ).classes("w-full mb-3").style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {theme.INFO}20; border-left: 4px solid {theme.INFO}; "
            f"box-shadow: 0 1px 6px rgba(61,104,236,0.08);"
        ) as self._filter_expansion:
            with ui.row().classes("items-center gap-3 mb-4 px-4 py-3 rounded-xl").style(
                f"background: {theme.INFO}08; border: 1px solid {theme.INFO}25"
            ):
                ui.icon("info").style(f"color: {theme.INFO}; font-size: 16px;")
                ui.label(
                    "Leeg = geen filter (alle data). Vul alleen in als je de pipeline "
                    "wilt beperken tot bepaalde opleidingen, herkomsten of examentypes."
                ).classes("text-sm").style(f"color: {theme.INFO}99")

            # --- Opleidingen ---
            with ui.column().classes("w-full gap-1 mb-4"):
                with ui.row().classes("items-center gap-2 mb-1"):
                    ui.label("Opleidingen").classes("text-sm font-semibold")
                    if self._filtering.get("programme"):
                        ui.badge(
                            f"{len(self._filtering['programme'])} geselecteerd"
                        ).props("color=accent outline").classes("text-xs")
                    else:
                        ui.badge("Alle").props("color=positive outline").classes("text-xs")
                ui.label(
                    "Leeg = alle opleidingen. Typ om te zoeken op isatcode of naam, "
                    "of voer een isatcode handmatig in."
                ).classes("text-xs opacity-50")
                self._filter_programme_select = (
                    ui.select(
                        options=dict(self._programme_options),
                        value=[
                            filtering_io.isatcode_str(c)
                            for c in self._filtering.get("programme", [])
                        ],
                        multiple=True,
                        with_input=True,
                        label="Opleidingen selecteren (isatcode)",
                    )
                    .props("use-chips new-value-mode=add-unique outlined")
                    .classes("w-full mt-1")
                )
                self._filter_programme_select.on_value_change(
                    self._on_filter_programme_change
                )

            # --- Herkomst & Examentype naast elkaar ---
            with ui.row().classes("w-full gap-4 no-wrap mb-4"):
                with ui.card().classes("grow").style(
                    "border: 1px solid #e8e8e8; box-shadow: none;"
                ):
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.label("Herkomst").classes("text-sm font-semibold")
                        selected_h = self._filtering.get("herkomst", [])
                        if selected_h:
                            ui.badge(
                                f"{len(selected_h)} geselecteerd"
                            ).props("color=accent outline").classes("text-xs")
                        else:
                            ui.badge("Alle").props("color=positive outline").classes("text-xs")
                    ui.label("Leeg = alle herkomsten.").classes("text-xs opacity-50 mb-2")
                    self._filter_herkomst_boxes = {}
                    for choice in filtering_io.HERKOMST_CHOICES:
                        cb = ui.checkbox(
                            choice,
                            value=choice in self._filtering.get("herkomst", []),
                        )
                        cb.on_value_change(self._on_filter_herkomst_change)
                        self._filter_herkomst_boxes[choice] = cb

                with ui.card().classes("grow").style(
                    "border: 1px solid #e8e8e8; box-shadow: none;"
                ):
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.label("Examentype").classes("text-sm font-semibold")
                        selected_e = self._filtering.get("examentype", [])
                        if selected_e:
                            ui.badge(
                                f"{len(selected_e)} geselecteerd"
                            ).props("color=accent outline").classes("text-xs")
                        else:
                            ui.badge("Alle").props("color=positive outline").classes("text-xs")
                    ui.label("Leeg = alle examentypes.").classes("text-xs opacity-50 mb-2")
                    self._filter_examentype_boxes = {}
                    for choice in filtering_io.EXAMENTYPE_CHOICES:
                        cb = ui.checkbox(
                            choice,
                            value=choice in self._filtering.get("examentype", []),
                        )
                        cb.on_value_change(self._on_filter_examentype_change)
                        self._filter_examentype_boxes[choice] = cb

            # --- Live preview ---
            self._filter_preview = ui.column().classes("w-full")
            self._update_filter_preview()

    def _on_filter_programme_change(self, e) -> None:
        # Normaliseer naar canonieke isatcode-strings (ook handmatig getypte
        # waarden), zodat opslag en pipeline-filter matchen.
        self._filtering["programme"] = [
            filtering_io.isatcode_str(v) for v in (e.value or []) if filtering_io.isatcode_str(v)
        ]
        self._mark_dirty()
        self._update_filter_preview()

    def _on_filter_herkomst_change(self, _e) -> None:
        self._filtering["herkomst"] = [
            c for c, box in self._filter_herkomst_boxes.items() if box.value
        ]
        self._mark_dirty()
        self._update_filter_preview()

    def _on_filter_examentype_change(self, _e) -> None:
        self._filtering["examentype"] = [
            c for c, box in self._filter_examentype_boxes.items() if box.value
        ]
        self._mark_dirty()
        self._update_filter_preview()

    def _update_filter_preview(self) -> None:
        self._filter_preview.clear()
        with self._filter_preview:
            if self._student_df is None:
                with ui.row().classes("items-center gap-2 px-3 py-2 rounded-lg").style(
                    "background: #f5f5f5; border: 1px solid #e0e0e0;"
                ):
                    ui.icon("info").style("color: #aaa; font-size: 14px;")
                    ui.label(
                        "Live preview niet beschikbaar — "
                        "draai eerst de pipeline om student_count te genereren."
                    ).classes("text-xs opacity-60")
                return

            roles = self._config.get("column_roles", {})
            remaining, total = filtering_io.count_programmes(
                self._student_df,
                programme_col=roles.get("programme", "Croho groepeernaam"),
                origin_col=roles.get("origin", "Herkomst"),
                exam_col=roles.get("exam_type", "Examentype"),
                programme=self._filtering.get("programme", []),
                herkomst=self._filtering.get("herkomst", []),
                examentype=self._filtering.get("examentype", []),
            )
            if total == 0:
                return

            pct = remaining / total
            if pct >= 0.999:
                color, icon = theme.POSITIVE, "check_circle"
                msg = f"Alle {total} opleidingen geselecteerd (geen filter actief)."
            elif remaining > 0:
                color, icon = theme.ACCENT, "filter_alt"
                msg = f"{remaining} van {total} opleidingen geselecteerd na filtering."
            else:
                color, icon = theme.NEGATIVE, "warning"
                msg = "Geen opleidingen geselecteerd — controleer je filterinstellingen."

            with ui.row().classes("items-center gap-2 px-4 py-2 rounded-lg").style(
                f"background: {color}12; border: 1px solid {color}35;"
            ):
                ui.icon(icon).style(f"color: {color}; font-size: 16px;")
                ui.label(msg).classes("text-sm font-medium").style(f"color: {color}")

    def _model_section(self) -> None:
        mc = self._config.setdefault("model_config", {})
        ts = mc.get("cumulative_timeseries", "sarima")
        with ui.expansion(f"Modelkeuze — {ts}", icon="model_training", value=False).classes(
            "w-full mb-3"
        ).style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {theme.ACCENT}20; border-left: 4px solid {theme.ACCENT}; "
            f"box-shadow: 0 1px 6px rgba(221,120,75,0.08);"
        ):
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

            ui.separator().classes("my-3")
            with ui.row().classes("items-end gap-4 flex-wrap"):
                my_inp = (
                    ui.number(
                        label="Vroegste trainingsjaar (min_training_year)",
                        value=int(mc.get("min_training_year", 2016)),
                        min=1990,
                        max=2100,
                        step=1,
                        precision=0,
                    )
                    .props("outlined")
                    .classes("w-64")
                )
                my_inp.tooltip(HELP["min_training_year"])

                def _on_min_training_year(e) -> None:
                    if e.value is None:
                        return
                    mc["min_training_year"] = int(e.value)
                    self._mark_dirty()

                my_inp.on_value_change(_on_min_training_year)

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
            "w-full mb-3"
        ).style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {theme.NPULS_GREEN}20; border-left: 4px solid {theme.NPULS_GREEN}; "
            f"box-shadow: 0 1px 6px rgba(0,175,129,0.08);"
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
        nf_label = (
            f"Numerus fixus — {len(self._nf_rows)} opleiding(en)"
            if self._nf_rows
            else "Numerus fixus — geen"
        )
        with ui.expansion(nf_label, icon="lock", value=False).classes("w-full mb-3").style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {theme.WARNING}25; border-left: 4px solid {theme.WARNING}; "
            f"box-shadow: 0 1px 6px rgba(230,160,32,0.08);"
        ):
            ui.label(
                "Opleidingen met een capaciteitslimiet. Gebruik exact dezelfde "
                "programmasleutel als in je data — de voorspelling wordt op dit maximum afgetopt."
            ).classes("text-sm opacity-60 mb-2")
            self._nf_container = ui.column().classes("w-full gap-2")
            self._render_nf_rows()
            ui.button("Rij toevoegen", icon="add", on_click=self._add_nf_row).props("flat")

    def _excl_section(self) -> None:
        excl_year_count = len({
            str(r.get("year")) for r in self._excl_rows if r.get("year") is not None
        })
        excl_label = (
            f"Uitsluitingsregels — {excl_year_count} jaar/jaren"
            if excl_year_count
            else "Uitsluitingsregels — geen"
        )
        with ui.expansion(excl_label, icon="block", value=False).classes(
            "w-full mb-3"
        ).style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {theme.NEGATIVE}18; border-left: 4px solid {theme.NEGATIVE}; "
            f"box-shadow: 0 1px 6px rgba(192,57,43,0.07);"
        ):
            ui.label(
                "Sluit bekende probleemjaren (bijv. COVID) of specifieke deelpopulaties "
                "uit de trainingsdata. Het voorspeljaar zelf is altijd beschermd."
            ).classes("text-sm opacity-60 mb-3")

            # Aanbevolen: COVID-jaren snel toevoegen — alleen tonen als de
            # data die jaren ook daadwerkelijk dekt (anders zou de knop
            # jaren toevoegen die niet in de training zitten).
            covid_selectable = [y for y in (2020, 2021) if y in self._selectable_years]
            if covid_selectable:
                covid_label = (
                    "Voeg 2020 & 2021 toe"
                    if len(covid_selectable) == 2
                    else f"Voeg {covid_selectable[0]} toe"
                )
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
                        covid_label,
                        icon="add_circle",
                        on_click=self._add_covid_years,
                    ).props("outline dense color=accent").classes("shrink-0")

            # Eenvoudige jaar-chips (quick add/remove)
            self._excl_years_chips = ui.row().classes("gap-2 flex-wrap mb-3 min-h-8")
            self._render_excl_year_chips()
            if self._selectable_years:
                with ui.row().classes("items-center gap-2 mb-1"):
                    opts = self._available_year_options()
                    self._new_year_input = (
                        ui.select(
                            options=opts,
                            value=(opts[0] if opts else None),
                            label="Jaar toevoegen",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )
                    ui.button(
                        "Toevoegen",
                        icon="add",
                        on_click=self._add_excl_year,
                    ).props("outline dense")
                ui.label(
                    "Alleen jaren met zowel tel- als oktoberdata "
                    f"({self._selectable_years[0]}–{self._selectable_years[-1]}) "
                    "kunnen worden uitgesloten."
                ).classes("text-xs opacity-50 mb-4")
            else:
                # Geen bruikbaar traindata-bereik bekend (data nog niet
                # geüpload): geen vrije invoer, zodat er geen jaar buiten de
                # data gekozen kan worden.
                self._new_year_input = None
                with ui.row().classes(
                    "items-center gap-2 mb-4 px-3 py-2 rounded-lg"
                ).style(f"background: {theme.INFO}0e; border: 1px solid {theme.INFO}30"):
                    ui.icon("info").style(f"color: {theme.INFO}; font-size: 16px;")
                    ui.label(
                        "Upload eerst tel- en oktoberbestanden; daarna kun je "
                        "uitsluitingsjaren kiezen binnen de beschikbare data."
                    ).classes("text-xs").style(f"color: {theme.INFO}")

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
        max_cores = os.cpu_count() or 1
        cpu_val = self._config.get("runtime", {}).get("cpu_count", None)
        runtime_label = f"Runtime — {cpu_val} cores" if cpu_val else "Runtime — standaard"
        with ui.expansion(runtime_label, icon="settings", value=False).classes("w-full mb-3").style(
            "background: white; border-radius: 10px; overflow: hidden; "
            "border: 1px solid #e0e0e0; border-left: 4px solid #9e9e9e; "
            "box-shadow: 0 1px 5px rgba(0,0,0,0.04);"
        ):
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
            with ui.column().classes("gap-1 mt-2"):
                current_cpu = runtime.get("cpu_count", None)
                cpu_inp = (
                    ui.number(
                        label="CPU-cores (leeg = automatisch)",
                        value=current_cpu,
                        min=1,
                        max=max_cores,
                        step=1,
                    )
                    .props("outlined clearable")
                    .classes("w-full max-w-xs")
                )
                cpu_inp.tooltip(HELP["cpu_count"])
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("memory").style("color: #aaa; font-size: 14px;")
                    ui.label(f"{max_cores} cores beschikbaar op deze machine").classes(
                        "text-xs"
                    ).style("color: #aaa;")

                def _on_cpu_change(e) -> None:
                    val = e.value
                    if val is None:
                        runtime["cpu_count"] = None
                    else:
                        clamped = max(1, min(int(val), max_cores))
                        runtime["cpu_count"] = clamped
                        if int(val) != clamped:
                            cpu_inp.set_value(clamped)
                    self._mark_dirty()

                cpu_inp.on_value_change(_on_cpu_change)

    def _adv_switch(self, label, target, key, default, *, help=None) -> None:
        sw = ui.switch(label, value=target.get(key, default))
        if help:
            sw.tooltip(help)

        def _on_change(e) -> None:
            target[key] = e.value
            self._mark_dirty()

        sw.on_value_change(_on_change)

    def _validation_section(self) -> None:
        telbestand_cfg = self._config.setdefault("validation", {}).setdefault(
            "telbestand", {}
        )
        sep_val = telbestand_cfg.get("separator", ";")
        with ui.expansion(
            f"Validatie — telbestand ({sep_val})", icon="fact_check", value=False
        ).classes("w-full mb-3").style(
            f"background: white; border-radius: 10px; overflow: hidden; "
            f"border: 1px solid {theme.INFO}20; border-left: 4px solid {theme.INFO}; "
            f"box-shadow: 0 1px 6px rgba(61,104,236,0.08);"
        ):
            ui.label(
                "Datakwaliteitscontrole vóór de pipeline start. Deze instellingen "
                "overschrijven de ingebouwde validatiedefaults voor telbestanden — "
                "handig als je databron een afwijkend formaat levert (bijv. UvA "
                "SQL i.p.v. legacy Studielink)."
            ).classes("text-sm opacity-60 mb-3")

            with ui.row().classes("w-full gap-4 flex-wrap mb-4"):
                sep_select = (
                    ui.select(
                        options={";": "; (legacy Studielink)", ",": ", (UvA SQL)"},
                        value=telbestand_cfg.get("separator", ";"),
                        label="Scheidingsteken",
                    )
                    .props("outlined")
                    .classes("w-64")
                )
                sep_select.tooltip(HELP["validation_separator"])

                def _on_sep(e) -> None:
                    telbestand_cfg["separator"] = e.value
                    self._mark_dirty()

                sep_select.on_value_change(_on_sep)

                col_input = (
                    ui.input(
                        value=telbestand_cfg.get("programme_column", "Groepeernaam"),
                        label="Programmakolom",
                        placeholder="bijv. Isatcode",
                    )
                    .props("outlined")
                    .classes("w-64")
                )
                col_input.tooltip(HELP["validation_programme_column"])

                def _on_col(e) -> None:
                    telbestand_cfg["programme_column"] = e.value
                    self._mark_dirty()

                col_input.on_value_change(_on_col)

            self._render_string_list_field(
                title="Toegestane herkomstcodes",
                help=HELP["validation_herkomst_allowed"],
                items=telbestand_cfg.setdefault("herkomst_allowed", ["N", "E", "R"]),
                placeholder="bijv. N",
            )
            self._render_string_list_field(
                title="Verplichte kolommen",
                help=HELP["validation_required_columns"],
                items=telbestand_cfg.setdefault(
                    "required_columns",
                    [
                        "Studiejaar", "Isatcode", "Groepeernaam", "Aantal",
                        "meercode_V", "Status", "Herinschrijving", "Hogerejaars",
                        "Herkomst",
                    ],
                ),
                placeholder="bijv. Isatcode",
            )

    def _render_string_list_field(
        self, *, title: str, help: str, items: list[str], placeholder: str
    ) -> None:
        """Bewerkbare lijst van strings: chips + een toevoeg-invoerveld.

        ``items`` is de *levende* lijst uit de configuratie zelf — muteren
        houdt ``self._config`` meteen in sync, zonder aparte write-back stap.
        """
        with ui.column().classes("w-full gap-1 mb-4"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.label(title).classes("text-sm font-semibold")
                count_badge = (
                    ui.badge(str(len(items)))
                    .props("color=accent outline")
                    .classes("text-xs")
                )
            chips_container = ui.row().classes("gap-2 flex-wrap mb-2 min-h-8")

            def _render_chips() -> None:
                chips_container.clear()
                count_badge.set_text(str(len(items)))
                with chips_container:
                    if not items:
                        ui.label(
                            "Leeg — gebruikt de ingebouwde standaard."
                        ).classes("text-sm opacity-40 italic")
                    for val in list(items):
                        with ui.row().classes(
                            "items-center gap-1 px-3 py-1 rounded-full no-wrap"
                        ).style(
                            f"background: {theme.INFO}12; "
                            f"border: 1px solid {theme.INFO}35;"
                        ):
                            ui.label(val).classes("text-sm font-mono")
                            ui.button(
                                icon="close",
                                on_click=lambda _e, v=val: _remove(v),
                            ).props("flat round dense").style(
                                f"color: {theme.INFO}; width: 20px; height: 20px;"
                            )

            def _remove(val: str) -> None:
                if val in items:
                    items.remove(val)
                _render_chips()
                self._mark_dirty()

            def _add() -> None:
                val = (new_input.value or "").strip()
                if not val:
                    return
                if val in items:
                    ui.notify(f"'{val}' staat al in de lijst.", type="warning")
                    return
                items.append(val)
                new_input.set_value("")
                _render_chips()
                self._mark_dirty()

            _render_chips()

            with ui.row().classes("items-center gap-2"):
                new_input = (
                    ui.input(placeholder=placeholder)
                    .props("dense outlined")
                    .classes("w-56")
                )
                new_input.on("keydown.enter", _add)
                ui.button("Toevoegen", icon="add", on_click=_add).props(
                    "dense outline"
                )
            if help:
                ui.label(help).classes("text-xs opacity-45 mt-1")

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
                    key_val = filtering_io.isatcode_str(row.get("key"))
                    key_opts = dict(self._programme_options)
                    if key_val and key_val not in key_opts:
                        key_opts[key_val] = key_val
                    key_in = (
                        ui.select(
                            options=key_opts,
                            value=key_val or None,
                            with_input=True,
                            label="Programmasleutel (isatcode)",
                        )
                        .props("dense outlined new-value-mode=add-unique")
                        .classes("grow")
                    )
                    key_in.tooltip(HELP["numerus_fixus"])
                    key_in.on_value_change(
                        lambda e, r=row: (
                            r.update(key=filtering_io.isatcode_str(e.value)),
                            self._mark_dirty(),
                        )
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
        ui.add_head_html(
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsoneditor@10'
            '/dist/jsoneditor.min.css">\n'
            '<script src="https://cdn.jsdelivr.net/npm/jsoneditor@10'
            '/dist/jsoneditor.min.js"></script>\n'
            + f"""<style>
#sp-json-ed {{height:560px}}
.jsoneditor {{border:none !important;border-radius:10px !important;overflow:hidden;
    box-shadow:0 0 0 1px #e0e0e0}}
.jsoneditor-menu {{background:{theme.PRIMARY} !important;
    border-bottom:2px solid {theme.ACCENT}44 !important}}
.jsoneditor-menu button {{color:rgba(255,255,255,.75) !important;
    border-radius:5px !important;transition:background .15s}}
.jsoneditor-menu button:hover {{background:rgba(255,255,255,.1) !important}}
.jsoneditor-menu button.jsoneditor-selected {{background:{theme.ACCENT} !important;
    color:#fff !important}}
.jsoneditor-navigation-bar {{background:#f8f9ff !important;
    border-bottom:1px solid #eaecf5 !important;
    font-size:11.5px !important;color:#5a607a !important}}
.jsoneditor-statusbar {{background:#f8f9ff !important;
    border-top:1px solid #eaecf5 !important;
    font-size:11px !important;color:#9da3bb !important}}
.jsoneditor-field {{color:#1a2030 !important;font-weight:500}}
.jsoneditor-value.jsoneditor-string {{color:{theme.NPULS_GREEN} !important}}
.jsoneditor-value.jsoneditor-number {{color:{theme.ACCENT} !important;font-weight:600}}
.jsoneditor-value.jsoneditor-boolean {{color:{theme.SECONDARY} !important;font-weight:600}}
.jsoneditor-value.jsoneditor-null {{color:#aab !important;font-style:italic}}
.jsoneditor-search input {{border-radius:4px !important;
    border:1px solid rgba(255,255,255,.25) !important;
    background:rgba(255,255,255,.1) !important;color:#fff !important;
    padding:2px 7px !important}}
.jsoneditor-search input::placeholder {{color:rgba(255,255,255,.4) !important}}
.jsoneditor-search .jsoneditor-results {{color:rgba(255,255,255,.6) !important}}
</style>"""
        )

        with ui.row().classes("items-center gap-3 mb-4 px-4 py-3 rounded-xl").style(
            "background: #1a1a1a08; border: 1px solid #1a1a1a18;"
        ):
            ui.icon("data_object").classes("text-xl opacity-40")
            with ui.column().classes("gap-0 grow"):
                ui.label("JSON-editor").classes("font-medium text-sm")
                ui.label(
                    "Bewerk de volledige configuratie rechtstreeks — ook secties "
                    "zonder eigen formulierkaart (bijv. model_features, columns, "
                    "cumulative_input). Klik ▶ om secties open/dicht te klappen. "
                    "Wijzigingen in de tabbladen Basis en Geavanceerd verschijnen "
                    "pas hier zodra je die kant op navigeert (of de pagina "
                    "herlaadt) — schrijf dus niet in beide tabbladen tegelijk."
                ).classes("text-sm opacity-50")

        with ui.row().classes("items-center gap-2 mb-3"):
            ui.icon("folder_open").style("color: #bbb; font-size: 14px;")
            ui.label(self._path).classes("text-xs font-mono opacity-40 break-all")

        ui.html('<div id="sp-json-ed"></div>')

        self._json_save_error = ui.column().classes("w-full items-end gap-1 mt-3")
        with ui.row().classes("w-full justify-end mt-2"):
            ui.button(
                "Opslaan vanuit JSON", icon="save", on_click=self._save_json_tab
            ).props("unelevated color=accent")

        self._run_json_editor_init()

    def _run_json_editor_init(self) -> None:
        config_json = json.dumps(self._config, ensure_ascii=False)
        ui.run_javascript(
            f"""
            (function tryInit(n) {{
                const el = document.getElementById('sp-json-ed');
                if (!el || typeof JSONEditor === 'undefined') {{
                    if (n > 0) {{ setTimeout(() => tryInit(n - 1), 150); return; }}
                    if (el) el.innerHTML =
                        '<p style="color:#c0392b;padding:16px;font-size:13px">'
                        + '&#9888; JSONEditor kon niet worden geladen'
                        + ' &#x2014; controleer je internetverbinding.</p>';
                    return;
                }}
                if (window.__spJE) {{ try {{ window.__spJE.destroy(); }} catch (_) {{}} }}
                window.__spJE = new JSONEditor(el, {{
                    mode: 'tree',
                    modes: ['tree', 'code'],
                    mainMenuBar: true,
                    navigationBar: true,
                    statusBar: true,
                    search: true,
                    enableSort: false,
                    enableTransform: false,
                    onError: function (e) {{ console.error('JSONEditor:', e); }},
                }});
                window.__spJE.set({config_json});
                window.__spJE.expandAll();
            }})(15);
            """
        )

    async def _save_json_tab(self) -> None:
        """Lees de JSON-editor uit, valideer en schrijf naar ``configuration.json``.

        De editor leeft alleen client-side (JSONEditor is een JS-widget), dus
        de inhoud moet via ``run_javascript`` worden opgehaald. Bij succes
        herladen we de pagina zodat alle tabbladen (Basis/Geavanceerd) de
        nieuw opgeslagen waarden tonen in plaats van hun oude in-memory state.
        """
        self._json_save_error.clear()
        try:
            raw = await ui.run_javascript(
                "try { return JSON.stringify(window.__spJE.get()); } "
                "catch (e) { return '__SP_JSON_ERROR__:' + e.message; }",
                timeout=5.0,
            )
        except TimeoutError:
            ui.notify(
                "JSON-editor reageert niet — herlaad de pagina en probeer opnieuw.",
                type="negative",
            )
            return

        if raw is None:
            ui.notify("JSON-editor is nog niet geladen.", type="warning")
            return
        if isinstance(raw, str) and raw.startswith("__SP_JSON_ERROR__:"):
            self._show_json_error(f"Ongeldige JSON: {raw.removeprefix('__SP_JSON_ERROR__:')}")
            return

        try:
            parsed = config_io.parse_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_json_error(f"Ongeldige JSON: {exc}")
            return

        errors = config_io.validate_config(parsed)
        if errors:
            for err in errors:
                self._show_json_error(err)
            return

        try:
            config_io.save_config(self._path, parsed)
        except OSError as exc:
            ui.notify(f"Opslaan mislukt: {exc}", type="negative")
            return

        STATE.config_saved = True
        ui.notify("Configuratie opgeslagen vanuit JSON.", type="positive")
        ui.navigate.reload()

    def _show_json_error(self, message: str) -> None:
        with self._json_save_error:
            with ui.row().classes("items-center gap-1"):
                ui.icon("error").style(f"color: {theme.NEGATIVE}; font-size: 16px;")
                ui.label(message).classes("text-sm").style(f"color: {theme.NEGATIVE}")

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

    def _save(self, *, notify: bool = True) -> bool:
        """Valideer en schrijf configuratie + filtering naar schijf.

        Args:
            notify: Toon een notificatie bij validatiefouten of schrijffouten.
                De debounced auto-save geeft ``False`` mee, zodat tussentijdse
                (nog onvolledige) waarden tijdens het typen geen foutmeldingen
                opleveren; alleen een expliciete opslag ('Volgende') is luid.

        Returns:
            ``True`` als er daadwerkelijk is opgeslagen, anders ``False``.
        """
        self._sync_dynamic_into_config()
        errors = config_io.validate_config(self._config)
        if errors:
            if notify:
                self._validate_ensemble()
                ui.notify(
                    "Kan niet opslaan: los eerst de validatiefouten op.",
                    type="negative",
                )
            return False
        filter_errors = filtering_io.validate_filtering(self._filtering_data)
        if filter_errors:
            if notify:
                for err in filter_errors:
                    ui.notify(err, type="negative")
            return False
        try:
            config_io.save_config(self._path, self._config)
            filtering_io.save_filtering(STATE.filtering_path, self._filtering_data)
        except OSError as exc:
            if notify:
                ui.notify(f"Opslaan mislukt: {exc}", type="negative")
            return False
        STATE.config_saved = True
        return True

    def _on_institution_change(self, e) -> None:
        self._config["institution_filter"] = [e.value] if e.value else []
        self._mark_dirty()

    # ─── Dirty-tracking ───────────────────────────────────────────────────────

    def _on_next_click(self) -> None:
        # Expliciete opslag: luid (toont eventuele validatiefouten). Invalideer
        # een eventueel geplande auto-save zodat die niet dubbel schrijft.
        self._dirty_gen += 1
        self._save(notify=True)
        ui.navigate.to(nav.next_route("/config"))

    def _mark_dirty(self) -> None:
        """Markeer de config als gewijzigd en plan een stille, debounced auto-save.

        Elke wijziging verschuift de save ``_AUTOSAVE_DELAY_S`` seconden naar
        achteren via een generatieteller: alleen de laatste geplande save van een
        reeks snelle wijzigingen voert daadwerkelijk uit. Zo valideert en schrijft
        de editor niet langer bij élke slider-tick of toetsaanslag.
        """
        self._dirty_gen += 1
        gen = self._dirty_gen
        ui.timer(_AUTOSAVE_DELAY_S, lambda: self._flush_autosave(gen), once=True)

    def _flush_autosave(self, gen: int) -> None:
        """Voer de auto-save uit als er ondertussen geen nieuwere wijziging kwam."""
        if gen == self._dirty_gen:
            self._save(notify=False)
