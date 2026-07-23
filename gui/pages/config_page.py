"""Configuratie-editor (#267).

Laadt ``configuration.json`` en maakt de parameters bewerkbaar. De veelgebruikte
instellingen krijgen passende widgets; het tabblad "Geavanceerd (JSON)" maakt
élke sleutel bewerkbaar. Ensemble-gewichten worden gevalideerd (som = 1.0) en
niet-opgeslagen wijzigingen leveren een waarschuwing op bij het verlaten.
"""

from __future__ import annotations

import json

from nicegui import ui

from gui import config_io, nav
from gui.components.layout import page_shell
from gui.components.states import empty_state, error_banner, section_title
from gui.state import STATE

#: Uitleg per configuratieveld — getoond als hover-tooltip. In lijn met
#: configuration.json en docs/configuratie-referentie.md.
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
        "de seizoensvolgorde van de weken, de voorspelhorizon en de reset-week in "
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

#: Korte tooltip per ensemble-gewichtgroep (welke weken het betreft).
_GROUP_HELP = {
    "master_week_17_23": "Masters in de vroege weken 17 t/m 23.",
    "week_30_34": "Weken 30 t/m 34.",
    "week_35_37": "Weken 35 t/m 37 (vlak voor de deadline).",
    "default": "Alle overige weken en combinaties.",
}


def _info_icon(text: str) -> None:
    """Render een klein help-icoon met een hover-tooltip."""
    ui.icon("help_outline").classes("text-sm opacity-50 cursor-help").tooltip(text)


def create() -> None:
    """Registreer de route ``/config``."""
    nav.register_route("/config")

    @ui.page("/config")
    def config_page() -> None:
        with page_shell(active="/config", title="Configuratie"):
            section_title("Configuratie", "Stel de model- en pipelineparameters in.")
            if not STATE.is_initialised:
                empty_state(
                    icon="folder_off",
                    title="Nog geen project",
                    message="Kies eerst een project voordat je de configuratie "
                    "kunt bewerken.",
                    action_label="Project opzetten",
                    on_action=lambda: ui.navigate.to("/wizard"),
                )
                return
            _ConfigView(STATE.config_path)


class _ConfigView:
    """Houdt de configuratie in het geheugen en rendert de editor."""

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
        self._build()

    # --- Opbouw ---------------------------------------------------------------

    def _build(self) -> None:
        with ui.row().classes("w-full items-center justify-between"):
            self._save_btn = ui.button(
                "Opslaan", icon="save", on_click=self._save
            ).props("unelevated")
            self._status = ui.label("").classes("text-sm")

        with ui.tabs().props("indicator-color=accent").classes("w-full") as tabs:
            tab_settings = ui.tab("Instellingen", icon="tune")
            tab_json = ui.tab("Geavanceerd (JSON)", icon="data_object")
        with ui.tab_panels(tabs, value=tab_settings).classes("w-full"):
            with ui.tab_panel(tab_settings):
                self._structured = ui.column().classes("w-full")
                self._render_structured()
            with ui.tab_panel(tab_json):
                self._build_json()
                # Synchroniseer de JSON-tekst zodra men het tabblad opent.
                tab_json.on("click", self._refresh_json_text)

    def _render_structured(self) -> None:
        """(Her)bouw de structured widgets vanuit ``self._config``."""
        self._structured.clear()
        with self._structured:
            self._model_section()
            self._ensemble_section()
            self._numerus_fixus_section()
            self._institution_section()
            self._excluded_section()
            self._misc_section()

    # --- Secties --------------------------------------------------------------

    def _model_section(self) -> None:
        mc = self._config.setdefault("model_config", {})
        with ui.expansion("Model", icon="model_training", value=True).classes("w-full"):
            with ui.grid(columns=2).classes("w-full gap-4"):
                self._select(
                    "Tijdreeksmodel (cumulatief)",
                    config_io.TIMESERIES_CHOICES,
                    mc,
                    "cumulative_timeseries",
                    "sarima",
                    help=HELP["cumulative_timeseries"],
                )
                self._select(
                    "Regressor (cumulatief)",
                    config_io.REGRESSOR_CHOICES,
                    mc,
                    "cumulative_regressor",
                    "xgboost",
                    help=HELP["cumulative_regressor"],
                )
                self._select(
                    "Classifier (individueel)",
                    config_io.CLASSIFIER_CHOICES,
                    mc,
                    "individual_classifier",
                    "xgboost",
                    help=HELP["individual_classifier"],
                )
                self._number(
                    "Min. trainingsjaar",
                    mc,
                    "min_training_year",
                    2016,
                    help=HELP["min_training_year"],
                )
                self._number(
                    "Laatste academische week",
                    mc,
                    "final_academic_week",
                    36,
                    help=HELP["final_academic_week"],
                )

    def _ensemble_section(self) -> None:
        weights = self._config.setdefault("ensemble_weights", {})
        with ui.expansion("Ensemble-gewichten", icon="balance", value=True).classes(
            "w-full"
        ):
            with ui.row().classes("items-center gap-1"):
                ui.label(
                    "Per weeksegment het gewicht van het individuele en cumulatieve "
                    "spoor. Elk paar moet optellen tot 1,0."
                ).classes("text-sm opacity-70")
                _info_icon(HELP["ensemble_weights"])
            self._ensemble_error = ui.column().classes("w-full")
            for group in config_io.ENSEMBLE_GROUPS:
                grp = weights.setdefault(group, {"individual": 0.5, "cumulative": 0.5})
                with ui.row().classes("w-full items-center gap-4 no-wrap"):
                    with ui.row().classes("items-center gap-1 w-48"):
                        ui.label(group).classes("text-sm font-mono")
                        _info_icon(_GROUP_HELP.get(group, ""))
                    self._weight_input(
                        grp, "individual", "individueel", help=HELP["weight_individual"]
                    )
                    self._weight_input(
                        grp, "cumulative", "cumulatief", help=HELP["weight_cumulative"]
                    )
            self._validate_ensemble()

    def _numerus_fixus_section(self) -> None:
        nf = self._config.setdefault("numerus_fixus", {})
        self._nf_rows = [{"key": k, "value": v} for k, v in nf.items()]
        with ui.expansion("Numerus fixus", icon="lock", value=False).classes("w-full"):
            with ui.row().classes("items-center gap-1"):
                ui.label(
                    "Opleidingen met een capaciteitslimiet (exacte programmasleutel → "
                    "aantal plaatsen)."
                ).classes("text-sm opacity-70")
                _info_icon(HELP["numerus_fixus"])
            self._nf_container = ui.column().classes("w-full gap-1")
            self._render_nf_rows()
            ui.button("Rij toevoegen", icon="add", on_click=self._add_nf_row).props(
                "flat"
            )

    def _institution_section(self) -> None:
        current = self._config.setdefault("institution_filter", [])
        with ui.expansion("Instellingsfilter", icon="business", value=False).classes(
            "w-full"
        ):
            with ui.row().classes("items-center gap-1"):
                ui.label(
                    "Beperk tot één of meer instellingen (Brincode of korte naam). "
                    "Leeg = alle instellingen."
                ).classes("text-sm opacity-70")
                _info_icon(HELP["institution_filter"])
            self._inst_select = (
                ui.select(
                    options=list(current),
                    value=list(current),
                    multiple=True,
                    label="Instellingen",
                )
                .props("use-chips new-value-mode=add-unique hide-dropdown-icon")
                .classes("w-full")
            )
            self._inst_select.tooltip(HELP["institution_filter"])
            self._inst_select.on_value_change(self._on_institution_change)

    def _excluded_section(self) -> None:
        excl = self._config.setdefault("excluded_data_points", [])
        self._excl_rows = [dict(item) for item in excl]
        with ui.expansion("Uitgesloten datapunten", icon="block", value=False).classes(
            "w-full"
        ):
            with ui.row().classes("items-center gap-1"):
                ui.label(
                    "Datapunten die uit de training worden gehouden (bijv. een "
                    "uitzonderlijk coronajaar)."
                ).classes("text-sm opacity-70")
                _info_icon(HELP["excluded_data_points"])
            self._excl_container = ui.column().classes("w-full gap-1")
            self._render_excl_rows()
            ui.button("Rij toevoegen", icon="add", on_click=self._add_excl_row).props(
                "flat"
            )

    def _misc_section(self) -> None:
        ci = self._config.setdefault("cumulative_input", {})
        with ui.expansion("Overig", icon="settings", value=False).classes("w-full"):
            self._switch(
                "Aggregeren (cumulatief)", ci, "aggregate", True, help=HELP["aggregate"]
            )
            self._switch(
                "Verwijderde rijen weglaten",
                ci,
                "drop_deleted",
                True,
                help=HELP["drop_deleted"],
            )
            runtime = self._config.setdefault("runtime", {})
            self._number(
                "CPU-cores (leeg = automatisch)",
                runtime,
                "cpu_count",
                None,
                allow_none=True,
                help=HELP["cpu_count"],
            )

    # --- Widget-helpers -------------------------------------------------------

    def _select(self, label, choices, target, key, default, *, help=None) -> None:
        value = target.get(key, default)
        options = list(choices)
        if value not in options:
            options.append(value)
        sel = ui.select(options, value=value, label=label).classes("w-full")
        if help:
            sel.tooltip(help)

        def _on_change(e) -> None:
            target[key] = e.value
            self._mark_dirty()

        sel.on_value_change(_on_change)

    def _number(
        self, label, target, key, default, *, allow_none=False, help=None
    ) -> None:
        value = target.get(key, default)
        inp = ui.number(label=label, value=value).classes("w-full")
        if help:
            inp.tooltip(help)

        def _on_change(e) -> None:
            val = e.value
            if val is None and not allow_none:
                return
            target[key] = int(val) if val is not None else None
            self._mark_dirty()

        inp.on_value_change(_on_change)

    def _switch(self, label, target, key, default, *, help=None) -> None:
        sw = ui.switch(label, value=target.get(key, default))
        if help:
            sw.tooltip(help)

        def _on_change(e) -> None:
            target[key] = e.value
            self._mark_dirty()

        sw.on_value_change(_on_change)

    def _weight_input(self, group_dict, key, caption, *, help=None) -> None:
        inp = (
            ui.number(label=caption, value=group_dict.get(key, 0.5), step=0.1)
            .props("dense")
            .classes("w-32")
        )
        if help:
            inp.tooltip(help)

        def _on_change(e) -> None:
            try:
                group_dict[key] = float(e.value) if e.value is not None else 0.0
            except (TypeError, ValueError):
                group_dict[key] = 0.0
            self._mark_dirty()
            self._validate_ensemble()

        inp.on_value_change(_on_change)

    # --- Dynamische tabellen --------------------------------------------------

    def _render_nf_rows(self) -> None:
        self._nf_container.clear()
        with self._nf_container:
            for row in self._nf_rows:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    key_in = (
                        ui.input(value=row["key"], placeholder="Programmasleutel")
                        .props("dense")
                        .classes("grow")
                    )
                    key_in.on_value_change(
                        lambda e, r=row: (r.update(key=e.value), self._mark_dirty())
                    )
                    val_in = (
                        ui.number(value=row["value"], placeholder="Plaatsen")
                        .props("dense")
                        .classes("w-32")
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

    def _remove_nf_row(self, row) -> None:
        self._nf_rows.remove(row)
        self._render_nf_rows()
        self._mark_dirty()

    _EXCL_FIELDS = ["year", "herkomst", "examentype", "opleiding"]

    def _render_excl_rows(self) -> None:
        self._excl_container.clear()
        with self._excl_container:
            for row in self._excl_rows:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    for field in self._EXCL_FIELDS:
                        inp = (
                            ui.input(value=str(row.get(field, "")), placeholder=field)
                            .props("dense")
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

    def _remove_excl_row(self, row) -> None:
        self._excl_rows.remove(row)
        self._render_excl_rows()
        self._mark_dirty()

    # --- Geavanceerd (JSON) ---------------------------------------------------

    def _build_json(self) -> None:
        ui.label(
            "Bewerk hier de volledige configuratie als JSON. 'JSON toepassen' "
            "laadt je wijzigingen in de bovenstaande velden."
        ).classes("text-sm opacity-70")
        self._json_area = (
            ui.textarea(value=json.dumps(self._config, ensure_ascii=False, indent=4))
            .props("outlined")
            .classes("w-full font-mono")
            .style("min-height: 24rem")
        )
        ui.button("JSON toepassen", icon="check", on_click=self._apply_json).props(
            "outline"
        )

    def _refresh_json_text(self) -> None:
        self._json_area.set_value(
            json.dumps(self._config, ensure_ascii=False, indent=4)
        )

    def _apply_json(self) -> None:
        try:
            parsed = config_io.parse_json(self._json_area.value)
        except (json.JSONDecodeError, ValueError) as exc:
            ui.notify(f"Ongeldige JSON: {exc}", type="negative")
            return
        self._config = parsed
        self._mark_dirty()
        # Herbouw de structured widgets vanuit de nieuwe config (geen reload,
        # zodat de nog niet opgeslagen wijzigingen behouden blijven).
        self._render_structured()
        ui.notify(
            "JSON toegepast. Sla op om de wijzigingen te bewaren.", type="positive"
        )

    # --- Validatie & opslaan --------------------------------------------------

    def _validate_ensemble(self) -> bool:
        self._ensemble_error.clear()
        errors = config_io.validate_ensemble_weights(
            self._config.get("ensemble_weights", {})
        )
        if errors:
            with self._ensemble_error:
                for err in errors:
                    ui.label(err).classes("text-sm").style("color: #c62828")
        return not errors

    def _sync_dynamic_into_config(self) -> None:
        """Zet de tabelrijen terug in de config-dict vóór opslaan."""
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
        ui.notify("Configuratie opgeslagen.", type="positive")

    def _on_institution_change(self, e) -> None:
        self._config["institution_filter"] = list(e.value or [])
        self._mark_dirty()

    # --- Dirty-tracking -------------------------------------------------------

    #: Eenmalige beforeunload-listener die een vlag checkt. Via addEventListener
    #: (niet window.onbeforeunload=) zodat NiceGUI's eigen handler intact blijft.
    _GUARD_JS = """
        if (!window.__spUnloadGuard) {
            window.__spDirty = false;
            window.__spUnloadGuard = (e) => {
                if (window.__spDirty) { e.preventDefault(); e.returnValue = ''; }
            };
            window.addEventListener('beforeunload', window.__spUnloadGuard);
        }
    """

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            ui.run_javascript(self._GUARD_JS + "window.__spDirty = true;")
        self._status.set_text("Niet-opgeslagen wijzigingen").style("color: #ed6c02")

    def _clear_dirty(self) -> None:
        self._dirty = False
        ui.run_javascript("window.__spDirty = false;")
        self._status.set_text("Opgeslagen").style("color: #2e7d32")
