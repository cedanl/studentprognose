"""Pipeline-runner (#268): stel parameters in en draai de voorspelling.

Bouwt een ``studentprognose``-commando, toont een live preview, vraagt
bevestiging en streamt de uitvoer live. Parameters worden per sessie bewaard.
"""

from __future__ import annotations

import datetime

from nicegui import app, ui

from gui import nav, process, theme, tracks
from gui.components import train_test_viz as tvz
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.progress_card import ProgressCard
from gui.components.states import empty_state, section_title
from gui.state import STATE

#: Sleutel waaronder de laatst gebruikte parameters bewaard worden.
_STORAGE_KEY = "run_settings"

# Jaarbereik voor prognosejaren: DATA_START is het eerste trainingsjaar
# (gedefinieerd in de vizmodule). Een geldig prognosejaar heeft ten minste
# één trainingsjaar nodig, dus de ondergrens is DATA_START+1.
_DATA_START: int = tvz.DATA_START
_CURRENT_YEAR: int = datetime.date.today().year
_FORECAST_YEARS: list[int] = list(range(_DATA_START + 1, _CURRENT_YEAR + 4))

def _default_week() -> str:
    return str(datetime.date.today().isocalendar()[1])


_DEFAULTS: dict = {
    "dataset": "Beide",
    "cohort": "Eerstejaars",
    "years": [_CURRENT_YEAR],
    "weeks": "38",
    "institutions": [],
    "skip_years": 0,
    "noetl": False,
    "dashboard": False,
    "no_warnings": False,
    "yes": True,
}

#: Aanbevolen waarden — worden visueel gemarkeerd in de UI.
_RECOMMENDED = {"weeks": "38"}

_WIZARD_MODE_TO_DATASET = {
    "cumulative": "Cumulatief",
    "individual": "Individueel",
    "both": "Beide",
}


# ---------------------------------------------------------------------------
# Validatiehulpfuncties (pure logica, geen NiceGUI)
# ---------------------------------------------------------------------------


def _validate_weeks(raw: str) -> str | None:
    """Valideer weekinvoer; geeft ``None`` terug als geldig.

    Accepteert: leeg (= alle weken), enkelvoudig getal (bijv. ``6``),
    meerdere getallen (``6 10 20``) en bereiksyntaxis (``1:38``).
    """
    if not raw.strip():
        return None
    for token in raw.strip().split():
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2:
                return f"Ongeldig bereik '{token}' — gebruik bijv. 1:38"
            try:
                n, m = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return f"Ongeldig bereik '{token}' — gebruik bijv. 1:38"
            if not (1 <= n <= 52 and 1 <= m <= 52):
                return f"Weekbereik '{token}': waarden moeten tussen 1 en 52 liggen"
            if n > m:
                return (
                    f"Weekbereik '{token}': begin ({n}) mag niet groter zijn dan einde ({m})"
                )
        else:
            try:
                w = int(token)
            except ValueError:
                return f"'{token}' is geen geldig weeknummer"
            if not (1 <= w <= 52):
                return f"Weeknummer {w} ligt buiten het geldige bereik (1–52)"
    return None


def _max_skip(min_forecast_year: int) -> int:
    """Bereken het maximum aantal te overslaan jaren voor backtesting.

    Minstens één trainingsjaar (``DATA_START``) moet overblijven.
    """
    return max(0, min_forecast_year - _DATA_START - 1)


def _parse_year_range(raw: str, valid: list[int]) -> list[int]:
    """Zet bereiknotatie (``2023:2025``) of losse jaren (``2023 2024``) om.

    Accepteert: ``2023:2025`` (bereik), ``2023 2024 2025`` (los) en
    combinaties. Retourneert alleen jaren die in ``valid`` staan.
    """
    tokens = raw.strip().replace(":", " ").split()
    result: list[int] = []
    i = 0
    while i < len(tokens):
        try:
            val = int(tokens[i])
        except ValueError:
            i += 1
            continue
        # Kijk of het volgende token ook een jaar is en het vorige een ':'
        # bevatte (bereik-detectie via ruwe string).
        result.append(val)
        i += 1

    # Her-evalueer als bereik als de ruwe string een ':' bevat.
    if ":" in raw:
        parts = raw.strip().split(":")
        if len(parts) == 2:
            try:
                start, end = int(parts[0].strip()), int(parts[1].strip())
                result = list(range(start, end + 1))
            except ValueError:
                pass

    return sorted({y for y in result if y in valid})


# ---------------------------------------------------------------------------
# Route-registratie
# ---------------------------------------------------------------------------


def create() -> None:
    """Registreer de route ``/run``."""
    nav.register_route("/run")

    @ui.page("/run")
    def run_page() -> None:
        with page_shell(active="/run", title="Uitvoeren"):
            section_title(
                "Uitvoeren", "Stel de voorspelling in en volg de voortgang live."
            )
            if not STATE.is_initialised:
                empty_state(
                    icon="folder_off",
                    title="Nog geen project",
                    message="Kies eerst een project voordat je de pipeline draait.",
                    action_label="Project opzetten",
                    on_action=lambda: ui.navigate.to("/wizard"),
                )
                return
            _RunView()


# ---------------------------------------------------------------------------
# Parameterformulier + runner
# ---------------------------------------------------------------------------


class _RunView:
    """Rendert het parameterformulier, de preview en de runner."""

    def __init__(self) -> None:
        stored = dict(app.storage.general.get(_STORAGE_KEY, {}))

        # Achterwaartse compatibiliteit: years was vroeger een string.
        raw_years = stored.get("years", _DEFAULTS["years"])
        if isinstance(raw_years, str):
            parsed = [int(x) for x in raw_years.strip().split() if x.isdigit()]
            raw_years = parsed if parsed else list(_DEFAULTS["years"])
        # Filter opgeslagen jaren tot het geldige bereik.
        stored["years"] = [y for y in raw_years if y in _FORECAST_YEARS] or list(
            _DEFAULTS["years"]
        )

        self._settings = {**_DEFAULTS, **stored}

        # Gebruik huidige weeknummer als standaard wanneer er nog niets is opgeslagen.
        if "weeks" not in stored:
            self._settings["weeks"] = _default_week()

        # Als er nog geen opgeslagen instellingen zijn maar de wizard heeft een
        # modus gekozen, gebruik die als standaard voor de dataset-dropdown.
        if not app.storage.general.get(_STORAGE_KEY):
            wizard_mode = getattr(STATE, "wizard_mode", None)
            if wizard_mode:
                self._settings["dataset"] = _WIZARD_MODE_TO_DATASET.get(
                    wizard_mode, _DEFAULTS["dataset"]
                )

        self._build()

    # ── UI-opbouw ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        with ui.card().classes("w-full"):
            with ui.grid(columns=2).classes("w-full gap-4"):

                # ── Dataset ─────────────────────────────────────────────────
                self._dataset = ui.select(
                    ["Cumulatief", "Individueel", "Beide"],
                    value=self._settings["dataset"],
                    label="Dataset (voorspelspoor)",
                ).classes("w-full")
                self._dataset.tooltip(tracks.dataset_tooltip())
                self._dataset.on_value_change(
                    lambda _e: (self._update_dataset_hint(), self._update_preview())
                )

                # ── Cohort ──────────────────────────────────────────────────
                self._cohort = ui.select(
                    ["Eerstejaars", "Hogerejaars", "Volume"],
                    value=self._settings["cohort"],
                    label="Cohort",
                ).classes("w-full")
                self._cohort.on_value_change(lambda _e: self._update_preview())

                # ── Prognosejaren (dropdown + bereik-helper) ─────────────────
                with ui.column().classes("w-full gap-0"):
                    self._years = (
                        ui.select(
                            options=_FORECAST_YEARS,
                            value=list(self._settings["years"]),
                            multiple=True,
                            label="Prognosejaren",
                        )
                        .props("use-chips")
                        .classes("w-full")
                    )
                    self._years.tooltip(
                        "Het academisch jaar waarvoor de prognose wordt gemaakt. "
                        "Meerdere jaren tegelijk zijn mogelijk."
                    )
                    self._years_error = (
                        ui.label("")
                        .classes("text-xs mt-0.5 font-medium")
                        .style(f"color: {theme.NEGATIVE}")
                    )
                    self._years_error.set_visibility(False)
                    self._years.on_value_change(lambda _e: self._on_years_change())

                    # Bereik-invoer — bijv. 2023:2025 of 2023 2024 2025
                    with ui.row().classes("items-center gap-2 mt-1 no-wrap"):
                        ui.label("of bereik:").classes("text-xs flex-none").style(
                            f"color:{theme.MUTED};"
                        )
                        self._year_range_input = (
                            ui.input(placeholder="2023:2025")
                            .props("dense outlined clearable")
                            .style("width:100px; font-size:13px;")
                        )
                        self._year_range_input.tooltip(
                            "Voer een bereik in (bijv. 2023:2025) of losse jaren "
                            "(bijv. 2023 2024) en druk op Enter of klik ➕."
                        )
                        self._year_range_add_btn = ui.button(
                            "", icon="add",
                            on_click=self._add_year_range,
                        ).props("unelevated dense size=sm color=primary")
                        self._year_range_add_btn.tooltip("Voeg bereik toe")
                        self._year_range_input.on("keydown.enter", self._add_year_range)

                # ── Weken (tekstveld + inline validatie + aanbevolen-hint) ──
                with ui.column().classes("w-full gap-0"):
                    with ui.row().classes("items-center gap-2 mb-0.5"):
                        pass  # spacer — label zit in het input-widget zelf
                    self._weeks = ui.input(
                        "Weken",
                        value=self._settings["weeks"],
                        placeholder="bijv. 6 of 1:38  (leeg = alle weken)",
                    ).classes("w-full")
                    self._weeks.tooltip(
                        "Weeknummer van de aanmeldpeildatum (1–52). "
                        "Gebruik bereiknotatie als 1:38 voor meerdere weken."
                    )
                    # Aanbevolen-hint (altijd zichtbaar als informatielabel)
                    with ui.row().classes("items-center gap-1 mt-0.5"):
                        ui.icon("star").classes("text-xs flex-none").style(
                            f"color: {theme.ACCENT}; font-size: 12px;"
                        )
                        self._weeks_rec_label = ui.label(
                            f"Aanbevolen: week {_RECOMMENDED['weeks']}"
                        ).classes("text-xs font-medium").style(
                            f"color: {theme.ACCENT}; opacity: 0.85"
                        )
                    self._weeks_error = (
                        ui.label("")
                        .classes("text-xs mt-0.5 font-medium")
                        .style(f"color: {theme.NEGATIVE}")
                    )
                    self._weeks_error.set_visibility(False)
                    self._weeks.on_value_change(lambda _e: self._on_weeks_change())

                # ── Instellingsfilter ────────────────────────────────────────
                self._institutions = (
                    ui.select(
                        options=list(self._settings["institutions"]),
                        value=list(self._settings["institutions"]),
                        multiple=True,
                        with_input=True,
                        label="Instellingsfilter (leeg = alle)",
                    )
                    .props("use-chips new-value-mode=add-unique")
                    .classes("w-full")
                )
                self._institutions.on_value_change(lambda _e: self._update_preview())

                # ── Jaren overslaan — max afhankelijk van jarenselectie ──────
                with ui.column().classes("w-full gap-0"):
                    self._skip_years = ui.number(
                        "Jaren overslaan (backtesting)",
                        value=self._settings["skip_years"],
                        min=0,
                    ).classes("w-full")
                    self._skip_years.tooltip(
                        "Aantal jaren vóór het prognosejaar dat als testset wordt "
                        "achtergehouden voor backtesting. 0 = geen backtest."
                    )
                    self._skip_hint = ui.label("").classes("text-xs opacity-60 mt-0.5")
                    self._skip_years.on_value_change(lambda _e: self._update_preview())

            # ── Dataset-hint ─────────────────────────────────────────────────
            with ui.row().classes("items-center gap-2 w-full"):
                self._dataset_hint_icon = (
                    ui.icon("info").classes("text-sm").style(f"color: {theme.ACCENT}")
                )
                self._dataset_hint = ui.label("").classes("text-sm opacity-80")
            self._update_dataset_hint()

            # ── Checkboxen ──────────────────────────────────────────────────
            with ui.row().classes("w-full gap-6 mt-2"):
                self._noetl = ui.checkbox(
                    "ETL overslaan", value=self._settings["noetl"]
                )
                self._noetl.tooltip(
                    "Sla ETL én validatie over (--noetl). "
                    "Gebruik dit alleen als de data al eerder is verwerkt."
                )
                self._dashboard = ui.checkbox(
                    "Dashboards genereren",
                    value=self._settings["dashboard"],
                )
                self._dashboard.tooltip(
                    "Genereer interactieve Plotly-dashboards in data/output/visualisations/ (--dashboard)."
                )
                self._no_warnings = ui.checkbox(
                    "Waarschuwingen onderdrukken", value=self._settings["no_warnings"]
                )
                self._no_warnings.tooltip(
                    "Onderdruk UserWarning-meldingen over historisch realisme en ontbrekende "
                    "lag-fallback (--no-warnings). Gebruik dit als de warnings bekend zijn."
                )
                self._yes = ui.checkbox(
                    "Validatieprompt overslaan", value=self._settings["yes"]
                )
                self._yes.tooltip(
                    "Sla de interactieve validatieprompt over (--yes). "
                    "Aanbevolen aan: een GUI-run heeft geen interactieve invoer."
                )
                for cb in (self._noetl, self._dashboard, self._no_warnings, self._yes):
                    cb.on_value_change(lambda _e: self._update_preview())

        # ── Dataverdeling-visualisatie ────────────────────────────────────────
        with ui.card().classes("w-full pr-16"):
            section_title("Dataverdeling", "Traindata · backtest · prognose")
            self._viz_html = ui.html("").classes("w-full")

        # ── Live command-preview ──────────────────────────────────────────────
        with ui.card().classes("w-full bg-grey-2"):
            ui.label("Commando").classes("text-xs uppercase opacity-60")
            self._preview = ui.label("").classes("font-mono text-sm break-all")

        with ui.row().classes("items-center gap-3"):
            self._start_btn = ui.button(
                "Start voorspelling", icon="play_arrow", on_click=self._confirm
            ).props("unelevated")
            self._result_slot = ui.row().classes("items-center gap-2")

        self._progress = ProgressCard()
        with ui.column().classes("w-full") as self._panel_container:
            self._panel = ProcessPanel()
        self._panel_container.set_visibility(False)

        # Initialiseer skip-hint, weeks-validatie en preview op basis van
        # opgeslagen waarden.
        self._on_weeks_change()
        self._on_years_change()  # roept ook _update_preview() aan

    # ── Parameterwijzigingen ──────────────────────────────────────────────────

    def _on_years_change(self) -> None:
        """Ververs skip-max, inline feedback en preview bij jarenselectie."""
        selected = sorted(self._years.value or [])

        if not selected:
            self._years_error.set_text("Selecteer minstens één prognosejaar.")
            self._years_error.set_visibility(True)
            self._skip_years.props("max=0")
            self._skip_years.set_value(0)
            self._skip_hint.set_text("")
        else:
            self._years_error.set_visibility(False)
            mx = _max_skip(min(selected))
            # Pas de max-prop aan zodat het veld zelf ook klaagt bij overschrijding.
            self._skip_years.props(f"max={mx}")
            # Klem de huidige waarde als die nu buiten het geldige bereik valt.
            cur = int(self._skip_years.value or 0)
            if cur > mx:
                self._skip_years.set_value(mx)
            # Informatieve hint over het backtesting-bereik.
            if mx == 0:
                self._skip_hint.set_text(
                    f"Backtest niet mogelijk — prognose {min(selected)} "
                    f"ligt direct na traindata ({_DATA_START})."
                )
            else:
                self._skip_hint.set_text(
                    f"Max. {mx} jaar  ·  traindata {_DATA_START}–{min(selected) - 1}."
                )

        self._update_preview()

    def _on_weeks_change(self) -> None:
        """Valideer weekinvoer, toon inline foutmelding en pas aanbevolen-hint aan."""
        val = (self._weeks.value or "").strip()
        err = _validate_weeks(val)
        if err:
            self._weeks_error.set_text(err)
            self._weeks_error.set_visibility(True)
        else:
            self._weeks_error.set_visibility(False)
        # Toon hint opvallender (groen) als waarde overeenkomt met aanbeveling.
        if val == _RECOMMENDED["weeks"]:
            self._weeks_rec_label.style(
                f"color: {theme.ACCENT}; opacity: 0.85; font-weight: 600;"
            )
        else:
            self._weeks_rec_label.style(
                f"color: {theme.ACCENT}; opacity: 0.45;"
            )
        self._update_preview()

    def _add_year_range(self, _e=None) -> None:
        """Voeg jaar(bereik) toe aan de jarenselectie vanuit het invoerveld."""
        raw = (self._year_range_input.value or "").strip()
        if not raw:
            return
        to_add = _parse_year_range(raw, _FORECAST_YEARS)
        if not to_add:
            return
        current = list(self._years.value or [])
        for y in to_add:
            if y not in current:
                current.append(y)
        self._years.set_value(sorted(current))
        self._year_range_input.set_value("")
        self._on_years_change()

    # ── Preview + visualisatie ────────────────────────────────────────────────

    def _years_as_str(self) -> str:
        """Geef de geselecteerde jaren terug als spatie-gescheiden string."""
        return " ".join(str(y) for y in sorted(self._years.value or []))

    def _current_args(self) -> list[str]:
        return process.build_run_args(
            dataset=self._dataset.value,
            cohort=self._cohort.value,
            years=self._years_as_str(),
            weeks=self._weeks.value or "",
            institutions=list(self._institutions.value or []),
            skip_years=int(self._skip_years.value or 0),
            noetl=self._noetl.value,
            dashboard=self._dashboard.value,
            no_warnings=self._no_warnings.value,
            yes=self._yes.value,
        )

    def _update_dataset_hint(self) -> None:
        t = tracks.track(self._dataset.value)
        if t is None:
            self._dataset_hint.set_text("")
            return
        suffix = " (aanbevolen)" if t.label == tracks.RECOMMENDED else ""
        self._dataset_hint.set_text(f"{t.label} — {t.short}{suffix}")

    def _update_preview(self) -> None:
        self._preview.set_text(process.preview_command(self._current_args()))
        self._update_viz()

    def _update_viz(self) -> None:
        skip = int(self._skip_years.value or 0)
        weeks = self._weeks.value or ""
        self._viz_html.set_content(tvz.render_v1(self._years_as_str(), skip, weeks))

    # ── Validatie ────────────────────────────────────────────────────────────

    def _validate_params(self) -> list[str]:
        """Valideer alle parameters; leeg = alles geldig."""
        errors: list[str] = []

        selected = sorted(self._years.value or [])
        if not selected:
            errors.append("Selecteer minstens één prognosejaar.")
        else:
            skip = int(self._skip_years.value or 0)
            mx = _max_skip(min(selected))
            if skip > mx:
                errors.append(
                    f"Jaren overslaan ({skip}) is te groot voor prognosejaar"
                    f" {min(selected)}. Maximum is {mx}"
                    f" (traindata begint in {_DATA_START})."
                )

        weeks_err = _validate_weeks(self._weeks.value or "")
        if weeks_err:
            errors.append(f"Weken: {weeks_err}")

        return errors

    # ── Persistentie ─────────────────────────────────────────────────────────

    def _persist(self) -> None:
        app.storage.general[_STORAGE_KEY] = {
            "dataset": self._dataset.value,
            "cohort": self._cohort.value,
            "years": list(self._years.value or []),
            "weeks": self._weeks.value or "",
            "institutions": list(self._institutions.value or []),
            "skip_years": int(self._skip_years.value or 0),
            "noetl": self._noetl.value,
            "dashboard": self._dashboard.value,
            "no_warnings": self._no_warnings.value,
            "yes": self._yes.value,
        }

    # ── Starten ──────────────────────────────────────────────────────────────

    def _confirm(self) -> None:
        # Valideer eerst; toon foutpopup bij ongeldige invoer.
        errors = self._validate_params()
        if errors:
            with ui.dialog() as err_dialog, ui.card().classes("max-w-lg w-full"):
                with ui.row().classes("items-center gap-2 mb-2"):
                    ui.icon("error_outline").props("color=negative size=sm")
                    ui.label("Ongeldige parameterinvoer").classes(
                        "text-base font-semibold"
                    )
                with ui.column().classes("w-full gap-2 my-1"):
                    for msg in errors:
                        with ui.row().classes("items-start gap-2 no-wrap"):
                            ui.icon("chevron_right").classes(
                                "text-sm flex-none mt-0.5"
                            ).style(f"color: {theme.NEGATIVE}")
                            ui.label(msg).classes("text-sm leading-snug")
                ui.separator().classes("my-1")
                with ui.row().classes("w-full justify-end"):
                    ui.button("Aanpassen", on_click=err_dialog.close).props(
                        "unelevated"
                    )
            err_dialog.open()
            return

        command = process.preview_command(self._current_args())
        with ui.dialog() as dialog, ui.card():
            ui.label("Deze voorspelling uitvoeren?").classes("text-lg font-medium")
            ui.label(command).classes("font-mono text-sm break-all")

            async def _run_confirmed() -> None:
                # Sluit de dialoog en start de run; als async handler zodat de
                # coroutine daadwerkelijk geawait wordt (een lambda zou 'm droppen).
                dialog.close()
                await self._start()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Annuleren", on_click=dialog.close).props("flat")
                ui.button("Uitvoeren", on_click=_run_confirmed).props("unelevated")
        dialog.open()

    async def _start(self) -> None:
        self._persist()
        self._result_slot.clear()
        self._start_btn.props("loading")
        self._panel_container.set_visibility(True)
        self._progress.start()
        try:
            args = self._current_args()
            returncode = await self._panel.run(
                args,
                cwd=STATE.project_dir,
                on_line=self._progress.on_line,
            )
            self._progress.complete(success=returncode == 0)
            if returncode == 0:
                self._show_results_link()
        finally:
            self._start_btn.props(remove="loading")

    def _show_results_link(self) -> None:
        with self._result_slot:
            ui.icon("check_circle").props("color=positive")
            if nav.is_available("/output"):
                ui.button(
                    "Bekijk resultaten",
                    icon="insights",
                    on_click=lambda: ui.navigate.to("/output"),
                ).props("unelevated")
            else:
                ui.label("Voorspelling klaar.").classes("text-positive")
