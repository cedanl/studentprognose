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

from gui import config_io, filtering_io, nav, theme
from gui.components.layout import page_shell
from gui.components.states import empty_state, error_banner, section_title
from gui.state import STATE


def _detect_overlap_years(project_dir: str) -> list[int] | None:
    """Detecteer jaren waarvoor zowel telbestanden als oktoberbestand aanwezig zijn.

    Leest alleen bestandsnamen voor telbestanden en uitsluitend de Collegejaar-kolom
    voor het oktoberbestand, zodat dit snel blijft.
    Geeft None terug als er onvoldoende data aanwezig is.
    """
    from studentprognose.utils.telbestand_filenames import (
        compile_patterns,
        match_telbestand,
    )

    tel_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    tel_years: set[int] = set()
    if os.path.isdir(tel_dir):
        try:
            patterns = compile_patterns(None)
            for fname in os.listdir(tel_dir):
                m = match_telbestand(fname, patterns)
                if m:
                    try:
                        tel_years.add(int(m.group("year")))
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass

    okt_path = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    okt_years: set[int] = set()
    if os.path.isfile(okt_path):
        try:
            import pandas as pd

            today = datetime.date.today()
            valid_lo, valid_hi = today.year - 20, today.year + 2
            df = pd.read_excel(okt_path, usecols=["Collegejaar"], engine="openpyxl")
            for raw in pd.to_numeric(df["Collegejaar"], errors="coerce").dropna().unique():
                y = int(raw)
                if valid_lo <= y <= valid_hi:
                    okt_years.add(y)
        except Exception:
            pass

    if not tel_years and not okt_years:
        return None
    if tel_years and okt_years:
        overlap = sorted(tel_years & okt_years)
        return overlap if overlap else sorted(tel_years)
    return sorted(tel_years or okt_years)


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
        "Beperk de teldata tot je eigen instelling(en) via Brincode. "
        "Leeg = alle instellingen. De meeste gebruikers zetten hier hun "
        "eigen Brincode."
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
        self._filtering_data = filtering_io.load_filtering(STATE.filtering_path)
        self._filtering = self._filtering_data["filtering"]
        self._student_df = self._load_student_count_df()
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

        with ui.row().classes("w-full justify-end mt-6"):
            with ui.element("div").on("click", self._on_next_click):
                self._next_btn = ui.button(
                    "Volgende", icon="arrow_forward"
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

        ui.button("Opslaan", icon="save", on_click=self._save).props("unelevated").classes("mt-3")

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

    def _year_card(self) -> None:
        mc = self._config.setdefault("model_config", {})
        _REC_YEAR = 2022
        _THIS_YEAR = datetime.date.today().year
        _SLIDER_MAX = _THIS_YEAR - 1

        # Bepaal het databereik op basis van aanwezige bestanden.
        try:
            project_dir = os.path.dirname(os.path.dirname(self._path))
            overlap_years = _detect_overlap_years(project_dir)
        except Exception:
            overlap_years = None
        _SLIDER_MIN = min(overlap_years) if overlap_years else 2010

        current_val = int(mc.get("min_training_year", _REC_YEAR))
        current_val = max(_SLIDER_MIN, min(current_val, _SLIDER_MAX))
        state = {"year": current_val}

        def years_back(yr: int) -> int:
            return max(0, _THIS_YEAR - yr)

        def build_presets() -> list[tuple[int, int]]:
            """Geeft lijst van (jaar, jaren-terug) voor de preset-tegels."""
            candidates = [_THIS_YEAR - n for n in (3, 4, 5, 6, 7, 10)]
            pts = [(yr, _THIS_YEAR - yr) for yr in candidates
                   if _SLIDER_MIN <= yr <= _SLIDER_MAX]
            # Voeg "alles" toe als oudste beschikbare data niet al in de lijst zit
            if not pts or _SLIDER_MIN < pts[-1][0]:
                pts.append((_SLIDER_MIN, _THIS_YEAR - _SLIDER_MIN))
            # Voeg huidige waarde toe als die niet in de presets zit
            preset_years = {yr for yr, _ in pts}
            if state["year"] not in preset_years:
                pts.insert(0, (state["year"], _THIS_YEAR - state["year"]))
            return pts

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
                        ui.badge("Essentieel").props("color=orange-8").classes("text-xs px-2")
                    ui.label(
                        "Hoeveel jaar terugkijken voor de training? Meer jaren geeft een "
                        "stabielere trend, maar heel vroege data weerspiegelt de "
                        "huidige situatie minder goed."
                    ).classes("text-sm opacity-60 mt-1")

                    if overlap_years:
                        with ui.row().classes("items-center gap-1.5 mt-1"):
                            ui.icon("folder_open").style(
                                f"color: {theme.INFO}; font-size: 14px;"
                            )
                            ui.label(
                                f"Beschikbare data: {min(overlap_years)}–{max(overlap_years)}"
                            ).classes("text-xs").style(f"color: {theme.INFO}")

                    @ui.refreshable
                    def _year_tiles() -> None:
                        A = theme.ACCENT
                        G = theme.NPULS_GREEN
                        presets = build_presets()
                        with ui.row().classes("gap-2 flex-wrap mt-3"):
                            for yr, yrs in presets:
                                is_sel = yr == state["year"]
                                is_rec = yr == _REC_YEAR
                                border = A if is_sel else ("#e0e0e0" if not is_rec else f"{A}55")
                                bg = f"{A}12" if is_sel else ("white" if not is_rec else f"{A}06")
                                label_col = A if is_sel else ("#555" if not is_rec else A)

                                tile = ui.element("div").style(
                                    f"padding:10px 16px;border-radius:10px;"
                                    f"border:2px solid {border};background:{bg};"
                                    f"cursor:pointer;text-align:center;min-width:72px;"
                                    f"transition:border-color 0.15s,background 0.15s,"
                                    f"box-shadow 0.15s;"
                                    + (f"box-shadow:0 0 0 3px {A}22;" if is_sel else "")
                                )

                                def _pick(y=yr) -> None:
                                    state["year"] = y
                                    mc["min_training_year"] = y
                                    self._mark_dirty()
                                    _year_tiles.refresh()

                                tile.on("click", _pick)
                                with tile:
                                    lbl = (
                                        "Alles"
                                        if yr == _SLIDER_MIN and yrs >= 10
                                        else f"{yrs} jaar"
                                    )
                                    ui.label(lbl).style(
                                        f"font-size:15px;font-weight:{'700' if is_sel else '600'};"
                                        f"color:{label_col};line-height:1.2;"
                                    )
                                    ui.label(f"vanaf {yr}").style(
                                        f"font-size:10px;color:{'#aaa' if not is_sel else A};"
                                        f"margin-top:2px;"
                                    )
                                    if is_rec:
                                        ui.label("★ aanbevolen").style(
                                            f"font-size:9px;color:{A};font-weight:600;"
                                            f"margin-top:3px;letter-spacing:0.03em;"
                                        )

                    _year_tiles()

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
        self._nf_section()
        self._excl_section()
        self._runtime_section()
        ui.button("Opslaan", icon="save", on_click=self._save).props("unelevated").classes("mt-4")

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

    def _filter_programme_options(self) -> list[str]:
        options = set(self._filtering.get("programme", []))
        if self._student_df is not None:
            col = self._config.get("column_roles", {}).get("programme", "Croho groepeernaam")
            if col in self._student_df.columns:
                options |= set(self._student_df[col].dropna().astype(str).unique())
        return sorted(options)

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
                    "Leeg = alle opleidingen. Typ om te zoeken of voer handmatig in."
                ).classes("text-xs opacity-50")
                prog_opts = self._filter_programme_options()
                self._filter_programme_select = (
                    ui.select(
                        options=prog_opts,
                        value=list(self._filtering.get("programme", [])),
                        multiple=True,
                        with_input=True,
                        label="Opleidingen selecteren",
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
        self._filtering["programme"] = list(e.value or [])
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
                ui.label("Directe JSON-bewerking").classes("font-medium text-sm")
                ui.label(
                    "Boom-modus: klik ▶ om secties open/dicht te klappen, "
                    "dubbelklik op een waarde om te bewerken. "
                    "Schakel linksboven naar Code-modus voor vrije tekstbewerking."
                ).classes("text-sm opacity-50")

        with ui.row().classes("items-center gap-2 mb-3"):
            ui.icon("folder_open").style("color: #bbb; font-size: 14px;")
            ui.label(self._path).classes("text-xs font-mono opacity-40 break-all")

        ui.html('<div id="sp-json-ed"></div>')

        self._json_error = ui.label("").classes("text-sm px-1 mt-1").style(
            f"color: {theme.NEGATIVE}"
        )

        with ui.row().classes("gap-2 mt-3 flex-wrap items-center").style(
            "border-top: 1px solid #f0f0f0; padding-top: 12px;"
        ):
            ui.button(
                "Valideer",
                icon="check_circle_outline",
                on_click=self._validate_json_editor,
            ).props("outline dense")
            ui.button(
                "Alles uitklappen",
                icon="unfold_more",
                on_click=lambda: ui.run_javascript(
                    "window.__spJE && window.__spJE.expandAll()"
                ),
            ).props("flat dense")
            ui.button(
                "Alles inklappen",
                icon="unfold_less",
                on_click=lambda: ui.run_javascript(
                    "window.__spJE && window.__spJE.collapseAll()"
                ),
            ).props("flat dense")
            ui.separator().props("vertical").classes("mx-1").style("height: 28px;")
            ui.button(
                "Toepassen",
                icon="check",
                on_click=self._apply_json_editor,
            ).props("unelevated dense")
            ui.button(
                "Opslaan",
                icon="save",
                on_click=self._save_json_editor,
            ).props("unelevated dense color=positive")

        self._run_json_editor_init()

    def _run_json_editor_init(self) -> None:
        config_json = json.dumps(self._config, ensure_ascii=False)
        ui.run_javascript(
            f"""
            (function tryInit(n) {{
                const el = document.getElementById('sp-json-ed');
                if (!el) return;
                if (typeof JSONEditor === 'undefined') {{
                    if (n > 0) {{ setTimeout(() => tryInit(n - 1), 150); return; }}
                    el.innerHTML = '<p style="color:#c0392b;padding:16px;font-size:13px">'
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

    def _refresh_json_text(self) -> None:
        config_json = json.dumps(self._config, ensure_ascii=False)
        ui.run_javascript(
            f"""
            if (window.__spJE) {{
                try {{
                    window.__spJE.set({config_json});
                    window.__spJE.expandAll();
                }} catch (e) {{ console.error('refresh error', e); }}
            }}
            """
        )

    async def _get_editor_json(self) -> str:
        result = await ui.run_javascript(
            "try { return JSON.stringify(window.__spJE.get()); }"
            " catch(e) { return '__ERR__:' + e.message; }",
            timeout=5.0,
        )
        if isinstance(result, str) and result.startswith("__ERR__:"):
            raise ValueError(result[8:])
        return result

    async def _validate_json_editor(self) -> None:
        try:
            json_text = await self._get_editor_json()
            config_io.parse_json(json_text)
            self._json_error.set_text("")
            ui.notify("JSON is geldig.", type="positive", position="top-right")
        except Exception as exc:
            self._json_error.set_text(f"Fout: {exc}")
            ui.notify("Ongeldige JSON — zie de melding hieronder.", type="negative")

    async def _apply_json_editor(self) -> bool:
        try:
            json_text = await self._get_editor_json()
            parsed = config_io.parse_json(json_text)
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_error.set_text(f"Ongeldige JSON: {exc}")
            ui.notify(f"Ongeldige JSON: {exc}", type="negative")
            return False
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
        ui.notify("JSON toegepast. Sla op om de wijzigingen te bewaren.", type="positive")
        return True

    async def _save_json_editor(self) -> None:
        if await self._apply_json_editor():
            self._save()

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
        filter_errors = filtering_io.validate_filtering(self._filtering_data)
        if filter_errors:
            for err in filter_errors:
                ui.notify(err, type="negative")
            return
        try:
            config_io.save_config(self._path, self._config)
            filtering_io.save_filtering(STATE.filtering_path, self._filtering_data)
        except OSError as exc:
            ui.notify(f"Opslaan mislukt: {exc}", type="negative")
            return
        self._clear_dirty()
        STATE.config_saved = True
        ui.notify("Configuratie opgeslagen.", type="positive", position="top")

    def _on_institution_change(self, e) -> None:
        self._config["institution_filter"] = [e.value] if e.value else []
        self._mark_dirty()

    # ─── Dirty-tracking ───────────────────────────────────────────────────────

    def _on_next_click(self) -> None:
        if self._dirty:
            ui.notify(
                "Sla de configuratie eerst op voordat je verder gaat.",
                type="warning",
                position="top",
            )
            return
        ui.navigate.to(nav.next_route("/config"))

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            ui.run_javascript(self._GUARD_JS + "window.__spDirty = true;")
        self._status.set_text("● Niet-opgeslagen wijzigingen").style(
            f"color: {theme.WARNING}"
        )
        self._next_btn.props(add="disabled")

    def _clear_dirty(self) -> None:
        self._dirty = False
        ui.run_javascript("window.__spDirty = false;")
        self._status.set_text("✓ Opgeslagen").style(f"color: {theme.POSITIVE}")
        self._next_btn.props(remove="disabled")
