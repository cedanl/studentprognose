"""Resultatenoverzicht (#269).

Toont na een run een gelikt overzicht: KPI-tiles, een op afwijking gesorteerde
opleidingentabel met kleurcodering (klikbaar voor een detailgrafiek), een
model-vergelijking en een audittrail-diff met de vorige run.
"""

from __future__ import annotations

import datetime
import os

import plotly.graph_objects as go
from nicegui import ui

from gui import nav, results_io, run_history
from gui.components.layout import page_shell
from gui.components.states import empty_state, error_banner, section_title
from gui.state import STATE
from gui.theme import ACCENT, INFO, NEGATIVE, POSITIVE, WARNING

_BUCKET_COLOR = {
    "positive": POSITIVE,
    "warning": WARNING,
    "negative": NEGATIVE,
    "grey": "#9e9e9e",
}


def create() -> None:
    """Registreer de route ``/output``."""
    nav.register_route("/output")

    @ui.page("/output")
    def output_page() -> None:
        with page_shell(active="/output", title="Resultaten"):
            section_title("Resultaten", "Overzicht van de laatste voorspelling.")
            _RunHistoryView()


class _RunHistoryView:
    """Run-selector + resultatenweergave.

    Bouwt een dropdown van alle beschikbare runs (huidige + historisch) en
    rendert de bijbehorende :class:`_ResultsView` in een wisselbaar container.
    """

    def __init__(self) -> None:
        # Huidige project-run
        current_files: list[tuple[str, str]] = []
        if STATE.is_initialised and STATE.output_dir:
            current_files = results_io.find_output_files(STATE.output_dir)

        # Historische runs uit tmp/
        history = run_history.find_historical_runs(
            exclude_dir=STATE.project_dir if STATE.is_initialised else None
        )

        # Bouw opties: {output_dir: label}
        options: dict[str, str] = {}
        if current_files and STATE.output_dir:
            name = os.path.basename(STATE.project_dir or "huidig")
            options[STATE.output_dir] = f"Huidige run — {name}"
        for entry in history:
            options[entry.output_dir] = entry.label

        if not options:
            self._show_empty()
            return

        first_dir = next(iter(options))
        self._current_output_dir = STATE.output_dir if STATE.is_initialised else None

        # Dropdown bovenaan — DOM-volgorde: selector → banner → resultaten.
        if len(options) > 1:
            with ui.card().classes("w-full").style("background:#f8f8f8"):
                with ui.row().classes("items-center gap-3 w-full no-wrap"):
                    ui.icon("history").classes("text-xl").style(f"color:{ACCENT}")
                    sel = ui.select(
                        options,
                        value=first_dir,
                        label="Run bekijken",
                        on_change=lambda e: self._load(e.value),
                    ).classes("flex-1")
                    sel.tooltip(
                        "Kies een eerdere run om de resultaten ervan te bekijken. "
                        "De actieve projectmap wijzigt niet."
                    )

        self._banner_slot = ui.row().classes("w-full")
        self._area = ui.column().classes("w-full gap-4")
        self._load(first_dir)

    def _load(self, output_dir: str) -> None:
        self._banner_slot.clear()
        self._area.clear()

        is_historical = self._current_output_dir is None or (
            os.path.realpath(output_dir)
            != os.path.realpath(self._current_output_dir)
        )
        if is_historical:
            with self._banner_slot:
                with ui.row().classes("items-center gap-2 py-1"):
                    ui.icon("history").classes("text-sm").style("color:#888")
                    ui.label("Historische run — niet het actieve project.").classes(
                        "text-sm opacity-60"
                    )

        with self._area:
            files = results_io.find_output_files(output_dir)
            if files:
                _ResultsView(files)
            else:
                error_banner(
                    "Geen outputbestanden gevonden voor deze run.",
                    "De run is mogelijk niet succesvol afgerond.",
                )

    def _show_empty(self) -> None:
        if not STATE.is_initialised:
            empty_state(
                icon="folder_off",
                title="Nog geen project",
                message="Kies eerst een project voordat je resultaten bekijkt.",
                action_label="Project opzetten",
                on_action=lambda: ui.navigate.to("/wizard"),
            )
        else:
            with ui.column().classes("w-full items-center text-center gap-3 py-12"):
                ui.icon("query_stats").classes("text-6xl opacity-40")
                ui.label("Nog geen resultaten").classes("text-xl font-medium")
                ui.label(
                    "Nog geen voorspelling uitgevoerd. Ga naar Uitvoeren om de "
                    "pipeline te starten. Na afloop verschijnen hier de resultaten "
                    "per opleiding."
                ).classes("text-sm opacity-70 max-w-md")
                ui.button(
                    "Naar Uitvoeren",
                    on_click=lambda: ui.navigate.to("/run"),
                ).props("unelevated")
                ui.button(
                    "Bekijk demo-resultaten",
                    icon="science",
                    on_click=lambda: ui.navigate.to("/"),
                ).props("flat")


class _ResultsView:
    """Rendert het resultatenoverzicht voor een gekozen outputbestand."""

    def __init__(self, files: list[tuple[str, str]]) -> None:
        self._files = files
        self._path = files[0][1]
        self._build()

    def _build(self) -> None:
        # Bestandskeuze (bij meerdere outputs).
        if len(self._files) > 1:
            ui.select(
                {path: label for label, path in self._files},
                value=self._path,
                label="Outputbestand",
                on_change=self._on_file_change,
            ).classes("w-full max-w-md")

        # Het overzicht rendert in een eigen container zodat een andere bestands-
        # keuze in-place kan herladen zonder de pagina (en de selectie) te resetten.
        self._body = ui.column().classes("w-full")
        self._render_body()

    def _render_body(self) -> None:
        self._body.clear()
        with self._body:
            try:
                df = results_io.load_output(self._path)
            except (OSError, ValueError, KeyError) as exc:
                error_banner(
                    "Het resultaatbestand kon niet worden gelezen.",
                    f"Bestand: {self._path}. Details: {exc}",
                )
                return

            self._render_kpis(df)
            self._render_model_comparison(df)
            self._render_programme_table(df)
            with ui.row().classes("items-center gap-2"):
                self._render_audit_button()
                self._render_share_button()

    def _on_file_change(self, e) -> None:
        # Herlaad het overzicht voor het gekozen bestand, mét behoud van de keuze.
        self._path = e.value
        self._render_body()

    # --- KPI-tiles ------------------------------------------------------------

    def _render_kpis(self, df) -> None:
        kpis = results_io.compute_kpis(df)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(self._path))
        with ui.row().classes("w-full gap-4 no-wrap"):
            self._kpi("Opleidingen", str(kpis["n_programmes"]), "school")
            best = kpis["best_model"]
            mape = kpis["best_mape"]
            self._kpi(
                "Beste model (MAPE)",
                f"{mape * 100:.1f}%" if mape is not None else "—",
                "emoji_events",
                subtitle=best or "",
            )
            self._kpi(
                "Voorspelling",
                f"{kpis['year']} · wk {kpis['week']}" if kpis["year"] else "—",
                "event",
            )
            self._kpi(
                "Laatst gegenereerd",
                mtime.strftime("%d-%m %H:%M"),
                "schedule",
            )

    def _kpi(self, label, value, icon, subtitle="") -> None:
        with ui.card().classes("grow items-start"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon(icon).classes("text-2xl").style(f"color: {INFO}")
                ui.label(label).classes("text-sm opacity-60")
            ui.label(value).classes("text-2xl font-bold")
            if subtitle:
                ui.label(subtitle).classes("text-xs opacity-60")

    # --- Model-vergelijking ---------------------------------------------------

    def _render_model_comparison(self, df) -> None:
        rows = results_io.prediction_rows(df)
        comparison = results_io.model_comparison(rows)
        if len(comparison) < 2:
            return
        with ui.card().classes("w-full"):
            ui.label("Model-vergelijking (gemiddelde MAE, lager = beter)").classes(
                "font-medium"
            )
            names = list(comparison.keys())
            values = [comparison[n] for n in names]
            best_idx = values.index(min(values))
            colors = [POSITIVE if i == best_idx else INFO for i in range(len(names))]
            fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                yaxis_title="Gemiddelde MAE",
            )
            ui.plotly(fig).classes("w-full")

    # --- Opleidingentabel -----------------------------------------------------

    def _render_programme_table(self, df) -> None:
        records = results_io.programme_rows(df)
        with ui.card().classes("w-full"):
            ui.label(
                "Opleidingen — gesorteerd op absolute afwijking (grootste eerst)"
            ).classes("font-medium")
            ui.label("Klik op een rij voor een detailgrafiek.").classes(
                "text-sm opacity-60"
            )
            columns = [
                {
                    "name": "opleiding",
                    "label": "Opleiding",
                    "field": "opleiding",
                    "align": "left",
                    "sortable": True,
                },
                {
                    "name": "herkomst",
                    "label": "Herkomst",
                    "field": "herkomst",
                    "sortable": True,
                },
                {
                    "name": "examentype",
                    "label": "Examentype",
                    "field": "examentype",
                    "sortable": True,
                },
                {
                    "name": "voorspelling",
                    "label": "Voorspelling",
                    "field": "voorspelling",
                    "sortable": True,
                },
                {
                    "name": "werkelijk",
                    "label": "Werkelijk",
                    "field": "werkelijk",
                    "sortable": True,
                },
                {
                    "name": "afwijking",
                    "label": "Afwijking",
                    "field": "afwijking",
                    "sortable": True,
                },
                {"name": "mape", "label": "MAPE", "field": "mape", "sortable": True},
            ]
            table = ui.table(
                columns=columns,
                rows=records,
                row_key="opleiding",
                pagination=15,
            ).classes("w-full")
            # Kleurgecodeerde MAPE-cel.
            table.add_slot(
                "body-cell-mape",
                r"""
                <q-td :props="props">
                    <q-badge :color="props.row.kleur" :label="props.value === null ? '—' : (props.value*100).toFixed(1)+'%'" />
                </q-td>
                """,
            )
            table.on("rowClick", self._on_row_click)
            self._records_by_key = {
                (r["opleiding"], r["herkomst"], r["examentype"]): r for r in records
            }

    def _on_row_click(self, e) -> None:
        # e.args = [evt, row, index]
        row = e.args[1] if len(e.args) > 1 else None
        if not row:
            return
        self._show_detail(row)

    def _show_detail(self, row) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-[32rem] max-w-full"):
            ui.label(
                f"{row['opleiding']} · {row['herkomst']} · {row['examentype']}"
            ).classes("text-lg font-medium")
            labels, values, colors = [], [], []
            if row.get("voorspelling") is not None:
                labels.append("Voorspelling")
                values.append(row["voorspelling"])
                colors.append(ACCENT)
            if row.get("werkelijk") is not None:
                labels.append("Werkelijk")
                values.append(row["werkelijk"])
                colors.append(POSITIVE)
            if labels:
                fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260)
                ui.plotly(fig).classes("w-full")
            else:
                ui.label("Geen numerieke waarden om te tonen.").classes(
                    "text-sm opacity-60"
                )
            if row.get("mape") is not None:
                ui.label(f"MAPE: {row['mape'] * 100:.1f}%").style(
                    f"color: {_BUCKET_COLOR.get(row['kleur'], '#000')}"
                )
            with ui.row().classes("w-full justify-end"):
                ui.button("Sluiten", on_click=dialog.close).props("flat")
        dialog.open()

    # --- Audittrail-diff ------------------------------------------------------

    def _render_audit_button(self) -> None:
        ui.button(
            "Vergelijk met vorige run",
            icon="compare_arrows",
            on_click=self._show_audit,
        ).props("outline")

    # --- Delen (placeholder) --------------------------------------------------

    #: Placeholder-ontvanger voor het delen van resultaten.
    _SHARE_RECIPIENT = "ceda@surf.nl"

    def _render_share_button(self) -> None:
        # Accent-oranje (npuls) knop — de CEDA-actie krijgt een eigen identiteit
        # naast de neutrale "Vergelijk met vorige run".
        ui.button(
            "Delen met CEDA",
            icon="share",
            on_click=self._show_share,
        ).props("outline color=accent")

    def _show_share(self) -> None:
        """Dummy: nog niet functioneel — verstuurt niets (#275)."""
        with ui.dialog() as dialog, ui.card().classes("w-[30rem] max-w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("share").style(f"color: {ACCENT}").classes("text-2xl")
                ui.label("Delen met CEDA").classes("text-lg font-medium")
            ui.label(
                f"Deze functie deelt de resultaten binnenkort met "
                f"{self._SHARE_RECIPIENT}. Ze is nog niet actief — er wordt nu "
                f"niets verzonden."
            ).classes("text-sm opacity-80")
            ui.badge("Binnenkort beschikbaar", color="warning")
            with ui.row().classes("w-full justify-end"):
                ui.button("Sluiten", on_click=dialog.close).props("flat")
        dialog.open()

    def _show_audit(self) -> None:
        totaal_path = self._totaal_path()
        if totaal_path is None or not os.path.isfile(totaal_path):
            ui.notify("Geen audittrail gevonden.", type="warning")
            return
        try:
            totaal = results_io.load_output(totaal_path)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Audittrail niet leesbaar: {exc}", type="negative")
            return
        diff = results_io.audit_diff(totaal)
        with ui.dialog() as dialog, ui.card().classes("w-[36rem] max-w-full"):
            ui.label("Vergelijking met vorige run").classes("text-lg font-medium")
            if not diff.get("available"):
                ui.label(
                    "Er is nog geen tweede run om mee te vergelijken. Draai de "
                    "pipeline nog eens om verschillen te zien."
                ).classes("text-sm opacity-70")
            else:
                ui.label(f"Nieuwe opleidingen: {len(diff['new_programmes'])}").classes(
                    "font-medium"
                )
                for p in diff["new_programmes"][:20]:
                    ui.label(f"• {p}").classes("text-sm")
                ui.separator()
                ui.label(f"Grote verschuivingen (≥10%): {len(diff['shifts'])}").classes(
                    "font-medium"
                )
                for s in diff["shifts"][:20]:
                    ui.label(
                        f"• {s['opleiding']}: {s['oud']} → {s['nieuw']} "
                        f"({s['verschil'] * 100:+.0f}%)"
                    ).classes("text-sm")
            with ui.row().classes("w-full justify-end"):
                ui.button("Sluiten", on_click=dialog.close).props("flat")
        dialog.open()

    def _totaal_path(self) -> str | None:
        """Leid het bijbehorende ``_totaal``-auditbestand af van het outputpad.

        Geeft ``None`` als het bestand niet de verwachte ``output_``-prefix heeft,
        zodat een onverwachte naam geen misvormd pad oplevert.
        """
        name = os.path.basename(self._path)  # output_first-years_cumulatief.xlsx
        prefix = "output_"
        if not name.startswith(prefix):
            return None
        totaal = "_totaal_" + name[len(prefix) :]
        return os.path.join(os.path.dirname(self._path), totaal)
