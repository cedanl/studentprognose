"""Init-wizard (#266, #data-upload): zet een nieuw project op en upload inputdata.

Vier sub-stappen:
  1. Projectmap kiezen
  2. Bevestigen wat aangemaakt wordt
  3. Aanmaken + optioneel demodata downloaden
  4. Modus kiezen + bijbehorende bestanden uploaden en valideren
"""

from __future__ import annotations

import asyncio
import datetime
import os
import tempfile
from collections.abc import Callable

from nicegui import ui

from gui import demodata, nav, theme
from gui.components.file_picker import DirectoryPicker
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.states import error_banner, info_banner, section_title
from gui.data_upload import (
    FileCheckResult,
    FileStatus,
    OverlapInfo,
    TelCoverage,
    compute_overlap,
    compute_tel_coverage,
    delete_individueel,
    delete_oktober,
    delete_telbestand,
    save_and_validate_individueel,
    save_and_validate_oktober,
    save_and_validate_telbestand,
    scan_existing_files,
)
from gui.state import STATE

_CREATED_DIRS = [
    "configuration/filtering/",
    "data/input/",
    "data/input_raw/telbestanden/",
    "data/output/",
]

# Formaatpreview per uploadzone: (kolomnaam, beschrijving, voorbeeldwaarde)
_TEL_PREVIEW: dict = {
    "title": "Verwacht formaat — Telbestand",
    "note": "CSV · puntkomma- of kommagescheiden · één bestand per rapportageweek",
    "filenames": [
        ("telbestandY2024W10.csv",                 "instellingsformaat"),
        ("telbestand_sl_20241007_v01_2024.csv",    "UvA SQL (datum eerst)"),
        ("Telbestand_SL_2020_V96_20210802.csv",    "UvA SQL (jaar eerst)"),
    ],
    "columns": [
        ("Studiejaar",      "Collegejaar (integer)",             "2024"),
        ("Isatcode",        "CROHO-opleidingscode",              "55604"),
        ("Aantal",          "Aantal aanmelders / inschrijvingen", "3"),
        ("meercode_V",      "Meerdere aanmeldingen",             "N / J"),
        ("Status",          "Aanmeldstatus-code",                "00"),
        ("Herinschrijving", "Herinschrijving",                   "N / J"),
        ("Hogerejaars",     "Hogerejaars",                       "N / J"),
        ("Herkomst",        "Nationaliteitscategorie",           "N / E / R"),
    ],
}

_OKT_PREVIEW: dict = {
    "title": "Verwacht formaat — Oktober-bestand",
    "note": "Excel (.xlsx) · peildatum 1 oktober · één rij per opleiding × herkomstgroep",
    "columns": [
        ("Collegejaar",                "Academisch jaar",             "2024"),
        ("Isatcode",                   "CROHO-opleidingscode",        "55604"),
        ("Aantal eerstejaars croho",   "Eerstejaars per opleiding",   "125"),
        ("EER-NL-nietEER",             "Herkomstgroep",               "NL / EER / niet-EER"),
        ("Examentype code",            "Type opleiding",              "B / M"),
        ("Aantal Hoofdinschrijvingen", "Totaal hoofdinschrijvingen",  "438"),
    ],
}


def _format_preview_html(preview: dict) -> str:
    """Genereer gestylde HTML-tabel voor de formaatpreview-tooltip."""
    filenames_html = ""
    if "filenames" in preview:
        fname_rows = ""
        for name, label in preview["filenames"]:
            fname_rows += (
                f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:3px;">'
                f'<span style="font-family:monospace;font-size:11px;color:#0070c9;'
                f'white-space:nowrap;">{name}</span>'
                f'<span style="font-size:10px;color:#999;">{label}</span>'
                f'</div>'
            )
        filenames_html = (
            f'<div style="margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #efefef;">'
            f'<div style="font-size:10px;color:#aaa;font-weight:500;text-transform:uppercase;'
            f'letter-spacing:.05em;margin-bottom:5px;">Bestandsnaming</div>'
            f'{fname_rows}'
            f'</div>'
        )

    rows = ""
    for col, desc, example in preview["columns"]:
        rows += (
            f'<tr>'
            f'<td style="padding:3px 14px 3px 0;font-family:monospace;font-size:11.5px;'
            f'color:#111;white-space:nowrap;vertical-align:top;">{col}</td>'
            f'<td style="padding:3px 14px 3px 0;font-size:11px;color:#666;'
            f'vertical-align:top;">{desc}</td>'
            f'<td style="padding:3px 0;font-family:monospace;font-size:11px;'
            f'color:#0070c9;white-space:nowrap;vertical-align:top;">{example}</td>'
            f'</tr>'
        )
    return (
        f'<div style="min-width:360px;max-width:520px;font-family:system-ui,sans-serif;">'
        f'<div style="font-weight:600;font-size:12.5px;color:#111;margin-bottom:10px;'
        f'padding-bottom:7px;border-bottom:1px solid #efefef;">{preview["title"]}</div>'
        f'{filenames_html}'
        f'<table style="border-collapse:collapse;width:100%;">'
        f'<thead><tr style="border-bottom:1px solid #efefef;">'
        f'<th style="text-align:left;padding:0 14px 5px 0;font-size:10px;color:#aaa;'
        f'font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Kolom</th>'
        f'<th style="text-align:left;padding:0 14px 5px 0;font-size:10px;color:#aaa;'
        f'font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Beschrijving</th>'
        f'<th style="text-align:left;padding:0 0 5px;font-size:10px;color:#aaa;'
        f'font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Voorbeeld</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
        f'<div style="margin-top:9px;font-size:10px;color:#bbb;border-top:1px solid #efefef;'
        f'padding-top:7px;">{preview["note"]}</div>'
        f'</div>'
    )


# Visuele configuratie per FileStatus.
_STATUS_VISUAL: dict[FileStatus, tuple[str, str, str]] = {
    FileStatus.CHECKING: ("hourglass_top", theme.INFO,     "Controleren…"),
    FileStatus.VALID:    ("check_circle",  theme.POSITIVE, "Geldig"),
    FileStatus.WARNINGS: ("warning",       theme.WARNING,  "Geldig (met opmerkingen)"),
    FileStatus.ERRORS:   ("error",         theme.NEGATIVE, "Fouten gevonden"),
}

# Beschikbare pipeline-modi met bijbehorende metadata.
# Tuple: (key, label, icon, cli_flag, beschrijving, aanbevolen)
_MODE_OPTS: list[tuple[str, str, str, str, str, bool]] = [
    ("cumulative", "Cumulatief",  "bar_chart", "-d cumulative", "Alleen telbestanden",       True),
    ("individual", "Individueel", "person",    "-d individual", "Alleen aanmelddata",         False),
    ("both",       "Beide",       "bolt",      "-d both",       "Telbestanden + aanmelddata", False),
]


def _coverage_html(cov: TelCoverage) -> str:
    """Genereer een zelfstandige HTML-visualisatie van de telbestand-dekking."""
    gap_set = set(cov.gaps)
    all_present = [(y, w) for y, ws in cov.present.items() for w in ws]
    min_yw = min(all_present)
    max_yw = max(all_present)

    n_gaps = len(cov.gaps)
    gap_color = theme.WARNING if n_gaps else theme.POSITIVE
    gap_label = (
        f"⚠ {n_gaps} {'gat' if n_gaps == 1 else 'gaten'}"
        if n_gaps
        else "✓ aaneengesloten"
    )

    header = (
        f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:10px;">'
        f'<span style="font-size:13px;font-weight:600;color:#1a1a1a;">'
        f'W{min_yw[1]} {min_yw[0]} → W{max_yw[1]} {max_yw[0]}'
        f'</span>'
        f'<span style="font-size:12px;color:#999;">'
        f'{cov.total} bestand{"en" if cov.total != 1 else ""}'
        f'</span>'
        f'<span style="font-size:12px;font-weight:500;color:{gap_color};">{gap_label}</span>'
        f'</div>'
    )

    rows_html: list[str] = []
    for year in cov.years:
        year_weeks = cov.present.get(year, set())
        min_year, max_year = cov.years[0], cov.years[-1]
        if year == min_year == max_year:
            w_start, w_end = min(year_weeks, default=1), max(year_weeks, default=1)
        elif year == min_year:
            w_start = min(year_weeks, default=1)
            w_end = 52
        elif year == max_year:
            w_start = 1
            w_end = max(year_weeks, default=52)
        else:
            w_start, w_end = 1, 52

        boxes: list[str] = []
        for w in range(1, 53):
            if w < w_start or w > w_end:
                bg = "#efefef"
                op = "0.25"
            elif (year, w) in gap_set:
                bg = "#FFD4B5"
                op = "1"
            else:
                bg = theme.ACCENT
                op = "1"
            boxes.append(
                f'<div title="W{w} {year}" style="'
                f'width:8px;height:14px;background:{bg};opacity:{op};'
                f'border-radius:2px;flex-shrink:0;"></div>'
            )

        rows_html.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="font-size:11px;font-family:monospace;width:36px;'
            f'color:#999;flex-shrink:0;">{year}</span>'
            f'<div style="display:flex;gap:2px;">{"".join(boxes)}</div>'
            f'<span style="font-size:10px;color:#bbb;margin-left:4px;flex-shrink:0;">'
            f'W{w_start}–{w_end}</span>'
            f'</div>'
        )

    gaps_html = ""
    if cov.gaps:
        shown = sorted(cov.gaps)[:10]
        text = ", ".join(f"W{w}/{y}" for y, w in shown)
        if len(cov.gaps) > 10:
            text += f" (+{len(cov.gaps) - 10} meer)"
        gaps_html = (
            f'<div style="margin-top:8px;padding-top:8px;'
            f'border-top:1px solid #efefef;'
            f'font-size:11px;color:{theme.WARNING};'
            f'display:flex;align-items:flex-start;gap:6px;">'
            f'<span style="flex-shrink:0;font-weight:500;">Gaten:</span>'
            f'<span style="word-break:break-all;line-height:1.5;">{text}</span>'
            f'</div>'
        )

    legend = (
        f'<div style="display:flex;gap:12px;margin-top:10px;'
        f'padding-top:8px;border-top:1px solid #efefef;">'
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:10px;height:10px;border-radius:2px;background:{theme.ACCENT};"></div>'
        f'<span style="font-size:10px;color:#999;">aanwezig</span></div>'
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:10px;height:10px;border-radius:2px;background:#FFD4B5;"></div>'
        f'<span style="font-size:10px;color:#999;">ontbreekt</span></div>'
        f'</div>'
    ) if n_gaps else ""

    return (
        f'<div style="background:#fafafa;border:1px solid #efefef;'
        f'border-radius:8px;padding:12px 14px;">'
        f'{header}'
        f'<div style="display:flex;flex-direction:column;gap:5px;">{"".join(rows_html)}</div>'
        f'{gaps_html}'
        f'{legend}'
        f'</div>'
    )


def _overlap_html(info: OverlapInfo) -> str:
    """Genereer een HTML-visualisatie van de jaar-overlap tussen telbestanden en oktober."""
    n = len(info.intersection)
    tel_set = set(info.tel_years)
    okt_set = set(info.okt_years)
    ovl_set = set(info.intersection)

    # ── Verdict ──────────────────────────────────────────────────────────────
    if n >= 6:
        vc, vi, vt = theme.POSITIVE, "✓", "Uitstekend"
        adv = f"{n} overlappende jaren — ruim voldoende data voor betrouwbare prognoses."
    elif n >= 4:
        vc, vi, vt = theme.POSITIVE, "✓", "Goed"
        adv = f"{n} overlappende jaren — goede basis voor modeltraining."
    elif n == 3:
        vc, vi, vt = theme.WARNING, "⚠", "Voldoende"
        adv = f"{n} overlappende jaren — acceptabel, maar meer historische data verbetert de nauwkeurigheid."
    elif n == 2:
        vc, vi, vt = theme.WARNING, "⚠", "Minimaal"
        adv = f"Slechts {n} overlappende jaren — voeg meer historische data toe voor betere prognoses."
    elif n == 1:
        vc, vi, vt = theme.NEGATIVE, "✗", "Onvoldoende"
        adv = "Slechts 1 overlappend jaar. Dit is te weinig voor een betrouwbaar model."
    else:
        vc, vi, vt = theme.NEGATIVE, "✗", "Geen overlap"
        adv = (
            "Geen overlappende jaren gevonden. "
            "Controleer of de datasets dezelfde periode beslaan."
        )

    if info.intersection:
        adv += f" Trainingsvenster: {info.intersection[0]}–{info.intersection[-1]}."

    # Jaar-bereik in badge tonen bij beperkte overlap zodat het direct leesbaar is.
    badge_range = ""
    if 0 < n <= 3 and info.intersection:
        badge_range = f" ({info.intersection[0]}–{info.intersection[-1]})"

    # ── Blokbreedte dynamisch op basis van jaarbereik ─────────────────────────
    # Beschikbare breedte is ~600px (container minus label 95px + gap 8px = 103px).
    # Blok = bw px + 3px gap; bw * n_years ≤ ~580px.
    n_yrs = len(info.year_range)
    bw = 40 if n_yrs <= 13 else 34 if n_yrs <= 16 else 28

    # ── Blokbouwers per dataset ───────────────────────────────────────────────
    def _tel_block(y: int) -> str:
        if y in tel_set:
            return (
                f'<div title="Telbestand aanwezig: {y}"'
                f' style="min-width:{bw}px;padding:5px 3px;'
                f'background:#DD784B18;border:1px solid #DD784B88;border-radius:6px;'
                f'text-align:center;font-size:11px;font-family:monospace;color:#DD784B;">'
                f'{y}</div>'
            )
        return (
            f'<div style="min-width:{bw}px;padding:5px 3px;background:transparent;'
            f'border:1px dashed #e4e4e4;border-radius:6px;text-align:center;'
            f'font-size:11px;font-family:monospace;color:#d4d4d4;">{y}</div>'
        )

    def _okt_block(y: int) -> str:
        if y in okt_set:
            return (
                f'<div title="Oktober aanwezig: {y}"'
                f' style="min-width:{bw}px;padding:5px 3px;'
                f'background:#3D68EC15;border:1px solid #3D68EC77;border-radius:6px;'
                f'text-align:center;font-size:11px;font-family:monospace;color:#3D68EC;">'
                f'{y}</div>'
            )
        return (
            f'<div style="min-width:{bw}px;padding:5px 3px;background:transparent;'
            f'border:1px dashed #e4e4e4;border-radius:6px;text-align:center;'
            f'font-size:11px;font-family:monospace;color:#d4d4d4;">{y}</div>'
        )

    def _ovl_block(y: int) -> str:
        if y in ovl_set:
            return (
                f'<div title="Overlappend jaar: {y}"'
                f' style="min-width:{bw}px;padding:5px 3px;'
                f'background:#00AF8118;border:1.5px solid {theme.POSITIVE};border-radius:6px;'
                f'text-align:center;font-size:11px;font-family:monospace;'
                f'font-weight:600;color:{theme.POSITIVE};">'
                f'{y}</div>'
            )
        # onzichtbare placeholder: behoudt kolomuitlijning zonder visuele ruis
        return (
            f'<div style="min-width:{bw}px;padding:5px 3px;border:1px solid transparent;'
            f'border-radius:6px;color:transparent;">{y}</div>'
        )

    def _row(label: str, blocks: str, bold: bool = False) -> str:
        fw = "600" if bold else "400"
        lc = "#333" if bold else "#999"
        return (
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="width:95px;text-align:right;flex-shrink:0;'
            f'font-size:11px;font-weight:{fw};color:{lc};">{label}</span>'
            f'<div style="display:flex;gap:3px;">{blocks}</div>'
            f'</div>'
        )

    tel_blocks = "".join(_tel_block(y) for y in info.year_range)
    okt_blocks = "".join(_okt_block(y) for y in info.year_range)
    ovl_blocks = "".join(_ovl_block(y) for y in info.year_range)

    return (
        f'<div style="background:#fafafa;border:1px solid #efefef;'
        f'border-radius:8px;padding:14px 16px;">'

        # ── Header ──────────────────────────────────────────────────────────
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:14px;">'
        f'<span style="font-size:13px;font-weight:600;color:#1a1a1a;">'
        f'Datadekking &amp; overlap</span>'
        f'<span style="font-size:12px;font-weight:600;padding:3px 10px;border-radius:12px;'
        f'background:{vc}20;color:{vc};">{vi}&nbsp;{vt} — {n} jaar{badge_range}</span>'
        f'</div>'

        # ── Rijen ────────────────────────────────────────────────────────────
        f'<div style="display:flex;flex-direction:column;gap:6px;">'
        + _row("Telbestanden", tel_blocks)
        + _row("Oktober", okt_blocks)
        + f'<div style="border-top:1px dashed #e8e8e8;margin:3px 0 3px 103px;"></div>'
        + _row("Overlap", ovl_blocks, bold=True)
        + f'</div>'

        # ── Advies ───────────────────────────────────────────────────────────
        + f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid #f0f0f0;'
        f'display:flex;align-items:flex-start;gap:8px;">'
        f'<span style="font-size:14px;color:{vc};flex-shrink:0;line-height:1.3;">{vi}</span>'
        f'<span style="font-size:12px;color:#555;line-height:1.5;">{adv}</span>'
        f'</div>'

        # ── Legenda ──────────────────────────────────────────────────────────
        + f'<div style="display:flex;gap:16px;margin-top:8px;padding-top:8px;'
        f'border-top:1px solid #f0f0f0;">'
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:10px;height:10px;background:#DD784B18;'
        f'border:1px solid #DD784B88;border-radius:2px;"></div>'
        f'<span style="font-size:10px;color:#999;">Telbestanden</span></div>'
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:10px;height:10px;background:#3D68EC15;'
        f'border:1px solid #3D68EC77;border-radius:2px;"></div>'
        f'<span style="font-size:10px;color:#999;">Oktober</span></div>'
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="width:10px;height:10px;background:#00AF8118;'
        f'border:1.5px solid {theme.POSITIVE};border-radius:2px;"></div>'
        f'<span style="font-size:10px;color:#999;">Overlap</span></div>'
        f'</div>'

        f'</div>'
    )


def create() -> None:
    """Registreer de route ``/wizard``."""
    nav.register_route("/wizard")

    @ui.page("/wizard")
    def wizard_page() -> None:
        with page_shell(active="/wizard", title="Project opzetten"):
            section_title(
                "Nieuw project",
                "Zet een projectmap op en upload je inputbestanden.",
            )
            _WizardView()


# ---------------------------------------------------------------------------
# Upload-zone component
# ---------------------------------------------------------------------------

class _UploadZone:
    """Upload-zone voor één bestandstype met directe validatiefeedback."""

    # Boven deze drempel wordt de bestandslijst ingeklapt getoond.
    _COLLAPSE_THRESHOLD = 5

    def __init__(
        self,
        *,
        title: str,
        description: str,
        hint: str,
        icon: str,
        required: bool,
        accept: str,
        multiple: bool,
        project_dir_getter: Callable[[], str],
        validate_fn: Callable[[str, str, bytes], FileCheckResult],
        on_change: Callable[[], None],
        delete_fn: Callable[[str, str], None] | None = None,
        format_preview: dict | None = None,
        allow_folder_mode: bool = False,
    ) -> None:
        self._project_dir_getter = project_dir_getter
        self._validate_fn = validate_fn
        self._on_change = on_change
        self._delete_fn = delete_fn
        self._format_preview = format_preview
        self._results: dict[str, FileCheckResult] = {}
        self._collapsed: bool = True
        self._folder_mode: bool = False
        self._allow_folder_mode = allow_folder_mode
        self._zone_uid = str(id(self))
        # Debounce-state voor map-upload samenvatting
        self._folder_batch_count: int = 0
        self._folder_batch_task: asyncio.Task | None = None
        # Placeholders ingevuld in _build
        self._hint_label = None
        self._folder_hint_label = None
        self._file_upload_el = None
        self._folder_container = None
        self._btn_files = None
        self._btn_folder = None
        self._build(title, description, hint, icon, required, accept, multiple)

    # --- Public interface ---------------------------------------------------

    def load_existing(
        self,
        existing: dict[str, FileCheckResult] | FileCheckResult | None,
    ) -> None:
        if isinstance(existing, dict):
            self._results = dict(existing)
        elif existing is not None:
            self._results[existing.filename] = existing
        self._refresh_results()

    def set_required(self, required: bool) -> None:
        """Wissel het 'Vereist'/'Optioneel'-badge live."""
        text = "Vereist" if required else "Optioneel"
        color = "accent" if required else "grey-6"
        self._badge.set_text(text)
        self._badge.props(f"color={color}")

    @property
    def has_valid(self) -> bool:
        return any(
            r.status in (FileStatus.VALID, FileStatus.WARNINGS)
            for r in self._results.values()
        )

    @property
    def count(self) -> int:
        return len(self._results)

    @property
    def valid_count(self) -> int:
        return sum(
            1 for r in self._results.values()
            if r.status in (FileStatus.VALID, FileStatus.WARNINGS)
        )

    # --- Build -------------------------------------------------------------

    def _build(
        self,
        title: str,
        description: str,
        hint: str,
        icon: str,
        required: bool,
        accept: str,
        multiple: bool,
    ) -> None:
        ui.add_css("""
            .sp-upload .q-uploader__list { display: none !important; }
            .sp-upload .q-uploader__subtitle { display: none !important; }
        """)
        with (
            ui.card()
            .classes("w-full")
            .style("border: 1px solid #e8e8e8; border-radius: 8px;")
        ):
            with ui.row().classes("w-full items-start justify-between gap-2 mb-1"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon(icon).classes("text-2xl").style(f"color: {theme.ACCENT}")
                    with ui.column().classes("gap-0"):
                        ui.label(title).classes("font-medium")
                        ui.label(description).classes("text-xs opacity-60")
                    if self._format_preview:
                        help_icon = ui.icon("help_outline").classes(
                            "text-base cursor-help flex-none"
                        ).style("color: #ccc; margin-top: 2px;")
                        with help_icon:
                            with ui.tooltip().classes(
                                "bg-white text-black shadow-4 q-pa-sm"
                            ).props("max-width=520px anchor='bottom left' self='top left'"):
                                ui.html(_format_preview_html(self._format_preview))
                badge_text = "Vereist" if required else "Optioneel"
                badge_color = "accent" if required else "grey-6"
                self._badge = ui.badge(badge_text).props(
                    f"color={badge_color}"
                ).classes("text-xs self-start mt-1 flex-none")

            # Mode-toggle: bestanden vs. map (alleen bij multiple uploads)
            if self._allow_folder_mode:
                with ui.row().classes("items-center gap-2 mb-2"):
                    ui.label("Uploaden via:").classes("text-xs opacity-50")
                    self._btn_files = (
                        ui.button("Bestanden", icon="upload_file",
                                  on_click=lambda: self._switch_mode(False))
                        .props("dense no-caps size=sm flat")
                        .style(f"background:{theme.ACCENT};color:white;border-radius:4px;")
                    )
                    self._btn_folder = (
                        ui.button("Map", icon="folder_open",
                                  on_click=lambda: self._switch_mode(True))
                        .props("dense no-caps size=sm flat")
                        .style("background:#f0f0f0;color:#888;border-radius:4px;")
                    )

            # Hint-tekst (bestandsmodus)
            if hint:
                self._hint_label = (
                    ui.label(hint)
                    .classes("text-xs opacity-50 mb-2")
                    .style("font-family:monospace")
                )

            # Hint-tekst (mapmodus, verborgen bij start)
            if self._allow_folder_mode:
                self._folder_hint_label = (
                    ui.label(
                        "Selecteer een map — alle CSV-bestanden met 'telbestand' "
                        "in de naam worden recursief verwerkt."
                    )
                    .classes("text-xs opacity-50 mb-2")
                    .style("font-family:monospace")
                )
                self._folder_hint_label.set_visibility(False)

            # Bestandsuploader (standaard zichtbaar)
            self._file_upload_el = (
                ui.upload(
                    on_upload=self._handle_upload,
                    multiple=multiple,
                    auto_upload=True,
                )
                .props(f"flat color=grey-3 text-color=grey-9 accept='{accept}'")
                .classes("w-full sp-upload")
                .style("border:2px dashed #d0d0d0;border-radius:6px;min-height:72px;")
            )

            # Mapuploader (verborgen bij start, alleen aangemaakt indien relevant)
            if self._allow_folder_mode:
                container_id = f"folder-upload-{self._zone_uid}"
                self._folder_container = (
                    ui.element("div")
                    .props(f'id="{container_id}"')
                    .classes("w-full")
                )
                with self._folder_container:
                    (
                        ui.upload(
                            on_upload=self._handle_folder_upload,
                            multiple=True,
                            auto_upload=True,
                        )
                        .props("flat color=grey-3 text-color=grey-9 label='Map kiezen'")
                        .classes("w-full sp-upload")
                        .style("border:2px dashed #d0d0d0;border-radius:6px;min-height:72px;")
                    )
                self._folder_container.set_visibility(False)
                # Injecteer webkitdirectory op de inner <input> na mount
                ui.timer(
                    0.4,
                    lambda cid=container_id: self._inject_webkitdirectory(cid),
                    once=True,
                )

            self._results_slot = ui.column().classes("w-full gap-1 mt-2")

    # --- Mode-switch en JavaScript-injectie --------------------------------

    def _switch_mode(self, folder_mode: bool) -> None:
        self._folder_mode = folder_mode
        self._file_upload_el.set_visibility(not folder_mode)
        self._folder_container.set_visibility(folder_mode)
        if self._hint_label:
            self._hint_label.set_visibility(not folder_mode)
        if self._folder_hint_label:
            self._folder_hint_label.set_visibility(folder_mode)
        active = f"background:{theme.ACCENT};color:white;border-radius:4px;"
        inactive = "background:#f5f5f5;color:#888;border-radius:4px;"
        self._btn_files.style(replace=inactive if folder_mode else active)
        self._btn_folder.style(replace=active if folder_mode else inactive)

    def _inject_webkitdirectory(self, container_id: str) -> None:
        ui.run_javascript(f"""
            (function() {{
                var c = document.getElementById("{container_id}");
                if (!c) return;
                var inp = c.querySelector(".q-uploader__input");
                if (!inp) return;
                inp.setAttribute("webkitdirectory", "");
                inp.setAttribute("mozdirectory", "");
                inp.removeAttribute("accept");
            }})();
        """)

    # --- Upload-handlers (async) ------------------------------------------

    async def _handle_upload(self, e) -> None:
        content = await e.file.read()
        await self._process_upload(e.file.name, content)

    async def _handle_folder_upload(self, e) -> None:
        fname = e.file.name
        matches = "telbestand" in fname.lower() and fname.lower().endswith(".csv")

        if matches:
            self._folder_batch_count += 1

        # Debounce: herstart de samenvattingstimer bij elk bestand
        if self._folder_batch_task and not self._folder_batch_task.done():
            self._folder_batch_task.cancel()
        self._folder_batch_task = asyncio.create_task(self._show_folder_summary())

        if not matches:
            return

        content = await e.file.read()
        await self._process_upload(fname, content)

    async def _show_folder_summary(self) -> None:
        try:
            await asyncio.sleep(1.5)
            count = self._folder_batch_count
            self._folder_batch_count = 0
            if count == 0:
                ui.notify(
                    "Geen telbestanden gevonden in de geselecteerde map. "
                    "Controleer of de bestanden 'telbestand' in de naam hebben en als .csv zijn opgeslagen.",
                    type="warning",
                    position="top",
                    close_button=True,
                    timeout=6000,
                )
        except asyncio.CancelledError:
            pass

    async def _process_upload(self, filename: str, content: bytes) -> None:
        self._results[filename] = FileCheckResult(
            filename=filename, status=FileStatus.CHECKING
        )
        self._refresh_results()

        result = await asyncio.to_thread(
            self._validate_fn,
            self._project_dir_getter(),
            filename,
            content,
        )

        self._results[filename] = result
        self._refresh_results()
        self._on_change()

    # --- Delete-handler ----------------------------------------------------

    def _delete_file(self, filename: str) -> None:
        if self._delete_fn is not None:
            try:
                self._delete_fn(self._project_dir_getter(), filename)
            except OSError:
                pass
        self._results.pop(filename, None)
        self._refresh_results()
        self._on_change()
        short = filename if len(filename) <= 40 else filename[:37] + "…"
        ui.notify(f"'{short}' verwijderd", type="warning", position="top",
                  close_button=True, timeout=3000)

    # --- UI-rendering -------------------------------------------------------

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        self._refresh_results()

    def _delete_all(self) -> None:
        filenames = list(self._results.keys())
        for filename in filenames:
            if self._delete_fn is not None:
                try:
                    self._delete_fn(self._project_dir_getter(), filename)
                except OSError:
                    pass
        self._results.clear()
        self._refresh_results()
        self._on_change()
        n = len(filenames)
        ui.notify(
            f"{n} bestand{'en' if n != 1 else ''} verwijderd",
            type="warning", position="top", close_button=True, timeout=3000,
        )

    def _refresh_results(self) -> None:
        self._results_slot.clear()
        count = len(self._results)
        if count == 0:
            return
        with self._results_slot:
            if count > self._COLLAPSE_THRESHOLD:
                self._render_summary_row()
                for result in self._results.values():
                    # In ingeklapte staat: alleen fouten en waarschuwingen tonen.
                    if not self._collapsed or result.status in (
                        FileStatus.ERRORS, FileStatus.WARNINGS
                    ):
                        self._render_file_row(result)
            else:
                with ui.row().classes("w-full justify-end"):
                    (
                        ui.button("Verwijder alles", icon="delete_sweep",
                                  on_click=self._delete_all)
                        .props("flat dense size=xs color=grey-6")
                        .tooltip("Alle bestanden verwijderen")
                    )
                for result in self._results.values():
                    self._render_file_row(result)

    def _render_summary_row(self) -> None:
        """Compacte samenvattingsbalk met inklapknop voor grote bestandslijsten."""
        total = len(self._results)
        valid = self.valid_count
        n_err = sum(1 for r in self._results.values() if r.status == FileStatus.ERRORS)
        n_warn = sum(1 for r in self._results.values() if r.status == FileStatus.WARNINGS)
        n_checking = sum(
            1 for r in self._results.values() if r.status == FileStatus.CHECKING
        )

        if n_err:
            icon_n, color = "error", theme.NEGATIVE
            msg = f"{n_err} met fouten — {valid} geldig"
        elif n_warn:
            icon_n, color = "warning", theme.WARNING
            msg = f"{n_warn} met opmerkingen — {valid - n_warn} puur geldig"
        elif n_checking:
            icon_n, color = "hourglass_top", theme.INFO
            msg = f"{valid} van {total} geldig · {n_checking} laden…"
        else:
            icon_n, color = "check_circle", theme.POSITIVE
            msg = f"{total} bestand{'en' if total != 1 else ''} geldig"

        btn_label = "Inklappen" if not self._collapsed else f"Toon alle ({total})"
        btn_icon = "expand_less" if not self._collapsed else "expand_more"

        with ui.row().classes("w-full items-center justify-between gap-2 py-0.5"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon(icon_n).classes("text-base flex-none").style(f"color: {color}")
                ui.label(msg).classes("text-sm font-medium").style(f"color: {color}")
            with ui.row().classes("items-center gap-1 no-wrap flex-none"):
                if self._delete_fn is not None:
                    (
                        ui.button("Verwijder alles", icon="delete_sweep",
                                  on_click=self._delete_all)
                        .props("flat dense size=sm color=grey-6")
                        .tooltip("Alle bestanden verwijderen")
                    )
                ui.button(
                    btn_label,
                    icon=btn_icon,
                    on_click=self._toggle_collapse,
                ).props("flat dense size=sm color=grey-7")

        # Subtiele scheidingslijn vóór de bestandsrijen als die zichtbaar zijn.
        n_visible = (
            sum(
                1 for r in self._results.values()
                if r.status in (FileStatus.ERRORS, FileStatus.WARNINGS)
            )
            if self._collapsed else total
        )
        if n_visible:
            ui.separator().classes("my-0.5 opacity-40")

    def _render_file_row(self, result: FileCheckResult) -> None:
        icon_name, color, status_text = _STATUS_VISUAL.get(
            result.status, ("radio_button_unchecked", theme.MUTED, ""),
        )
        is_checking = result.status == FileStatus.CHECKING

        with ui.column().classes("w-full gap-0"):
            with ui.row().classes("w-full items-center gap-2 no-wrap py-1"):
                if is_checking:
                    ui.spinner(size="xs").style(f"color: {color}")
                else:
                    ui.icon(icon_name).style(f"color: {color}").classes("text-base flex-none")
                ui.label(result.filename).classes("text-sm font-mono grow truncate")
                meta_parts = [status_text]
                if result.row_count is not None:
                    meta_parts.append(f"{result.row_count:,} rijen".replace(",", "."))
                ui.label(" · ".join(meta_parts)).classes("text-xs flex-none").style(
                    f"color: {color}"
                )
                if self._delete_fn is not None and not is_checking:
                    (
                        ui.button(
                            icon="delete_outline",
                            on_click=lambda fn=result.filename: self._delete_file(fn),
                        )
                        .props("flat dense round size=xs")
                        .style("color: #ccc; flex-shrink: 0;")
                        .tooltip("Verwijderen")
                    )

            if result.hard_errors or result.soft_errors:
                with ui.column().classes("ml-6 gap-0.5 mb-1"):
                    for msg in result.hard_errors + result.soft_errors:
                        with ui.row().classes("items-start gap-1 no-wrap"):
                            ui.icon("subdirectory_arrow_right").classes(
                                "text-xs flex-none mt-0.5 opacity-40"
                            )
                            ui.label(msg).classes("text-xs leading-snug").style(
                                f"color: {theme.NEGATIVE}"
                            )

            if result.warnings:
                with ui.column().classes("ml-6 gap-0.5 mb-1"):
                    for msg in result.warnings:
                        with ui.row().classes("items-start gap-1 no-wrap"):
                            ui.icon("subdirectory_arrow_right").classes(
                                "text-xs flex-none mt-0.5 opacity-40"
                            )
                            ui.label(msg).classes("text-xs leading-snug").style(
                                f"color: {theme.WARNING}"
                            )


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class _WizardView:
    """Houdt de wizard-state en rendert de vier stappen."""

    def __init__(self) -> None:
        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self._project_dir = os.path.join(os.getcwd(), "tmp", f"studentprognose{stamp}")
        self._mode: str = "cumulative"
        self._picker = DirectoryPicker(on_select=self._on_dir_selected)
        self._build()

    def _build(self) -> None:
        with ui.stepper().props("vertical").classes("w-full") as self._stepper:
            self._build_step1()
            self._build_step2()
            self._build_step3()

    # ── Stap 1: map kiezen ──────────────────────────────────────────────────

    def _build_step1(self) -> None:
        with ui.step("Projectmap kiezen"):
            ui.label(
                "Kies de map waarin het project wordt aangemaakt. "
                "Een bestaande configuratie wordt niet overschreven."
            ).classes("text-sm opacity-70")
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                self._path_input = (
                    ui.input("Projectmap", value=self._project_dir)
                    .props("outlined dense")
                    .classes("grow")
                )
                ui.button(
                    "Bladeren",
                    icon="folder_open",
                    on_click=lambda: self._picker.open(self._path_input.value),
                ).props("outline")
            with ui.stepper_navigation():
                ui.button("Volgende", on_click=self._goto_create)

    # ── Stap 2: aanmaken ────────────────────────────────────────────────────

    def _build_step2(self) -> None:
        with ui.step("Aanmaken"):
            self._conflict_slot = ui.column().classes("w-full")
            self._confirm_path = ui.label().classes("text-sm font-mono opacity-60 mb-2")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                self._demo_checkbox = ui.checkbox(
                    "Demo Studielink-data downloaden (≈4 MB)",
                    value=False,
                )
                # Vertikale scheidingslijn
                ui.html(
                    '<div style="width:1px;height:20px;background:#e0e0e0;'
                    'align-self:center;flex-shrink:0"></div>'
                )
                # Studielink-knop: visueel greyed-out, tooltip + klikfeedback
                with ui.row().classes("items-center gap-1.5").style(
                    "opacity:0.5; cursor:not-allowed"
                ):
                    ui.tooltip(
                        "Directe koppeling met het Studielink-portaal"
                        " — binnenkort beschikbaar"
                    )
                    (
                        ui.button(
                            "Download vanuit Studielink",
                            icon="cloud_download",
                            on_click=self._on_studielink_click,
                        )
                        .props("flat color=grey dense")
                        .style("pointer-events:auto; cursor:not-allowed")
                    )
                    ui.label("binnenkort").classes("text-xs rounded-full px-2").style(
                        "background:#f0f0f0; color:#999;"
                        " border:1px solid #ddd; font-style:italic;"
                        " white-space:nowrap"
                    )
            self._demo_progress = ui.linear_progress(value=0.0, show_value=False)
            self._demo_progress.set_visibility(False)
            self._panel = ProcessPanel()
            with ui.stepper_navigation():
                ui.button("Terug", on_click=self._stepper.previous).props("flat")
                self._create_btn = ui.button(
                    "Project aanmaken",
                    icon="build",
                    on_click=self._create_project,
                )
            self._create_feedback = ui.column().classes("mt-2 w-full")

    # ── Stap 3: modus kiezen + data uploaden ────────────────────────────────

    def _build_step3(self) -> None:
        with ui.step("Data uploaden"):

            # ── Modus-selectie ─────────────────────────────────────────────
            ui.label("Welke data ga je gebruiken?").classes("font-medium")
            ui.label(
                "De gekozen modus bepaalt welke bestanden verplicht zijn."
            ).classes("text-sm opacity-60 mb-3")

            self._mode_cards: dict[str, ui.card] = {}
            with ui.row().classes("w-full gap-3 mb-5"):
                for key, label, icon, cli_flag, desc, recommended in _MODE_OPTS:
                    with (
                        ui.card()
                        .classes("flex-1 cursor-pointer")
                        .style("border-radius: 8px; border: 2px solid #e8e8e8;")
                    ) as card:
                        card.on("click", lambda k=key: self._select_mode(k))
                        with ui.column().classes("items-center text-center gap-1"):
                            ui.icon(icon).classes("text-3xl").style(
                                f"color: {theme.ACCENT}"
                            )
                            ui.label(label).classes("font-medium text-sm")
                            ui.label(cli_flag).classes("text-xs font-mono opacity-40")
                            ui.label(desc).classes("text-xs opacity-60")
                            if recommended:
                                ui.badge("★ Aanbevolen").props(
                                    "color=accent outline"
                                ).classes("text-xs mt-1")
                    self._mode_cards[key] = card

            # ── Upload-zones ───────────────────────────────────────────────
            self._tel_wrapper = ui.column().classes("w-full gap-3")
            with self._tel_wrapper:
                self._zone_tel = _UploadZone(
                    title="Telbestanden",
                    description="Weekelijkse Studielink-exports (één CSV per week).",
                    hint="bijv. telbestandY2024W10.csv, telbestand_sl_20241007_v01_2024.csv of Telbestand_SL_2020_V96_20210802.csv",
                    icon="bar_chart",
                    required=True,
                    accept=".csv",
                    multiple=True,
                    project_dir_getter=lambda: self._project_dir,
                    validate_fn=save_and_validate_telbestand,
                    on_change=self._on_tel_change,
                    delete_fn=delete_telbestand,
                    format_preview=_TEL_PREVIEW,
                    allow_folder_mode=True,
                )
                self._coverage_slot = ui.column().classes("w-full")

            # Ruimte tussen tel en ind: alleen zichtbaar als beide secties actief zijn.
            self._space_tel_ind = ui.space().classes("h-3")

            self._ind_wrapper = ui.column().classes("w-full")
            with self._ind_wrapper:
                self._zone_ind = _UploadZone(
                    title="Individuele aanmelddata",
                    description="Eén CSV-bestand met aanmeldinformatie per student.",
                    hint="Wordt opgeslagen als: individuele_aanmelddata.csv",
                    icon="person",
                    required=True,
                    accept=".csv",
                    multiple=False,
                    project_dir_getter=lambda: self._project_dir,
                    validate_fn=save_and_validate_individueel,
                    on_change=self._refresh_summary,
                    delete_fn=delete_individueel,
                )

            ui.space().classes("h-3")

            self._zone_okt = _UploadZone(
                title="Oktober-bestand",
                description="Werkelijke inschrijvingen per opleiding, peildatum 1 oktober.",
                hint="Wordt opgeslagen als: oktober_bestand.xlsx",
                icon="calendar_month",
                required=True,
                accept=".xlsx",
                multiple=False,
                project_dir_getter=lambda: self._project_dir,
                validate_fn=save_and_validate_oktober,
                on_change=self._on_okt_change,
                delete_fn=delete_oktober,
                format_preview=_OKT_PREVIEW,
            )

            ui.space().classes("h-4")

            # ── Overlap-visualisatie (tel vs. oktober) ─────────────────────
            self._overlap_slot = ui.column().classes("w-full gap-0")

            # ── Statuskaart ────────────────────────────────────────────────
            self._summary_card = (
                ui.card()
                .classes("w-full")
                .style("border: 1px solid #e8e8e8; border-radius: 8px;")
            )

            with ui.stepper_navigation():
                ui.button("Terug", on_click=self._stepper.previous).props("flat")
                ui.button(
                    "Overslaan",
                    icon="skip_next",
                    on_click=lambda: ui.navigate.to("/config"),
                ).props("flat color=grey")
                self._proceed_btn = ui.button(
                    "Naar configuratie",
                    icon="arrow_forward",
                    on_click=lambda: ui.navigate.to("/config"),
                ).props("unelevated color=accent")
                self._proceed_btn.set_visibility(False)

            # Initialiseer visuele staat nadat alle elementen bestaan.
            self._apply_mode()

    # ── Telbestand-dekking ───────────────────────────────────────────────────

    def _on_tel_change(self) -> None:
        self._refresh_coverage()
        self._refresh_summary()
        self._refresh_overlap()

    def _on_okt_change(self) -> None:
        self._refresh_summary()
        self._refresh_overlap()

    def _refresh_coverage(self) -> None:
        cov = compute_tel_coverage(self._zone_tel._results)
        self._coverage_slot.clear()
        if cov is not None:
            with self._coverage_slot:
                ui.html(_coverage_html(cov))

    def _refresh_overlap(self) -> None:
        """Toon overlap-visualisatie zodra zowel telbestanden als oktober geldig zijn."""
        self._overlap_slot.clear()
        if self._mode not in ("cumulative", "both"):
            return
        cov = compute_tel_coverage(self._zone_tel._results)
        okt_result = next(iter(self._zone_okt._results.values()), None)
        info = compute_overlap(cov, okt_result)
        if info is not None:
            with self._overlap_slot:
                ui.html(_overlap_html(info))
                ui.space().classes("h-4")

    # ── Modus-logica ────────────────────────────────────────────────────────

    def _select_mode(self, mode: str) -> None:
        self._mode = mode
        self._apply_mode()

    def _apply_mode(self) -> None:
        """Pas kaartrand, zichtbaarheid van zones en badge-teksten aan."""
        STATE.wizard_mode = self._mode

        for key, card in self._mode_cards.items():
            if key == self._mode:
                card.style(
                    f"border-radius: 8px; border: 2px solid {theme.ACCENT};"
                    f"background: {theme.ACCENT}0d;"
                )
            else:
                card.style("border-radius: 8px; border: 2px solid #e8e8e8;")

        needs_tel = self._mode in ("cumulative", "both")
        needs_ind = self._mode in ("individual", "both")

        self._tel_wrapper.set_visibility(needs_tel)
        self._ind_wrapper.set_visibility(needs_ind)
        # Tussenruimte alleen tonen als beide secties zichtbaar zijn.
        self._space_tel_ind.set_visibility(needs_tel and needs_ind)

        self._refresh_summary()
        self._refresh_overlap()

    # ── Samenvattingskaart ───────────────────────────────────────────────────

    def _refresh_summary(self) -> None:
        tel_ok = self._zone_tel.has_valid
        ind_ok = self._zone_ind.has_valid
        okt_ok = self._zone_okt.has_valid

        needs_tel = self._mode in ("cumulative", "both")
        needs_ind = self._mode in ("individual", "both")

        # Oktober is altijd vereist (bevat de labels voor het model).
        ready = (
            (not needs_tel or tel_ok)
            and (not needs_ind or ind_ok)
            and okt_ok
        )

        self._summary_card.clear()
        with self._summary_card:
            rows: list[tuple[str, bool, str]] = []

            if needs_tel:
                if tel_ok and self._zone_tel.count > self._zone_tel.valid_count:
                    detail = (
                        f"{self._zone_tel.valid_count} van "
                        f"{self._zone_tel.count} bestanden geldig"
                    )
                elif tel_ok:
                    detail = f"{self._zone_tel.valid_count} bestand(en) geldig"
                else:
                    detail = "Nog uploaden"
                rows.append(("Telbestanden", tel_ok, detail))

            if needs_ind:
                rows.append((
                    "Individuele aanmelddata",
                    ind_ok,
                    "Aanwezig" if ind_ok else "Nog uploaden",
                ))

            rows.append((
                "Oktober-bestand",
                okt_ok,
                "Aanwezig" if okt_ok else "Nog uploaden",
            ))

            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("checklist").classes("text-lg").style(f"color: {theme.ACCENT}")
                ui.label("Status geselecteerde modus").classes("font-medium")

            for label, ok, detail in rows:
                icon_name = "check_circle" if ok else "radio_button_unchecked"
                color = theme.POSITIVE if ok else theme.MUTED
                with ui.row().classes("items-center gap-2 py-0.5"):
                    ui.icon(icon_name).style(f"color: {color}").classes("text-base flex-none")
                    ui.label(label).classes("text-sm flex-none w-52")
                    ui.label(detail).classes("text-xs opacity-60")

            if not ready:
                ui.separator().classes("my-2")
                with ui.row().classes("items-center gap-2"):
                    ui.icon("info").style(f"color: {theme.INFO}").classes("text-base")
                    ui.label(
                        "Upload de vereiste bestanden of klik 'Overslaan' als je "
                        "ze al handmatig hebt geplaatst."
                    ).classes("text-xs opacity-70")

        self._proceed_btn.set_visibility(ready)

    # ── Stap-overgangen ─────────────────────────────────────────────────────

    def _goto_create(self) -> None:
        self._project_dir = os.path.abspath(self._path_input.value.strip())
        self._confirm_path.set_text(self._project_dir)

        self._conflict_slot.clear()
        if os.path.isfile(
            os.path.join(self._project_dir, "configuration", "configuration.json")
        ):
            with self._conflict_slot:
                info_banner(
                    "Deze map bevat al een project. "
                    "De bestaande configuratie blijft behouden — "
                    "er wordt niets overschreven."
                )

        self._stepper.next()

    def _on_dir_selected(self, path: str) -> None:
        self._path_input.set_value(path)

    # ── Project aanmaken ────────────────────────────────────────────────────

    async def _create_project(self) -> None:
        self._create_btn.props("loading")
        self._create_feedback.clear()
        try:
            os.makedirs(self._project_dir, exist_ok=True)
            returncode = await self._panel.run(["init"], cwd=self._project_dir)

            if returncode == 0:
                with self._create_feedback:
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.icon("check_circle").style(f"color: {theme.POSITIVE}").classes("text-base")
                        ui.label("Projectmap aangemaakt").classes("text-sm font-medium").style(f"color: {theme.POSITIVE}")
                    with ui.column().classes("gap-0 ml-6"):
                        for d in _CREATED_DIRS:
                            with ui.row().classes("items-center gap-1"):
                                ui.icon("folder").classes("text-amber-8 text-sm")
                                ui.label(d).classes("text-xs font-mono opacity-60")
                        with ui.row().classes("items-center gap-1"):
                            ui.icon("description").classes("opacity-40 text-sm")
                            ui.label("configuration/configuration.json").classes("text-xs font-mono opacity-60")

            if returncode == 0 and self._demo_checkbox.value:
                await self._download_demodata()

            if returncode == 0:
                STATE.project_dir = self._project_dir
                await self._goto_upload_step()
        finally:
            self._create_btn.props(remove="loading")

    async def _goto_upload_step(self) -> None:
        self._stepper.next()
        existing = await asyncio.to_thread(scan_existing_files, self._project_dir)
        self._zone_tel.load_existing(existing["telbestanden"])
        self._zone_ind.load_existing(existing["individueel"])
        self._zone_okt.load_existing(existing["oktober"])
        self._refresh_coverage()
        self._refresh_summary()
        self._refresh_overlap()

    def _on_studielink_click(self) -> None:
        ui.notify(
            "Studielink-koppeling is nog niet beschikbaar"
            " — wordt toegevoegd in een volgende versie.",
            type="info",
            icon="cloud_off",
            timeout=4000,
        )

    async def _download_demodata(self) -> None:
        dest = os.path.join(self._project_dir, "data", "input_raw")
        self._demo_progress.set_visibility(True)
        holder: dict = {"value": 0.0}
        timer = ui.timer(
            0.1,
            lambda: self._demo_progress.set_value(max(holder["value"], 0.0)),
        )

        def _work() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = os.path.join(tmp, "demo-data.zip")
                demodata.download_file(
                    demodata.DEMO_URL,
                    zip_path,
                    progress_cb=lambda f: holder.__setitem__("value", f),
                )
                demodata.extract_zip(zip_path, dest)

        try:
            await asyncio.to_thread(_work)
            self._demo_progress.set_value(1.0)
            with self._create_feedback:
                info_banner("Demodata gedownload naar data/input_raw/.")
        except Exception as exc:  # noqa: BLE001
            with self._create_feedback:
                error_banner(
                    "Demodata downloaden mislukt.",
                    f"Controleer je internetverbinding. Details: {exc}",
                )
        finally:
            timer.cancel()
