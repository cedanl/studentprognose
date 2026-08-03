"""Startpagina van de GUI.

Toont voor nieuwe gebruikers de zakelijke meerwaarde van instroom-prognoses
en de drie voorspelsporen; voor terugkerende gebruikers een projectdashboard
met snelkoppelingen. Bevat ook een één-klik demo die de volledige flow
automatisch met demodata uitvoert in een tijdelijke map.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Callable

from nicegui import ui

from gui import demodata, filtering_io, nav, theme
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.progress_card import ProgressCard
from gui.components.states import error_banner
from gui.state import STATE

#: Jaar/week voor de demo-voorspelling (de demodata dekt 2020–2026).
_DEMO_YEAR = "2024"
_DEMO_WEEK = "6"

#: De demo scopt bewust op een subset (Master + Niet-EER) zodat de volledige
#: pipeline in seconden klaar is in plaats van minuten.
_DEMO_FILTERING = {
    "filtering": {"programme": [], "herkomst": ["Niet-EER"], "examentype": ["Master"]}
}

#: Vijf concrete toepassingen van instroom-prognoses — op volgorde van
#: herkenbaarheid voor een beleidsmaker.
_USE_CASES: list[tuple[str, str, str]] = [
    (
        "account_balance",
        "Begroting",
        "Weet al in februari wat september brengt — stuur begrotingsposten bij "
        "terwijl het nog kan.",
    ),
    (
        "groups",
        "Capaciteit",
        "Plan docenten tijdig en stuur wervingscampagnes bij op data — "
        "weet vroeg of je tekortkomt.",
    ),
    (
        "lock",
        "Numerus fixus",
        "Zet selectiegrenzen op data — geen gissen naar omzettingspercentages.",
    ),
    (
        "trending_down",
        "Early warning",
        "Signaleer een dalende opleiding 1–2 jaar eerder — intervenieer terwijl het nog kan.",
    ),
    (
        "fact_check",
        "Accreditatie",
        "Onderbouw groeiprognoses kwantitatief voor NVAO-visitaties en jaarverslagen.",
    ),
    (
        "leaderboard",
        "Benchmark",
        "Vergelijk jouw instellingstrend met vergelijkbare instellingen — zie waar je staat.",
    ),
]


def create() -> None:
    """Registreer de route ``/``."""
    nav.register_route("/")

    @ui.page("/")
    def home_page() -> None:
        ui.add_head_html("""<style>
@keyframes sp-shimmer {
    0%   { background-position: 150% 50%; }
    100% { background-position: -50% 50%; }
}
.sp-demo-btn {
    background: linear-gradient(100deg,
        #a84a15 0%, #dd784b 30%, #ffd166 52%, #dd784b 70%, #a84a15 100%) !important;
    background-size: 300% 100% !important;
    animation: sp-shimmer 1.6s linear infinite !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 18px rgba(221,120,75,0.55), inset 0 1px 0 rgba(255,255,255,0.18) !important;
}
.sp-action-card {
    transition: box-shadow 0.2s, transform 0.15s;
    cursor: pointer;
}
.sp-action-card:hover {
    box-shadow: 0 10px 28px rgba(0,0,0,0.14) !important;
    transform: translateY(-3px);
}
/* Uitgeschakelde CTA-knop: ziet er inactief uit maar blijft klikbaar zodat
   een klik uitlegt waarom (er is al een project actief). De dubbele
   .q-btn-selector verhoogt de specificiteit zodat deze wint van Quasar's
   eigen bg-/text-classes ongeacht de laadvolgorde van de stylesheets. */
.q-btn.sp-btn-locked {
    background: #e4e4e4 !important;
    color: #9a9a9a !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    opacity: 1 !important;
    animation: none !important;
}
.q-btn.sp-btn-locked:hover {
    background: #dcdcdc !important;
    box-shadow: none !important;
}
.q-btn.sp-btn-locked .q-icon {
    color: #9a9a9a !important;
}
.sp-demo-step-dot {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 12px; font-weight: 700; color: #fff;
}
.sp-use-chip {
    padding: 10px 14px;
    border-radius: 10px;
    background: rgba(221, 120, 75, 0.08);
    border: 1px solid rgba(221, 120, 75, 0.15);
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, box-shadow 0.15s, transform 0.1s;
    min-width: 76px;
    text-align: center;
    user-select: none;
}
.sp-use-chip .q-icon {
    display: block !important;
    margin: 0 auto 4px auto !important;
}
.sp-use-chip:hover {
    background: rgba(221, 120, 75, 0.16);
    border-color: rgba(221, 120, 75, 0.45);
    box-shadow: 0 3px 12px rgba(221, 120, 75, 0.18);
    transform: translateY(-2px);
}
</style>""")
        with page_shell(active="/", title="Start", show_stepper=False):
            _HomeView()


class _HomeView:
    """Beheert de startpagina: waardepropositie, intro/CTA's en de één-klik demo."""

    def __init__(self) -> None:
        self._container = ui.column().classes("w-full gap-4")
        self._step_n: int = 0
        self._render_intro()

    def _render_intro(self) -> None:
        self._container.clear()
        with self._container:
            # De startpagina toont altijd dezelfde waardepropositie + CTA.
            # Bestaat er al een actief project, dan blijven de CTA-knoppen
            # zichtbaar maar uitgeschakeld: een klik legt uit dat er eerst
            # gereset moet worden (via de Reset-knop rechtsboven).
            self._render_value_proposition()
            self._render_cta(locked=STATE.is_initialised)

    # ── Waardepropositie ──────────────────────────────────────────────────────

    def _render_value_proposition(self) -> None:
        """Toon de vijf zakelijke toepassingen van instroom-prognoses (hero-variant)."""
        A = theme.ACCENT
        with ui.card().classes("w-full").style(
            f"background: radial-gradient(ellipse at 50% -5%, {A}11 0%, transparent 62%);"
            "border: 1px solid rgba(0,0,0,0.07);"
        ):
            with ui.column().classes("w-full items-center text-center gap-3 py-1"):

                # Logo-lockup: het beeldmerk (baret boven stijgende pijl) met
                # de woordmerk-tekst. Zet meteen de merkidentiteit neer.
                with ui.column().classes("items-center gap-1"):
                    ui.image("/gui-assets/logo.svg").classes("w-16 h-16").style(
                        "filter: drop-shadow(0 2px 6px rgba(0,0,0,0.10));"
                    )
                    ui.html(
                        '<div style="font-size:20px;font-weight:700;'
                        'letter-spacing:-0.3px;line-height:1;">'
                        f'Student<span style="color:{A}">prognose</span>'
                        "</div>"
                    )

                # Headline + subtitel
                ui.label("Weet in maart wat september brengt").classes(
                    "text-2xl font-bold"
                ).style("letter-spacing:-0.3px;line-height:1.2;")
                ui.label(
                    "Studielink-aanmeldingen omzetten naar beslissingsklare "
                    "prognoses — maanden vóór 1 september."
                ).classes("text-sm opacity-55").style(
                    "max-width:460px;line-height:1.65;"
                )

                # Use-case chips
                self._render_use_cases()

                # Methodologie-link
                ui.button(
                    "Hoe werkt het model?",
                    icon="alt_route",
                    on_click=lambda: ui.navigate.to("/methodologie"),
                ).props("flat dense color=accent").classes("mt-1 text-xs")

    def _render_use_cases(self) -> None:
        """Render de vijf use-cases als klikbare chips — click opent uitgewerkt voorbeeld."""
        with ui.row().classes("gap-2 flex-wrap my-1 justify-center"):
            for icon, title, desc in _USE_CASES:
                chip = ui.element("div").classes("sp-use-chip")
                chip.on("click", lambda i=icon, t=title: self._open_use_case(i, t))
                with chip:
                    ui.tooltip(f"{desc} · Klik voor uitgewerkt voorbeeld").style(
                        "max-width: 230px; white-space: normal; "
                        "line-height: 1.45; font-size: 12px;"
                    )
                    ui.icon(icon).classes("text-2xl").style(f"color: {theme.ACCENT}")
                    ui.label(title).classes("text-xs font-semibold")

    # ── Use-case detail dialogen ──────────────────────────────────────────────

    _UC_SCENARIOS: dict[str, str] = {
        "account_balance": "400 eerstejaars HBO Informatica · week 8",
        "groups":          "400 eerstejaars HBO Informatica · week 8",
        "lock":            "350 plaatsen Geneeskunde · omzettingsfactor 67%",
        "trending_down":   "Opleiding Communicatie · instroom 2020–2025",
        "fact_check":      "NVAO-visitatie HBO Informatica · 2025",
        "leaderboard":     "HBO Informatica · instroom 2021–2025 · 5 vergelijkbare instellingen",
    }

    def _open_use_case(self, icon_key: str, title: str) -> None:
        """Maak een dialoogvenster met uitgewerkt voorbeeld voor de gegeven use case."""
        render_fns = {
            "account_balance": self._uc_begroting,
            "groups": self._uc_personeel_werving,
            "lock": self._uc_numerus,
            "trending_down": self._uc_early_warning,
            "fact_check": self._uc_accreditatie,
            "leaderboard": self._uc_benchmark,
        }
        render_fn = render_fns.get(icon_key)
        if not render_fn:
            return

        scenario = self._UC_SCENARIOS.get(icon_key, "Voorbeeld-scenario")

        with ui.dialog() as dialog, ui.card().style(
            "width:560px;max-width:96vw;border-radius:16px;"
            "overflow:hidden;padding:0;"
        ):
            # Header
            with ui.row().classes(
                "w-full items-center justify-between px-5 pt-4 pb-3"
            ).style("border-bottom:1px solid #f0f0f0;"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    with ui.element("div").style(
                        f"width:32px;height:32px;border-radius:8px;"
                        f"background:{theme.ACCENT}15;display:flex;"
                        "align-items:center;justify-content:center;"
                    ):
                        ui.icon(icon_key).classes("text-lg").style(
                            f"color:{theme.ACCENT}"
                        )
                    ui.label(title).classes("font-semibold text-base")
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense color=grey-6"
                )
            # Scenario-badge
            with ui.element("div").classes("px-5 pt-3"):
                ui.html(
                    f'<span style="display:inline-flex;align-items:center;gap:5px;'
                    f'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;'
                    f'background:{theme.ACCENT}12;color:{theme.ACCENT};'
                    f'border:1px solid {theme.ACCENT}30;">'
                    f"Voorbeeld · {scenario}"
                    f"</span>"
                )
            # Content
            with ui.column().classes("w-full px-5 pt-3 pb-5 gap-2").style(
                "overflow-y:auto;max-height:75vh;"
            ):
                render_fn()

        dialog.open()

    def _uc_section(self, label: str) -> None:
        ui.label(label).classes("text-xs font-semibold uppercase mt-1 mb-1").style(
            "color:#aaa;letter-spacing:0.06em;"
        )

    def _uc_comparison(self, without: str, with_forecast: str) -> None:
        """Met/zonder-prognose vergelijking — gedeeld patroon voor alle dialogen."""
        ui.separator().classes("my-2 opacity-20")
        with ui.row().classes("w-full gap-2"):
            with ui.element("div").style(
                "flex:1;padding:10px 12px;border-radius:8px;"
                "background:#f7f7f7;border:1px solid #ebebeb;"
            ):
                ui.label("Zonder prognose").classes(
                    "text-xs font-semibold mb-1"
                ).style("color:#bbb;")
                ui.label(without).classes("text-xs leading-relaxed").style(
                    "color:#999;"
                )
            with ui.element("div").style(
                f"flex:1;padding:10px 12px;border-radius:8px;"
                f"background:{theme.ACCENT}0a;border:1px solid {theme.ACCENT}25;"
            ):
                ui.label("Met prognose").classes(
                    "text-xs font-semibold mb-1"
                ).style(f"color:{theme.ACCENT};")
                ui.label(with_forecast).classes(
                    "text-xs leading-relaxed"
                ).style("color:#555;")

    def _uc_begroting(self) -> None:
        A = theme.ACCENT
        ui.label(
            "Stel: je begroot op vorig jaar (362 studenten) en er komen er 400. "
            "Dat is €419.880 rijksbijdrage die je niet had ingepland — te laat om bij te sturen. "
            "Met de prognose weet je dit in februari, ruim vóór de deadline in maart."
        ).classes("text-xs opacity-60 leading-relaxed mb-1")

        self._uc_section("Financieel effect (400 eerstejaars)")
        ui.html(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px;">
          <div style="background:#f9f9f9;border-radius:8px;padding:10px 12px;border:1px solid #efefef;">
            <div style="font-size:10px;color:#bbb;margin-bottom:2px;">Collegegeld</div>
            <div style="font-size:15px;font-weight:700;color:#1a1a1a;">€925.600</div>
            <div style="font-size:10px;color:#ccc;">400 × €2.314</div>
          </div>
          <div style="background:#f9f9f9;border-radius:8px;padding:10px 12px;border:1px solid #efefef;">
            <div style="font-size:10px;color:#bbb;margin-bottom:2px;">Rijksbijdrage</div>
            <div style="font-size:15px;font-weight:700;color:#1a1a1a;">€3.592.000</div>
            <div style="font-size:10px;color:#ccc;">400 × €8.980</div>
          </div>
          <div style="background:{A}0d;border-radius:8px;padding:10px 12px;border:1px solid {A}30;">
            <div style="font-size:10px;color:{A};margin-bottom:2px;font-weight:500;">Verwacht totaal</div>
            <div style="font-size:15px;font-weight:700;color:{A};">€4.517.600</div>
            <div style="font-size:10px;color:{A}99;">+€419.880 vs. vorig jaar</div>
          </div>
        </div>
        """)

        self._uc_section("Actietimeline")
        ui.html(f"""
        <div style="display:flex;align-items:center;margin-bottom:10px;">
          <div style="text-align:center;min-width:90px;">
            <div style="width:30px;height:30px;border-radius:50%;background:{A};color:#fff;
                 display:flex;align-items:center;justify-content:center;font-weight:700;
                 margin:0 auto 4px;font-size:10px;">feb</div>
            <div style="font-size:11px;color:#333;font-weight:500;">Prognose</div>
            <div style="font-size:10px;color:#aaa;">week 8</div>
          </div>
          <div style="flex:1;height:2px;background:{A}33;margin-bottom:18px;"></div>
          <div style="text-align:center;min-width:90px;">
            <div style="width:30px;height:30px;border-radius:50%;background:{A}cc;color:#fff;
                 display:flex;align-items:center;justify-content:center;font-weight:700;
                 margin:0 auto 4px;font-size:10px;">mrt</div>
            <div style="font-size:11px;color:#333;font-weight:500;">Bijstelling</div>
            <div style="font-size:10px;color:#aaa;">begroting</div>
          </div>
          <div style="flex:1;height:2px;background:{A}33;margin-bottom:18px;"></div>
          <div style="text-align:center;min-width:90px;">
            <div style="width:30px;height:30px;border-radius:50%;background:#00AF81;color:#fff;
                 display:flex;align-items:center;justify-content:center;font-weight:700;
                 margin:0 auto 4px;font-size:10px;">sep</div>
            <div style="font-size:11px;color:#333;font-weight:500;">Instroom</div>
            <div style="font-size:10px;color:#aaa;">bevestigd ✓</div>
          </div>
        </div>
        """)

        self._uc_comparison(
            "Begroting op 362 (vorig jaar). In september tekort ontdekken → noodbudget aanvragen.",
            "Prognose in februari → bijstelling in maart → optimale ruimte voor extra FTE en materiaal.",
        )

    def _uc_personeel_werving(self) -> None:
        A = theme.ACCENT
        NEG = theme.NEGATIVE
        WARN = "#E07040"
        POS = "#00AF81"
        ui.label(
            "Week 8: de prognose voorspelt 400 eerstejaars Informatica. "
            "Twee vragen moeten nu beantwoord worden: heb je genoeg docenten? "
            "En staan er ook genoeg studenten op te komen?"
        ).classes("text-xs opacity-60 leading-relaxed mb-1")

        self._uc_section("Docentbezetting (400 eerstejaars)")
        ui.html(f"""
        <div style="background:#fafafa;border:1px solid #efefef;border-radius:10px;
             padding:14px 16px;margin-bottom:12px;">
          <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr;
               gap:6px;align-items:center;text-align:center;margin-bottom:12px;">
            <div>
              <div style="font-size:26px;font-weight:800;color:#1a1a1a;">400</div>
              <div style="font-size:10px;color:#aaa;">studenten</div>
            </div>
            <div style="font-size:20px;color:#ccc;padding:0 4px;">÷</div>
            <div>
              <div style="font-size:26px;font-weight:800;color:#1a1a1a;">25</div>
              <div style="font-size:10px;color:#aaa;">studenten/FTE</div>
            </div>
            <div style="font-size:20px;color:#ccc;padding:0 4px;">=</div>
            <div style="background:{A}0d;border-radius:8px;padding:8px 4px;">
              <div style="font-size:26px;font-weight:800;color:{A};">16</div>
              <div style="font-size:10px;color:{A}99;">FTE nodig</div>
            </div>
          </div>
          <div style="border-top:1px solid #efefef;padding-top:10px;
               display:flex;flex-direction:column;gap:4px;">
            <div style="display:flex;justify-content:space-between;font-size:12px;">
              <span style="color:#555;">Huidige bezetting</span>
              <span style="font-weight:700;color:#1a1a1a;">14 FTE</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:12px;">
              <span style="color:#555;">Tekort</span>
              <span style="font-weight:700;color:{NEG};">−2 FTE</span>
            </div>
            <div style="margin-top:6px;padding:7px 10px;border-radius:6px;
                 background:{NEG}0d;border:1px solid {NEG}30;font-size:11px;color:{NEG};">
              Actie: 2 gastdocenten of tijdelijke contracten werven vóór april
            </div>
          </div>
        </div>
        """)

        self._uc_section("Wervingssignaal (week 8 — aanmeldingen HBO Informatica)")
        ui.html(f"""
        <div style="background:#fafafa;border:1px solid #efefef;border-radius:10px;
             padding:14px 16px;margin-bottom:12px;">
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div>
              <div style="display:flex;justify-content:space-between;font-size:11px;
                   color:#999;margin-bottom:4px;">
                <span>Huidig: <strong style="color:#333;">312</strong></span>
                <span>Target: <strong style="color:#333;">400</strong></span>
              </div>
              <div style="background:#efefef;border-radius:4px;height:10px;overflow:hidden;">
                <div style="width:78%;height:100%;background:{A};border-radius:4px;"></div>
              </div>
              <div style="font-size:10px;color:#aaa;margin-top:2px;">78% van target · tekort: −88</div>
            </div>
            <div>
              <div style="font-size:11px;color:#999;margin-bottom:4px;">
                Prognose <em>zonder</em> actie: <strong style="color:{WARN};">342</strong>
                &nbsp;(tekort: −58)
              </div>
              <div style="background:#efefef;border-radius:4px;height:10px;overflow:hidden;">
                <div style="width:85.5%;height:100%;background:{WARN};border-radius:4px;opacity:0.7;"></div>
              </div>
            </div>
            <div>
              <div style="font-size:11px;color:#999;margin-bottom:4px;">
                Prognose <em>met</em> campagne: <strong style="color:{POS};">387</strong>
                &nbsp;(97% van target)
              </div>
              <div style="background:#efefef;border-radius:4px;height:10px;overflow:hidden;">
                <div style="width:96.75%;height:100%;background:{POS};border-radius:4px;"></div>
              </div>
            </div>
          </div>
        </div>
        """)

        self._uc_comparison(
            "Docenttekort pas zichtbaar in augustus; aanmeldingstekort pas in mei — beide te laat voor actie.",
            "Prognose in week 8: docenten geworven in april, campagne gestart in week 9 → geen noodscenario's.",
        )

    def _uc_numerus(self) -> None:
        A = theme.ACCENT
        POS = theme.POSITIVE
        ui.label(
            "Bij numerus fixus moet de selectiegrens juridisch houdbaar zijn. "
            "Dat vereist een data-gedreven omzettingspercentage."
        ).classes("text-xs opacity-60 leading-relaxed mb-1")

        self._uc_section("Berekening selectiegrens")
        ui.html(f"""
        <div style="background:#fafafa;border:1px solid #efefef;border-radius:10px;
             padding:14px 16px;margin-bottom:14px;">
          <div style="display:flex;flex-direction:column;gap:7px;">
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:12px;color:#555;">Aanmeldingen t/m week 10</span>
              <span style="font-size:13px;font-weight:700;color:#1a1a1a;">520</span>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:12px;color:#555;">Historisch omzettingspercentage</span>
              <span style="font-size:13px;font-weight:700;color:#1a1a1a;">67%</span>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:12px;color:#555;">Capaciteit opleiding</span>
              <span style="font-size:13px;font-weight:700;color:#1a1a1a;">350 plaatsen</span>
            </div>
            <div style="border-top:1px dashed #e8e8e8;padding-top:8px;
                 display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:12px;color:#333;font-weight:500;">Verwachte instroom (520 × 0.67)</span>
              <span style="font-size:15px;font-weight:800;color:{POS};">348 ✓</span>
            </div>
            <div style="font-size:11px;color:#aaa;font-style:italic;margin-top:-4px;">
              Prognose valt nét binnen capaciteit — beslissing met marge
            </div>
          </div>
          <div style="margin-top:12px;padding:10px 12px;border-radius:8px;
               background:{A}0a;border:1px solid {A}25;">
            <div style="font-size:12px;color:{A};font-weight:600;margin-bottom:3px;">
              Advies: selectiegrens op 523 aanmeldingen
            </div>
            <div style="font-size:11px;color:#666;">
              523 × 0.67 = 350 — precies de capaciteitsgrens
            </div>
          </div>
        </div>
        """)

        self._uc_comparison(
            "Onduidelijke grens → te vroeg/laat selecteren → bezwaren en rechtszaken van afgewezen kandidaten.",
            "Data-gedreven grens (523) → juridisch houdbaar → actief doorverwijzen bij overschrijding.",
        )



    def _uc_early_warning(self) -> None:
        A = theme.ACCENT
        NEG = theme.NEGATIVE
        data = [(2020, 312), (2021, 298), (2022, 271), (2023, 244), (2024, 218), (2025, 191)]
        ui.label(
            "Communicatie daalt al 5 jaar op rij. Zonder prognose-monitoring zie je "
            "dat pas als de schade al gedaan is. Met de tool signaleert het systeem "
            "de dalende trend automatisch — en geeft de instelling 2 jaar de tijd om te handelen."
        ).classes("text-xs opacity-60 leading-relaxed mb-1")

        self._uc_section("Instroom Communicatie — dalende trend")
        peak = 312
        bars_html = ""
        for year, val in data:
            pct = round(val / peak * 100)
            is_forecast = year == 2025
            is_signal = year == 2023
            bar_col = f"{NEG}55" if is_forecast else NEG
            label_col = "#aaa" if is_forecast else "#555"
            suffix = " (prognose)" if is_forecast else ""
            signal_badge = (
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'padding:1px 7px;border-radius:10px;font-size:9px;font-weight:600;'
                f'background:{A}15;color:{A};border:1px solid {A}30;margin-left:4px;">'
                f'⚠ prognose-signaal</span>'
            ) if is_signal else ""
            bars_html += f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
              <span style="width:32px;font-size:10px;color:#aaa;text-align:right;">{year}</span>
              <div style="flex:1;background:#efefef;border-radius:3px;height:16px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{bar_col};border-radius:3px;"></div>
              </div>
              <span style="width:auto;font-size:10px;color:{label_col};white-space:nowrap;">{val}{suffix}{signal_badge}</span>
            </div>"""

        ui.html(f"""
        <div style="background:#fafafa;border:1px solid #efefef;border-radius:10px;
             padding:14px 16px;margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="font-size:12px;font-weight:600;color:#1a1a1a;">Instroom per jaar</span>
            <span style="font-size:10px;padding:2px 8px;border-radius:10px;
                   background:{NEG}12;color:{NEG};border:1px solid {NEG}30;">−8%/jaar</span>
          </div>
          {bars_html}
        </div>
        """)

        self._uc_section("Wanneer zichtbaar?")
        ui.html(f"""
        <div style="display:flex;gap:8px;margin-bottom:6px;">
          <div style="flex:1;padding:10px 12px;border-radius:8px;
               background:{A}0a;border:1px solid {A}25;">
            <div style="font-size:11px;font-weight:600;color:{A};margin-bottom:3px;">Met prognose</div>
            <div style="font-size:11px;color:#555;line-height:1.6;">
              Trend zichtbaar in <strong>2023</strong><br>
              → 2 jaar eerder<br>
              → Nog tijd voor interventie
            </div>
          </div>
          <div style="flex:1;padding:10px 12px;border-radius:8px;
               background:#f7f7f7;border:1px solid #ebebeb;">
            <div style="font-size:11px;font-weight:600;color:#bbb;margin-bottom:3px;">Zonder prognose</div>
            <div style="font-size:11px;color:#aaa;line-height:1.6;">
              Crisis pas merkbaar in <strong>2025</strong><br>
              → Te weinig studenten<br>
              → Opleiding niet-rendabel
            </div>
          </div>
        </div>
        """)

        self._uc_section("Mogelijke interventies (2023–2024)")
        ui.html(f"""
        <div style="display:flex;flex-direction:column;gap:5px;margin-bottom:6px;">
          <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#555;">
            <span style="color:{A};font-size:14px;">•</span>
            Samenvoegen met verwante opleiding (bijv. Journalistiek)
          </div>
          <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#555;">
            <span style="color:{A};font-size:14px;">•</span>
            Vernieuwd curriculum + herpositionering in arbeidsmarkt
          </div>
          <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#555;">
            <span style="color:{A};font-size:14px;">•</span>
            Gerichte wervingscampagne op ondervertegenwoordigde regio's
          </div>
        </div>
        """)

        self._uc_comparison(
            "Daling pas merkbaar als instroom al kritiek laag is → noodscenario, mogelijk sluiting.",
            "Dalende trend 2 jaar eerder zichtbaar → tijdig interventie → opleiding behoudt levensvatbaarheid.",
        )


    def _uc_accreditatie(self) -> None:
        A = theme.ACCENT
        POS = theme.POSITIVE
        ui.label(
            "NVAO-visitaties vereisen kwantitatieve onderbouwing van strategische keuzes. "
            "Een instroom-prognose maakt groei- en stabilisatiescenario's aantoonbaar."
        ).classes("text-xs opacity-60 leading-relaxed mb-1")

        self._uc_section("Wat verwacht NVAO?")
        ui.html(f"""
        <div style="background:#fafafa;border:1px solid #efefef;border-radius:10px;
             padding:14px 16px;margin-bottom:14px;">
          <div style="display:flex;flex-direction:column;gap:8px;">
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="font-size:16px;color:{POS};flex-shrink:0;line-height:1.2;">✓</span>
              <div style="font-size:11px;color:#555;line-height:1.5;">
                <strong>Standaard 1 – Beoogde leerresultaten:</strong> aantonen dat de
                opleiding voldoende draagvlak heeft op basis van verwachte instroom
              </div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="font-size:16px;color:{POS};flex-shrink:0;line-height:1.2;">✓</span>
              <div style="font-size:11px;color:#555;line-height:1.5;">
                <strong>Standaard 3 – Toetsing:</strong> stabiele cohortgrootte onderbouwen
                voor betrouwbare normering en ijking
              </div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="font-size:16px;color:{POS};flex-shrink:0;line-height:1.2;">✓</span>
              <div style="font-size:11px;color:#555;line-height:1.5;">
                <strong>Jaarverslag / Strategisch plan:</strong> groeiprognose als
                kwantitatief bewijs bij nieuw programma of uitbreiding
              </div>
            </div>
          </div>
        </div>
        """)

        self._uc_section("Wat staat er nu in het dossier?")
        ui.html(f"""
        <div style="display:flex;gap:8px;margin-bottom:8px;">
          <div style="flex:1;padding:12px 14px;border-radius:10px;
               background:#f7f7f7;border:1px solid #ebebeb;">
            <div style="font-size:11px;font-weight:600;color:#bbb;margin-bottom:6px;">
              Zonder prognose
            </div>
            <div style="display:flex;flex-direction:column;gap:5px;">
              <div style="font-size:11px;color:#aaa;">✓ Historische instroom</div>
              <div style="font-size:11px;color:#aaa;">✓ Rendementen vorige cohorten</div>
              <div style="font-size:11px;color:#d0d0d0;">✗ Toekomstverwachting</div>
              <div style="font-size:11px;color:#d0d0d0;">✗ Meerjarentrend + bandbreedte</div>
            </div>
            <div style="margin-top:8px;font-size:10px;font-style:italic;color:#ccc;">
              Commissie vraagt: "Hoe gaat dit zich ontwikkelen?"
            </div>
          </div>
          <div style="flex:1;padding:12px 14px;border-radius:10px;
               background:{POS}08;border:1px solid {POS}30;">
            <div style="font-size:11px;font-weight:600;margin-bottom:6px;"
                 style="color:{POS};">
              <span style="color:{POS};">Met prognose</span>
            </div>
            <div style="display:flex;flex-direction:column;gap:5px;">
              <div style="font-size:11px;color:#555;">✓ Historische instroom</div>
              <div style="font-size:11px;color:#555;">✓ Rendementen vorige cohorten</div>
              <div style="font-size:11px;color:{POS};font-weight:500;">✓ Toekomstverwachting (kwantitatief)</div>
              <div style="font-size:11px;color:{POS};font-weight:500;">✓ Meerjarentrend + bandbreedte</div>
            </div>
            <div style="margin-top:8px;padding:5px 8px;border-radius:6px;
                 background:{POS}12;font-size:10px;color:{POS};font-weight:600;">
              Dossier volledig → NVAO-oordeel: Goed
            </div>
          </div>
        </div>
        """)

        self._uc_section("Kerncijfers")
        ui.html(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:6px;">
          <div style="background:#f9f9f9;border-radius:8px;padding:10px 12px;border:1px solid #efefef;">
            <div style="font-size:10px;color:#bbb;margin-bottom:2px;">Historische instroom</div>
            <div style="font-size:15px;font-weight:700;color:#1a1a1a;">362 → 400</div>
            <div style="font-size:10px;color:#ccc;">2024 → 2025 (prognose)</div>
          </div>
          <div style="background:#f9f9f9;border-radius:8px;padding:10px 12px;border:1px solid #efefef;">
            <div style="font-size:10px;color:#bbb;margin-bottom:2px;">5-jaar trend</div>
            <div style="font-size:15px;font-weight:700;color:#1a1a1a;">+9% / jaar</div>
            <div style="font-size:10px;color:#ccc;">structureel, niet conjunctureel</div>
          </div>
          <div style="background:{A}0d;border-radius:8px;padding:10px 12px;border:1px solid {A}30;">
            <div style="font-size:10px;color:{A};margin-bottom:2px;font-weight:500;">NVAO-oordeel</div>
            <div style="font-size:15px;font-weight:700;color:{A};">Goed</div>
            <div style="font-size:10px;color:{A}99;">stabiel draagvlak aangetoond</div>
          </div>
        </div>
        """)

        self._uc_comparison(
            "Groei onderbouwen met alleen historische cijfers → visitatiecommissie vraagt naar toekomstverwachting.",
            "Prognose + trendanalyse als bijlage → commissie ziet kwantitatief bewijs → sterker dossier.",
        )

    def _uc_benchmark(self) -> None:
        A = theme.ACCENT
        POS = theme.POSITIVE
        NEG = theme.NEGATIVE
        ui.label(
            "Groeit jouw instelling sneller of langzamer dan vergelijkbare instellingen? "
            "Door je eigen prognose naast die van peers te leggen zie je waar je staat "
            "en wat realistisch is om van te leren."
        ).classes("text-xs opacity-60 leading-relaxed mb-1")

        self._uc_section("HBO Informatica — instroom 2021–2025")
        # Fictieve benchmark: 5 instellingen, eigen instelling = HvA
        bench_data = [
            ("Jouw instelling (HvA)", 245, 278, 312, 356, 400, True),
            ("Hogeschool Rotterdam",  220, 241, 259, 271, 280, False),
            ("Fontys Hogeschool",     310, 328, 345, 370, 395, False),
            ("Saxion Hogeschool",     180, 188, 195, 204, 210, False),
            ("Windesheim",            260, 272, 284, 301, 318, False),
        ]
        years = [2021, 2022, 2023, 2024, 2025]
        peak = 400
        rows_html = ""
        for inst, *vals, is_own in bench_data:
            last = vals[-1]
            growth = round((vals[-1] / vals[0] - 1) * 100)
            growth_str = f"+{growth}%" if growth >= 0 else f"{growth}%"
            growth_col = POS if growth >= 10 else (NEG if growth < 0 else "#888")
            bar_bg = A if is_own else "#c8c8c8"
            font_w = "700" if is_own else "400"
            name_col = "#1a1a1a" if is_own else "#555"
            own_badge = (
                f'<span style="font-size:9px;padding:1px 6px;border-radius:8px;'
                f'background:{A}18;color:{A};border:1px solid {A}30;'
                f'margin-left:5px;font-weight:600;">jij</span>'
            ) if is_own else ""
            bar_pct = round(last / peak * 100)
            rows_html += f"""
            <div style="margin-bottom:9px;">
              <div style="display:flex;justify-content:space-between;
                   align-items:center;margin-bottom:3px;">
                <span style="font-size:11px;color:{name_col};font-weight:{font_w};">
                  {inst}{own_badge}
                </span>
                <span style="font-size:11px;font-weight:600;color:{growth_col};">
                  {growth_str}
                </span>
              </div>
              <div style="background:#efefef;border-radius:4px;height:10px;overflow:hidden;">
                <div style="width:{bar_pct}%;height:100%;background:{bar_bg};
                     border-radius:4px;{'opacity:0.5;' if not is_own else ''}"></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                   font-size:9px;color:#bbb;margin-top:2px;">
                {'  '.join(f'<span>{v}</span>' for v in vals)}
              </div>
            </div>"""

        ui.html(f"""
        <div style="background:#fafafa;border:1px solid #efefef;border-radius:10px;
             padding:14px 16px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;
               margin-bottom:12px;">
            <span style="font-size:12px;font-weight:600;color:#1a1a1a;">
              Instroom eerstejaars
            </span>
            <div style="display:flex;gap:12px;">
              {'  '.join(f'<span style="font-size:10px;color:#aaa;">{y}</span>' for y in years)}
            </div>
          </div>
          {rows_html}
        </div>
        """)

        self._uc_section("Wat zie je?")
        ui.html(f"""
        <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:10px;">
          <div style="display:flex;gap:10px;align-items:flex-start;padding:9px 12px;
               border-radius:8px;background:{A}08;border:1px solid {A}20;">
            <span style="font-size:14px;flex-shrink:0;">📈</span>
            <div style="font-size:11px;color:#444;line-height:1.55;">
              <strong>Jij groeit het snelst (+63%)</strong> — maar Fontys heeft meer absolute volume.
              Schaalvoordelen zijn daar mogelijk groter.
            </div>
          </div>
          <div style="display:flex;gap:10px;align-items:flex-start;padding:9px 12px;
               border-radius:8px;background:#f7f7f7;border:1px solid #e8e8e8;">
            <span style="font-size:14px;flex-shrink:0;">💡</span>
            <div style="font-size:11px;color:#555;line-height:1.55;">
              Windesheim groeit stabiel (+22%) zonder grote pieken — mogelijk een
              stabieler aanmeldpatroon om van te leren voor prognose-nauwkeurigheid.
            </div>
          </div>
          <div style="display:flex;gap:10px;align-items:flex-start;padding:9px 12px;
               border-radius:8px;background:#f7f7f7;border:1px solid #e8e8e8;">
            <span style="font-size:14px;flex-shrink:0;">⚠️</span>
            <div style="font-size:11px;color:#555;line-height:1.55;">
              Saxion en Rotterdam groeien nauwelijks — vergelijkbare regio, andere uitkomst.
              Aanleiding voor strategisch gesprek.
            </div>
          </div>
        </div>
        """)

        self._uc_comparison(
            "Jouw groei ziet er sterk uit — maar je weet niet of je achterloopt of voorloopt op peers.",
            "Benchmarkprognose toont: jij groeit het snelst. Leer van stabiele peers voor betere voorspelkwaliteit.",
        )

    # ── CTA (start / al-actief) ───────────────────────────────────────────────

    def _render_cta(self, *, locked: bool = False) -> None:
        """Toon de start-CTA.

        Args:
            locked: Is er al een project actief, dan blijven de knoppen zichtbaar
                maar uitgeschakeld; een klik opent een uitleg-dialoog dat er eerst
                gereset moet worden.
        """
        with ui.card().classes("w-full"):
            with ui.column().classes("w-full items-center text-center gap-3 py-12"):
                ui.icon("rocket_launch").classes("text-6xl opacity-40")
                ui.label("Klaar om te starten?").classes("text-xl font-medium")
                ui.label(
                    "Zet een projectmap op met jouw Studielink-data en "
                    "draai je eerste prognose."
                ).classes("text-sm opacity-70 max-w-md")
                with ui.row().classes(
                    "gap-3 items-center justify-center flex-wrap"
                ):
                    self._cta_button(
                        "Project opzetten",
                        icon=None,
                        locked=locked,
                        on_click=lambda: ui.navigate.to("/wizard"),
                    )
                    self._cta_button(
                        "Probeer direct met demodata",
                        icon="bolt",
                        locked=locked,
                        on_click=self._run_demo,
                        active_classes="sp-demo-btn",
                    )
                if locked:
                    self._render_active_project_hint()

    def _cta_button(
        self,
        label: str,
        *,
        icon: str | None,
        locked: bool,
        on_click: Callable[[], object],
        active_classes: str = "",
    ) -> None:
        """Bouw één CTA-knop — normaal, of uitgeschakeld-maar-klikbaar bij ``locked``."""
        if locked:
            btn = ui.button(
                label, icon=icon, on_click=self._warn_project_active
            ).props("unelevated")
            # Inline !important omdat Quasar de knopkleur in een CSS-@layer zet;
            # inline stijl wint gegarandeerd van een gelaagde stylesheet-regel.
            btn.classes("sp-btn-locked")
            btn.style(
                "background:#e4e4e4 !important;color:#9a9a9a !important;"
                "box-shadow:none !important;cursor:not-allowed !important;"
            )
            btn.tooltip("Er is al een project actief — reset eerst om opnieuw te beginnen")
            return
        btn = ui.button(label, icon=icon, on_click=on_click).props("unelevated")
        if active_classes:
            btn.classes(active_classes)

    def _render_active_project_hint(self) -> None:
        """Subtiele regel onder de knoppen die het actieve project benoemt."""
        A = theme.ACCENT
        project_name = os.path.basename(STATE.project_dir or "") or "Naamloos project"
        with ui.row().classes("items-center gap-2 no-wrap mt-1 px-3 py-1 rounded-lg").style(
            f"background:{A}0d;border:1px solid {A}22;"
        ):
            ui.icon("lock").classes("text-sm").style(f"color:{A}")
            ui.html(
                f'Project <strong>{project_name}</strong> is actief · reset via '
                f'de knop <span style="white-space:nowrap;">'
                f'<span class="material-icons" style="font-size:13px;'
                f'vertical-align:text-bottom;">restart_alt</span> Reset</span> '
                f"rechtsboven om opnieuw te beginnen."
            ).classes("text-xs").style("color:#666;")

    def _warn_project_active(self) -> None:
        """Uitleg-dialoog: er is al een project actief; eerst resetten."""
        A = theme.ACCENT
        project_name = os.path.basename(STATE.project_dir or "") or "je huidige project"
        with ui.dialog() as dialog, ui.card().style(
            "width:460px;max-width:95vw;border-radius:16px;overflow:hidden;padding:0;"
        ):
            # Header
            with ui.row().classes(
                "w-full items-center justify-between px-5 pt-4 pb-3"
            ).style("border-bottom:1px solid #f0f0f0;"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    with ui.element("div").style(
                        f"width:32px;height:32px;border-radius:8px;background:{A}15;"
                        "display:flex;align-items:center;justify-content:center;"
                    ):
                        ui.icon("lock").classes("text-lg").style(f"color:{A}")
                    ui.label("Er is al een project actief").classes(
                        "font-semibold text-base"
                    )
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense color=grey-6"
                )
            # Body
            with ui.column().classes("w-full px-5 pt-4 pb-5 gap-3"):
                ui.html(
                    f"Je werkt op dit moment in het project "
                    f"<strong>{project_name}</strong>. Er kan maar één project "
                    f"tegelijk open zijn. Wil je een nieuw project opzetten of de "
                    f"demo draaien, reset dan eerst het huidige project."
                ).classes("text-sm leading-relaxed").style("color:#555;")
                with ui.element("div").classes("w-full").style(
                    f"padding:10px 12px;border-radius:8px;background:{A}0a;"
                    f"border:1px solid {A}25;"
                ):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("restart_alt").classes("text-lg").style(f"color:{A}")
                        ui.label(
                            "Gebruik de knop “Reset” rechtsboven, of reset "
                            "direct hieronder."
                        ).classes("text-xs leading-relaxed").style("color:#555;")
                with ui.row().classes("w-full justify-end gap-2 no-wrap mt-1"):
                    ui.button("Sluiten", on_click=dialog.close).props(
                        "flat color=grey-7"
                    )

                    def _reset() -> None:
                        STATE.project_dir = None
                        STATE.config_saved = False
                        dialog.close()
                        ui.navigate.to("/")

                    ui.button(
                        "Project resetten", icon="restart_alt", on_click=_reset
                    ).props("unelevated color=accent")
        dialog.open()

    # ── Demo-uitvoering ───────────────────────────────────────────────────────

    async def _run_demo(self) -> None:
        """Voer init → demodata → pipeline automatisch uit in een tijdelijke map."""
        project_dir = tempfile.mkdtemp(prefix="sp-demo-")
        STATE.project_dir = project_dir
        self._step_n = 0

        self._container.clear()
        with self._container:
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center gap-3 no-wrap mb-3"):
                    with ui.element("div").style(
                        f"width:44px;height:44px;border-radius:10px;"
                        f"background:{theme.ACCENT}1a;"
                        "display:flex;align-items:center;justify-content:center;flex-shrink:0"
                    ):
                        ui.icon("rocket_launch").classes("text-2xl").style(
                            f"color:{theme.ACCENT}"
                        )
                    with ui.column().classes("gap-0"):
                        ui.label("Demo wordt uitgevoerd").classes(
                            "text-lg font-semibold"
                        )
                        ui.label(project_dir).classes("text-xs font-mono opacity-50")

                ui.separator().classes("my-1")

                self._steps = ui.column().classes("w-full gap-2 my-2")
                self._progress = ui.linear_progress(value=0.0, show_value=False)
                self._progress.set_visibility(False)
                self._pipeline_progress = ProgressCard()

                with ui.expansion("Uitvoerlog", icon="terminal").props(
                    "dense"
                ).classes("w-full mt-2"):
                    panel = ProcessPanel()

                self._error = ui.column().classes("w-full")

        try:
            self._step("Projectmap aanmaken…")
            if await panel.run(["init"], cwd=project_dir) != 0:
                return

            filtering_io.save_filtering(
                os.path.join(project_dir, "configuration", "filtering", "base.json"),
                _DEMO_FILTERING,
            )

            self._step("Demodata downloaden…")
            await self._download_demodata(project_dir)

            self._step("Voorspelling draaien (cumulatief spoor)…")
            self._pipeline_progress.start()
            rc = await panel.run(
                ["-d", "c", "-w", _DEMO_WEEK, "-y", _DEMO_YEAR, "--yes"],
                cwd=project_dir,
                on_line=self._pipeline_progress.on_line,
            )
            self._pipeline_progress.complete(success=rc == 0)
        except Exception as exc:  # noqa: BLE001
            with self._error:
                error_banner("De demo kon niet worden voltooid.", f"Details: {exc}")
            return

        if rc == 0:
            self._on_demo_done()

    def _on_demo_done(self) -> None:
        self._step("Klaar!", done=True)
        with self._steps:
            ui.button(
                "Bekijk resultaten",
                icon="insights",
                on_click=lambda: ui.navigate.to("/output"),
            ).props("unelevated").classes("mt-1")
        ui.timer(0.8, lambda: ui.navigate.to("/output"), once=True)

    def _step(self, text: str, *, done: bool = False) -> None:
        self._step_n += 1
        n = self._step_n
        with self._steps:
            with ui.row().classes("items-center gap-3 py-1"):
                if done:
                    ui.icon("check_circle").props("color=positive").classes("text-2xl")
                else:
                    with ui.element("div").classes("sp-demo-step-dot").style(
                        f"background:{theme.ACCENT}"
                    ):
                        ui.label(str(n)).style(
                            "color:white;font-size:12px;font-weight:700"
                        )
                ui.label(text).classes("text-sm font-medium" if done else "text-sm")

    async def _download_demodata(self, project_dir: str) -> None:
        dest = os.path.join(project_dir, "data", "input_raw")
        self._progress.props("indeterminate")
        self._progress.set_visibility(True)

        def _work() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = os.path.join(tmp, "demo-data.zip")
                demodata.download_file(demodata.DEMO_URL, zip_path)
                demodata.extract_zip(zip_path, dest)

        try:
            await asyncio.to_thread(_work)
        finally:
            self._progress.set_visibility(False)
