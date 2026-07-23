"""ProgressCard: geanimeerde fase-voortgangsbalk voor de pipeline-runner.

Detecteert bekende log-markers van ``studentprognose`` en beweegt een
voortgangsbalk van fase naar fase. Gebruikt ``ui.timer`` voor vloeiende
animatie en een verstreken/geschatte tijdweergave.
"""

from __future__ import annotations

import time

from nicegui import ui

from gui import theme

#: Fasen: (weergavenaam, doel-progress bij bereik, log-marker-substring).
_PHASES: list[tuple[str, float, str]] = [
    ("Configuratie laden", 0.06, "Loading configuration"),
    ("Data laden", 0.22, "Loading data"),
    ("Voorbereiden", 0.42, "Preprocessing"),
    ("Voorspellen", 0.64, "Predicting first-years"),
    ("Nabewerking", 0.88, "Postprocessing"),
    ("Opslaan", 0.97, "Saving output"),
]

_CREEP_PER_TICK = 0.0007  # kruipsnelheid per 200ms-tick (~0.35%/s)
_EASE_RUNNING = 0.22  # ease-factor tijdens run
_EASE_DONE = 0.55  # snellere ease-factor bij afronden
_TICK_MS = 200


def _fmt(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"


#: CSS eenmalig in de <head> injecteren voor de pulse-ring animatie.
_PULSE_CSS = """
<style>
@keyframes _sp_pulse {
  0%   { box-shadow: 0 0 0 0   rgba(221,120,75,.50); }
  65%  { box-shadow: 0 0 0 7px rgba(221,120,75,.00); }
  100% { box-shadow: 0 0 0 0   rgba(221,120,75,.00); }
}
._sp_dot_active { animation: _sp_pulse 1.6s ease-out infinite; }
</style>
"""


class ProgressCard:
    """Fase-voortgangsbalk die log-regels van de pipeline verwerkt.

    Example::
        card = ProgressCard()
        card.start()
        await panel.run(args, cwd=..., on_line=card.on_line)
        card.complete(success=True)
    """

    def __init__(self) -> None:
        self._phase_idx: int = -1
        self._target: float = 0.0
        self._display: float = 0.0
        self._start_time: float | None = None
        self._done: bool = False
        self._success: bool | None = None
        self._timer: ui.timer | None = None
        self._build()
        # Timer created here (in build context) and kept inactive until start().
        # Creating it in start() fails when the triggering element's context slot
        # has been deleted by container.clear() just before start() is called.
        self._timer = ui.timer(_TICK_MS / 1000, self._tick, active=False)

    # ------------------------------------------------------------------
    # Publieke API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Reset de kaart, maak hem zichtbaar en start de timer."""
        self._phase_idx = -1
        self._target = 0.0
        self._display = 0.0
        self._done = False
        self._success = None
        self._start_time = time.time()

        self._card.set_visibility(True)
        self._phase_lbl.set_text("Starten…")
        self._phase_lbl.style(f"color:{theme.ACCENT}; font-weight:600")
        self._step_lbl.set_text("")
        self._timer_lbl.set_text("")
        self._apply_bar(0.0)
        self._render_stepper()

        if self._timer is not None:
            self._timer.interval = _TICK_MS / 1000
            self._timer.active = True

    def on_line(self, line: str) -> None:
        """Verwerk één logregel en spring naar de bijbehorende fase."""
        for i, (label, target, marker) in enumerate(_PHASES):
            if marker in line and i > self._phase_idx:
                self._phase_idx = i
                self._target = target
                self._phase_lbl.set_text(label + "…")
                self._step_lbl.set_text(f"Stap {i + 1} van {len(_PHASES)}")
                self._render_stepper()
                break

    def complete(self, *, success: bool) -> None:
        """Markeer de run als afgerond."""
        self._done = True
        self._success = success
        self._step_lbl.set_text("")
        if success:
            self._target = 1.0  # succesvolle run: vul de balk helemaal
            self._phase_lbl.set_text("Klaar")
            self._phase_lbl.style(f"color:{theme.POSITIVE}; font-weight:600")
        else:
            # Mislukt: stop op huidige positie — eerlijk beeld van hoever het kwám
            self._phase_lbl.set_text("Mislukt")
            self._phase_lbl.style(f"color:{theme.NEGATIVE}; font-weight:600")
        self._render_stepper()

    # ------------------------------------------------------------------
    # Interne opbouw
    # ------------------------------------------------------------------

    def _build(self) -> None:
        ui.add_head_html(_PULSE_CSS)

        with ui.card().classes("w-full") as self._card:
            self._card.set_visibility(False)

            # Stepper (HTML, volledig vervangen bij fase-overgang)
            self._stepper_el = ui.html("").classes("w-full")

            # Voortgangsbalk — iets hoger voor meer gewicht
            with ui.element("div").style(
                "width:100%; height:10px; background:#eeeeee; "
                "border-radius:9999px; overflow:hidden; margin:2px 0 6px"
            ):
                self._bar_el = ui.element("div").style(
                    f"height:100%; width:0%; border-radius:9999px; "
                    f"background:linear-gradient(90deg,{theme.ACCENT},{theme.SECONDARY}); "
                    "transition:width 0.35s ease-out"
                )

            # Onderschrift: fase + stapnummer links, timer rechts
            with ui.row().classes("w-full items-center justify-between gap-2"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    self._phase_lbl = (
                        ui.label("")
                        .classes("text-sm")
                        .style(f"color:{theme.ACCENT}; font-weight:600")
                    )
                    self._step_lbl = (
                        ui.label("").classes("text-xs").style(f"color:{theme.MUTED}")
                    )
                self._timer_lbl = (
                    ui.label("")
                    .classes("text-xs text-right")
                    .style(f"color:{theme.MUTED}")
                )

    # ------------------------------------------------------------------
    # Animatie
    # ------------------------------------------------------------------

    def _apply_bar(self, value: float) -> None:
        pct = value * 100
        if self._success is True:
            gradient = f"linear-gradient(90deg,{theme.POSITIVE},{theme.NPULS_BLUE})"
        elif self._success is False:
            gradient = f"linear-gradient(90deg,{theme.NEGATIVE},{theme.NEGATIVE}cc)"
        else:
            gradient = f"linear-gradient(90deg,{theme.ACCENT},{theme.SECONDARY})"
        self._bar_el.style(
            f"height:100%; width:{pct:.1f}%; border-radius:9999px; "
            f"background:{gradient}; transition:width 0.35s ease-out"
        )

    def _tick(self) -> None:  # noqa: C901
        try:
            self._tick_inner()
        except RuntimeError:
            # Element deleted (browser tab closed mid-run).
            if self._timer:
                self._timer.active = False

    def _tick_inner(self) -> None:
        ease = _EASE_DONE if self._done else _EASE_RUNNING

        # Stop zodra done + op eindpositie geconvergeerd
        converged = (
            self._display >= 0.997                        # succes → 100%
            if self._success
            else abs(self._display - self._target) < 0.002  # fout → huidige positie
        )
        if self._done and converged:
            if self._timer:
                self._timer.active = False
            self._apply_bar(1.0 if self._success else self._display)
            if self._start_time:
                elapsed = time.time() - self._start_time
                label = "Klaar in" if self._success else "Gestopt na"
                self._timer_lbl.set_text(f"{label} {_fmt(elapsed)}")
            return

        # Langzaam kruipen binnen de huidige fase
        if not self._done and self._phase_idx >= 0:
            next_i = self._phase_idx + 1
            ceiling = (
                _PHASES[next_i][1] - 0.04
                if next_i < len(_PHASES)
                else _PHASES[self._phase_idx][1]
            )
            if self._target < ceiling:
                self._target = min(self._target + _CREEP_PER_TICK, ceiling)

        # Vloeiend bewegen richting doel
        gap = self._target - self._display
        self._display = max(0.0, min(1.0, self._display + gap * ease))
        self._apply_bar(self._display)

        # Tijdlabel
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if not self._done:
                if self._display > 0.10 and elapsed > 6:
                    remaining = (elapsed / self._display) * (1.0 - self._display)
                    self._timer_lbl.set_text(
                        f"{_fmt(elapsed)} verstreken · ~{_fmt(remaining)} resterend"
                    )
                else:
                    self._timer_lbl.set_text(f"{_fmt(elapsed)} verstreken")

    # ------------------------------------------------------------------
    # Stepper HTML
    # ------------------------------------------------------------------

    def _render_stepper(self) -> None:
        self._stepper_el.set_content(self._build_stepper_html())

    def _build_stepper_html(self) -> str:
        parts: list[str] = []
        n = len(_PHASES)
        for i, (label, _prog, _marker) in enumerate(_PHASES):
            is_done = i < self._phase_idx or (self._done and i <= self._phase_idx)
            is_active = i == self._phase_idx and not self._done

            if is_done:
                dot_bg, dot_fg, dot_inner = theme.POSITIVE, "white", "✓"
                lbl_color, lbl_weight = theme.POSITIVE, "500"
                extra_class = ""
            elif is_active:
                dot_bg, dot_fg, dot_inner = theme.ACCENT, "white", str(i + 1)
                lbl_color, lbl_weight = theme.ACCENT, "600"
                extra_class = "_sp_dot_active"
            else:
                dot_bg, dot_fg, dot_inner = "#e2e2e2", "#aaaaaa", str(i + 1)
                lbl_color, lbl_weight = "#aaaaaa", "400"
                extra_class = ""

            dot_html = (
                f'<div class="{extra_class}" '
                f'style="width:28px;height:28px;border-radius:50%;'
                f"background:{dot_bg};color:{dot_fg};"
                f"display:flex;align-items:center;justify-content:center;"
                f'font-size:11px;font-weight:700;flex-shrink:0">'
                f"{dot_inner}</div>"
            )
            lbl_html = (
                f'<span style="font-size:11px;line-height:1.3;text-align:center;'
                f"color:{lbl_color};font-weight:{lbl_weight};"
                f'display:block;max-width:72px">'
                f"{label}</span>"
            )
            col_html = (
                f'<div style="display:flex;flex-direction:column;'
                f'align-items:center;gap:5px;flex-shrink:0">'
                f"{dot_html}{lbl_html}</div>"
            )
            parts.append(col_html)

            if i < n - 1:
                line_color = theme.POSITIVE if is_done else "#e8e8e8"
                line_html = (
                    f'<div style="flex:1;height:2px;background:{line_color};'
                    f'margin-top:13px;min-width:8px"></div>'
                )
                parts.append(line_html)

        return (
            '<div style="display:flex;align-items:flex-start;'
            'width:100%;padding:4px 0 8px">' + "".join(parts) + "</div>"
        )
