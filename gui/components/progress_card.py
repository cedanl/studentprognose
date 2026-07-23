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
    ("Data laden",         0.22, "Loading data"),
    ("Voorbereiden",       0.42, "Preprocessing"),
    ("Voorspellen",        0.64, "Predicting first-years"),
    ("Nabewerking",        0.88, "Postprocessing"),
    ("Opslaan",            0.97, "Saving output"),
]

_CREEP_PER_TICK = 0.0007  # kruipsnelheid per 200ms-tick (~0.35% per seconde)
_EASE           = 0.22    # deel van de kloof dat per tick wordt gedicht
_TICK_MS        = 200     # timer-interval in milliseconden


def _fmt(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"


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
        self._phase_lbl.style(f"color:{theme.ACCENT}")
        self._timer_lbl.set_text("")
        self._apply_bar(0.0)
        self._render_stepper()

        if self._timer is not None:
            self._timer.cancel()
        self._timer = ui.timer(_TICK_MS / 1000, self._tick)

    def on_line(self, line: str) -> None:
        """Verwerk één logregel en spring naar de bijbehorende fase."""
        for i, (label, target, marker) in enumerate(_PHASES):
            if marker in line and i > self._phase_idx:
                self._phase_idx = i
                self._target = target
                self._phase_lbl.set_text(label + "…")
                self._render_stepper()
                break

    def complete(self, *, success: bool) -> None:
        """Markeer de run als afgerond."""
        self._done = True
        self._success = success
        self._target = 1.0
        if success:
            self._phase_lbl.set_text("Klaar")
            self._phase_lbl.style(f"color:{theme.POSITIVE}")
        else:
            self._phase_lbl.set_text("Mislukt")
            self._phase_lbl.style(f"color:{theme.NEGATIVE}")
        self._render_stepper()

    # ------------------------------------------------------------------
    # Interne opbouw
    # ------------------------------------------------------------------

    def _build(self) -> None:
        with ui.card().classes("w-full").style("gap:12px") as self._card:
            self._card.set_visibility(False)

            # Stepper (HTML, volledig vervangen bij fase-overgang)
            self._stepper_el = ui.html("").classes("w-full")

            # Voortgangsbalk
            with ui.element("div").style(
                "width:100%; height:8px; background:#f0f0f0; "
                "border-radius:9999px; overflow:hidden; margin:4px 0"
            ):
                self._bar_el = ui.element("div").style(
                    f"height:100%; width:0%; border-radius:9999px; "
                    f"background:linear-gradient(90deg,{theme.ACCENT},{theme.SECONDARY}); "
                    "transition:width 0.3s ease-out"
                )

            # Onderschrift: fase links, timer rechts
            with ui.row().classes("w-full items-center justify-between"):
                self._phase_lbl = (
                    ui.label("")
                    .classes("text-sm font-medium")
                    .style(f"color:{theme.ACCENT}")
                )
                self._timer_lbl = (
                    ui.label("")
                    .classes("text-xs")
                    .style(f"color:{theme.MUTED}")
                )

    # ------------------------------------------------------------------
    # Animatie
    # ------------------------------------------------------------------

    def _apply_bar(self, value: float) -> None:
        pct = value * 100
        if self._success is True:
            gradient = (
                f"linear-gradient(90deg,{theme.POSITIVE},{theme.SECONDARY})"
            )
        elif self._success is False:
            gradient = f"linear-gradient(90deg,{theme.NEGATIVE},{theme.NEGATIVE})"
        else:
            gradient = (
                f"linear-gradient(90deg,{theme.ACCENT},{theme.SECONDARY})"
            )
        self._bar_el.style(
            f"height:100%; width:{pct:.1f}%; border-radius:9999px; "
            f"background:{gradient}; transition:width 0.3s ease-out"
        )

    def _tick(self) -> None:
        # Stop zodra done + volledig geanimeerd
        if self._done and self._display >= 0.997:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._apply_bar(1.0)
            if self._start_time:
                elapsed = time.time() - self._start_time
                self._timer_lbl.set_text(f"Klaar in {_fmt(elapsed)}")
            return

        # Kruip langzaam binnen de huidige fase
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
        self._display = max(0.0, min(1.0, self._display + gap * _EASE))
        self._apply_bar(self._display)

        # Tijdlabel
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if self._done:
                pass  # afgehandeld in stop-conditie
            elif self._display > 0.10 and elapsed > 6:
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
            # is_future = not is_done and not is_active  # implied

            if is_done:
                dot_bg, dot_fg, dot_inner = theme.POSITIVE, "white", "✓"
                lbl_color, lbl_weight = theme.POSITIVE, "500"
            elif is_active:
                dot_bg, dot_fg, dot_inner = theme.ACCENT, "white", str(i + 1)
                lbl_color, lbl_weight = theme.ACCENT, "600"
            else:
                dot_bg, dot_fg, dot_inner = "#e0e0e0", "#9e9e9e", str(i + 1)
                lbl_color, lbl_weight = "#9e9e9e", "400"

            dot_html = (
                f'<div style="width:26px;height:26px;border-radius:50%;'
                f'background:{dot_bg};color:{dot_fg};'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:11px;font-weight:700;flex-shrink:0;'
                f'transition:background 0.35s,color 0.35s">'
                f"{dot_inner}</div>"
            )
            lbl_html = (
                f'<span style="font-size:10px;line-height:1.3;text-align:center;'
                f"color:{lbl_color};font-weight:{lbl_weight};"
                f'display:block;max-width:68px;'
                f'transition:color 0.35s">'
                f"{label}</span>"
            )
            col_html = (
                f'<div style="display:flex;flex-direction:column;'
                f'align-items:center;gap:4px;flex-shrink:0">'
                f"{dot_html}{lbl_html}</div>"
            )
            parts.append(col_html)

            if i < n - 1:
                line_color = theme.POSITIVE if is_done else "#e8e8e8"
                line_html = (
                    f'<div style="flex:1;height:2px;background:{line_color};'
                    f'margin-top:12px;min-width:8px;'
                    f'transition:background 0.35s"></div>'
                )
                parts.append(line_html)

        return (
            '<div style="display:flex;align-items:flex-start;'
            'width:100%;padding:2px 0">'
            + "".join(parts)
            + "</div>"
        )
