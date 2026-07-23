"""ProcessPanel: draai een CLI-commando en stream de uitvoer live naar de UI.

Gedeeld door de init-wizard (#266), de pipeline-runner (#268) en de benchmark-tab
(#270). Toont een statusbadge, een live-log en een stop-knop, en vertaalt fouten
via :func:`gui.errors.humanize_error`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from nicegui import ui

from gui import theme
from gui.errors import humanize_error
from gui.process import locate_cli

#: Status → (label, kleur, icoon).
_STATUS = {
    "idle": ("Nog niet gestart", theme.DARK, "radio_button_unchecked"),
    "running": ("Bezig…", theme.INFO, "hourglass_top"),
    "success": ("Geslaagd", theme.POSITIVE, "task_alt"),
    "failed": ("Mislukt", theme.NEGATIVE, "error"),
}


class ProcessPanel:
    """UI-paneel dat één subprocess tegelijk draait en de uitvoer streamt.

    Example:
        panel = ProcessPanel()
        await panel.run(["init"], cwd=project_dir, on_success=go_next)
    """

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._error_slot: ui.element
        self._all_lines: list[str] = []
        self._build()

    def output_text(self) -> str:
        """De volledige verzamelde uitvoer van de laatste run (voor parsing)."""
        return "\n".join(self._all_lines)

    def _build(self) -> None:
        with ui.column().classes("w-full gap-2"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    self._icon = ui.icon("radio_button_unchecked")
                    self._label = ui.label("Nog niet gestart").classes("text-sm")
                self._stop_btn = ui.button(
                    "Stop", icon="stop", on_click=self.stop
                ).props("flat color=negative")
                self._stop_btn.set_visibility(False)
            self._log = ui.log(max_lines=2000).classes(
                "w-full h-64 bg-grey-10 text-white rounded p-2 text-xs"
            )
            self._error_slot = ui.column().classes("w-full")

    def _set_status(self, state: str) -> None:
        label, color, icon = _STATUS[state]
        self._label.set_text(label)
        self._icon.props(f"name={icon}")
        self._icon.style(f"color: {color}")
        self._label.style(f"color: {color}")
        self._stop_btn.set_visibility(state == "running")

    def stop(self) -> None:
        """Beëindig het lopende subprocess (zo mogelijk)."""
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            self._log.push("— gestopt door gebruiker —")

    async def run(
        self,
        args: list[str],
        *,
        cwd: str,
        on_success: Callable[[], None] | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> int:
        """Draai ``studentprognose <args>`` in ``cwd`` en stream de uitvoer.

        Args:
            args: Argumenten ná ``studentprognose``.
            cwd: Werkmap voor het subprocess.
            on_success: Callback bij exitcode 0.

        Returns:
            De exitcode van het proces (of ``-1`` bij een startfout).
        """
        self._error_slot.clear()
        self._log.clear()
        self._all_lines = []
        self._set_status("running")

        try:
            exe = locate_cli()
        except FileNotFoundError as exc:
            self._set_status("failed")
            self._render_error(str(exc))
            return -1

        try:
            self._process = await asyncio.create_subprocess_exec(
                exe,
                *args,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            self._set_status("failed")
            self._render_error(str(exc))
            return -1

        assert self._process.stdout is not None
        tail: list[str] = []
        async for raw in self._process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            self._log.push(line)
            self._all_lines.append(line)
            tail.append(line)
            if on_line is not None:
                on_line(line)
            if len(tail) > 40:
                tail.pop(0)

        returncode = await self._process.wait()

        if returncode == 0:
            self._set_status("success")
            if on_success is not None:
                on_success()
        else:
            self._set_status("failed")
            self._render_error("\n".join(tail))
        return returncode

    def _render_error(self, raw: str) -> None:
        """Toon een actiegerichte foutmelding onder de log."""
        from gui.components.states import error_banner

        friendly = humanize_error(raw)
        with self._error_slot:
            error_banner(friendly.message, friendly.hint)
