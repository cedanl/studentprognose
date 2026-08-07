"""UX Flow pagina — /uxflow.

Interactieve visualisatie van de twee gebruikersroutes door de applicatie:
nieuwe gebruiker (via wizard) en terugkerende gebruiker (snel-start).
"""

from __future__ import annotations

from nicegui import ui

from gui import nav
from gui.components.layout import page_shell


def create() -> None:
    """Registreer de route /uxflow."""
    nav.register_route("/uxflow")

    @ui.page("/uxflow")
    def _page() -> None:
        with page_shell(active="/uxflow", title="UX Flow"):
            ui.add_head_html(_CSS)
            ui.html(_HTML)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* UX Flow — alle klassen met uxf- prefix om Quasar niet te verstoren */

@keyframes uxf-fadein {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.uxf-root {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  max-width: 880px;
  margin: 0 auto;
  padding: 0 0 56px;
  color: #1a1a1a;
  /* Subtle dot grid background */
  background-image: radial-gradient(circle, #e0e0e0 1px, transparent 1px);
  background-size: 24px 24px;
  background-position: 0 0;
  border-radius: 16px;
  padding: 24px 28px 56px;
}

/* ── Header ── */
.uxf-header {
  text-align: center;
  padding: 4px 0 28px;
  animation: uxf-fadein 0.5s ease both;
}
.uxf-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #1a1a1a;
  color: white;
  border-radius: 100px;
  padding: 5px 16px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}
.uxf-title {
  font-size: 24px;
  font-weight: 700;
  color: #111;
  margin: 0 0 7px;
  letter-spacing: -0.3px;
}
.uxf-subtitle {
  color: #777;
  font-size: 13.5px;
  margin: 0;
}

/* ── Legenda ── */
.uxf-legend {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  padding: 11px 18px;
  background: white;
  border-radius: 12px;
  border: 1px solid #e8e8e8;
  margin-bottom: 32px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  animation: uxf-fadein 0.5s 0.05s ease both;
}
.uxf-leg {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  font-weight: 500;
  color: #555;
}
.uxf-dot {
  width: 11px;
  height: 11px;
  border-radius: 3px;
  flex-shrink: 0;
}
.uxf-dot-screen   { background: #3D68EC; }
.uxf-dot-wizard   { background: #DD784B; }
.uxf-dot-decision { background: #8b5cf6; border-radius: 2px; transform: rotate(45deg); }
.uxf-dot-process  { background: #8b5cf6; opacity: 0.6; }
.uxf-dot-success  { background: #00AF81; }
.uxf-dot-error    { background: #C0392B; }

/* ── Fase-label ── */
.uxf-phase {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 28px 0 12px;
}
.uxf-phase::before,
.uxf-phase::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, transparent, #ddd, transparent);
}
.uxf-phase span {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #b0b0b0;
  white-space: nowrap;
  background: white;
  padding: 3px 12px;
  border-radius: 100px;
  border: 1px solid #e8e8e8;
}

/* ── Verticale connector ── */
.uxf-vcon {
  width: 3px;
  height: 32px;
  background: linear-gradient(to bottom, #c8c8c8, #b0b0b0);
  margin: 0 auto;
  position: relative;
  border-radius: 2px;
}
.uxf-vcon::after {
  content: '';
  position: absolute;
  bottom: -7px;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: #b0b0b0;
}

/* ── Knooppunt-kaart ── */
.uxf-card {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  background: white;
  border-radius: 14px;
  padding: 16px 20px;
  border-left: 4px solid transparent;
  box-shadow: 0 2px 6px rgba(0,0,0,0.07), 0 6px 18px rgba(0,0,0,0.04);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  position: relative;
  animation: uxf-fadein 0.4s ease both;
}
.uxf-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.1), 0 12px 32px rgba(0,0,0,0.06);
}
.uxf-card-sm { padding: 12px 16px; }

.uxf-c-screen  { border-left-color: #3D68EC; }
.uxf-c-action  { border-left-color: #DD784B; }
.uxf-c-process { border-left-color: #8b5cf6; }
.uxf-c-success { border-left-color: #00AF81; }
.uxf-c-error   { border-left-color: #C0392B; }

.uxf-card-icon {
  font-size: 22px;
  flex-shrink: 0;
  line-height: 1;
  margin-top: 2px;
  width: 32px;
  text-align: center;
}
.uxf-card-body { flex: 1; min-width: 0; }

.uxf-tag {
  display: inline-block;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.09em;
  padding: 2px 8px;
  border-radius: 4px;
  margin-bottom: 5px;
}
.uxf-tag-screen  { background: #e8eeff; color: #2f5bd8; }
.uxf-tag-action  { background: #fff0e6; color: #c45f2a; }
.uxf-tag-process { background: #f3eeff; color: #6d28d9; }
.uxf-tag-success { background: #e3f7ef; color: #008059; }
.uxf-tag-error   { background: #fde8e6; color: #a12315; }
.uxf-tag-wizard  { background: #fff0e6; color: #c45f2a; }
.uxf-tag-optional { background: #f0f0f0; color: #888; }

.uxf-card-title {
  font-size: 15px;
  font-weight: 700;
  color: #111;
  margin-bottom: 4px;
  line-height: 1.3;
}
.uxf-card-desc {
  font-size: 12.5px;
  color: #555;
  line-height: 1.6;
}
.uxf-card-hint {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11.5px;
  color: #888;
  margin-top: 7px;
  font-style: italic;
  background: #f9f9f7;
  padding: 3px 9px;
  border-radius: 5px;
}
.uxf-route {
  font-size: 10.5px;
  font-weight: 700;
  font-family: 'SFMono-Regular', 'Consolas', monospace;
  color: #444;
  background: #efefef;
  border: 1.5px solid #d0d0d0;
  border-radius: 5px;
  padding: 3px 9px;
  flex-shrink: 0;
  margin-top: 2px;
  align-self: flex-start;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

/* ── Diamond beslissingsknoop ── */
.uxf-decision {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0 6px;
}
.uxf-diamond {
  width: 92px;
  height: 92px;
  background: linear-gradient(135deg, #9d5cf6, #7c3aed);
  border-radius: 8px;
  transform: rotate(45deg);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 6px 20px rgba(124,58,237,0.3), 0 2px 6px rgba(124,58,237,0.2);
}
.uxf-diamond-text {
  transform: rotate(-45deg);
  font-size: 12px;
  font-weight: 800;
  color: white;
  text-align: center;
  line-height: 1.4;
  letter-spacing: 0.01em;
}

/* ── Branch-balk ── */
.uxf-branch {
  display: flex;
  align-items: center;
  gap: 0;
  margin: 8px 0 0;
  position: relative;
}
.uxf-branch-lbl {
  font-size: 11px;
  font-weight: 700;
  padding: 4px 13px;
  border-radius: 100px;
  white-space: nowrap;
  border: 1.5px solid transparent;
}
.uxf-branch-lbl-no {
  color: #b85a28;
  background: #fff3ec;
  border-color: #DD784B55;
}
.uxf-branch-lbl-yes {
  color: #2a52c5;
  background: #edf0ff;
  border-color: #3D68EC55;
}
.uxf-branch-line {
  flex: 1;
  height: 2px;
  background: linear-gradient(to right, #DD784B44, #e0e0e0, #3D68EC44);
}

/* ── Twee rijbanen ── */
.uxf-lanes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.uxf-lane {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.uxf-lane-hdr {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  font-weight: 700;
  padding: 8px 14px;
  border-radius: 10px;
  letter-spacing: 0.01em;
}
.uxf-lane-hdr-new {
  background: linear-gradient(to right, #fff3ec, #fff8f5);
  color: #b85a28;
  border: 1px solid #ffe0cc;
}
.uxf-lane-hdr-return {
  background: linear-gradient(to right, #edf0ff, #f3f6ff);
  color: #2a52c5;
  border: 1px solid #d4dcff;
}

/* ── Wizard blok ── */
.uxf-wizard {
  background: white;
  border-radius: 14px;
  border: 1px solid #ffd9c0;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  flex: 1;
}
.uxf-wizard-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 11px 15px;
  background: linear-gradient(to right, #fffaf6, #fff8f2);
  border-bottom: 1px solid #ffd9c0;
}
.uxf-wizard-hdr-title {
  font-size: 12.5px;
  font-weight: 700;
  color: #c05a20;
}
.uxf-wstep {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 15px;
  border-bottom: 1px solid #faf3ef;
  transition: background 0.12s;
  cursor: default;
}
.uxf-wstep:last-child { border-bottom: none; }
.uxf-wstep:hover { background: #fdf7f3; }
.uxf-wstep-num {
  width: 22px;
  height: 22px;
  background: #DD784B;
  color: white;
  border-radius: 50%;
  font-size: 10.5px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 1px;
}
.uxf-wstep-warn .uxf-wstep-num {
  background: linear-gradient(135deg, #E6A020, #d4900a);
}
.uxf-wstep-title {
  font-size: 13px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 2px;
}
.uxf-wstep-desc {
  font-size: 11.5px;
  color: #666;
  line-height: 1.45;
}
.uxf-wstep-note {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: #c27f10;
  margin-top: 4px;
  background: #fffbf0;
  padding: 2px 7px;
  border-radius: 4px;
  border-left: 2px solid #E6A020;
}
.uxf-wstep-body { flex: 1; }

/* ── Keuze-blok (terugkerende gebruiker) ── */
.uxf-choices {
  background: white;
  border-radius: 14px;
  border: 1px solid #d4dcff;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  flex: 1;
}
.uxf-choices-hdr {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #aaa;
  padding: 10px 15px 8px;
  border-bottom: 1px solid #f0f0f0;
  background: #fafbff;
}
.uxf-choice {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 11px 15px;
  border-bottom: 1px solid #f5f5f5;
  position: relative;
  transition: background 0.12s;
  cursor: default;
}
.uxf-choice:last-child { border-bottom: none; }
.uxf-choice:hover { background: #f7f9ff; }
.uxf-choice-primary {
  background: linear-gradient(to right, #f5f8ff, #f0f4ff);
}
.uxf-choice-primary:hover { background: linear-gradient(to right, #eef2ff, #e8eeff); }
.uxf-choice-icon { font-size: 18px; flex-shrink: 0; }
.uxf-choice-title {
  font-size: 13px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 2px;
}
.uxf-choice-desc { font-size: 11px; color: #888; }
.uxf-choice-badge {
  position: absolute;
  right: 11px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 9px;
  font-weight: 800;
  background: #3D68EC;
  color: white;
  border-radius: 5px;
  padding: 3px 8px;
  letter-spacing: 0.04em;
  box-shadow: 0 2px 6px rgba(61,104,236,0.3);
}

/* ── Merge-balk ── */
.uxf-merge {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 20px 0 0;
}
.uxf-merge-line {
  flex: 1;
  height: 2px;
  background: linear-gradient(to right, transparent, #c8c8c8, transparent);
  border-radius: 2px;
}
.uxf-merge-lbl {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: #999;
  white-space: nowrap;
  padding: 5px 16px;
  border: 1.5px solid #dedede;
  border-radius: 100px;
  background: white;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

/* ── Gedeeld-pad sectie ── */
.uxf-shared {
  background: linear-gradient(to bottom, #fafafa, white);
  border-radius: 16px;
  border: 1px solid #ebebeb;
  padding: 0 20px 20px;
  margin-top: 0;
  box-shadow: 0 2px 12px rgba(0,0,0,0.03);
}
.uxf-shared-intro {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 0 8px;
  font-size: 11px;
  font-weight: 700;
  color: #aaa;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.uxf-shared-intro::before,
.uxf-shared-intro::after {
  content: '';
  width: 24px;
  height: 2px;
  background: #ddd;
  border-radius: 2px;
}

/* ── Uitkomst-lanes (succes / fout) ── */
.uxf-outcomes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.uxf-outcome { display: flex; flex-direction: column; gap: 8px; }
.uxf-outcome-hdr {
  font-size: 12px;
  font-weight: 700;
  text-align: center;
  padding: 6px 12px;
  border-radius: 8px;
}
.uxf-outcome-success .uxf-outcome-hdr {
  background: linear-gradient(to right, #e2f7ef, #d8f4ea);
  color: #006b44;
  border: 1px solid #b5e8d5;
}
.uxf-outcome-error .uxf-outcome-hdr {
  background: linear-gradient(to right, #fdecea, #fde5e2);
  color: #9b1c0e;
  border: 1px solid #f5c0bb;
}

/* ── Fout-herstelstappen ── */
.uxf-recovery {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-top: 8px;
}
.uxf-recovery-item {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: 11.5px;
  color: #6b6b6b;
  padding: 6px 10px;
  background: #fff9f8;
  border-radius: 6px;
  border: 1px solid #f5ddd9;
  line-height: 1.4;
  transition: background 0.12s;
}
.uxf-recovery-item:hover {
  background: #fff3f1;
  border-color: #C0392B55;
}

/* ── Footer ── */
.uxf-footer {
  text-align: center;
  padding: 28px 0 0;
  font-size: 11.5px;
  color: #ccc;
  border-top: 1px solid #ebebeb;
  margin-top: 44px;
}
</style>
"""

# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = """
<div class="uxf-root">

  <!-- ══ Header ══ -->
  <div class="uxf-header">
    <div class="uxf-badge">&#9654; UX Flow</div>
    <h1 class="uxf-title">Studentprognose</h1>
    <p class="uxf-subtitle">Van eerste bezoek tot prognoseresultaat &mdash; twee gebruikersroutes</p>
  </div>

  <!-- ══ Legenda ══ -->
  <div class="uxf-legend">
    <div class="uxf-leg"><div class="uxf-dot uxf-dot-screen"></div>Scherm / Pagina</div>
    <div class="uxf-leg"><div class="uxf-dot uxf-dot-wizard"></div>Wizard-stap</div>
    <div class="uxf-leg"><div class="uxf-dot uxf-dot-decision"></div>Beslissing</div>
    <div class="uxf-leg"><div class="uxf-dot uxf-dot-process"></div>Achtergrondproces</div>
    <div class="uxf-leg"><div class="uxf-dot uxf-dot-success"></div>Succes</div>
    <div class="uxf-leg"><div class="uxf-dot uxf-dot-error"></div>Fout</div>
  </div>

  <!-- ════════════════════════════════════
       FASE 1 · TOEGANG
  ════════════════════════════════════════ -->
  <div class="uxf-phase"><span>Fase 1 &middot; Toegang</span></div>

  <div class="uxf-card uxf-c-screen">
    <div class="uxf-card-icon">&#127968;</div>
    <div class="uxf-card-body">
      <div class="uxf-tag uxf-tag-screen">SCHERM</div>
      <div class="uxf-card-title">Home</div>
      <div class="uxf-card-desc">
        Startscherm van de applicatie. Toont het actieve project of een call-to-action
        om de wizard te starten als er nog geen project is gekozen.
      </div>
    </div>
    <div class="uxf-route">/</div>
  </div>

  <div class="uxf-vcon"></div>

  <!-- Beslissing: project aanwezig? -->
  <div class="uxf-decision">
    <div class="uxf-diamond">
      <div class="uxf-diamond-text">Project<br>geladen?</div>
    </div>
  </div>

  <!-- Branch-balk -->
  <div style="height:12px; width:3px; background:#b0b0b0; margin:0 auto; border-radius:2px;"></div>
  <div class="uxf-branch">
    <div class="uxf-branch-lbl uxf-branch-lbl-no">&#8592; Nee &nbsp;&middot;&nbsp; Eerste bezoek</div>
    <div class="uxf-branch-line"></div>
    <div class="uxf-branch-lbl uxf-branch-lbl-yes">Terugkerend gebruik &nbsp;&middot;&nbsp; Ja &#8594;</div>
  </div>

  <!-- ════════════════════════════════════
       FASE 2 · ONBOARDING OF SNEL-START
  ════════════════════════════════════════ -->
  <div class="uxf-phase"><span>Fase 2 &middot; Onboarding of Snel-start</span></div>

  <div class="uxf-lanes">

    <!-- ─ Linker rijbaan: Nieuwe gebruiker ─ -->
    <div class="uxf-lane">
      <div class="uxf-lane-hdr uxf-lane-hdr-new">
        &#128196; Eerste bezoek
      </div>

      <div class="uxf-wizard">
        <div class="uxf-wizard-hdr">
          <div class="uxf-wizard-hdr-title">&#129497; Wizard &mdash; Project opzetten</div>
          <div class="uxf-tag uxf-tag-wizard">4 STAPPEN</div>
        </div>

        <div class="uxf-wstep">
          <div class="uxf-wstep-num">1</div>
          <div class="uxf-wstep-body">
            <div class="uxf-wstep-title">Projectmap kiezen</div>
            <div class="uxf-wstep-desc">Bestaande werkmap selecteren of een nieuwe aanmaken</div>
          </div>
        </div>

        <div class="uxf-wstep">
          <div class="uxf-wstep-num">2</div>
          <div class="uxf-wstep-body">
            <div class="uxf-wstep-title">Configuratie</div>
            <div class="uxf-wstep-desc">Standaardinstellingen worden automatisch gegenereerd</div>
          </div>
        </div>

        <div class="uxf-wstep">
          <div class="uxf-wstep-num">3</div>
          <div class="uxf-wstep-body">
            <div class="uxf-wstep-title">Filteren</div>
            <div class="uxf-wstep-desc">Relevante opleidingen, instellingen en jaren selecteren</div>
          </div>
        </div>

        <div class="uxf-wstep uxf-wstep-warn">
          <div class="uxf-wstep-num">4</div>
          <div class="uxf-wstep-body">
            <div class="uxf-wstep-title">Data uploaden &#9888;</div>
            <div class="uxf-wstep-desc">Vijf verplichte inputbestanden uploaden; elk bestand wordt direct gevalideerd</div>
            <div class="uxf-wstep-note">Fout? &#8594; Correcte bestanden aanleveren en opnieuw uploaden</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ─ Rechter rijbaan: Terugkerende gebruiker ─ -->
    <div class="uxf-lane">
      <div class="uxf-lane-hdr uxf-lane-hdr-return">
        &#128257; Terugkerend gebruik
      </div>

      <div class="uxf-card uxf-c-screen uxf-card-sm">
        <div class="uxf-card-icon">&#9989;</div>
        <div class="uxf-card-body">
          <div class="uxf-tag uxf-tag-optional">STATUS</div>
          <div class="uxf-card-title">Project geladen</div>
          <div class="uxf-card-desc">Configuratie en data zijn al aanwezig in de projectmap</div>
        </div>
      </div>

      <div class="uxf-choices">
        <div class="uxf-choices-hdr">Volgende stap kiezen</div>

        <div class="uxf-choice">
          <div class="uxf-choice-icon">&#9881;&#65039;</div>
          <div>
            <div class="uxf-choice-title">Configuratie aanpassen</div>
            <div class="uxf-choice-desc">Als instellingen of data zijn gewijzigd</div>
          </div>
        </div>

        <div class="uxf-choice uxf-choice-primary">
          <div class="uxf-choice-icon">&#9654;&#65039;</div>
          <div>
            <div class="uxf-choice-title">Direct uitvoeren</div>
            <div class="uxf-choice-desc">Meestgekozen pad &mdash; config is al klaar</div>
          </div>
          <div class="uxf-choice-badge">Meest gekozen</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Merge-punt -->
  <div class="uxf-merge">
    <div class="uxf-merge-line"></div>
    <div class="uxf-merge-lbl">Paden komen samen</div>
    <div class="uxf-merge-line"></div>
  </div>

  <!-- ══════════════════════════════════════════════════════
       GEDEELD PAD — Configureren · Uitvoeren · Resultaat
  ══════════════════════════════════════════════════════════ -->
  <div class="uxf-shared">
    <div class="uxf-shared-intro">Gedeeld pad voor beide gebruikersroutes</div>

    <!-- Fase 3 -->
    <div class="uxf-phase" style="margin-top:4px;"><span>Fase 3 &middot; Configureren</span></div>

    <div class="uxf-vcon"></div>

    <div class="uxf-card uxf-c-screen">
      <div class="uxf-card-icon">&#9881;&#65039;</div>
      <div class="uxf-card-body">
        <div class="uxf-tag uxf-tag-screen">SCHERM</div>
        <div class="uxf-card-title">Configuratie</div>
        <div class="uxf-card-desc">
          Drie tabbladen: <strong>Basis</strong> &mdash; instelling, peildatum, trainingsjaar en uitsluitingen;
          <strong>Geavanceerd</strong> &mdash; modelkeuze en ensemble-gewichten;
          <strong>JSON</strong> &mdash; volledige configuratie als tekst.
        </div>
        <div class="uxf-card-hint">&#128161; Basis-tab is voldoende voor de meeste gebruikers</div>
      </div>
      <div class="uxf-route">/config</div>
    </div>

    <!-- Fase 4 -->
    <div class="uxf-phase"><span>Fase 4 &middot; Uitvoeren</span></div>

    <div class="uxf-vcon"></div>

    <div class="uxf-card uxf-c-action">
      <div class="uxf-card-icon">&#9654;&#65039;</div>
      <div class="uxf-card-body">
        <div class="uxf-tag uxf-tag-action">SCHERM</div>
        <div class="uxf-card-title">Uitvoeren</div>
        <div class="uxf-card-desc">
          Parameters instellen (academisch jaar, peildatum, dataset), preview van de run bekijken
          en de voorspelling starten via een bevestigingsdialoog.
        </div>
      </div>
      <div class="uxf-route">/run</div>
    </div>

    <div class="uxf-vcon"></div>

    <div class="uxf-card uxf-c-process">
      <div class="uxf-card-icon">&#8987;</div>
      <div class="uxf-card-body">
        <div class="uxf-tag uxf-tag-process">PROCES</div>
        <div class="uxf-card-title">Voorspelling draait</div>
        <div class="uxf-card-desc">
          Live logstream &mdash; SARIMA, XGBoost en het ratio-model worden parallel
          berekend en samengesteld tot een gewogen ensemble-prognose.
        </div>
      </div>
    </div>

    <div class="uxf-vcon"></div>

    <!-- Beslissing: geslaagd? -->
    <div class="uxf-decision">
      <div class="uxf-diamond">
        <div class="uxf-diamond-text">Geslaagd?</div>
      </div>
    </div>

    <!-- Fase 5 -->
    <div class="uxf-phase"><span>Fase 5 &middot; Resultaat</span></div>

    <div class="uxf-outcomes">

    <!-- Succes -->
    <div class="uxf-outcome uxf-outcome-success">
      <div class="uxf-outcome-hdr">&#10003; Ja &mdash; Succes</div>
      <div class="uxf-card uxf-c-success uxf-card-sm">
        <div class="uxf-card-icon">&#128202;</div>
        <div class="uxf-card-body">
          <div class="uxf-tag uxf-tag-success">SUCCES</div>
          <div class="uxf-card-title">Resultaten</div>
          <div class="uxf-card-desc">
            Interactieve grafieken per opleiding, weekprognose,
            vergelijking met vorig jaar en downloadopties (Excel, CSV).
          </div>
        </div>
        <div class="uxf-route">/output</div>
      </div>
    </div>

    <!-- Fout -->
    <div class="uxf-outcome uxf-outcome-error">
      <div class="uxf-outcome-hdr">&#10007; Nee &mdash; Fout</div>
      <div class="uxf-card uxf-c-error uxf-card-sm">
        <div class="uxf-card-icon">&#9888;&#65039;</div>
        <div class="uxf-card-body">
          <div class="uxf-tag uxf-tag-error">FOUT</div>
          <div class="uxf-card-title">Fout in de run</div>
          <div class="uxf-card-desc">
            Zie de errorlog voor de oorzaak &mdash; ontbrekende data,
            onjuiste filterinstellingen of configuratieconflict.
          </div>
          <div class="uxf-recovery">
            <div class="uxf-recovery-item">&#9881;&#65039;&nbsp; Configuratie controleren &rarr; <code style="font-size:10.5px;background:#f5f5f5;padding:0 4px;border-radius:3px;">/config</code></div>
            <div class="uxf-recovery-item">&#128193;&nbsp; Data controleren &rarr; wizard stap&nbsp;4</div>
            <div class="uxf-recovery-item">&#9654;&#65039;&nbsp; Opnieuw uitvoeren &rarr; <code style="font-size:10.5px;background:#f5f5f5;padding:0 4px;border-radius:3px;">/run</code></div>
          </div>
        </div>
      </div>
    </div>

  </div>

  </div><!-- /uxf-shared -->

  <!-- ══ Footer ══ -->
  <div class="uxf-footer">
    Studentprognose &middot; CEDA / Npuls &middot; open-source tool voor eerstejaarsinstroom HO
  </div>

</div>
"""
