"""Interactieve schematische weergave van de drie voorspelsporen (inline SVG).

Eén on-brand SVG-flow. Elk spoor (cumulatief / individueel / beide) is
hoverbaar — dan verschijnt een korte uitleg — en klikbaar: het linkt door naar de
uitgebreide methodologie-documentatie. Bedoeld om inline gerenderd te worden
(``ui.html``) zodat de CSS-hover en links werken.
"""

from __future__ import annotations

#: Basis-URL van de methodologie-documentatie.
_DOCS = "https://cedanl.github.io/studentprognose/methodologie"

#: Per spoor de diepe-link naar de bijbehorende methodologiepagina.
DOC_CUMULATIEF = f"{_DOCS}/sarima/"
DOC_INDIVIDUEEL = f"{_DOCS}/individueel/"
DOC_BEIDE = f"{_DOCS}/ensemble/"

#: Algemene methodologie-link (gebruikt onder het schema).
DOCS = f"{_DOCS}/"


def flow_svg() -> str:
    """Geef de interactieve drie-sporen-SVG als HTML-string.

    Hover over een spoor toont een korte uitleg; klikken opent de uitgebreide
    methodologie. Klassen zijn geprefixt met ``sp-`` om botsingen te voorkomen.
    """
    return f"""
<svg width="100%" viewBox="0 0 940 306" xmlns="http://www.w3.org/2000/svg"
     font-family="'Segoe UI',system-ui,Arial,sans-serif">
  <defs>
    <marker id="aB" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L6.5,3 L0,6 Z" fill="#3D68EC"/></marker>
    <marker id="aG" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L6.5,3 L0,6 Z" fill="#00AF81"/></marker>
    <marker id="aO" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L6.5,3 L0,6 Z" fill="#DD784B"/></marker>
    <marker id="aK" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L6.5,3 L0,6 Z" fill="#111"/></marker>
    <filter id="sh" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.2" flood-color="#000" flood-opacity="0.12"/></filter>
    <style>
      .sp-hz {{ cursor: pointer; }}
      .sp-hz text {{ text-decoration: none; }}
      .sp-band {{ transition: stroke-opacity .12s, fill-opacity .12s; }}
      .sp-hz:hover .sp-band {{ stroke-opacity: 1; fill-opacity: .12; }}
      .sp-tip {{ opacity: 0; transition: opacity .12s ease; pointer-events: none; }}
      .sp-hz:hover .sp-tip {{ opacity: 1; }}
    </style>
  </defs>

  <!-- ===================== CUMULATIEF (klikbaar + hover) ===================== -->
  <a class="sp-hz" href="{DOC_CUMULATIEF}" target="_blank" rel="noopener">
    <rect class="sp-band" x="12" y="40" width="512" height="92" rx="16" fill="#3D68EC0d"
          stroke="#3D68EC" stroke-width="1.4" stroke-opacity="0.5" fill-opacity="0.05" stroke-dasharray="5 4"/>
    <text x="30" y="62" font-size="11.5" font-weight="700" fill="#3D68EC" letter-spacing="1.5">CUMULATIEF SPOOR</text>
    <g filter="url(#sh)">
      <rect x="30" y="72" width="150" height="48" rx="10" fill="#fff" stroke="#3D68EC" stroke-width="1.5"/>
      <rect x="222" y="72" width="118" height="48" rx="10" fill="#fff" stroke="#3D68EC" stroke-width="1.5"/>
      <rect x="382" y="72" width="118" height="48" rx="10" fill="#fff" stroke="#3D68EC" stroke-width="1.5"/></g>
    <text x="105" y="92" text-anchor="middle" font-size="12.5" font-weight="600" fill="#111">Wekelijkse</text>
    <text x="105" y="108" text-anchor="middle" font-size="12.5" font-weight="600" fill="#111">aanmeldcurve</text>
    <text x="281" y="94" text-anchor="middle" font-size="13" font-weight="600" fill="#111">SARIMA</text>
    <text x="281" y="110" text-anchor="middle" font-size="10" fill="#6b6b6b">tijdreeks</text>
    <text x="441" y="94" text-anchor="middle" font-size="13" font-weight="600" fill="#111">XGBoost</text>
    <text x="441" y="110" text-anchor="middle" font-size="10" fill="#6b6b6b">regressie</text>
    <line x1="180" y1="96" x2="218" y2="96" stroke="#3D68EC" stroke-width="1.8" marker-end="url(#aB)"/>
    <line x1="340" y1="96" x2="378" y2="96" stroke="#3D68EC" stroke-width="1.8" marker-end="url(#aB)"/>
    <!-- tooltip -->
    <g class="sp-tip">
      <rect x="30" y="138" width="404" height="50" rx="8" fill="#111" fill-opacity="0.95"/>
      <text x="46" y="158" font-size="11.5" font-weight="700" fill="#7aa2ff">Cumulatief spoor</text>
      <text x="46" y="176" font-size="11" fill="#e8e8ea">Voorspelt uit de wekelijkse aanmeldcurve (SARIMA → XGBoost).</text>
    </g>
  </a>

  <!-- ===================== INDIVIDUEEL (klikbaar + hover) ===================== -->
  <a class="sp-hz" href="{DOC_INDIVIDUEEL}" target="_blank" rel="noopener">
    <rect class="sp-band" x="12" y="204" width="432" height="92" rx="16" fill="#00AF810d"
          stroke="#00AF81" stroke-width="1.4" stroke-opacity="0.5" fill-opacity="0.05" stroke-dasharray="5 4"/>
    <text x="30" y="226" font-size="11.5" font-weight="700" fill="#00AF81" letter-spacing="1.5">INDIVIDUEEL SPOOR</text>
    <g filter="url(#sh)">
      <rect x="30" y="236" width="150" height="48" rx="10" fill="#fff" stroke="#00AF81" stroke-width="1.5"/>
      <rect x="222" y="236" width="140" height="48" rx="10" fill="#fff" stroke="#00AF81" stroke-width="1.5"/></g>
    <text x="105" y="256" text-anchor="middle" font-size="12.5" font-weight="600" fill="#111">Individuele</text>
    <text x="105" y="272" text-anchor="middle" font-size="12.5" font-weight="600" fill="#111">aanmeldingen</text>
    <text x="292" y="258" text-anchor="middle" font-size="13" font-weight="600" fill="#111">Classifier</text>
    <text x="292" y="274" text-anchor="middle" font-size="10" fill="#6b6b6b">inschrijfkans</text>
    <line x1="180" y1="260" x2="218" y2="260" stroke="#00AF81" stroke-width="1.8" marker-end="url(#aG)"/>
    <g class="sp-tip">
      <rect x="30" y="146" width="404" height="50" rx="8" fill="#111" fill-opacity="0.95"/>
      <text x="46" y="166" font-size="11.5" font-weight="700" fill="#4be3b8">Individueel spoor</text>
      <text x="46" y="184" font-size="11" fill="#e8e8ea">Per aanmelding de inschrijfkans, opgeteld (classifier).</text>
    </g>
  </a>

  <!-- convergentie -->
  <path d="M500,96 C560,96 545,168 596,168" fill="none" stroke="#3D68EC" stroke-width="1.9" marker-end="url(#aO)"/>
  <path d="M362,260 C520,260 545,184 596,184" fill="none" stroke="#00AF81" stroke-width="1.9" marker-end="url(#aO)"/>

  <!-- ===================== BEIDE / ENSEMBLE (klikbaar + hover) ===================== -->
  <a class="sp-hz" href="{DOC_BEIDE}" target="_blank" rel="noopener">
    <g filter="url(#sh)"><rect x="600" y="140" width="156" height="72" rx="13"
          fill="#DD784B" stroke="#DD784B"/></g>
    <text x="678" y="170" text-anchor="middle" font-size="15.5" font-weight="700" fill="#fff">Ensemble</text>
    <text x="678" y="191" text-anchor="middle" font-size="11.5" fill="#fff" opacity="0.92">&#171; Beide &#187;</text>
    <rect x="635" y="116" width="86" height="19" rx="9.5" fill="#fff" stroke="#DD784B" stroke-width="1.2"/>
    <text x="678" y="129" text-anchor="middle" font-size="10" font-weight="700" fill="#DD784B">aanbevolen</text>
    <g class="sp-tip">
      <rect x="516" y="222" width="404" height="50" rx="8" fill="#111" fill-opacity="0.95"/>
      <text x="532" y="242" font-size="11.5" font-weight="700" fill="#f0a06b">Beide · aanbevolen</text>
      <text x="532" y="260" font-size="11" fill="#e8e8ea">Combineert beide sporen tot één ensemble — nauwkeurigst.</text>
    </g>
  </a>

  <!-- instroom -->
  <g filter="url(#sh)"><rect x="806" y="146" width="120" height="60" rx="13" fill="#111"/></g>
  <text x="866" y="172" text-anchor="middle" font-size="12.5" font-weight="700" fill="#fff">Voorspelde</text>
  <text x="866" y="190" text-anchor="middle" font-size="12.5" font-weight="700" fill="#fff">instroom</text>
  <line x1="756" y1="176" x2="802" y2="176" stroke="#111" stroke-width="1.9" marker-end="url(#aK)"/>
</svg>
"""
