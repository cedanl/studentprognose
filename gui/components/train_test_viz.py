"""Train/test/prognose tijdlijn-visualisaties voor de run-pagina.

Biedt vier visuele varianten om de dataverdeling (traindata, backtest,
prognose) inzichtelijk te tonen. Elke ``render_vN``-functie geeft een
volledig zelfstandige HTML-string terug die via
``ui.html().set_content(html)`` in NiceGUI gezet kan worden.
"""

from __future__ import annotations

#: Fallback-startjaar voor training wanneer het echte databereik (nog) niet
#: uit de projectbestanden afgeleid kan worden. In de praktijk overschrijft de
#: run-pagina dit met het werkelijke overlap-bereik tussen telbestanden en het
#: oktober-bestand (zie ``gui.data_upload.scan_data_year_bounds``).
DATA_START: int = 2016

_T = "#3D68EC"   # traindata — Npuls blauw
_V = "#E6A020"   # backtest  — amber
_P = "#DD784B"   # prognose  — Npuls oranje
_MU = "#6b6b6b"  # muted tekst
_EXCL = "#C0392B"  # uitgesloten jaar — warm rood (theme.NEGATIVE)


# ── Gedeelde hulpfuncties ──────────────────────────────────────────────────


def _parse(raw: str) -> list[int]:
    """Zet een spatie-gescheiden jarenstring om naar een gesorteerde lijst."""
    years: list[int] = []
    for part in raw.strip().split():
        try:
            years.append(int(part))
        except ValueError:
            pass
    return sorted(years)


def _compute(
    years_str: str,
    skip_years: int,
    data_start: int = DATA_START,
    data_end: int | None = None,
) -> dict | None:
    """Bereken train-/test-/prognose-verdeling.

    Args:
        years_str: Prognosejaren als spatie-gescheiden string, bijv. "2024"
            of "2023 2024".
        skip_years: Aantal backtest-jaren vóór de prognose.
        data_start: Eerste jaar met trainingsdata (overlap tussen telbestanden
            en oktober-bestand). Standaard :data:`DATA_START` als fallback.
        data_end: Laatste jaar met realisatiedata (laatste overlap-jaar).
            Training reikt nooit verder dan dit jaar; ``None`` = onbekend
            (geen bovengrens).

    Returns:
        Dict met ``train``, ``test``, ``pred`` tuples en bijbehorende
        ``_n``-tellers, of ``None`` bij ongeldige invoer.
    """
    years = _parse(years_str)
    if not years:
        return None

    pred_start = min(years)
    pred_end = max(years)
    # Use actual selected year count — not the range width — so non-contiguous
    # selections (e.g. 2023 + 2025) get a proportional bar that reflects the
    # real number of prediction years, not the gap-inflated range.
    pred_n = len(years)

    test: tuple[int, int] | None = None
    test_n: int = 0
    if skip_years > 0:
        test_start = pred_start - skip_years
        test_end = pred_start - 1
        # Backtest kan alleen jaren beslaan waarvoor realisatiedata (truth)
        # bestaat — dus nooit voorbij het laatste overlap-jaar.
        if data_end is not None:
            test_end = min(test_end, data_end)
        if test_end >= test_start:
            test = (test_start, test_end)
            test_n = test_end - test_start + 1
        train_end = test_start - 1
    else:
        train_end = pred_start - 1

    train_start = data_start
    # Training kan niet verder reiken dan het laatste jaar met realisatiedata:
    # zonder telbestand + oktober-overlap valt er niets te leren.
    if data_end is not None:
        train_end = min(train_end, data_end)
    train_n = train_end - train_start + 1
    if train_n <= 0:
        return None

    contiguous = (pred_end - pred_start + 1) == pred_n

    return {
        "train": (train_start, train_end),
        "test": test,
        "pred": (pred_start, pred_end),
        "pred_years": years,        # exacte selectie (voor niet-aaneengesloten jaren)
        "pred_contiguous": contiguous,
        "train_n": train_n,
        "test_n": test_n,
        "pred_n": pred_n,
    }


def _yr(n: int) -> str:
    """'1 jaar' of 'N jaren'."""
    return "1 jaar" if n == 1 else f"{n} jaren"


def _rng(start: int, end: int) -> str:
    """Jaarbereik-label: enkel jaar of 'start–end'."""
    return str(start) if start == end else f"{start}–{end}"


def _empty_state(msg: str = "Voer een prognosejaar in om de dataverdeling te zien.") -> str:
    return (
        f'<div style="padding:18px 20px; text-align:center; color:{_MU}; font-size:13px;'
        f' font-style:italic; border:1px dashed #ddd; border-radius:8px;">{msg}</div>'
    )


# ── V1 — Horizon ────────────────────────────────────────────────────────────


def render_v1(
    years_str: str,
    skip_years: int,
    weeks_str: str = "",
    data_start: int = DATA_START,
    data_end: int | None = None,
    excluded_years: list[int] | None = None,
) -> str:
    """V1 — Horizon: proportionele gesegmenteerde tijdlijn met week-annotatie.

    Args:
        excluded_years: Jaren die via ``excluded_data_points`` uit de
            trainingsdata zijn gesloten (configuratie). Jaren die binnen het
            traindata-bereik vallen worden als gearceerde marker op de balk
            getoond.
    """
    d = _compute(years_str, skip_years, data_start, data_end)
    if d is None:
        return _empty_state()

    train_s, train_e = d["train"]
    pred_s, pred_e = d["pred"]
    train_n, test_n, pred_n = d["train_n"], d["test_n"], d["pred_n"]
    total_n = train_n + test_n + pred_n

    # Eerste weeknummer uit de invoer, voor de peildatum-badge.
    week_num: int | None = None
    for tok in (weeks_str or "").strip().split():
        if tok != ":" and ":" not in tok:
            try:
                week_num = int(tok)
                break
            except ValueError:
                pass

    # ── Procentuele grenzen voor tikmarkeringen ──────────────────────────────
    train_pct = train_n / total_n * 100
    test_pct = test_n / total_n * 100 if test_n > 0 else 0.0

    # ── Labelrij (boven de balk) ─────────────────────────────────────────────
    def _label(flex: int, txt: str, color: str, delay: float) -> str:
        return (
            f'<div style="flex:{flex}; min-width:0; font-size:9px; font-weight:700;'
            f' letter-spacing:0.09em; text-transform:uppercase; color:{color};'
            f' white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'
            f' animation:sp-v1-up 0.38s {delay:.2f}s cubic-bezier(0.22,1,0.36,1) both;">'
            f'{txt}</div>'
        )

    labels_html = _label(train_n, "Traindata", _T, 0.0)
    if d["test"] is not None:
        labels_html += _label(test_n, "Backtest", _V, 0.07)
    labels_html += _label(pred_n, "Prognose", _P, 0.14)

    # ── Balk-segmenten ───────────────────────────────────────────────────────
    def _seg(flex: int, color: str, label: str, delay: float, glow: bool = False) -> str:
        shadow = f"box-shadow:0 0 18px {color}66;" if glow else ""
        return (
            f'<div style="flex:{flex}; min-width:24px; background:{color};'
            f' display:flex; align-items:center; justify-content:center; {shadow}'
            f' animation:sp-v1-grow 0.45s {delay:.2f}s cubic-bezier(0.22,1,0.36,1) both;">'
            f'<span style="font-size:12px; font-weight:600; color:#fff;'
            f' letter-spacing:0.01em; text-shadow:0 1px 3px rgba(0,0,0,0.3);">{label}</span>'
            f'</div>'
        )

    # Label in de prognose-balk: aaneengesloten → bereiknotatie, anders individuele jaren
    # (max 3 getoond om overflow te voorkomen).
    pred_years: list[int] = d["pred_years"]
    if d["pred_contiguous"]:
        pred_bar_label = _rng(pred_s, pred_e)
    elif pred_n <= 3:
        pred_bar_label = " · ".join(str(y) for y in pred_years)
    else:
        pred_bar_label = f"{pred_n} jaar"

    bar_html = _seg(train_n, _T, _rng(train_s, train_e), 0.0)
    if d["test"] is not None:
        ts, te = d["test"]
        bar_html += _seg(test_n, _V, _rng(ts, te), 0.1)
    bar_html += _seg(pred_n, _P, pred_bar_label, 0.2, glow=True)

    # ── Uitgesloten-jaren markers (over de traindata-balk) ───────────────────
    # Alleen jaren binnen het traindata-bereik worden gemarkeerd: dat is waar
    # ``excluded_data_points`` daadwerkelijk op toepast (zie configuratie).
    excl_in_train = sorted(
        y for y in (excluded_years or []) if train_s <= y <= train_e
    )

    def _excl_marker(year: int, delay: float) -> str:
        left = train_pct * (year - train_s) / train_n
        width = max(train_pct / train_n, 1.2)
        return (
            f'<div title="Jaar {year} uitgesloten van training" '
            f'style="position:absolute; left:{left:.2f}%; width:{width:.2f}%; top:0; bottom:0;'
            f' background:repeating-linear-gradient(45deg, {_EXCL}66, {_EXCL}66 3px,'
            f' transparent 3px, transparent 7px); border-left:1px solid {_EXCL};'
            f' border-right:1px solid {_EXCL};'
            f' animation:sp-v1-up 0.3s {delay:.2f}s ease both;"></div>'
        )

    excl_overlay_html = "".join(
        _excl_marker(y, 0.3 + i * 0.05) for i, y in enumerate(excl_in_train)
    )

    # ── Tikmarkeringen op de jaaras ──────────────────────────────────────────
    # Toont de beginjaren van elk segment + het eindjaar van de prognose als
    # de prognose meer dan één jaar beslaat.
    ticks: list[tuple[float, int, str]] = []  # (pct, year, align)
    ticks.append((0.0, train_s, "left"))
    if d["test"] is not None:
        ts, te = d["test"]
        ticks.append((train_pct, ts, "center"))
        ticks.append((train_pct + test_pct, pred_s, "center"))
    else:
        ticks.append((train_pct, pred_s, "center"))
    if pred_n > 1:
        ticks.append((100.0, pred_e, "right"))

    def _tick(pct: float, year: int, align: str, delay: float) -> str:
        tx = {"left": "0", "center": "-50%", "right": "-100%"}[align]
        return (
            f'<div style="position:absolute; left:{pct:.1f}%; bottom:0;">'
            f'<div style="width:1px; height:6px; background:#d0d0d0; margin-bottom:2px;"></div>'
            f'<span style="display:block; font-size:10px; color:{_MU}; white-space:nowrap;'
            f' transform:translateX({tx});">'
            f'{year}</span></div>'
        )

    ticks_html = "".join(
        _tick(p, y, a, 0.35 + i * 0.05) for i, (p, y, a) in enumerate(ticks)
    )

    # ── Peildatum-badge (week) ───────────────────────────────────────────────
    week_badge = ""
    if week_num is not None:
        week_badge = (
            f'<span style="flex-shrink:0; display:inline-flex; align-items:center; gap:4px;'
            f' background:#f4f4f4; border:1px solid #e2e2e2; border-radius:20px;'
            f' padding:2px 9px 2px 6px; font-size:10px; color:{_MU}; white-space:nowrap;">'
            f'<span style="font-size:11px;">&#128197;</span> week&nbsp;{week_num}'
            f'</span>'
        )

    # ── Legenda ──────────────────────────────────────────────────────────────
    train_pct_disp = train_n / total_n * 100

    def _dot(color: str, txt: str, pct: float) -> str:
        return (
            f'<span style="display:inline-flex; align-items:center; gap:5px;">'
            f'<span style="width:9px; height:9px; border-radius:50%; background:{color};'
            f' flex-shrink:0;"></span>'
            f'<span style="font-size:11px; color:#333;">{txt}</span>'
            f'<span style="font-size:10px; color:{_MU};">({pct:.0f}%)</span>'
            f'</span>'
        )

    if d["pred_contiguous"]:
        pred_legend_label = f"Prognose {_rng(pred_s, pred_e)}"
    elif pred_n <= 4:
        pred_legend_label = "Prognose " + ", ".join(str(y) for y in pred_years)
    else:
        pred_legend_label = f"Prognose ({pred_n} jaar)"

    legend_parts = [_dot(_T, f"{_yr(train_n)} traindata", train_pct_disp)]
    if test_n > 0:
        legend_parts.append(_dot(_V, f"{_yr(test_n)} backtest", test_n / total_n * 100))
    legend_parts.append(_dot(_P, pred_legend_label, pred_n / total_n * 100))

    sep = '<span style="color:#ddd; margin:0 3px; font-size:13px;">·</span>'
    legend_html = sep.join(legend_parts)

    excl_note_html = ""
    if excl_in_train:
        excl_years_lbl = ", ".join(str(y) for y in excl_in_train)
        excl_note_html = (
            f'  <div style="margin-top:6px; display:flex; align-items:center; gap:5px;">'
            f'<span style="font-size:11px; color:{_EXCL};">&#9888;</span>'
            f'<span style="font-size:11px; color:{_EXCL};">'
            f'Uitgesloten van training: {excl_years_lbl}</span>'
            f'  </div>'
        )

    return (
        "<style>"
        "@keyframes sp-v1-up {"
        " from { opacity:0; transform:translateY(7px); }"
        " to   { opacity:1; transform:translateY(0); }"
        "}"
        "@keyframes sp-v1-grow {"
        " from { opacity:0; transform:scaleX(0.88); transform-origin:left; }"
        " to   { opacity:1; transform:scaleX(1); }"
        "}"
        "</style>"
        '<div style="font-family:inherit; padding:4px 0 6px;">'

        # Labels + week badge (zelfde flex-verdeling)
        f'  <div style="display:flex; align-items:flex-end; gap:3px; margin-bottom:5px;">'
        f'    <div style="flex:1; min-width:0; display:flex; gap:3px;">{labels_html}</div>'
        f'    {week_badge}'
        f'  </div>'

        # Gesegmenteerde balk (+ uitgesloten-jaren overlay)
        '  <div style="position:relative;">'
        '    <div style="display:flex; height:52px; border-radius:8px;'
        '     overflow:hidden; gap:2px;">'
        f'      {bar_html}'
        '    </div>'
        f'    {excl_overlay_html}'
        '  </div>'

        # Jaaras met tikmarkeringen
        f'  <div style="position:relative; height:26px; margin-top:1px;">'
        f'    {ticks_html}'
        f'  </div>'

        # Legenda — geen opacity-animatie zodat de tekst ook in headless mode zichtbaar is
        f'  <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:10px; align-items:center;">'
        f'    {legend_html}'
        f'  </div>'

        f'{excl_note_html}'

        "</div>"
    )


# ── V2 — Jaar-chips ──────────────────────────────────────────────────────────


def render_v2(
    years_str: str,
    skip_years: int,
    data_start: int = DATA_START,
    data_end: int | None = None,
) -> str:
    """V2 — Jaar-chips: elk jaar als individuele pil."""
    d = _compute(years_str, skip_years, data_start, data_end)
    if d is None:
        return _empty_state()

    train_s, train_e = d["train"]
    pred_s, pred_e = d["pred"]
    train_n, test_n, pred_n = d["train_n"], d["test_n"], d["pred_n"]

    _base = (
        "display:inline-flex; align-items:center; justify-content:center;"
        " border-radius:6px; padding:4px 10px; font-size:12px; font-weight:600;"
        " font-family:inherit; white-space:nowrap; letter-spacing:0.01em;"
    )

    def _train_chip(yr: int, delay: float) -> str:
        return (
            f'<span style="{_base} border:1.5px solid {_T}; color:{_T}; background:transparent;'
            f' animation:sp-v2-pop 0.3s {delay:.2f}s cubic-bezier(0.22,1,0.36,1) both;">'
            f'{yr}</span>'
        )

    def _test_chip(yr: int, delay: float) -> str:
        return (
            f'<span style="{_base} background:{_V}; color:#fff; border:1.5px solid {_V};'
            f' animation:sp-v2-pop 0.3s {delay:.2f}s cubic-bezier(0.22,1,0.36,1) both;">'
            f'{yr}</span>'
        )

    def _pred_chip(yr: int, delay: float) -> str:
        return (
            f'<span style="{_base} background:{_P}; color:#fff; border:1.5px solid {_P};'
            f' box-shadow:0 0 10px {_P}66, 0 2px 8px rgba(221,120,75,0.3);'
            f' animation:sp-v2-pop 0.3s {delay:.2f}s cubic-bezier(0.22,1,0.36,1) both;">'
            f'&#9658; {yr}</span>'
        )

    def _header(icon: str, label: str, n: int, color: str, delay: float) -> str:
        return (
            f'<div style="font-size:10px; font-weight:700; letter-spacing:0.08em;'
            f' text-transform:uppercase; color:{color}; margin-bottom:8px;'
            f' animation:sp-v2-pop 0.3s {delay:.2f}s ease both;">'
            f'{icon} {label} <span style="font-weight:400; opacity:0.7;">({_yr(n)})</span>'
            f'</div>'
        )

    counter = 0.0
    step = 0.04

    # Traindata groep
    train_chips = ""
    for yr in range(train_s, train_e + 1):
        train_chips += _train_chip(yr, counter)
        counter += step

    sep_delay = counter
    counter += step

    # Backtest groep
    test_block = ""
    if d["test"] is not None:
        ts, te = d["test"]
        test_header = _header("\U0001f52c", "Backtest", test_n, _V, sep_delay)
        test_chips = ""
        for yr in range(ts, te + 1):
            test_chips += _test_chip(yr, counter)
            counter += step
        sep_delay2 = counter
        counter += step
        test_block = (
            '<div style="width:1px; background:#e0e0e0; margin:0 4px; align-self:stretch;'
            f' animation:sp-v2-pop 0.3s {sep_delay:.2f}s ease both;"></div>'
            f'<div style="padding-right:4px;">'
            f'{test_header}'
            f'<div style="display:flex; flex-wrap:wrap; gap:6px;">{test_chips}</div>'
            f'</div>'
        )
        sep_delay = sep_delay2

    pred_header = _header("\U0001f3af", "Prognose", pred_n, _P, sep_delay)
    pred_chips = ""
    for yr in range(pred_s, pred_e + 1):
        pred_chips += _pred_chip(yr, counter)
        counter += step

    pred_block = (
        '<div style="width:1px; background:#e0e0e0; margin:0 4px; align-self:stretch;'
        f' animation:sp-v2-pop 0.3s {sep_delay:.2f}s ease both;"></div>'
        f'<div>'
        f'{pred_header}'
        f'<div style="display:flex; flex-wrap:wrap; gap:6px;">{pred_chips}</div>'
        f'</div>'
    )

    train_header = _header("\U0001f4da", "Traindata", train_n, _T, 0.0)

    return (
        "<style>"
        "@keyframes sp-v2-pop {"
        " from { opacity:0; transform:scale(0.82) translateY(6px); }"
        " to   { opacity:1; transform:none; }"
        "}"
        "</style>"
        '<div style="font-family:inherit; padding:4px 0 6px;">'
        '  <div style="display:flex; align-items:flex-start; flex-wrap:wrap; gap:0;">'
        f'    <div style="padding-right:4px;">'
        f'      {train_header}'
        f'      <div style="display:flex; flex-wrap:wrap; gap:6px;">{train_chips}</div>'
        f'    </div>'
        f'    {test_block}'
        f'    {pred_block}'
        "  </div>"
        "</div>"
    )


# ── V3 — Stat Cards ───────────────────────────────────────────────────────────


def render_v3(
    years_str: str,
    skip_years: int,
    data_start: int = DATA_START,
    data_end: int | None = None,
) -> str:
    """V3 — Stat Cards: dashboard-metriekkaarten."""
    d = _compute(years_str, skip_years, data_start, data_end)
    if d is None:
        return _empty_state()

    train_s, train_e = d["train"]
    pred_s, pred_e = d["pred"]
    train_n, test_n, pred_n = d["train_n"], d["test_n"], d["pred_n"]
    total_n = train_n + test_n + pred_n

    train_pct = train_n / total_n * 100
    test_pct = test_n / total_n * 100 if test_n > 0 else 0
    pred_pct = pred_n / total_n * 100

    def _card(
        icon: str, role: str, color: str, rng_label: str,
        count: int, pct: float, delay: float,
    ) -> str:
        return (
            f'<div style="flex:1; min-width:140px; background:#fff; border-radius:8px;'
            f' box-shadow:0 2px 10px rgba(0,0,0,0.07); overflow:hidden;'
            f' animation:sp-v3-fadein 0.4s {delay:.2f}s ease both;">'
            f'<div style="height:4px; background:{color};"></div>'
            f'<div style="padding:14px 16px 16px;">'
            f'<div style="font-size:24px; margin-bottom:8px; line-height:1;">{icon}</div>'
            f'<div style="font-size:11px; font-weight:700; letter-spacing:0.09em;'
            f' text-transform:uppercase; color:{color}; margin-bottom:4px;">{role}</div>'
            f'<div style="font-size:22px; font-weight:700; color:#1a1a1a;'
            f' letter-spacing:-0.01em; margin-bottom:2px;">{rng_label}</div>'
            f'<div style="font-size:13px; color:{_MU}; margin-bottom:14px;">{_yr(count)}</div>'
            f'<div style="height:4px; background:#f0f0f0; border-radius:2px; overflow:hidden;">'
            f'<div style="height:4px; background:{color}; border-radius:2px; width:{pct:.1f}%;'
            f' transform-origin:left; animation:sp-v3-barfill 0.65s {delay + 0.15:.2f}s'
            f' cubic-bezier(0.22,1,0.36,1) both;">'
            f'</div></div>'
            f'</div></div>'
        )

    cards = _card("\U0001f4da", "Traindata", _T, _rng(train_s, train_e), train_n, train_pct, 0.0)
    if d["test"] is not None:
        ts, te = d["test"]
        cards += _card("\U0001f52c", "Backtest", _V, _rng(ts, te), test_n, test_pct, 0.12)
    cards += _card("\U0001f3af", "Prognose", _P, _rng(pred_s, pred_e), pred_n, pred_pct, 0.24)

    # Mini totaalstaaf
    mini_segs = (
        f'<div style="flex:{train_n}; background:{_T}; border-radius:4px 0 0 4px;'
        f' display:flex; align-items:center; justify-content:center;">'
        f'<span style="font-size:10px; color:#fff; font-weight:600; opacity:0.9;">{train_n}j</span>'
        f'</div>'
    )
    if d["test"] is not None:
        mini_segs += (
            f'<div style="flex:{test_n}; background:{_V}; display:flex; align-items:center; justify-content:center;">'
            f'<span style="font-size:10px; color:#fff; font-weight:600; opacity:0.9;">{test_n}j</span>'
            f'</div>'
        )
    mini_segs += (
        f'<div style="flex:{pred_n}; background:{_P}; border-radius:0 4px 4px 0;'
        f' display:flex; align-items:center; justify-content:center;">'
        f'<span style="font-size:10px; color:#fff; font-weight:600; opacity:0.9;">{pred_n}j</span>'
        f'</div>'
    )

    return (
        "<style>"
        "@keyframes sp-v3-fadein {"
        " from { opacity:0; transform:translateY(8px); }"
        " to   { opacity:1; transform:none; }"
        "}"
        "@keyframes sp-v3-barfill {"
        " from { transform:scaleX(0); }"
        " to   { transform:scaleX(1); }"
        "}"
        "</style>"
        '<div style="font-family:inherit; padding:4px 0 6px;">'
        f'  <div style="display:flex; gap:12px; flex-wrap:wrap;">{cards}</div>'
        '  <div style="margin-top:12px; height:24px; border-radius:5px; display:flex; gap:2px;'
        '   overflow:hidden; animation:sp-v3-fadein 0.5s 0.45s ease both;">'
        f'    {mini_segs}'
        '  </div>'
        "</div>"
    )


# ── V4 — Dark Timeline ───────────────────────────────────────────────────────


def render_v4(
    years_str: str,
    skip_years: int,
    data_start: int = DATA_START,
    data_end: int | None = None,
) -> str:
    """V4 — Dark Timeline: premium donker tijdlijnkaart."""
    d = _compute(years_str, skip_years, data_start, data_end)
    if d is None:
        return (
            '<div style="background:#0d1117; border-radius:10px; padding:20px 24px;">'
            '<div style="text-align:center; color:rgba(255,255,255,0.35); font-size:13px;'
            ' font-style:italic;">'
            'Voer een prognosejaar in om de dataverdeling te zien.'
            '</div></div>'
        )

    train_s, train_e = d["train"]
    pred_s, pred_e = d["pred"]
    train_n, test_n, pred_n = d["train_n"], d["test_n"], d["pred_n"]

    # ── Labels rij (zelfde flex-proporties als balk) ────────────────────────
    label_segs = (
        f'<div style="flex:{train_n}; min-width:0;">'
        f'<span style="font-size:9px; font-weight:700; letter-spacing:0.1em;'
        f' color:{_T}; text-transform:uppercase; white-space:nowrap;'
        f' animation:sp-v4-in 0.45s 0s ease both;">'
        f'Traindata</span></div>'
    )
    if d["test"] is not None:
        ts, te = d["test"]
        label_segs += (
            f'<div style="flex:{test_n}; min-width:0;">'
            f'<span style="font-size:9px; font-weight:700; letter-spacing:0.1em;'
            f' color:{_V}; text-transform:uppercase; white-space:nowrap;'
            f' animation:sp-v4-in 0.45s 0.15s ease both;">'
            f'Backtest</span></div>'
        )
    label_segs += (
        f'<div style="flex:{pred_n}; min-width:0;">'
        f'<span style="font-size:9px; font-weight:700; letter-spacing:0.1em;'
        f' color:{_P}; text-transform:uppercase; white-space:nowrap;'
        f' animation:sp-v4-in 0.45s 0.3s ease both;">'
        f'Prognose</span></div>'
    )

    # ── Tick marks rij ──────────────────────────────────────────────────────
    def _tick_seg(flex: int, year: int, delay: float) -> str:
        return (
            f'<div style="flex:{flex}; min-width:0; position:relative; height:20px;">'
            f'<div style="position:absolute; left:0; bottom:0; display:flex;'
            f' flex-direction:column; align-items:center; gap:2px;'
            f' animation:sp-v4-in 0.4s {delay:.2f}s ease both;">'
            f'<span style="font-size:9px; color:rgba(255,255,255,0.5);'
            f' white-space:nowrap;">{year}</span>'
            f'<div style="width:1px; height:6px; background:rgba(255,255,255,0.3);"></div>'
            f'</div></div>'
        )

    tick_segs = _tick_seg(train_n, train_s, 0.05)
    if d["test"] is not None:
        ts, te = d["test"]
        tick_segs += _tick_seg(test_n, ts, 0.20)
    tick_segs += _tick_seg(pred_n, pred_s, 0.35)

    # ── Jaarbereik-labels onder de balk ─────────────────────────────────────
    range_segs = (
        f'<div style="flex:{train_n}; min-width:0;">'
        f'<span style="font-size:11px; color:rgba(255,255,255,0.45);">{_rng(train_s, train_e)}</span>'
        f'</div>'
    )
    if d["test"] is not None:
        ts, te = d["test"]
        range_segs += (
            f'<div style="flex:{test_n}; min-width:0;">'
            f'<span style="font-size:11px; color:rgba(255,255,255,0.55);">{_rng(ts, te)}</span>'
            f'</div>'
        )
    range_segs += (
        f'<div style="flex:{pred_n}; min-width:0;">'
        f'<span style="font-size:11px; color:rgba(255,255,255,0.7); font-weight:600;">'
        f'{_rng(pred_s, pred_e)}</span>'
        f'</div>'
    )

    # ── Stats footer ─────────────────────────────────────────────────────────
    stat_parts = [
        f'<span style="color:{_T}; font-size:12px;">&#9679;</span>'
        f'<span style="font-size:12px; color:rgba(255,255,255,0.75); margin-left:4px;">'
        f'{_yr(train_n)} traindata</span>',
    ]
    if test_n > 0:
        stat_parts.append(
            f'<span style="color:{_V}; font-size:12px;">&#9679;</span>'
            f'<span style="font-size:12px; color:rgba(255,255,255,0.75); margin-left:4px;">'
            f'{_yr(test_n)} backtest</span>'
        )
    stat_parts.append(
        f'<span style="color:{_P}; font-size:12px;">&#9679;</span>'
        f'<span style="font-size:12px; color:rgba(255,255,255,0.75); margin-left:4px;">'
        f'Prognose {_rng(pred_s, pred_e)}</span>'
    )
    stats_html = (
        ' <span style="color:rgba(255,255,255,0.2); margin:0 6px;">&#183;</span> '.join(stat_parts)
    )

    # ── Staggered segment slide-in (als wrapper over de balk) ────────────────
    bar_html = (
        f'<div style="flex:{train_n}; background:{_T};'
        f' box-shadow:0 0 12px {_T}44;'
        f' animation:sp-v4-slide 0.4s 0s ease both;">'
        f'</div>'
    )
    if d["test"] is not None:
        bar_html += (
            f'<div style="flex:{test_n}; background:{_V};'
            f' box-shadow:0 0 12px {_V}44;'
            f' animation:sp-v4-slide 0.4s 0.2s ease both;">'
            f'</div>'
        )
    bar_html += (
        f'<div style="flex:{pred_n}; background:{_P};'
        f' box-shadow:0 0 20px {_P}88, 0 0 40px {_P}44;'
        f' animation:sp-v4-slide 0.4s 0.4s ease both;">'
        f'</div>'
    )

    return (
        "<style>"
        "@keyframes sp-v4-in {"
        " from { opacity:0; transform:translateY(6px); }"
        " to   { opacity:1; transform:none; }"
        "}"
        "@keyframes sp-v4-slide {"
        " from { opacity:0; transform:translateX(-18px); }"
        " to   { opacity:1; transform:none; }"
        "}"
        "</style>"
        '<div style="background:#0d1117; border-radius:10px; padding:20px 24px 18px;'
        ' font-family:inherit; overflow:hidden;">'

        # Labels
        f'  <div style="display:flex; gap:2px; margin-bottom:6px;">'
        f'    {label_segs}'
        f'  </div>'

        # Tick marks
        f'  <div style="display:flex; gap:2px; margin-bottom:4px;">'
        f'    {tick_segs}'
        f'  </div>'

        # Bar
        '  <div style="display:flex; gap:2px; height:40px; border-radius:6px; overflow:hidden;">'
        f'    {bar_html}'
        '  </div>'

        # Jaarbereiken
        f'  <div style="display:flex; gap:2px; margin-top:8px;">'
        f'    {range_segs}'
        f'  </div>'

        # Stats
        '  <div style="margin-top:14px; padding-top:12px;'
        '   border-top:1px solid rgba(255,255,255,0.08);'
        '   display:flex; gap:6px; flex-wrap:wrap; align-items:center;'
        '   animation:sp-v4-in 0.5s 0.55s ease both;">'
        f'    {stats_html}'
        '  </div>'

        "</div>"
    )
