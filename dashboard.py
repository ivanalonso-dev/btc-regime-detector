#!/usr/bin/env python3
"""
btc-regime-detector · dashboard

Genera dashboard.html (un solo archivo, se abre con doble clic) a partir de los
resultados de regime_detector.py:
  - ESTADO OFICIAL v2 de BTC (alcista fuerte / alcista débil-lateral / no alcista)
  - Estado de cada detector individual y consenso
  - Indicadores de contexto: precio, MVRV y su percentil, distancia a la MA200, ER90
  - Precio (escala log) sombreado por estado, MVRV y curva de capital
  - Comportamiento de BTC en cada estado, métricas por detector, retrasos frente a
    techos/suelos y el registro diario en vivo

Uso:
    python regime_detector.py && python dashboard.py
    python dashboard.py --synthetic        # para probar
Requisitos: pandas, numpy, plotly
"""
from __future__ import annotations

import argparse
import html
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from regime_detector import load_data, strategy_returns, synthetic_data

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # azul, naranja, aqua (paleta categórica validada)
GOOD, WARN_TXT, CRITICAL = "#0ca30c", "#a86a00", "#d03b3b"  # estado: siempre con icono + texto
BAND_STRONG = "rgba(12,163,12,0.14)"
BAND_WEAK = "rgba(250,178,25,0.18)"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e1", "#fcfcfb"

STATE_STYLE = {
    "ALCISTA FUERTE": ("good", "▲"),
    "ALCISTA DÉBIL / LATERAL": ("warn", "◆"),
    "NO ALCISTA": ("bad", "▼"),
    "ALCISTA": ("good", "▲"),
}


def fmt_price(v: float) -> str:
    return f"{v:,.0f} $" if v >= 100 else f"{v:,.2f} $"


def base_layout(title: str, log_y: bool = False, height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height, margin=dict(l=60, r=20, t=50, b=70), autosize=True,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=INK2),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor=GRID, showspikes=True, spikemode="across",
                   spikethickness=1, spikecolor=INK2),
        yaxis=dict(type="log" if log_y else "linear", gridcolor=GRID, zeroline=False),
    )


def _html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%",
                       config={"responsive": True, "displaylogo": False})


def state_shapes(states: pd.Series) -> list[dict]:
    """Rectángulos verticales por tramo: verde = alcista fuerte, ámbar = débil/lateral."""
    shapes = []
    s = states.fillna(-1)
    for code, color in [(2, BAND_STRONG), (1, BAND_WEAK)]:
        on = s == code
        starts = s.index[on & ~on.shift(1, fill_value=False)]
        ends = s.index[on & ~on.shift(-1, fill_value=False)]
        shapes += [dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1,
                        fillcolor=color, line_width=0, layer="below") for a, b in zip(starts, ends)]
    return shapes


def chart_price(df: pd.DataFrame, states: pd.Series) -> str:
    c = df["price"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.index, y=c, name="Precio BTC", line=dict(color=SERIES[0], width=2),
                             hovertemplate="%{y:,.2f} $"))
    fig.add_trace(go.Scatter(x=c.index, y=c.rolling(200).mean(), name="Media 200 días",
                             line=dict(color=SERIES[1], width=2), hovertemplate="%{y:,.2f} $"))
    fig.update_layout(**base_layout("Precio de BTC (escala log) · verde = alcista fuerte · "
                                    "ámbar = alcista débil / lateral · sin color = no alcista",
                                    log_y=True, height=460), shapes=state_shapes(states))
    return _html(fig)


def chart_mvrv(df: pd.DataFrame) -> str | None:
    if "mvrv" not in df or df["mvrv"].notna().sum() < 100:
        return None
    m = df["mvrv"].dropna()
    m = m[m.index >= "2011-01-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m.index, y=m, name="MVRV", line=dict(color=SERIES[0], width=2),
                             hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Scatter(x=m.index, y=m.rolling(200).mean(), name="Media 200 días del MVRV",
                             line=dict(color=SERIES[1], width=2), hovertemplate="%{y:.2f}"))
    fig.add_hline(y=1, line=dict(color=INK2, width=1, dash="dot"))
    fig.update_layout(**base_layout("MVRV (valor de mercado / valor realizado) · línea punteada = 1",
                                    log_y=True))
    return _html(fig)


def chart_equity(df: pd.DataFrame, states: pd.Series, start: str, cost: float) -> str:
    p = df["price"][df.index >= start]
    st = states.reindex(p.index)
    curves = {
        "BTC en ALCISTA (fuerte + débil)": (1 + strategy_returns(p, (st >= 1).astype(float), cost)).cumprod(),
        "BTC solo en ALCISTA FUERTE": (1 + strategy_returns(p, (st == 2).astype(float), cost)).cumprod(),
        "Comprar y mantener": p / p.iloc[0],
    }
    fig = go.Figure()
    for (name, eq), color in zip(curves.items(), [SERIES[0], SERIES[2], SERIES[1]]):
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=dict(color=color, width=2),
                                 hovertemplate="x%{y:,.1f}"))
    fig.update_layout(**base_layout(f"Capital multiplicado desde {start} (escala log, neto de costes)",
                                    log_y=True))
    return _html(fig)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "—" if np.isnan(v) else f"{v:,.2f}"
    return str(v)


def table_html(df: pd.DataFrame, highlight: str | None = None, index: bool = True) -> str:
    head = ([df.index.name or ""] if index else []) + list(df.columns)
    rows = []
    for idx, r in df.iterrows():
        cls = ' class="sel"' if highlight is not None and idx == highlight else ""
        cells = ([f"<th>{html.escape(str(idx))}</th>"] if index else []) + [
            f"<td>{badge(v) if isinstance(v, str) and v in STATE_STYLE else html.escape(_fmt(v))}</td>"
            for v in r.values]
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return ("<div class='tw'><table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def badge(s: str) -> str:
    cls, icon = STATE_STYLE.get(s, ("", ""))
    return f"<span class='badge {cls}'>{icon} {html.escape(s)}</span>"


def _read(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if os.path.exists(path) else pd.DataFrame()


def build(df: pd.DataFrame, res: str, out: str, log_path: str):
    st = json.load(open(os.path.join(res, "estado_actual.json"), encoding="utf-8"))
    states = pd.read_csv(os.path.join(res, "estados_oficiales.csv"), index_col=0, parse_dates=True)["estado"]
    metrics = pd.read_csv(os.path.join(res, "metricas.csv"), index_col=0)
    lags = _read(os.path.join(res, "retrasos.csv"))
    by_state = _read(os.path.join(res, "estados_btc.csv"))
    strat = _read(os.path.join(res, "estados_estrategias.csv"))
    log = _read(log_path)
    cfg = st.get("config", {})
    base, v1 = st["detector_oficial"], st["detector"]
    start, split, cost = cfg.get("start", "2012-01-01"), cfg.get("split", "2020-01-01"), cfg.get("cost_pct", 0.1)
    cls, icon = STATE_STYLE[st["estado_oficial"]]

    er = st["er90"]
    cards = [
        ("Precio", fmt_price(st["precio"]), st["fecha"]),
        ("Días en el estado", f"{st['dias_en_estado']}", f"desde {st['estado_desde']}"),
        ("Fuerza de tendencia (ER90)", f"{er:.2f}",
         f"{'por encima' if er > st['er_umbral'] else 'por debajo'} del umbral {st['er_umbral']}"),
        ("Distancia a la MA200", f"{st['dist_ma200_%']:+.1f}%", "precio frente a su media de 200 días"),
    ]
    if "mvrv" in st:
        cards.append(("MVRV", f"{st['mvrv']:.2f}", f"percentil histórico {st['mvrv_percentil']:.0f}"))
    if "precio_realizado" in st:
        cards.append(("Precio realizado", fmt_price(st["precio_realizado"]), "coste medio de todas las monedas"))
    cards_html = "".join(f"<div class='card'><div class='k'>{html.escape(k)}</div>"
                         f"<div class='v'>{html.escape(v)}</div><div class='s'>{html.escape(s)}</div></div>"
                         for k, v, s in cards)

    det_rows = "".join(f"<tr{' class=sel' if k == base else ''}><th>{html.escape(k)}</th><td>{badge(v)}</td></tr>"
                       for k, v in st["detectores"].items())
    n_bull = sum(v == "ALCISTA" for v in st["detectores"].values())
    n_tot = sum(v != "n/d" for v in st["detectores"].values())

    keep = ["CAGR_%", "maxDD_%", "IS Calmar", "OOS CAGR_%", "OOS maxDD_%", "OOS Calmar",
            "%_invertido", "cambios/año"]
    mt = metrics[[c for c in keep if c in metrics]]
    mt.index.name = "detector"
    lag_cols = ["evento", "fecha", "precio"] + [c for c in [base, v1] if c in lags]
    lag_tbl = lags[lag_cols] if len(lags) else pd.DataFrame({"info": ["sin ciclos completos"]})
    log_tbl = (log.iloc[::-1].head(15)[["fecha", "precio", "estado", "mvrv", "er90"]]
               if len(log) else pd.DataFrame({"info": ["el registro empieza con la próxima ejecución"]}))
    log_since = log["fecha"].iloc[0] if len(log) else "—"

    mvrv_chart = chart_mvrv(df)
    page = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Regime Detector</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{--surface:{SURFACE};--ink:{INK};--ink2:{INK2};--grid:{GRID};--good:{GOOD};--warn:{WARN_TXT};--bad:{CRITICAL}}}
*{{box-sizing:border-box}}
body{{margin:0;background:#f4f3ef;color:var(--ink);font:14px/1.45 Inter,"Segoe UI",system-ui,sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px}}
.sub{{color:var(--ink2);margin:0 0 20px}}
.hero{{display:flex;flex-wrap:wrap;gap:16px;align-items:stretch;margin-bottom:16px}}
.regime{{flex:1 1 340px;background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:20px}}
.regime .lbl{{color:var(--ink2);font-size:13px}}
.regime .big{{font-size:30px;font-weight:700;margin:6px 0 4px;line-height:1.15}}
.good{{color:var(--good)}} .warn{{color:var(--warn)}} .bad{{color:var(--bad)}}
.cards{{flex:2 1 520px;align-self:start;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.card{{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:14px}}
.card .k{{color:var(--ink2);font-size:12px}} .card .v{{font-size:22px;font-weight:650;margin:4px 0}}
.card .s{{color:var(--ink2);font-size:12px}}
.panel{{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:12px;margin-bottom:16px;min-width:0}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}
.tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} thead th{{color:var(--ink2);font-weight:600}}
tr.sel{{background:#eef5fd}} tr.sel th{{font-weight:700}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--grid)}}
.badge.good{{border-color:var(--good)}} .badge.warn{{border-color:var(--warn)}} .badge.bad{{border-color:var(--bad)}}
details{{margin-top:12px}} summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
.note{{color:var(--ink2);font-size:12px;margin-top:8px}}
</style></head><body><main>
<h1>BTC Regime Detector</h1>
<p class="sub">Estado de mercado de Bitcoin con datos hasta {st['fecha']}. Solo usa datos pasados; señal al cierre, aplicada al día siguiente.</p>

<div class="hero">
  <div class="regime">
    <div class="lbl">Estado oficial (v2)</div>
    <div class="big {cls}">{icon} {html.escape(st['estado_oficial'])}</div>
    <div class="lbl">Base: {html.escape(base)} + fuerza de tendencia ER90 (umbral {st['er_umbral']})</div>
    <div class="lbl">Consenso de detectores individuales: {n_bull} de {n_tot} en alcista</div>
    <details><summary>Ver detectores individuales</summary>
      <div class="tw"><table><tbody>{det_rows}</tbody></table></div></details>
  </div>
  <div class="cards">{cards_html}</div>
</div>

<div class="panel">{chart_price(df, states)}</div>
<div class="grid2">
  {'<div class="panel">' + mvrv_chart + '</div>' if mvrv_chart else ''}
  <div class="panel">{chart_equity(df, states, start, cost)}</div>
</div>

<h2>Comportamiento de BTC en cada estado</h2>
<div>
  <div class="panel">{table_html(by_state, index=False)}
  <p class="note">Rentabilidad de BTC al día siguiente de cada estado, anualizada. IS = {start} a {split}; OOS = desde {split}.</p></div>
  <div class="panel">{table_html(strat, index=False)}
  <p class="note">Tener BTC según el estado y liquidez el resto, con {cost}% de coste por cambio. Calmar = CAGR / |drawdown máximo|.</p></div>
</div>

<h2>Registro en vivo (examen real del detector)</h2>
<div class="panel">{table_html(log_tbl, index=False)}
<p class="note">Una fila por día desde {log_since}, escrita en el momento y nunca modificada después. El detector v2 se adoptó tras ver el out-of-sample, así que este registro es su validación honesta.</p></div>

<h2>Métricas por detector individual</h2>
<div class="panel">{table_html(mt, highlight=base)}
<p class="note">Selección pre-registrada v1 (mejor Calmar in-sample): <b>{html.escape(v1)}</b>, que en out-of-sample quedó por debajo de comprar y mantener. La v2 usa <b>{html.escape(base)}</b> (fila resaltada).</p></div>

<h2>Retraso frente a techos y suelos de ciclo</h2>
<div class="panel">{table_html(lag_tbl, index=False)}
<p class="note">Días desde el techo/suelo (identificado a posteriori: máximo histórico seguido de una caída ≥ 60%) hasta que el detector cambia, y % que se movió el precio en ese tiempo.</p></div>

<p class="note">Datos: Coin Metrics Community API. Proyecto 006 · github.com/ivanalonso-dev</p>
</main>
<script>window.addEventListener("load",()=>document.querySelectorAll(".js-plotly-plot").forEach(e=>Plotly.Plots.resize(e)));</script>
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Dashboard generado: {out}  (ábrelo con doble clic)")


def main():
    ap = argparse.ArgumentParser(description="Dashboard del régimen de BTC")
    ap.add_argument("--data", default=os.path.join("data", "btc_daily.csv"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", default="resultados")
    ap.add_argument("--log", default=os.path.join("registro", "regime_log.csv"))
    ap.add_argument("--out", default="dashboard.html")
    a = ap.parse_args()
    df = synthetic_data() if a.synthetic else load_data(a.data)
    build(df, a.results, a.out, a.log)


if __name__ == "__main__":
    main()
