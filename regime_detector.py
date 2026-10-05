#!/usr/bin/env python3
"""
btc-regime-detector · detectores y evaluación

Clasifica cada día de Bitcoin como ALCISTA o NO ALCISTA con varios detectores
que solo usan datos hasta el cierre de ese día, y los evalúa por su VALOR
ECONÓMICO: qué habría pasado teniendo BTC solo en régimen alcista y liquidez el
resto (posición aplicada al día siguiente, coste por cambio incluido).

DETECTORES
----------
  MA200         cierre > media de 200 días
  MA200_SLOPE   cierre > media 200 y la media sube (vs hace 20 días)
  GOLDEN_CROSS  media 50 > media 200
  MVRV_TREND    MVRV > su media de 200 días (on-chain: el beneficio no realizado
                del mercado crece)                       [requiere columna mvrv]
  VOTE          mayoría de MA200_SLOPE, GOLDEN_CROSS y MVRV_TREND
                (sin MVRV: mayoría de MA200, MA200_SLOPE y GOLDEN_CROSS)
Cada uno en versión inmediata y con confirmación de N días (por defecto 5):
el régimen solo cambia si la señal se mantiene N cierres seguidos.

EVALUACIÓN
----------
  - CAGR, drawdown máximo, Sharpe, Calmar, % de tiempo invertido, cambios/año
  - Rentabilidad anualizada de BTC dentro de cada régimen
  - Retraso frente a techos y suelos de ciclo (identificados a posteriori:
    máximo histórico seguido de una caída >= 60%) solo como referencia
  - In-sample (< --split) y out-of-sample (>= --split)
  - Selección PRE-REGISTRADA: el detector con mejor Calmar in-sample
    (desempate: menos cambios/año). El out-of-sample es el examen.
  - Diagnóstico de lateralidad: ¿cambia el comportamiento de BTC dentro de
    cada régimen según la fuerza de tendencia (efficiency ratio de 90 días)?

DETECTOR OFICIAL v2 (3 estados)
-------------------------------
La regla v1 (mejor Calmar in-sample) eligió MA200 (5d), que en out-of-sample
quedó por debajo de comprar y mantener. La familia MVRV_TREND fue la única que
batió a comprar y mantener en 2020-2026 en todas sus variantes (media de 100 a
365 días), así que la v2 adopta MVRV_TREND (5d) como base. Esta decisión se tomó
DESPUÉS de ver el out-of-sample: por eso el examen real es el registro diario en
vivo (registro/regime_log.csv) desde la fecha de adopción.
  ALCISTA FUERTE           MVRV_TREND alcista y ER90 > 0.14 (confirmado 5 días)
  ALCISTA DÉBIL / LATERAL  MVRV_TREND alcista y ER90 <= 0.14
  NO ALCISTA               MVRV_TREND no alcista
El umbral 0.14 es la mediana del ER90 en el periodo in-sample (2012-2019).

Uso:
    python regime_detector.py                       # usa data/btc_daily.csv
    python regime_detector.py --confirm 5 --split 2020-01-01 --cost 0.1
    python regime_detector.py --synthetic           # prueba sin descargar nada
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

YEAR = 365
ER_STRONG = 0.14                     # mediana del ER90 en 2012-2019 (in-sample)
STATE_NAMES = {2: "ALCISTA FUERTE", 1: "ALCISTA DÉBIL / LATERAL", 0: "NO ALCISTA"}
V2_ADOPTED = "2026-10-05"            # fecha de adopción del detector v2


# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ["date", "time", "timestamp"] if c in df.columns)
    df.index = pd.to_datetime(df[tcol]).dt.tz_localize(None).dt.normalize()
    df = df.drop(columns=[tcol])
    if "price" not in df:
        pc = next((c for c in ["close", "priceusd"] if c in df.columns), None)
        if pc is None:
            raise ValueError(f"Falta columna price/close en {list(df.columns)}")
        df = df.rename(columns={pc: "price"})
    if "mvrv" not in df and {"market_cap", "realized_cap"} <= set(df.columns):
        df["mvrv"] = df["market_cap"] / df["realized_cap"]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["price"] > 0]
    # las métricas on-chain se publican con 1-2 días de retraso: se arrastra el último
    # valor conocido (máx. 3 días) para no dejar los detectores on-chain en n/d
    for col in ["mvrv", "market_cap", "realized_cap", "realized_price"]:
        if col in df:
            df[col] = df[col].ffill(limit=3)
    return df


def synthetic_data(seed: int = 3) -> pd.DataFrame:
    """Serie con ciclos alcistas/bajistas tipo BTC para probar el código."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2011-01-01", "2026-09-30", freq="D")
    n = len(idx)
    cycle = np.sin(2 * np.pi * np.arange(n) / (4 * 365) - 1.2)
    drift = 0.0035 * cycle + 0.0008
    r = drift + rng.standard_t(4, n) * 0.025
    price = 0.5 * np.exp(np.cumsum(r))
    realized = pd.Series(price).ewm(span=500).mean().values * 0.8
    return pd.DataFrame({"price": price, "mvrv": price / realized}, index=idx)


# -----------------------------------------------------------------------------
# Detectores
# -----------------------------------------------------------------------------
def confirm(raw: pd.Series, n: int) -> pd.Series:
    """El estado solo cambia tras n cierres seguidos con la nueva señal."""
    if n <= 1:
        return raw.astype(float)
    vals = raw.values
    out = np.full(len(vals), np.nan)
    state, run_val, run_len = np.nan, None, 0
    for i, v in enumerate(vals):
        if np.isnan(v):
            out[i] = state
            continue
        if np.isnan(state):
            state = v
        if v == run_val:
            run_len += 1
        else:
            run_val, run_len = v, 1
        if v != state and run_len >= n:
            state = v
        out[i] = state
    return pd.Series(out, index=raw.index)


def raw_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["price"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()

    def b(cond, *req):
        s = cond.astype(float)
        for r in req:
            s[r.isna()] = np.nan
        return s

    sig = {
        "MA200": b(c > sma200, sma200),
        "MA200_SLOPE": b((c > sma200) & (sma200 > sma200.shift(20)), sma200.shift(20)),
        "GOLDEN_CROSS": b(sma50 > sma200, sma200),
    }
    if "mvrv" in df and df["mvrv"].notna().sum() > 400:
        m = df["mvrv"]
        m200 = m.rolling(200).mean()
        sig["MVRV_TREND"] = b(m > m200, m200)
        members = ["MA200_SLOPE", "GOLDEN_CROSS", "MVRV_TREND"]
    else:
        members = ["MA200", "MA200_SLOPE", "GOLDEN_CROSS"]
    stack = pd.concat([sig[k] for k in members], axis=1)
    vote = (stack.sum(axis=1) >= 2).astype(float)
    vote[stack.isna().any(axis=1)] = np.nan
    sig["VOTE"] = vote
    return sig


def build_regimes(df: pd.DataFrame, n_confirm: int) -> pd.DataFrame:
    out = {}
    for name, s in raw_signals(df).items():
        out[name] = s
        if n_confirm > 1:
            out[f"{name} ({n_confirm}d)"] = confirm(s, n_confirm)
    return pd.DataFrame(out)


def efficiency_ratio(c: pd.Series, n: int = 90) -> pd.Series:
    """Kaufman: 1 = tendencia limpia, 0 = ruido/lateral."""
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


# -----------------------------------------------------------------------------
# Evaluación
# -----------------------------------------------------------------------------
def strategy_returns(price: pd.Series, regime: pd.Series, cost_pct: float) -> pd.Series:
    r = price.pct_change().fillna(0)
    pos = regime.shift(1).fillna(0)           # señal al cierre de t -> posición en t+1
    trades = pos.diff().abs().fillna(pos.abs())
    return pos * r - trades * cost_pct / 100


def perf(rets: pd.Series) -> dict:
    if len(rets) < 30:
        return {}
    eq = (1 + rets).cumprod()
    years = len(rets) / YEAR
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    vol = rets.std() * np.sqrt(YEAR)
    sharpe = rets.mean() * YEAR / vol if vol > 0 else np.nan
    return {"CAGR_%": 100 * cagr, "maxDD_%": 100 * dd, "Sharpe": sharpe,
            "Calmar": cagr / abs(dd) if dd < 0 else np.nan}


def evaluate(df: pd.DataFrame, regimes: pd.DataFrame, start: str, split: str,
             cost_pct: float) -> pd.DataFrame:
    price = df["price"]
    nxt = price.pct_change().shift(-1)        # rentabilidad del día siguiente
    rows = []
    bh = price.pct_change().fillna(0)
    for name in ["BUY_HOLD"] + list(regimes.columns):
        if name == "BUY_HOLD":
            rets, reg = bh, pd.Series(1.0, index=price.index)
        else:
            reg = regimes[name]
            rets = strategy_returns(price, reg, cost_pct)
        row = {"detector": name}
        for tag, lo, hi in [("", start, None), ("IS ", start, split), ("OOS ", split, None)]:
            m = (rets.index >= lo) & ((rets.index < hi) if hi else True)
            p = perf(rets[m])
            for k, v in p.items():
                row[f"{tag}{k}"] = v
        m = reg.index >= start
        rg = reg[m]
        row["%_invertido"] = 100 * rg.mean()
        row["cambios/año"] = rg.diff().abs().sum() / (m.sum() / YEAR) if name != "BUY_HOLD" else 0
        if name != "BUY_HOLD":
            bull = nxt[m][rg == 1].dropna()
            bear = nxt[m][rg == 0].dropna()
            row["BTC_alcista_%anual"] = 100 * bull.mean() * YEAR if len(bull) else np.nan
            row["BTC_no_alcista_%anual"] = 100 * bear.mean() * YEAR if len(bear) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("detector")


def cycle_extremes(price: pd.Series, min_dd: float = 0.60, min_rally: float = 1.0):
    """Techos = máximos históricos seguidos de caída >= min_dd. Suelos = mínimo entre un
    techo y el siguiente máximo histórico (o fin de datos si ya subió >= min_rally).
    Identificados A POSTERIORI: solo sirven para medir retrasos."""
    tops, bottoms = [], []
    peak_val, peak_date, marked = -np.inf, None, False
    for d, v in price.items():
        if v > peak_val:
            if marked:  # nuevo máximo histórico tras un techo marcado -> cerrar ciclo
                seg = price[tops[-1][0]:d]
                bottoms.append((seg.idxmin(), seg.min()))
            peak_val, peak_date, marked = v, d, False
        elif not marked and v <= peak_val * (1 - min_dd):
            tops.append((peak_date, peak_val))
            marked = True
    if marked:
        seg = price[tops[-1][0]:]
        bmin, bdate = seg.min(), seg.idxmin()
        if price[bdate:].max() >= bmin * (1 + min_rally):
            bottoms.append((bdate, bmin))
    return tops, bottoms


def lags(price: pd.Series, regimes: pd.DataFrame, start: str) -> pd.DataFrame:
    tops, bottoms = cycle_extremes(price)
    rows = []
    for kind, events, target in [("TECHO", tops, 0.0), ("SUELO", bottoms, 1.0)]:
        for d, v in events:
            if d < pd.Timestamp(start):
                continue
            row = {"evento": kind, "fecha": d.date(), "precio": round(v, 2)}
            for name in regimes.columns:
                after = regimes[name][d:]
                hit = after[after == target]
                if len(hit):
                    hd = hit.index[0]
                    days = (hd - d).days
                    move = 100 * (price[hd] / v - 1)
                    row[name] = f"{days}d ({move:+.0f}%)"
                else:
                    row[name] = "pendiente"
            rows.append(row)
    return pd.DataFrame(rows)


def lateral_diagnostic(df: pd.DataFrame, regime: pd.Series, start: str) -> pd.DataFrame:
    er = efficiency_ratio(df["price"])
    nxt = df["price"].pct_change().shift(-1)
    m = (df.index >= pd.Timestamp(start)) & er.notna() & regime.notna() & nxt.notna()
    thr = er[m].median()
    rows = []
    for reg_name, rv in [("ALCISTA", 1.0), ("NO ALCISTA", 0.0)]:
        for er_name, cond in [("tendencia fuerte", er > thr), ("débil / lateral", er <= thr)]:
            sel = nxt[m & (regime == rv) & cond]
            if len(sel) < 30:
                continue
            ann, vol = sel.mean() * YEAR, sel.std() * np.sqrt(YEAR)
            rows.append({"régimen": reg_name, "fuerza (ER90)": er_name, "días": len(sel),
                         "BTC_%anual": round(100 * ann, 1), "vol_%": round(100 * vol, 1),
                         "Sharpe": round(ann / vol, 2) if vol > 0 else np.nan})
    out = pd.DataFrame(rows)
    out.attrs["umbral_er"] = float(thr)
    return out


def select_detector(metrics: pd.DataFrame) -> str:
    cand = metrics.drop(index="BUY_HOLD")
    cand = cand.sort_values(["IS Calmar", "cambios/año"], ascending=[False, True])
    return cand.index[0]


def official_base(regimes: pd.DataFrame, n_confirm: int) -> str:
    sfx = f" ({n_confirm}d)" if n_confirm > 1 else ""
    for name in [f"MVRV_TREND{sfx}", f"MA200{sfx}"]:
        if name in regimes and regimes[name].notna().any():
            return name
    return regimes.columns[0]


def official_states(df: pd.DataFrame, regimes: pd.DataFrame, n_confirm: int) -> tuple[pd.Series, str]:
    base_name = official_base(regimes, n_confirm)
    base = regimes[base_name]
    er = efficiency_ratio(df["price"])
    strong_raw = (er > ER_STRONG).astype(float)
    strong_raw[er.isna()] = np.nan
    strong = confirm(strong_raw, n_confirm)
    st = pd.Series(np.where(base == 1, 1 + strong.fillna(0), 0.0), index=df.index)
    st[base.isna()] = np.nan
    return st, base_name


def evaluate_states(df: pd.DataFrame, states: pd.Series, start: str, split: str,
                    cost_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = df["price"]
    nxt = price.pct_change().shift(-1)
    m0 = df.index >= pd.Timestamp(start)
    rows = []
    for code in [2, 1, 0]:
        for tag, lo, hi in [("TOTAL", start, None), ("IS", start, split), ("OOS", split, None)]:
            m = m0 & (df.index >= pd.Timestamp(lo)) & ((df.index < pd.Timestamp(hi)) if hi else True)
            sel = nxt[m & (states == code)].dropna()
            if len(sel) < 20:
                continue
            ann, vol = sel.mean() * YEAR, sel.std() * np.sqrt(YEAR)
            rows.append({"estado": STATE_NAMES[code], "tramo": tag, "días": len(sel),
                         "%_tiempo": round(100 * len(sel) / m.sum(), 1),
                         "BTC_%anual": round(100 * ann, 1), "vol_%": round(100 * vol, 1),
                         "Sharpe": round(ann / vol, 2) if vol > 0 else np.nan})
    by_state = pd.DataFrame(rows)

    strat_rows = []
    variants = {"Comprar y mantener": pd.Series(1.0, index=df.index),
                "BTC en ALCISTA (fuerte + débil)": (states >= 1).astype(float),
                "BTC solo en ALCISTA FUERTE": (states == 2).astype(float)}
    for name, reg in variants.items():
        rets = strategy_returns(price, reg, cost_pct) if name != "Comprar y mantener" \
            else price.pct_change().fillna(0)
        row = {"estrategia": name}
        for tag, lo, hi in [("IS", start, split), ("OOS", split, None)]:
            m = (rets.index >= lo) & ((rets.index < hi) if hi else True)
            for k, v in perf(rets[m]).items():
                row[f"{tag} {k}"] = round(v, 2)
        strat_rows.append(row)
    return by_state, pd.DataFrame(strat_rows)


def update_log(path: str, df: pd.DataFrame, states: pd.Series, base_name: str) -> pd.DataFrame:
    """Registro en vivo: una fila por cada fecha nueva, nunca se reescribe el pasado."""
    cols = ["fecha", "precio", "estado", "detector_base", "mvrv", "er90", "registrado_en"]
    log = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)
    last = df.index[-1]
    fecha = str(last.date())
    if fecha not in set(log["fecha"].astype(str)) and pd.notna(states.iloc[-1]):
        er = efficiency_ratio(df["price"]).iloc[-1]
        row = {"fecha": fecha, "precio": round(float(df["price"].iloc[-1]), 2),
               "estado": STATE_NAMES[int(states.iloc[-1])], "detector_base": base_name,
               "mvrv": round(float(df["mvrv"].iloc[-1]), 4) if "mvrv" in df else np.nan,
               "er90": round(float(er), 4),
               "registrado_en": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        log.to_csv(path, index=False)
    return log


def current_state(df: pd.DataFrame, regimes: pd.DataFrame, selected: str,
                  states: pd.Series | None = None, base_name: str = "") -> dict:
    last = df.index[-1]
    reg = regimes[selected].dropna()
    cur = reg.iloc[-1]
    changes = reg[reg.diff() != 0]
    since = changes.index[-1] if len(changes) else reg.index[0]
    c = df["price"]
    sma200 = c.rolling(200).mean()
    er = efficiency_ratio(c)
    st = {
        "fecha": str(last.date()),
        "precio": float(c.iloc[-1]),
        "detector": selected,
        "regimen": "ALCISTA" if cur == 1 else "NO ALCISTA",
        "desde": str(since.date()),
        "dias_en_regimen": int((last - since).days),
        "dist_ma200_%": float(100 * (c.iloc[-1] / sma200.iloc[-1] - 1)),
        "er90": float(er.iloc[-1]),
        "detectores": {k: ("ALCISTA" if v == 1 else "NO ALCISTA" if v == 0 else "n/d")
                       for k, v in regimes.iloc[-1].items()},
    }
    if states is not None:
        s = states.dropna()
        ch = s[s.diff() != 0]
        since3 = ch.index[-1] if len(ch) else s.index[0]
        st.update({"estado_oficial": STATE_NAMES[int(s.iloc[-1])], "detector_oficial": base_name,
                   "estado_desde": str(since3.date()), "dias_en_estado": int((last - since3).days),
                   "er_umbral": ER_STRONG, "v2_adoptado": V2_ADOPTED})
    if "mvrv" in df and pd.notna(df["mvrv"].iloc[-1]):
        m = df["mvrv"].dropna()
        st["mvrv"] = float(m.iloc[-1])
        st["mvrv_percentil"] = float(100 * (m < m.iloc[-1]).mean())
    if "realized_price" in df and pd.notna(df["realized_price"].iloc[-1]):
        st["precio_realizado"] = float(df["realized_price"].iloc[-1])
    return st


def run(df: pd.DataFrame, n_confirm: int, start: str, split: str, cost: float, out: str):
    regimes = build_regimes(df, n_confirm)
    metrics = evaluate(df, regimes, start, split, cost)
    selected = select_detector(metrics)
    lag = lags(df["price"], regimes, start)
    lat = lateral_diagnostic(df, regimes[selected], start)
    states, base_name = official_states(df, regimes, n_confirm)
    by_state, strat = evaluate_states(df, states, start, split, cost)
    state = current_state(df, regimes, selected, states, base_name)

    os.makedirs(out, exist_ok=True)
    metrics.round(3).to_csv(os.path.join(out, "metricas.csv"))
    regimes.to_csv(os.path.join(out, "regimenes.csv"))
    lag.to_csv(os.path.join(out, "retrasos.csv"), index=False)
    lat.to_csv(os.path.join(out, "lateralidad.csv"), index=False)
    states.rename("estado").to_csv(os.path.join(out, "estados_oficiales.csv"))
    by_state.to_csv(os.path.join(out, "estados_btc.csv"), index=False)
    strat.to_csv(os.path.join(out, "estados_estrategias.csv"), index=False)
    with open(os.path.join(out, "estado_actual.json"), "w", encoding="utf-8") as f:
        json.dump({**state, "config": {"confirm": n_confirm, "start": start, "split": split,
                                       "cost_pct": cost}}, f, indent=2, ensure_ascii=False)
    return regimes, metrics, selected, lag, lat, state, states, base_name, by_state, strat


def main():
    ap = argparse.ArgumentParser(description="Detector de régimen de BTC")
    ap.add_argument("--data", default=os.path.join("data", "btc_daily.csv"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--confirm", type=int, default=5, help="días de confirmación")
    ap.add_argument("--start", default="2012-01-01", help="inicio de la evaluación")
    ap.add_argument("--split", default="2020-01-01", help="inicio del out-of-sample")
    ap.add_argument("--cost", type=float, default=0.10, help="coste por cambio de posición, %%")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--log", default=os.path.join("registro", "regime_log.csv"),
                    help="registro diario en vivo (no se escribe con --synthetic)")
    a = ap.parse_args()

    df = synthetic_data() if a.synthetic else load_data(a.data)
    print(f"Datos: {len(df):,} días  {df.index[0].date()} -> {df.index[-1].date()}  "
          f"(MVRV: {'sí' if 'mvrv' in df else 'no'})")
    (regimes, metrics, selected, lag, lat, st,
     states, base_name, by_state, strat) = run(df, a.confirm, a.start, a.split, a.cost, a.out)
    log = None if a.synthetic else update_log(a.log, df, states, base_name)

    show = ["CAGR_%", "maxDD_%", "Calmar", "IS Calmar", "OOS CAGR_%", "OOS maxDD_%", "OOS Calmar",
            "%_invertido", "cambios/año", "BTC_alcista_%anual", "BTC_no_alcista_%anual"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:,.2f}".format):
        print(f"\n=== MÉTRICAS (evaluación desde {a.start}; IS < {a.split} <= OOS; coste {a.cost}% por cambio) ===")
        print(metrics[[c for c in show if c in metrics]].to_string())
        print(f"\n=== SELECCIÓN PRE-REGISTRADA v1 (mejor Calmar in-sample): {selected} ===")
        print("\n=== RETRASOS frente a techos/suelos de ciclo (días hasta la señal, % movido) ===")
        shown = [c for c in regimes.columns if a.confirm <= 1 or c.endswith(f"({a.confirm}d)")]
        if selected not in shown:
            shown = [selected] + shown
        print(lag[["evento", "fecha", "precio"] + shown].to_string(index=False)
              if len(lag) else "  (sin ciclos completos en el periodo)")
        print("  (tabla completa en retrasos.csv)")
        print(f"\n=== LATERALIDAD con {selected} (ER90 umbral = {lat.attrs.get('umbral_er', float('nan')):.2f}) ===")
        print(lat.to_string(index=False))
        print("  -> Si dentro de ALCISTA el Sharpe de 'débil / lateral' es muy inferior al de 'tendencia fuerte',")
        print("     un tercer estado LATERAL aporta valor. Si son parecidos, basta con 2 estados.")

        print(f"\n=== DETECTOR OFICIAL v2: {base_name} + ER90 (umbral {ER_STRONG}) ===")
        print(by_state.to_string(index=False))
        print()
        print(strat.to_string(index=False))

    print(f"\n=== ESTADO ACTUAL ({st['fecha']}) ===")
    print(f"  ESTADO OFICIAL: {st['estado_oficial']}  (desde {st['estado_desde']}, {st['dias_en_estado']} días)")
    print(f"  Régimen v1: {st['regimen']}  (detector {st['detector']}, desde {st['desde']}, {st['dias_en_regimen']} días)")
    print(f"  Precio: {st['precio']:,.2f} $   distancia a MA200: {st['dist_ma200_%']:+.1f}%   ER90: {st['er90']:.2f}")
    if "mvrv" in st:
        print(f"  MVRV: {st['mvrv']:.2f} (percentil histórico {st['mvrv_percentil']:.0f})")
    print("  Detectores:", ", ".join(f"{k}={v}" for k, v in st["detectores"].items()))
    if log is not None:
        print(f"\nRegistro en vivo: {a.log} ({len(log)} días registrados desde {log['fecha'].iloc[0]})")
    print(f"\nFicheros en ./{a.out}/  -> ahora ejecuta: python dashboard.py")


if __name__ == "__main__":
    main()
