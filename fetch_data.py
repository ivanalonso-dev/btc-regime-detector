#!/usr/bin/env python3
"""
btc-regime-detector · descarga de datos

Descarga la historia diaria de Bitcoin desde la API gratuita de Coin Metrics
(community-api.coinmetrics.io): precio, capitalización, capitalización realizada
y MVRV, desde 2010. Guarda data/btc_daily.csv.

Cada métrica se pide por separado: si alguna no está disponible en el plan
gratuito, se omite sin romper la descarga. Si falta MVRV pero existen las dos
capitalizaciones, se calcula como CapMrktCurUSD / CapRealUSD.

Uso:
    python fetch_data.py
    python fetch_data.py --start 2010-07-18 --out data/btc_daily.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = {
    "PriceUSD": "price",
    "CapMrktCurUSD": "market_cap",
    "CapRealUSD": "realized_cap",
    "CapMVRVCur": "mvrv",
}
HEADERS = {"User-Agent": "btc-regime-detector/1.0 (research)", "Accept": "application/json"}


def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                raise
            time.sleep(2 * (i + 1))
        except urllib.error.URLError:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}")


def fetch_metric(metric: str, start: str) -> pd.Series:
    params = {"assets": "btc", "metrics": metric, "frequency": "1d",
              "page_size": 10000, "start_time": start}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    rows = []
    while url:
        payload = _get(url)
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_numeric(df[metric], errors="coerce").values, index=idx, name=metric)


def main():
    ap = argparse.ArgumentParser(description="Descarga BTC diario (Coin Metrics)")
    ap.add_argument("--start", default="2010-07-18")
    ap.add_argument("--out", default=os.path.join("data", "btc_daily.csv"))
    a = ap.parse_args()

    cols = {}
    for metric, name in METRICS.items():
        try:
            s = fetch_metric(metric, a.start)
            if s.notna().sum() > 0:
                cols[name] = s
                print(f"  OK  {metric:15s} {s.notna().sum():>6,} días  ({s.first_valid_index().date()} -> {s.last_valid_index().date()})")
            else:
                print(f"  --  {metric:15s} sin datos")
        except Exception as e:  # métrica no incluida en el plan gratuito, etc.
            print(f"  --  {metric:15s} no disponible ({e})")

    if "price" not in cols:
        raise SystemExit("No se pudo descargar el precio. Revisa la conexión o usa un CSV propio "
                         "(columnas: date, price[, mvrv]) con regime_detector.py --data.")

    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    if "mvrv" not in df and {"market_cap", "realized_cap"} <= set(df.columns):
        df["mvrv"] = df["market_cap"] / df["realized_cap"]
        print("  MVRV calculado como market_cap / realized_cap")
    if "realized_cap" in df and "market_cap" in df:
        df["realized_price"] = df["price"] * df["realized_cap"] / df["market_cap"]
    df = df[df["price"].notna()]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out)
    print(f"\nGuardado {a.out}: {len(df):,} días, {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Columnas: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
