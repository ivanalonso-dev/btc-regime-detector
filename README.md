# btc-regime-detector

Bitcoin market regime detection using price-trend and on-chain cycle indicators, validated by **economic value** on the full BTC history (2010–2026), with an HTML dashboard and a **live daily log** that serves as the out-of-sample test going forward.

Project 006 of a three-part series:
- **006 · btc-regime-detector** — which regime is BTC in? *(this repo)*
- **007 · crypto-bull-regime-system** — trading system for bull regimes
- **008 · crypto-bear-regime-system** — trading system for bear regimes

> **TL;DR:** Price-based regime filters (200-day MA, golden cross) did **not** beat buy-and-hold on a risk-adjusted basis in 2020–2026. An **on-chain** filter — MVRV above its own 200-day average — did, in every parameter variant tested: in 2020–2026 it returned **48% CAGR with a −33% max drawdown, vs 45% / −77% for buy-and-hold**, and flagged the last three cycle tops 2–4× faster than the moving averages. Because it was adopted after seeing those results, its real test is the live log started on 2026-10-05.

## Official regime (v2, three states)

| State | Rule | Used by |
|---|---|---|
| **BULL – STRONG** | MVRV > its 200-day average **and** 90-day Kaufman efficiency ratio > 0.14 | 007 (trend-following) |
| **BULL – WEAK / SIDEWAYS** | MVRV > its 200-day average **and** ER90 ≤ 0.14 | 007 (mean reversion) |
| **NOT BULL** | MVRV ≤ its 200-day average | 008 |

All signals use data up to each day's close only and require **5 consecutive closes** to switch. The 0.14 threshold is the median ER90 of the in-sample period (2012–2019).

## Methodology

- **Candidates** (each raw and with 5-day confirmation): `MA200`, `MA200_SLOPE` (price > MA200 and MA200 rising), `GOLDEN_CROSS` (MA50 > MA200), `MVRV_TREND` (MVRV > its 200-day average), `VOTE` (majority).
- **Economic test:** hold BTC only in the bull regime, cash otherwise; signal at close, position next day, 0.10% cost per switch.
- **Metrics:** CAGR, max drawdown, Sharpe, Calmar, % invested, switches/year, BTC return inside each regime, lag versus cycle tops/bottoms (identified ex-post as an all-time high followed by a ≥ 60% drawdown).
- **Validation:** in-sample 2012–2019, out-of-sample 2020–2026, with a **pre-registered selection rule** (best in-sample Calmar).

Data: [Coin Metrics Community API](https://docs.coinmetrics.io/api) — daily price, market cap and MVRV since 2010.

## Results

### Detectors (strategy: BTC only in bull regime)

| Detector | In-sample Calmar | OOS CAGR | OOS max DD | OOS Calmar | Switches / yr |
|---|---|---|---|---|---|
| Buy & hold | 1.77 | 44.5% | −76.7% | 0.58 | 0 |
| MA200 (5d) — *v1 pick* | **2.54** | 28.5% | −54.4% | 0.52 | 2.1 |
| MA200_SLOPE (5d) | 2.26 | 32.2% | −61.9% | 0.52 | 2.0 |
| GOLDEN_CROSS (5d) | 2.01 | 26.2% | −56.8% | 0.46 | 1.7 |
| VOTE (5d) | 2.36 | 37.0% | −53.0% | 0.70 | 1.8 |
| **MVRV_TREND (5d)** | 2.36 | **48.4%** | **−33.2%** | **1.46** | 2.9 |

**The pre-registered rule failed:** it picked MA200 (5d), which under-performed buy-and-hold out-of-sample. We report this rather than hide it.

### Robustness of the on-chain signal (5-day confirmation, OOS Calmar)

| MVRV average | 100d | 150d | 200d | 250d | 300d | 365d |
|---|---|---|---|---|---|---|
| OOS Calmar | 0.65 | 0.79 | 1.46 | 0.95 | 1.14 | 0.77 |

Every MVRV variant beat buy-and-hold (0.58); no price moving average did (MA150/200/250: 0.68 / 0.52 / 0.55). 200 days is the best and likely partly lucky — a realistic expectation is a Calmar around 1.

### Lag to cycle tops (days until the detector turns "not bull", price change meanwhile)

| Top | MVRV_TREND (5d) | MA200 (5d) |
|---|---|---|
| 2013-12-04 | 20d (−43%) | 117d (−60%) |
| 2017-12-16 | 16d (−31%) | 92d (−58%) |
| 2021-11-08 | 30d (−25%) | 54d (−30%) |

At cycle bottoms all detectors lag similarly (2–5 months).

### BTC behaviour in each v2 state (next-day returns, annualized)

| State | % of time | BTC return | Sharpe (IS / OOS) |
|---|---|---|---|
| Bull – strong | 32% | 214% | 2.53 / 1.98 |
| Bull – weak / sideways | 20% | 99% | 2.47 / 1.51 |
| Not bull | 48% | 14% | 0.15 / 0.24 |

| Strategy | OOS CAGR | OOS max DD | OOS Calmar |
|---|---|---|---|
| Buy & hold | 44.5% | −76.7% | 0.58 |
| BTC in any bull state | 48.4% | −33.2% | 1.46 |
| BTC only in bull – strong | 30.6% | −25.6% | 1.20 |

**Reading:** the weak/sideways sub-state is still profitable, so it should **not** be used to exit; its value is to tell project 007 *which kind* of system to run (trend-following vs mean reversion). Being out of the market is not free: BTC still rose ~14% a year in the "not bull" state — the detector's value lies in avoiding deep drawdowns.

## Limitations

- Only ~3 complete BTC cycles in the data; every statistic rests on few independent episodes.
- The v2 detector was chosen after seeing out-of-sample results → the live log is the only clean test.
- On-chain metrics are published with a 1–2 day delay (the last value is carried forward up to 3 days).
- Cycle tops/bottoms used for lag measurement are identified ex-post.

## Usage

```bash
pip install -r requirements.txt
python fetch_data.py          # BTC daily since 2010 (Coin Metrics)
python regime_detector.py     # detectors, evaluation, v2 states, live log -> resultados/, registro/
python dashboard.py           # -> dashboard.html
```

On Windows, `run_daily.bat` runs all three; schedule it once a day (Task Scheduler) to keep the live log growing. `--synthetic` runs everything on simulated data without downloading.

Outputs: `resultados/` (metrics, regimes, states, lags, current state) and `registro/regime_log.csv` — **one row per day, written at the time and never modified**, committed to the repo as a public forward record.

---
*Author: Ivan Alonso · [github.com/ivanalonso-dev](https://github.com/ivanalonso-dev)*
