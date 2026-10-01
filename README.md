# gta-transit-delay-forecasting

**1-day-ahead forecasting of system-wide TTC bus delay minutes**, joined with Toronto weather from Open-Meteo. Built as a GTA-focused portfolio project for data / ML new-grad roles (York U BDS Computation).

## Problem

Can we predict **tomorrow’s total reported TTC bus delay minutes** better than naive calendars/lags, using only information available at the end of today — and do it with a leakage-safe time split?

Incident-level delay logs are noisy and heavy-tailed. This project aggregates them to a **daily system total**, engineers lag / rolling / calendar / weather features, and compares honest baselines to a stronger tree model.

## Demo

```text
$ python -m src.run_all

=== Validation MAE ===
  naive_lag1                mae=1039.5  …
  seasonal_lag7             mae=1201.0  …
  ridge                     mae=865.4   …
  hist_gradient_boosting    mae=860.5   … *

=== Test metrics ===
  naive_lag1                mae=966.8   rmse=1304.2  mape=32.0%
  seasonal_lag7             mae=1203.8  rmse=1526.7  mape=41.7%
  ridge                     mae=864.9   rmse=1087.8  mape=29.9%
  hist_gradient_boosting    mae=869.4   rmse=1086.1  mape=31.4% *
```

Selected by **lowest validation MAE** → report **test** once. Figures land in `reports/figures/` (parity, residuals, MAE by weekday, model bars).

## Data sources (documented)

| Source | What we use | License / terms |
|--------|-------------|-----------------|
| [TTC Bus Delay Data](https://open.toronto.ca/dataset/ttc-bus-delay-data/) (City of Toronto Open Data / CKAN) | Incident rows for **2023** and **2024** yearly workbooks; aggregated to daily `total_delay_min`, `n_incidents`, … | [Open Government Licence – Toronto](https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-licence/) |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Daily Toronto weather (temp, precip, snow, wind), **2023-01-01 → 2024-12-31** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — attribution: Weather data by Open-Meteo.com |

**Honest data notes**

- Rows are **reported delay incidents**, not GTFS-Realtime vehicle lateness. Coverage and coding practices can change year to year.
- We forecast the **system-wide daily sum** of `Min Delay`, not per-route ETA. Route-level models are a natural extension.
- The City of Toronto portal UI sometimes marks related pages oddly; the **CKAN package `ttc-bus-delay-data` remains active** and is what this repo downloads. If the feed is unreachable, use `python -m src.run_all --fixture` (synthetic daily series) or the checked-in `data/processed/*.csv` offline cache from a prior ingest.
- Weather enters as **lag-1** covariates only (known at forecast time). Same-day weather is **not** used as a model feature.

## Method

1. **Ingest transit** — CKAN `package_show` → download yearly XLSX → normalize columns → daily aggregates.
2. **Ingest weather** — Open-Meteo archive daily fields for Toronto City Hall vicinity.
3. **Inner-join on date** (731 overlapping days for 2023–2024).
4. **Features (no same-day leakage)**  
   - Delay lags: 1, 7, 14  
   - Rolling mean / std / max over 7 and 14 days on **shifted** series  
   - Calendar: DOW, month, weekend, month-start, day-of-year, coarse statutory holiday flag  
   - Weather lag-1: mean temp, precip, snow, max wind  
5. **Time-based split** (config): train ≤ 2024-04-30 · val ≤ 2024-08-31 · test = remainder. **No random shuffle.**
6. **Models** — `naive_lag1`, `seasonal_lag7`, `Ridge` (scaled), `HistGradientBoostingRegressor` with **early stopping on the chronological val set**.
7. **Selection** — lowest val MAE; **error analysis** on test (by weekday, precip bucket, worst days).

## Results

Checked-in run (seed **42**, `configs/default.yaml`):

| Model | Val MAE | Test MAE | Test RMSE | Test MAPE |
|-------|--------:|---------:|----------:|----------:|
| naive_lag1 | 1039.5 | 966.8 | 1304.2 | 32.0% |
| seasonal_lag7 | 1201.0 | 1203.8 | 1526.7 | 41.7% |
| ridge | 865.4 | **864.9** | 1087.8 | **29.9%** |
| **hist_gradient_boosting** ★ | **860.5** | 869.4 | 1086.1 | 31.4% |

★ Selected on validation. On this holdout, Ridge is essentially tied / slightly better on test MAE — an honest outcome for a noisy daily sum. Both beat lag baselines by ~10% MAE.

**Error analysis highlights:** weekends have lower MAE than mid-week; large misses cluster on extreme delay days (e.g. 2024-11-20) and holiday-adjacent schedule shocks (Christmas). R² vs a mean baseline is weak — delay totals are bursty — so we emphasize **MAE vs lag baselines**, not R² alone.

## Stack

- Python 3.11+
- pandas, NumPy, scikit-learn, matplotlib, PyYAML, requests, openpyxl, joblib, pytest

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Full pipeline (uses data/processed cache if present; else downloads TTC + Open-Meteo)
python -m src.run_all

# Or step-by-step
python -m src.train --config configs/default.yaml
python -m src.evaluate

# Offline / CI (synthetic fixtures)
python -m src.run_all --fixture
# equivalent: python -m src.train --config configs/fixture.yaml

pytest -q
```

To force re-download of transit/weather, delete `data/processed/ttc_bus_daily_delays.csv` and `data/processed/toronto_daily_weather.csv` (and optionally `data/raw/*.xlsx`).

## Project layout

```text
configs/default.yaml     # splits, features, model hyperparams, source URLs
configs/fixture.yaml     # offline CI config
src/ingest_transit.py    # CKAN TTC bus delays → daily
src/ingest_weather.py    # Open-Meteo → daily
src/features.py          # join, lags, rolling, calendar, time split
src/models.py            # baselines + HGB
src/train.py             # train / select / plots / metrics JSON
src/evaluate.py          # print metrics + error analysis
src/run_all.py           # one command
tests/                   # unit + fixture pipeline tests
data/processed/          # small daily CSVs (safe to commit)
reports/                 # metrics, predictions, figures
```

## Limitations

- **Not operational ETA prediction** — daily system totals, not vehicle-level arrival times.
- **Incident reporting ≠ passenger-experienced delay** — under/over-reporting and diversion coding matter.
- **Holiday calendar is incomplete** (fixed dates + simple observed Mondays); full Ontario holiday logic would need a dedicated library.
- **Weather is lag-1 observed**, not a proper NWP forecast product.
- **Distribution shift** — construction seasons, network changes, and labour events can break stationarity across years.
- Metrics can drift slightly across sklearn versions; seed and YAML config are fixed for this report.

## License

MIT for this repository’s code.

- TTC / City of Toronto data: Open Government Licence – Toronto (retain attribution).
- Weather: Open-Meteo, CC BY 4.0 (“Weather data by Open-Meteo.com”).

## Cite

```text
Toronto Transit Commission / City of Toronto Open Data — TTC Bus Delay Data
  https://open.toronto.ca/dataset/ttc-bus-delay-data/
Open-Meteo Historical Weather API — https://open-meteo.com/
```
