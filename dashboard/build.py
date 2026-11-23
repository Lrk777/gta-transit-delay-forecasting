"""Build dashboard/index.html from committed data and reports (no network, no extra packages).

Usage: python dashboard/build.py   (reads data/processed and reports/robustness.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from _lib import page

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"naive_lag1": "Yesterday's total", "rolling_mean7": "7-day average", "ridge": "Ridge (with weather)",
               "ridge_no_weather": "Ridge (no weather)", "hist_gradient_boosting": "Gradient boosting (with weather)",
               "hist_gradient_boosting_no_weather": "Gradient boosting (no weather)"}
CI_ROWS = [("ridge_no_weather_vs_naive_lag1", "Ridge vs yesterday's total"),
           ("ridge_no_weather_vs_rolling_mean7", "Ridge vs 7-day average"),
           ("ridge_with_vs_without_weather", "Adding weather to Ridge")]


def build_data() -> dict:
    daily = pd.read_csv(ROOT / "data" / "processed" / "ttc_bus_daily_delays.csv", parse_dates=["date"]).set_index("date")
    y = daily.total_delay_min
    roll = y.rolling(7).mean()
    forecast7 = y.shift(1).rolling(7).mean()
    rob = json.loads((ROOT / "reports" / "robustness.json").read_text())
    met = json.loads((ROOT / "reports" / "metrics.json").read_text())
    test_start, test_end = met["split_ranges"]["test"]
    t = (y.index >= test_start) & (y.index <= test_end)
    dow = y.groupby(y.index.dayofweek).mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in y.index], "daily": [int(v) for v in y],
        "roll7": [None if pd.isna(v) else round(float(v)) for v in roll],
        "test_dates": [d.strftime("%Y-%m-%d") for d in y.index[t]], "test_actual": [int(v) for v in y[t]],
        "test_forecast": [round(float(v)) for v in forecast7[t]],
        "splits": met["split_ranges"], "selected": rob["selected_on_validation"],
        "mae": sorted([{"key": k, "name": MODEL_NAMES[k], "test": round(v, 1), "val": round(rob["val_mae"][k], 1)}
                       for k, v in rob["test_mae"].items()], key=lambda r: r["test"]),
        "cis": [{"name": label, "diff": round(rob["bootstrap_95ci"][k]["mean_diff"], 1), "lo": round(rob["bootstrap_95ci"][k]["ci_low"], 1),
                 "hi": round(rob["bootstrap_95ci"][k]["ci_high"], 1), "share": rob["bootstrap_95ci"][k]["share_a_better"]} for k, label in CI_ROWS],
        "dow": [{"label": names[i], "v": round(float(dow[i]))} for i in range(7)],
        "kpi": {"days": int(len(y)), "incidents": int(daily.n_incidents.sum()), "mean": round(float(y.mean())),
                "first": y.index.min().strftime("%b %Y"), "last": y.index.max().strftime("%b %Y"),
                "mae7": round(rob["test_mae"]["rolling_mean7"], 1), "mae_naive": round(rob["test_mae"]["naive_lag1"], 1),
                "mae_ridge": round(rob["test_mae"]["ridge_no_weather"], 1), "test_days": rob["test_days"]},
    }


BODY = """
<h1>TTC bus delays: can a model beat a simple average?</h1>
<p class="sub">Forecasting tomorrow's total reported TTC bus delay minutes from TTC open data and Toronto weather. Models are chosen on a validation period and scored once on the last four months of 2024.</p>
<div class="kpis">
 <div class="kpi"><div class="v" id="k1"></div><div class="l" id="k1l"></div></div>
 <div class="kpi"><div class="v" id="k2"></div><div class="l" id="k2l"></div></div>
 <div class="kpi"><div class="v" id="k3"></div><div class="l" id="k3l"></div></div>
 <div class="kpi"><div class="v">7-day avg</div><div class="l">recommended forecast: as accurate as the best model and far simpler to run</div></div>
</div>
<div class="grid">
 <div class="card wide"><h2>Daily bus delay minutes, 2023–2024</h2><p class="note">Gray = each day, blue = 7-day average. Dashed lines mark where validation and test periods begin.</p>
  <div class="legend"><span><i style="background:var(--gray)"></i>daily total</span><span><i style="background:var(--c1)"></i>7-day average</span></div><div id="c1"></div><details><summary>Table</summary><div id="t1"></div></details></div>
 <div class="card wide"><h2>Test period: actual delays vs the 7-day-average forecast</h2><p class="note" id="n2"></p>
  <div class="legend"><span><i style="background:var(--gray)"></i>actual</span><span><i style="background:var(--c1)"></i>forecast (average of the previous 7 days)</span></div><div id="c2"></div><details><summary>Table</summary><div id="t2"></div></details></div>
 <div class="card"><h2>Forecast error by method</h2><p class="note">Average miss per day on the test period, in delay minutes. Lower is better. Blue = chosen on the validation period.</p><div id="c3"></div><details><summary>Table</summary><div id="t3"></div></details></div>
 <div class="card"><h2>Is the difference real?</h2><p class="note">Difference in average daily error (minutes) with a 95% range from 2,000 block-bootstrap resamples. Left of 0 = the first method is better. A range that crosses 0 means the gap could be noise.</p><div id="c4"></div><details><summary>Table</summary><div id="t4"></div></details></div>
 <div class="card wide"><h2>Average delay minutes by day of week</h2><p class="note">Midweek days run highest; Sundays are lowest.</p><div id="c5"></div><details><summary>Table</summary><div id="t5"></div></details></div>
</div>
<footer>Sources: Toronto Transit Commission, TTC Bus Delay Data (City of Toronto Open Data, Open Government Licence – Toronto), 2023–2024; Open-Meteo historical weather for Toronto. Code and method: <a href="https://github.com/Lrk777/gta-transit-delay-forecasting">github.com/Lrk777/gta-transit-delay-forecasting</a>.</footer>
"""

SCRIPT = r"""
const mon = d => d.slice(0, 7);
function render() {
  const K = D.kpi;
  $("#k1").textContent = fmtN(K.mean); $("#k1l").textContent = "delay minutes on an average day (" + fmtN(K.incidents) + " incidents, " + K.first + " – " + K.last + ")";
  $("#k2").textContent = fmtN(K.mae_ridge, 0); $("#k2l").textContent = "best model's average daily miss, vs " + fmtN(K.mae7, 0) + " for a 7-day average";
  $("#k3").textContent = "−" + Math.round((1 - K.mae_ridge / K.mae_naive) * 100) + "%"; $("#k3l").textContent = "error vs using yesterday's total (" + fmtN(K.mae_naive, 0) + ")";
  lineChart("#c1", D.dates, [{name: "Daily", color: css("--gray"), vals: D.daily, width: 1}, {name: "7-day avg", color: css("--c1"), vals: D.roll7, width: 2.5, strong: true}],
    {label: "Daily delay minutes", xFmt: mon, maxTicks: 12, noEndLabels: true, right: 20, marks: [{at: D.splits.val[0], label: "validation"}, {at: D.splits.test[0], label: "test"}]});
  table("#t1", ["Month", "Average daily delay minutes"], Object.entries(D.dates.reduce((a, d, i) => { const k = mon(d); (a[k] = a[k] || []).push(D.daily[i]); return a; }, {})).map(([k, v]) => [k, fmtN(v.reduce((s, x) => s + x, 0) / v.length)]));
  $("#n2").textContent = K.test_days + " days, Sep–Dec 2024. Average miss: " + fmtN(K.mae7, 1) + " minutes a day. Big one-off spikes can't be forecast by any method here.";
  lineChart("#c2", D.test_dates, [{name: "Actual", color: css("--gray"), vals: D.test_actual, width: 1.5}, {name: "Forecast", color: css("--c1"), vals: D.test_forecast, width: 2.5, strong: true}],
    {label: "Actual vs forecast", maxTicks: 8, xFmt: d => d.slice(5), right: 120});
  table("#t2", ["Date", "Actual", "Forecast", "Miss"], D.test_dates.map((d, i) => [d, fmtN(D.test_actual[i]), fmtN(D.test_forecast[i]), fmtN(D.test_actual[i] - D.test_forecast[i])]));
  hbars("#c3", D.mae.map(m => ({label: m.name, v: m.test, color: m.key === D.selected ? css("--c1") : m.key === "rolling_mean7" ? css("--c3") : css("--gray"), strong: m.key === D.selected || m.key === "rolling_mean7",
    extra: [[css("--gray"), "validation error", fmtN(m.val, 1)]]})), {left: 210, metric: "test error (min/day)", fmt: v => fmtN(v, 0)});
  table("#t3", ["Method", "Validation error", "Test error"], D.mae.map(m => [m.name, fmtN(m.val, 1), fmtN(m.test, 1)]));
  intervals("#c4", D.cis.map(c => ({label: c.name, v: c.diff, lo: c.lo, hi: c.hi, color: c.hi < 0 || c.lo > 0 ? css("--c2") : css("--c1"),
    extra: [[css("--gray"), "resamples where first is better", Math.round(c.share * 100) + "%"]]})), {left: 190, min: -400, max: 100, step: 100, metric: "difference", rangeName: "95% range", fmt: v => (v > 0 ? "+" : "") + fmtN(v, 0)});
  table("#t4", ["Comparison", "Difference", "95% range", "First better in"], D.cis.map(c => [c.name, fmtN(c.diff, 1), fmtN(c.lo, 1) + " to " + fmtN(c.hi, 1), Math.round(c.share * 100) + "%"]));
  vbars("#c5", D.dow.map(d => ({label: d.label, v: d.v, color: css("--c1")})), {metric: "average delay minutes", height: 220});
  table("#t5", ["Day", "Average delay minutes"], D.dow.map(d => [d.label, fmtN(d.v)]));
}
render();
"""


def main() -> None:
    out = ROOT / "dashboard" / "index.html"
    out.write_text(page("TTC bus delays: can a model beat a simple average?", BODY, SCRIPT, build_data()), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
