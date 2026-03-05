"""Build an interactive cyberpunk-theme Plotly dashboard with dropdown navigation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        out: dict[str, object] = {}
        for k, v in row.items():
            if v is None:
                out[k] = ""
                continue
            vv = v.strip()
            if vv == "":
                out[k] = ""
                continue
            if vv.lower() in {"true", "false"}:
                out[k] = vv.lower() == "true"
                continue
            try:
                out[k] = float(vv)
            except Exception:
                out[k] = vv
        normalized.append(out)
    return normalized


def build_html(data: dict[str, list[dict[str, object]]]) -> str:
    payload = json.dumps(data, indent=2)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NER to JSON - Interactive Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #070a14;
      --panel: #0d1326;
      --ink: #e8f7ff;
      --muted: #9abbd9;
      --line: #27426c;
      --accent: #00eaff;
      --accent2: #ff2bd6;
      --radius: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Rajdhani", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(920px 420px at -8% -10%, rgba(0, 234, 255, .16), transparent),
        radial-gradient(920px 420px at 112% -8%, rgba(255, 43, 214, .13), transparent),
        var(--bg);
    }
    .app {
      height: 100vh;
      padding: 10px 10px 16px;
      overflow: hidden;
    }
    .main {
      max-width: 1320px;
      margin: 0 auto;
    }
    .header {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px 12px;
      box-shadow: 0 0 0 1px rgba(0,234,255,0.13), 0 10px 24px rgba(0, 0, 0, .45);
    }
    .header h2 { margin: 0; font-size: 1.16rem; color: #e8f7ff; letter-spacing: .4px; }
    .header p { margin: 4px 0 0; color: var(--muted); font-size: .92rem; }
    .controls {
      margin-top: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 8px 10px;
      box-shadow: 0 0 0 1px rgba(0,234,255,0.10), 0 8px 18px rgba(0, 0, 0, .35);
    }
    .controls label {
      display: block;
      font-size: .88rem;
      color: #b8d7f4;
      font-weight: 650;
      margin-bottom: 4px;
    }
    .controls select {
      width: 100%;
      max-width: 560px;
      border: 1px solid #2a4f7f;
      background: #0c1730;
      color: #e6f7ff;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: .94rem;
      font-weight: 600;
      outline: none;
    }
    .controls select:focus {
      border-color: #00eaff;
      box-shadow: 0 0 0 3px rgba(0, 234, 255, 0.14);
    }

    .panel {
      display: none;
      margin-top: 8px;
      margin-bottom: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 0 0 1px rgba(0,234,255,0.10), 0 10px 22px rgba(0, 0, 0, .38);
      padding: 6px;
      height: calc(100vh - 178px);
    }
    .panel.active { display: block; }
    .panel-head { padding: 8px 9px 0; }
    .panel-head h3 { margin: 0; font-size: .98rem; color: #dff7ff; }
    .panel-head p { margin: 3px 0 4px; color: var(--muted); font-size: .86rem; }
    .plot { width: 100%; height: calc(100% - 54px); min-height: 240px; }
    @media (max-width: 1000px) {
      .app { padding: 8px; }
      .panel { height: calc(100vh - 168px); }
      .plot { min-height: 220px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <main class="main">
      <section class="header">
        <h2>Experiment Results (Interactive)</h2>
        <p>One chart is shown at a time for focused analysis.</p>
      </section>
      <section class="controls">
        <label for="plot-select">Select Experiment Track</label>
        <select id="plot-select">
          <option value="p1">1. JSON Validate vs Temperature</option>
          <option value="p2">2. Output Format Comparison</option>
          <option value="p3">3. Generation Mode Comparison</option>
          <option value="p4">4. Data Prep Variants</option>
          <option value="p5">5. Final Test Comparison</option>
          <option value="p6">6. Baseline vs Data Prep (Test)</option>
        </select>
      </section>

      <section id="p1" class="panel active"><div class="panel-head"><h3>JSON Validate vs Temperature</h3><p>Dual-line trend (yes vs no) with marker labels at each temperature.</p></div><div id="plot1" class="plot"></div></section>
      <section id="p2" class="panel"><div class="panel-head"><h3>Output Format Comparison</h3><p>Horizontal bar chart for JSON, XML, and plain outputs.</p></div><div id="plot2" class="plot"></div></section>
      <section id="p3" class="panel"><div class="panel-head"><h3>Generation Mode Comparison</h3><p>Donut chart comparing F1 share between free and constrained generation.</p></div><div id="plot3" class="plot"></div></section>
      <section id="p4" class="panel"><div class="panel-head"><h3>Data Prep Variants</h3><p>Line comparison across variants: F1, validity, and repair%.</p></div><div id="plot4" class="plot"></div></section>
      <section id="p5" class="panel"><div class="panel-head"><h3>Final Test Comparison</h3><p>Grouped bars for precision, recall, and F1.</p></div><div id="plot5" class="plot"></div></section>
      <section id="p6" class="panel"><div class="panel-head"><h3>Baseline vs Data Prep (Test)</h3><p>Slope-style line comparing baseline and improved data prep across metrics.</p></div><div id="plot6" class="plot"></div></section>
    </main>
  </div>

  <script>
    const DATA = __DASHBOARD_DATA__;

    const palette = ["#00eaff", "#ff2bd6", "#1aff9c", "#ff9d00", "#7b61ff", "#ff4d6d"];
    const themedLayout = {
      paper_bgcolor: "#0b1326",
      plot_bgcolor: "#101938",
      font: { color: "#e6f8ff" },
      legend: { font: { color: "#cde8ff" } },
      xaxis: {
        gridcolor: "#273f69",
        linecolor: "#35598e",
        tickfont: { color: "#cde8ff" },
        titlefont: { color: "#e6f8ff" },
      },
      yaxis: {
        gridcolor: "#273f69",
        linecolor: "#35598e",
        tickfont: { color: "#cde8ff" },
        titlefont: { color: "#e6f8ff" },
      },
    };

    function fnum(v, n = 4) {
      const x = Number(v);
      if (Number.isNaN(x)) return 0;
      return Number(x.toFixed(n));
    }

    function renderPlot1() {
      const rows = [...(DATA.json_validity || [])];
      rows.sort((a, b) => Number(a.temperature) - Number(b.temperature));
      const temps = [...new Set(rows.map(r => String(r.temperature)))];
      const byTemp = new Map(rows.map(r => [`${r.json_validate}|${r.temperature}`, r]));
      const yesVals = temps.map(t => fnum((byTemp.get(`yes|${t}`) || {}).f1 || 0));
      const noVals = temps.map(t => fnum((byTemp.get(`no|${t}`) || {}).f1 || 0));
      const allVals = [...yesVals, ...noVals].filter(v => v > 0);
      const minVal = allVals.length ? Math.min(...allVals) : 0.0;
      const maxVal = allVals.length ? Math.max(...allVals) : 1.0;
      const pad = 0.003;
      const yMin = Math.max(0, Number((minVal - pad).toFixed(4)));
      const yMax = Math.min(1, Number((maxVal + pad).toFixed(4)));
      const dtick = 0.002;

      const traces = [{
        type: "scatter",
        mode: "lines+markers+text",
        name: "validate=yes",
        x: temps.map(t => `t=${t}`),
        y: yesVals,
        line: { color: "#00eaff", width: 3 },
        marker: { color: "#00eaff", size: 10 },
        text: yesVals.map(v => v.toFixed(4)),
        textposition: "top center"
      }];
      if (noVals.some(v => v > 0)) {
        traces.push({
          type: "scatter",
          mode: "lines+markers+text",
          name: "validate=no",
          x: temps.map(t => `t=${t}`),
          y: noVals,
          line: { color: "#ff2bd6", width: 3, dash: "dot" },
          marker: { color: "#ff2bd6", size: 10 },
          text: noVals.map(v => v.toFixed(4)),
          textposition: "bottom center"
        });
      }

      Plotly.newPlot("plot1", traces, {
        ...themedLayout,
        margin: { t: 20, r: 20, b: 70, l: 85 },
        yaxis: {
          ...themedLayout.yaxis,
          title: "F1 Score",
          range: [yMin, yMax],
          dtick: dtick,
          tickformat: ".3f",
          gridcolor: "#273f69",
          zeroline: false
        },
        xaxis: {
          ...themedLayout.xaxis,
          title: "Temperature",
          type: "category",
          showline: true,
          linecolor: "#35598e"
        },
        legend: { orientation: "h", y: 1.12 }
      }, { responsive: true });
    }

    function renderPlot2() {
      const rows = DATA.format_cmp || [];
      const sorted = [...rows].sort((a, b) => Number(a.f1) - Number(b.f1));
      Plotly.newPlot("plot2", [{
        type: "bar",
        orientation: "h",
        x: sorted.map(r => fnum(r.f1)),
        y: sorted.map(r => String(r.format).toUpperCase()),
        marker: { color: sorted.map((_, i) => palette[i % palette.length]), line: { color: "#315b90", width: 1 } },
        text: sorted.map(r => fnum(r.f1).toFixed(4)),
        textposition: "outside",
        hovertemplate: "Format=%{y}<br>F1=%{x}<extra></extra>"
      }], {
        ...themedLayout,
        margin: { t: 20, r: 50, b: 50, l: 90 },
        xaxis: { ...themedLayout.xaxis, title: "F1", range: [0.0, 1.0], gridcolor: "#273f69" },
        yaxis: { ...themedLayout.yaxis, title: "Format" }
      }, { responsive: true });
    }

    function renderPlot3() {
      const rows = DATA.gen_mode || [];
      Plotly.newPlot("plot3", [{
        type: "pie",
        hole: 0.52,
        labels: rows.map(r => String(r.mode)),
        values: rows.map(r => fnum(r.f1)),
        marker: { colors: ["#00eaff", "#1aff9c", "#ff9d00"] },
        textinfo: "label+percent",
        hovertemplate: "Mode=%{label}<br>F1=%{value:.4f}<extra></extra>"
      }], {
        ...themedLayout,
        margin: { t: 20, r: 20, b: 30, l: 20 },
        paper_bgcolor: "#0b1326",
        plot_bgcolor: "#0b1326",
        font: { color: "#e6f8ff" },
        showlegend: true,
        legend: { orientation: "h", y: -0.08 }
      }, { responsive: true });
    }

    function renderPlot4() {
      const rows = [...(DATA.data_prep_val || [])];
      rows.sort((a, b) => Number(b.f1) - Number(a.f1));
      const variants = rows.map(r => String(r.variant));
      const f1Vals = rows.map(r => fnum(r.f1));
      const validityVals = rows.map(r => fnum(r.validity));
      const repairPctVals = rows.map(r => {
        const total = Number(r.total_examples || 0);
        const repaired = Number(r.repaired_json_count || 0);
        return total > 0 ? Number(((repaired / total) * 100).toFixed(3)) : 0;
      });

      const traces = [
        {
          type: "scatter",
          mode: "lines+markers+text",
          name: "F1",
          x: variants,
          y: f1Vals,
          line: { color: "#1aff9c", width: 3 },
          marker: { color: "#1aff9c", size: 9 },
          text: f1Vals.map(v => v.toFixed(4)),
          textposition: "top center",
          yaxis: "y"
        },
        {
          type: "scatter",
          mode: "lines+markers+text",
          name: "Validity",
          x: variants,
          y: validityVals,
          line: { color: "#00eaff", width: 3, dash: "dash" },
          marker: { color: "#00eaff", size: 9 },
          text: validityVals.map(v => v.toFixed(4)),
          textposition: "bottom center",
          yaxis: "y"
        },
        {
          type: "scatter",
          name: "Repair %",
          x: variants,
          y: repairPctVals,
          mode: "lines+markers+text",
          text: repairPctVals.map(v => `${v.toFixed(3)}%`),
          textposition: "top center",
          line: { color: "#ff4d6d", width: 3, dash: "dot" },
          marker: { size: 9 },
          yaxis: "y2"
        }
      ];

      Plotly.newPlot("plot4", traces, {
        ...themedLayout,
        margin: { t: 20, r: 65, b: 65, l: 70 },
        xaxis: { ...themedLayout.xaxis, title: "Data Prep Variant" },
        yaxis: { ...themedLayout.yaxis, title: "F1", range: [0.84, 0.96], gridcolor: "#273f69" },
        yaxis2: {
          title: "Repair %",
          overlaying: "y",
          side: "right",
          rangemode: "tozero",
          gridcolor: "#273f69",
          tickfont: { color: "#cde8ff" },
          titlefont: { color: "#e6f8ff" },
        },
        legend: { orientation: "h", y: 1.12 }
      }, { responsive: true });
    }

    function renderPlot5() {
      const rows = DATA.final_test || [];
      const x = rows.map(r => String(r.run_name).replace("final_test_", ""));
      const metrics = [
        { key: "precision", name: "Precision", color: "#00eaff" },
        { key: "recall", name: "Recall", color: "#1aff9c" },
        { key: "f1", name: "F1", color: "#ff9d00" }
      ];
      const traces = metrics.map(m => ({
        type: "bar",
        name: m.name,
        x,
        y: rows.map(r => fnum(r[m.key])),
        marker: { color: m.color },
        text: rows.map(r => fnum(r[m.key]).toFixed(4)),
        textposition: "outside"
      }));

      Plotly.newPlot("plot5", traces, {
        ...themedLayout,
        barmode: "group",
        margin: { t: 20, r: 20, b: 80, l: 70 },
        yaxis: { ...themedLayout.yaxis, title: "Score", range: [0.82, 0.95], gridcolor: "#273f69" },
        xaxis: { ...themedLayout.xaxis, title: "Configuration" },
        legend: { orientation: "h", y: 1.1 }
      }, { responsive: true });
    }

    function renderPlot6() {
      const rows = DATA.data_prep_test || [];
      const ordered = [...rows].sort((a, b) => String(a.source).localeCompare(String(b.source)));
      const metrics = ["precision", "recall", "f1"];
      const traces = ordered.map((r, idx) => ({
        type: "scatter",
        mode: "lines+markers+text",
        name: String(r.source),
        x: metrics,
        y: metrics.map(m => fnum(r[m])),
        text: metrics.map(m => fnum(r[m]).toFixed(4)),
        textposition: "top center",
        line: { width: 4, color: idx === 0 ? "#ff4d6d" : "#00eaff" },
        marker: { size: 10 }
      }));

      Plotly.newPlot("plot6", traces, {
        ...themedLayout,
        margin: { t: 20, r: 20, b: 60, l: 70 },
        yaxis: { ...themedLayout.yaxis, title: "Score", range: [0.82, 0.93], gridcolor: "#273f69" },
        xaxis: { ...themedLayout.xaxis, title: "Metric" },
        legend: { orientation: "h", y: 1.12 }
      }, { responsive: true });
    }

    const panels = ["p1", "p2", "p3", "p4", "p5", "p6"];
    const rendered = new Set();

    function ensureRendered(id) {
      if (rendered.has(id)) return;
      if (id === "p1") renderPlot1();
      if (id === "p2") renderPlot2();
      if (id === "p3") renderPlot3();
      if (id === "p4") renderPlot4();
      if (id === "p5") renderPlot5();
      if (id === "p6") renderPlot6();
      rendered.add(id);
    }

    function showPanel(id) {
      panels.forEach(pid => {
        const panel = document.getElementById(pid);
        if (panel) panel.classList.toggle("active", pid === id);
      });
      ensureRendered(id);
      window.dispatchEvent(new Event("resize"));
    }

    const selector = document.getElementById("plot-select");
    if (selector) {
      selector.addEventListener("change", (e) => {
        showPanel(e.target.value);
      });
    }

    showPanel("p1");
  </script>
</body>
</html>
"""

    return template.replace("__DASHBOARD_DATA__", payload)


def main() -> None:
    data = {
        "json_validity": normalize_rows(
            read_csv(ROOT / "experiments/qwen2_5_1_5B_masked_tuned/json_validity_f1_experiment_results.csv")
        ),
        "format_cmp": normalize_rows(
            read_csv(ROOT / "experiments/qwen2_5_1_5B_masked_tuned/fmt_format_comparison_temp_0p0_validate_yes.csv")
        ),
        "gen_mode": normalize_rows(
            read_csv(ROOT / "experiments/qwen2_5_1_5B_masked_tuned/gen_mode_comparison_temp_0p0_validate_yes_format_json.csv")
        ),
        "data_prep_val": normalize_rows(
            read_csv(ROOT / "experiments/data_prep_comparison/data_prep_comparison_temp_0p0_mode_constrained.csv")
        ),
        "final_test": normalize_rows(
            read_csv(ROOT / "experiments/qwen2_5_1_5B_masked_tuned/final_test_comparison.csv")
        ),
        "data_prep_test": normalize_rows(
            read_csv(ROOT / "experiments/data_prep_comparison/data_prep_test_compare.csv")
        ),
    }

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "index.html"
    out.write_text(build_html(data), encoding="utf-8")
    print(f"Interactive dashboard generated: {out}")


if __name__ == "__main__":
    main()
