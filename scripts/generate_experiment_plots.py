"""Generate viewer-friendly SVG charts from experiment CSV files (no external deps)."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def save_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def header(title: str, subtitle: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{width/2}" y="42" text-anchor="middle" font-size="30" font-family="Arial" fill="#0f172a" font-weight="700">{title}</text>',
        f'<text x="{width/2}" y="68" text-anchor="middle" font-size="16" font-family="Arial" fill="#475569">{subtitle}</text>',
    ]


# ---------------------------------------------------------------------
# Chart types
# ---------------------------------------------------------------------
def line_chart_svg(title: str, labels: list[str], values: list[float], color: str = "#0ea5e9") -> str:
    width, height = 1200, 700
    ml, mr, mt, mb = 90, 40, 100, 130
    cw, ch = width - ml - mr, height - mt - mb
    ymax = 1.0

    svg = header(title, "F1 Score", width, height)

    for i in range(6):
        t = i / 5
        y = mt + ch - t * ch
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{ml-10}" y="{y+5:.1f}" text-anchor="end" font-size="12" fill="#334155">{(t*ymax):.3f}</text>')

    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{width-mr}" y2="{mt+ch}" stroke="#334155"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ch}" stroke="#334155"/>')

    n = max(2, len(values))
    step = cw / (n - 1)
    pts = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = ml + i * step
        y = mt + ch - (val / ymax) * ch
        pts.append((x, y))
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        svg.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="12" fill="#0f172a" font-weight="700">{val:.4f}</text>')
        svg.append(f'<text x="{x:.1f}" y="{mt+ch+24:.1f}" text-anchor="middle" font-size="12" fill="#334155">{lab}</text>')

    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>')

    svg.append("</svg>")
    return "\n".join(svg)


def hbar_chart_svg(title: str, labels: list[str], values: list[float], color: str = "#22c55e") -> str:
    width, height = 1200, 700
    ml, mr, mt, mb = 220, 40, 100, 60
    cw, ch = width - ml - mr, height - mt - mb
    ymax = 1.0

    svg = header(title, "F1 Score", width, height)
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#334155"/>')

    n = max(1, len(values))
    step = ch / n
    bar_h = max(22, step * 0.5)

    for i, (lab, val) in enumerate(zip(labels, values)):
        y = mt + i * step + (step - bar_h) / 2
        w = (val / ymax) * cw
        svg.append(f'<rect x="{ml}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="7" fill="{color}"/>')
        svg.append(f'<text x="{ml-10}" y="{y+bar_h*0.7:.1f}" text-anchor="end" font-size="14" fill="#0f172a">{lab}</text>')
        svg.append(f'<text x="{ml+w+8:.1f}" y="{y+bar_h*0.7:.1f}" font-size="12" fill="#0f172a" font-weight="700">{val:.4f}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def lollipop_chart_svg(title: str, labels: list[str], values: list[float], color: str = "#f59e0b") -> str:
    width, height = 1200, 700
    ml, mr, mt, mb = 100, 40, 100, 120
    cw, ch = width - ml - mr, height - mt - mb
    ymax = 1.0

    svg = header(title, "F1 Score", width, height)
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#334155"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ch}" stroke="#334155"/>')

    n = max(1, len(values))
    step = cw / (n + 1)
    for i, (lab, val) in enumerate(zip(labels, values), start=1):
        x = ml + i * step
        y = mt + ch - (val / ymax) * ch
        svg.append(f'<line x1="{x:.1f}" y1="{mt+ch:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#fcd34d" stroke-width="5"/>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}"/>')
        svg.append(f'<text x="{x:.1f}" y="{y-14:.1f}" text-anchor="middle" font-size="12" fill="#0f172a" font-weight="700">{val:.4f}</text>')
        svg.append(f'<text x="{x:.1f}" y="{mt+ch+26:.1f}" text-anchor="middle" font-size="13" fill="#334155">{lab}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def dot_plot_svg(title: str, labels: list[str], values: list[float], color: str = "#8b5cf6") -> str:
    width, height = 1200, 700
    ml, mr, mt, mb = 260, 40, 100, 70
    cw, ch = width - ml - mr, height - mt - mb
    ymax = 1.0

    svg = header(title, "F1 Score", width, height)

    n = max(1, len(values))
    step = ch / n
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = mt + i * step + step / 2
        x = ml + (val / ymax) * cw
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{ml-12}" y="{y+5:.1f}" text-anchor="end" font-size="14" fill="#0f172a">{lab}</text>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>')
        svg.append(f'<text x="{x+14:.1f}" y="{y+5:.1f}" font-size="12" fill="#0f172a" font-weight="700">{val:.4f}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def grouped_bar_svg(title: str, labels: list[str], f1: list[float], validity: list[float]) -> str:
    width, height = 1200, 700
    ml, mr, mt, mb = 100, 60, 100, 170
    cw, ch = width - ml - mr, height - mt - mb
    ymax = 1.0

    svg = header(title, "F1 and Validity on Test Split", width, height)
    svg.append(f'<rect x="{ml}" y="{mt}" width="{cw}" height="{ch}" fill="#ffffff" stroke="#e2e8f0" rx="10"/>')
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#334155" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ch}" stroke="#334155" stroke-width="1.5"/>')

    # Grid + y-axis labels
    for i in range(6):
        t = i / 5
        y = mt + ch - t * ch
        val = t * ymax
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{ml-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#334155">{val:.2f}</text>')

    n = max(1, len(labels))
    step = cw / n
    bw = max(20, step * 0.22)

    for i, lab in enumerate(labels):
        gx = ml + i * step + step / 2
        y1 = mt + ch - (f1[i] / ymax) * ch
        y2 = mt + ch - (validity[i] / ymax) * ch
        svg.append(
            f'<rect x="{gx-bw-5:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{(mt+ch-y1):.1f}" '
            f'rx="6" fill="#ef4444"/>'
        )
        svg.append(
            f'<rect x="{gx+5:.1f}" y="{y2:.1f}" width="{bw:.1f}" height="{(mt+ch-y2):.1f}" '
            f'rx="6" fill="#0ea5e9"/>'
        )

        short_lab = lab.replace("final_test_", "").replace("_", " ")
        svg.append(
            f'<text x="{gx:.1f}" y="{mt+ch+24:.1f}" text-anchor="middle" font-size="12" '
            f'fill="#334155">{short_lab}</text>'
        )
        svg.append(
            f'<text x="{gx:.1f}" y="{min(y1, y2)-10:.1f}" text-anchor="middle" font-size="12" '
            f'fill="#0f172a" font-weight="700">F1 {f1[i]:.4f}</text>'
        )

    # legend
    legend_x = width - 260
    legend_y = 98
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="16" height="16" fill="#ef4444" rx="3"/>')
    svg.append(f'<text x="{legend_x+24}" y="{legend_y+13}" font-size="13" fill="#334155">F1</text>')
    svg.append(f'<rect x="{legend_x+80}" y="{legend_y}" width="16" height="16" fill="#0ea5e9" rx="3"/>')
    svg.append(f'<text x="{legend_x+104}" y="{legend_y+13}" font-size="13" fill="#334155">Validity</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def slope_chart_svg(title: str, labels: list[str], values: list[float]) -> str:
    width, height = 1200, 700
    ml, mr, mt, mb = 180, 180, 100, 90
    cw, ch = width - ml - mr, height - mt - mb
    ymax = 1.0

    svg = header(title, "F1 Score", width, height)

    x_left = ml
    x_right = ml + cw

    for i in range(6):
        t = i / 5
        y = mt + ch - t * ch
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#e2e8f0"/>')

    points = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = x_left if i == 0 else x_right
        y = mt + ch - (val / ymax) * ch
        points.append((x, y, lab, val))

    if len(points) == 2:
        svg.append(f'<line x1="{points[0][0]}" y1="{points[0][1]:.1f}" x2="{points[1][0]}" y2="{points[1][1]:.1f}" stroke="#14b8a6" stroke-width="4"/>')

    for x, y, lab, val in points:
        svg.append(f'<circle cx="{x}" cy="{y:.1f}" r="10" fill="#0f766e"/>')
        anchor = "end" if x == x_left else "start"
        tx = x - 14 if x == x_left else x + 14
        svg.append(f'<text x="{tx}" y="{y-6:.1f}" text-anchor="{anchor}" font-size="13" fill="#0f172a" font-weight="700">{lab}</text>')
        svg.append(f'<text x="{tx}" y="{y+12:.1f}" text-anchor="{anchor}" font-size="12" fill="#334155">F1={val:.4f}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) JSON validate x temperature -> LINE CHART
    p1 = ROOT / "experiments/qwen2_5_1_5B_masked_tuned/json_validity_f1_experiment_results.csv"
    rows = read_csv(p1)
    if rows:
        yes_rows = [r for r in rows if r.get("json_validate") == "yes"]
        yes_rows.sort(key=lambda r: to_float(r.get("temperature", "0")))
        labels = [f"t={r['temperature']}" for r in yes_rows]
        vals = [to_float(r["f1"]) for r in yes_rows]
        svg = line_chart_svg("Validation Trend: Temperature (json_validate=yes)", labels, vals)
        save_svg(PLOTS_DIR / "01_json_validate_temperature_f1.svg", svg)

    # 2) Output format -> HORIZONTAL BAR
    p2 = ROOT / "experiments/qwen2_5_1_5B_masked_tuned/fmt_format_comparison_temp_0p0_validate_yes.csv"
    rows = read_csv(p2)
    if rows:
        labels = [r["format"] for r in rows]
        vals = [to_float(r["f1"]) for r in rows]
        svg = hbar_chart_svg("Validation: Output Format Comparison", labels, vals)
        save_svg(PLOTS_DIR / "02_format_comparison_f1.svg", svg)

    # 3) Generation mode -> LOLLIPOP
    p3 = ROOT / "experiments/qwen2_5_1_5B_masked_tuned/gen_mode_comparison_temp_0p0_validate_yes_format_json.csv"
    rows = read_csv(p3)
    if rows:
        labels = [r["mode"] for r in rows]
        vals = [to_float(r["f1"]) for r in rows]
        svg = lollipop_chart_svg("Validation: Free vs Constrained Generation", labels, vals)
        save_svg(PLOTS_DIR / "03_generation_mode_f1.svg", svg)

    # 4) Data prep variants -> DOT PLOT
    p4 = ROOT / "experiments/data_prep_comparison/data_prep_comparison_temp_0p0_mode_constrained.csv"
    rows = read_csv(p4)
    if rows:
        labels = [r["variant"] for r in rows]
        vals = [to_float(r["f1"]) for r in rows]
        svg = dot_plot_svg("Validation: Data Prep Variant Comparison", labels, vals)
        save_svg(PLOTS_DIR / "04_data_prep_variants_f1.svg", svg)

    # 5) Final test comparison -> GROUPED BARS (F1 + Validity)
    p5 = ROOT / "experiments/qwen2_5_1_5B_masked_tuned/final_test_comparison.csv"
    rows = read_csv(p5)
    if rows:
        labels = [r["run_name"] for r in rows]
        f1_vals = [to_float(r["f1"]) for r in rows]
        val_vals = [to_float(r["validity"]) for r in rows]
        svg = grouped_bar_svg("Test: Final Configuration Comparison", labels, f1_vals, val_vals)
        save_svg(PLOTS_DIR / "05_final_test_comparison_f1.svg", svg)

    # 6) Baseline vs data prep test -> SLOPE CHART
    p6 = ROOT / "experiments/data_prep_comparison/data_prep_test_compare.csv"
    rows = read_csv(p6)
    if rows:
        labels = [r["source"] for r in rows]
        vals = [to_float(r["f1"]) for r in rows]
        svg = slope_chart_svg("Test: Baseline vs Data Prep", labels, vals)
        save_svg(PLOTS_DIR / "06_baseline_vs_data_prep_test_f1.svg", svg)

    index_md = """# Plots\n\nMixed chart types generated for viewer clarity:\n\n1. `01_json_validate_temperature_f1.svg` (line chart)\n2. `02_format_comparison_f1.svg` (horizontal bar chart)\n3. `03_generation_mode_f1.svg` (lollipop chart)\n4. `04_data_prep_variants_f1.svg` (dot plot)\n5. `05_final_test_comparison_f1.svg` (grouped bars)\n6. `06_baseline_vs_data_prep_test_f1.svg` (slope chart)\n"""
    (PLOTS_DIR / "README.md").write_text(index_md)

    print(f"Plots generated in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
