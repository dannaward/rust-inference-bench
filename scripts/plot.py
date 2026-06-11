#!/usr/bin/env python3
"""Render benchmark plots as dependency-free SVG (Python stdlib only).

Reads the latest results/*.json and writes results/plots/{latency,throughput,
speedup}.svg. SVG keeps the repo reproducible (no matplotlib) and renders inline
on GitHub.

Usage: python3 scripts/plot.py [results/<file>.json]
"""
import glob
import json
import sys

W, H = 680, 380
PAD_L, PAD_R, PAD_T, PAD_B = 70, 20, 40, 60
CANDLE_C = "#2c7fb8"
BURN_C = "#de8a26"


def load():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("results/*.json"))[-1]
    with open(path) as f:
        return json.load(f), path


def svg(body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="sans-serif" font-size="13">\n'
        f'<rect width="{W}" height="{H}" fill="white"/>\n'
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15" '
        f'font-weight="bold">{title}</text>\n{body}</svg>\n'
    )


def axes(ymax, ylabel, xticks):
    px0, px1 = PAD_L, W - PAD_R
    py0, py1 = H - PAD_B, PAD_T
    s = f'<line x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}" stroke="#333"/>\n'
    s += f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="#333"/>\n'
    for i in range(6):
        v = ymax * i / 5
        y = py0 - (py0 - py1) * i / 5
        s += f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#eee"/>\n'
        s += f'<text x="{px0-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{v:.0f}</text>\n'
    s += (
        f'<text x="18" y="{(py0+py1)/2:.0f}" text-anchor="middle" fill="#555" '
        f'transform="rotate(-90 18 {(py0+py1)/2:.0f})">{ylabel}</text>\n'
    )
    n = len(xticks)
    for i, t in enumerate(xticks):
        x = px0 + (px1 - px0) * (i + 0.5) / n
        s += f'<text x="{x:.1f}" y="{py0+20}" text-anchor="middle" fill="#555">{t}</text>\n'
    return s, (px0, px1, py0, py1)


def legend(x, y):
    return (
        f'<rect x="{x}" y="{y}" width="12" height="12" fill="{CANDLE_C}"/>'
        f'<text x="{x+16}" y="{y+11}">candle-cpu</text>'
        f'<rect x="{x+110}" y="{y}" width="12" height="12" fill="{BURN_C}"/>'
        f'<text x="{x+126}" y="{y+11}">burn-ndarray</text>\n'
    )


def grouped_bars(records, key_label, val, err, ymax, ylabel, title):
    xticks = [r[key_label] for r in records]
    body, (px0, px1, py0, py1) = axes(ymax, ylabel, xticks)
    n = len(records)
    slot = (px1 - px0) / n
    bw = slot * 0.32
    for i, r in enumerate(records):
        cx = px0 + slot * (i + 0.5)
        for j, (agg, color) in enumerate([(r[val[0]], CANDLE_C), (r[val[1]], BURN_C)]):
            x = cx + (j - 1) * bw - bw / 2 + bw / 2
            x = cx - bw + j * bw
            h = (py0 - py1) * agg["median"] / ymax
            body += f'<rect x="{x:.1f}" y="{py0-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}"/>\n'
            # IQR whisker
            elo = (py0 - py1) * agg["p25"] / ymax
            ehi = (py0 - py1) * agg["p75"] / ymax
            xm = x + bw / 2
            body += f'<line x1="{xm:.1f}" y1="{py0-elo:.1f}" x2="{xm:.1f}" y2="{py0-ehi:.1f}" stroke="#333"/>\n'
    body += legend(px1 - 230, PAD_T - 4)
    return svg(body, title)


def line_chart(records, xkey, val, ymax, ylabel, title):
    xticks = [f'b={r[xkey]}' for r in records]
    body, (px0, px1, py0, py1) = axes(ymax, ylabel, xticks)
    n = len(records)
    for series, color in [(val[0], CANDLE_C), (val[1], BURN_C)]:
        pts = []
        for i, r in enumerate(records):
            x = px0 + (px1 - px0) * (i + 0.5) / n
            y = py0 - (py0 - py1) * r[series]["median"] / ymax
            pts.append(f"{x:.1f},{y:.1f}")
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>\n'
        body += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>\n'
    body += legend(px1 - 230, PAD_T - 4)
    return svg(body, title)


def speedup_bars(latency, throughput, title):
    items = [(r["seq_label"], r["candle_speedup_x"]["median"], r["distinguishable"]) for r in latency]
    items += [(f'thr b={r["batch"]}', r["candle_speedup_x"]["median"], r["distinguishable"]) for r in throughput]
    ymax = max(2.0, max(v for _, v, _ in items) * 1.1)
    xticks = [k for k, _, _ in items]
    body, (px0, px1, py0, py1) = axes(ymax, "Candle speedup (x)", xticks)
    # reference line at 1.0
    y1 = py0 - (py0 - py1) * 1.0 / ymax
    body += f'<line x1="{px0}" y1="{y1:.1f}" x2="{px1}" y2="{y1:.1f}" stroke="#c00" stroke-dasharray="4"/>\n'
    body += f'<text x="{px1-2}" y="{y1-4:.1f}" text-anchor="end" fill="#c00">1.0 (tie)</text>\n'
    n = len(items)
    slot = (px1 - px0) / n
    bw = slot * 0.5
    for i, (_, v, dist) in enumerate(items):
        cx = px0 + slot * (i + 0.5)
        h = (py0 - py1) * v / ymax
        color = CANDLE_C if v >= 1 else BURN_C
        op = "1" if dist else "0.4"
        body += f'<rect x="{cx-bw/2:.1f}" y="{py0-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}" opacity="{op}"/>\n'
    return svg(body, title)


def main():
    data, path = load()
    lat = data["latency"]
    thr = data["throughput"]
    cpu = data["environment"]["cpu"]
    sub = f"{cpu}, 1 thread, AC — {data['config']['trials']} trials (median, IQR whiskers)"

    lat_ymax = max(r["burn_ms"]["p75"] for r in lat) * 1.15
    thr_ymax = max(max(r["candle_sps"]["median"], r["burn_sps"]["median"]) for r in thr) * 1.2

    import os
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results/plots"
    os.makedirs(outdir, exist_ok=True)
    open(f"{outdir}/latency.svg", "w").write(
        grouped_bars(lat, "seq_label", ("candle_ms", "burn_ms"), None, lat_ymax,
                     "p50 latency (ms)", f"Latency (lower is better) — {sub}")
    )
    open(f"{outdir}/throughput.svg", "w").write(
        line_chart(thr, "batch", ("candle_sps", "burn_sps"), thr_ymax,
                   "sentences / sec", f"Throughput (higher is better) — {sub}")
    )
    open(f"{outdir}/speedup.svg", "w").write(
        speedup_bars(lat, thr, f"Candle speedup vs Burn (faded = tie) — {sub}")
    )
    print(f"plotted from {path} -> {outdir}/{{latency,throughput,speedup}}.svg")


if __name__ == "__main__":
    main()
