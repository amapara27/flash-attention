"""Charts for the write-up, from results/summary.csv -> charts/<name>_<light|dark>.{png,svg}

  progression  best ncu time per stage (1 head and 4 heads)
  stalls       warp latency per stage, split into stall reasons (they sum to the total)
  tiles        tile-sweep heatmaps per stage, each relative to its own best tile

usage: python make_charts.py        (needs matplotlib + numpy)
Palettes are the dataviz reference palette; categorical and ramp choices were run through its validator.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).parent
OUT = ROOT / "charts"
DPI = 200
PX = DPI / 100            # 1 "CSS px" in output pixels

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", other="#b9b8b1",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
                  ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", other="#5f5e5a",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
                 ramp=["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]),
}

STAGES = [   # (summary.csv stage, label)
    ("01_tiled", "1 · Tiled (non-causal)"),
    ("02_causal_mask", "2 · Causal mask"),
    ("03_tile_skipping", "3 · Tile skipping"),
    ("04_multihead", "4 · Multi-head"),
    ("05_s_registers", "5 · S in registers"),
    ("06_warp_partitioning", "6 · Warp partitioning"),
    ("07_fp16_smem", "7 · fp16 storage"),
    ("08_double_buffering", "8 · Double buffering"),
    ("09b_wmma_qk_fixed", "9 · Tensor cores (WMMA)"),
]
FOOT = "RTX 3060 Ti · N = 4096, d = 64 · Nsight Compute duration, clocks locked to base · stage 9 = WMMA with the race fixed"

plt.rcParams.update({"font.family": "Segoe UI", "svg.fonttype": "path", "axes.unicode_minus": False})


# ---------------------------------------------------------------- data
def load():
    with open(ROOT / "results" / "summary.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "PASS"]


def best(rows, stage, wl):
    c = [r for r in rows if r["stage"] == stage and r["workload"] == wl]
    return min(c, key=lambda r: float(r["ncu_ms"])) if c else None


# ---------------------------------------------------------------- drawing helpers
def figure(t, w, h):
    fig = plt.figure(figsize=(w, h), dpi=DPI, facecolor=t["surface"])
    return fig


def header(fig, t, title, subtitle, y=0.955):
    fig.text(0.04, y, title, fontsize=17, fontweight="semibold", color=t["ink"], va="top")
    fig.text(0.04, y - 0.062, subtitle, fontsize=10.5, color=t["ink2"], va="top")


def footer(fig, t, text=FOOT):
    fig.text(0.04, 0.025, text, fontsize=8.5, color=t["muted"], va="bottom")


def style_axis(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9, length=0, pad=6)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def px_per_unit(ax):
    o = ax.transData.transform((0, 0))
    return ax.transData.transform((1, 0))[0] - o[0], abs(ax.transData.transform((0, 1))[1] - o[1])


def hbar(ax, y, x0, x1, h, color, round_end=True, gap_after=False):
    """horizontal bar: 4px rounded data-end, square at the baseline.
    gap_after leaves a 2px surface gap before the next stacked segment (inset, not a stroke)."""
    pxx, pxy = px_per_unit(ax)
    if gap_after:
        x1 -= 2 * PX / pxx
    if x1 <= x0:
        return
    rx, ry = 4 * PX / pxx, 4 * PX / pxy                # 4px radius, in each axis's data units
    y0, y1 = y - h / 2, y + h / 2
    if round_end and (x1 - x0) > rx:
        # one path, rounded only at the data end, so there's no seam at the baseline
        verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1),
                 (x0, y1), (x0, y0)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3,
                 MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, linewidth=0))
    else:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=color, linewidth=0))


def text_width(fig, txt):
    """rendered width of a Text, in figure fraction"""
    return txt.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width


def legend_row(fig, t, items, x, y, gap=0.018, size=9.5):
    """swatch + label pairs in a row (text stays in ink, the swatch carries identity)"""
    r = fig.canvas.get_renderer()
    for color, label in items:
        fig.patches.append(FancyBboxPatch((x, y - 0.009), 0.011, 0.018, boxstyle="round,pad=0,rounding_size=0.002",
                                          transform=fig.transFigure, facecolor=color, linewidth=0))
        txt = fig.text(x + 0.016, y, label, fontsize=size, color=t["ink2"], va="center")
        x += 0.016 + txt.get_window_extent(r).width / fig.bbox.width + gap
    return x


def save(fig, name, theme):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}_{theme}.{ext}", facecolor=fig.get_facecolor(), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 1. progression
def progression(rows, theme):
    t = THEMES[theme]
    fig = figure(t, 13, 6.6)
    a_end, b_end = best(rows, "09b_wmma_qk_fixed", "A"), best(rows, "09b_wmma_qk_fixed", "B")
    a0, b0 = best(rows, "02_causal_mask", "A"), best(rows, "04_multihead", "B")
    header(fig, t, "Best time per optimisation stage",
           f"Causal attention: {float(a0['ncu_ms']):.2f} → {float(a_end['ncu_ms']):.2f} ms on 1 head "
           f"({float(a0['ncu_ms']) / float(a_end['ncu_ms']):.1f}×), {float(b0['ncu_ms']):.2f} → "
           f"{float(b_end['ncu_ms']):.2f} ms on 4 heads ({float(b0['ncu_ms']) / float(b_end['ncu_ms']):.1f}×). "
           "Best tile per stage; not every stage helped.")
    legend_row(fig, t, [(t["series"][0], "Faster than the previous stage"),
                        (t["series"][1], "Slower than the previous stage"),
                        (t["other"], "Non-causal reference")], 0.04, 0.80)

    n = len(STAGES)
    panels = [("1 head", "A", 0.225, 0.34, 1.25, 2), ("4 heads", "B", 0.635, 0.32, 1.25, 5)]
    for title, wl, left, width, pad, step in panels:
        ax = fig.add_axes([left, 0.12, width, 0.6])
        style_axis(ax, t)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
        vals = []
        for i, (stage, _) in enumerate(STAGES):
            wl_i = "A0" if (wl == "A" and stage == "01_tiled") else wl
            b = best(rows, stage, wl_i)
            vals.append(float(b["ncu_ms"]) if b else None)
        xmax = max(v for v in vals if v) * pad
        ax.set_xlim(0, xmax)
        ax.set_ylim(n - 0.4, -0.6)
        ax.set_yticks([])
        ax.set_xlabel("ms", color=t["muted"], fontsize=9, loc="right", labelpad=2)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="semibold", color=t["ink"], pad=10)
        fig.canvas.draw()
        h = min(0.62, 24 * PX / px_per_unit(ax)[1])
        prev = None
        pxx = px_per_unit(ax)[0]
        for i, v in enumerate(vals):
            if v is None:
                if i == 1:
                    ax.text(0, i, "single-head stages only", fontsize=9, color=t["muted"], va="center")
                continue
            stage = STAGES[i][0]
            if stage == "01_tiled":
                color = t["other"]
            else:
                color = t["series"][1] if (prev is not None and v > prev) else t["series"][0]
            hbar(ax, i, 0, v, h, color)
            x = v + 6 * PX / pxx
            ms = ax.text(x, i, f"{v:.2f} ms", fontsize=9.5, color=t["ink"], va="center")
            if prev is not None and stage != "01_tiled":
                x += text_width(fig, ms) * fig.bbox.width / pxx + 6 * PX / pxx
                ax.text(x, i, f"{prev / v:.2f}×", fontsize=9.5, color=t["ink2"], va="center")
            prev = v
    # shared stage labels on the left
    ax0 = fig.axes[0]
    for i, (_, label) in enumerate(STAGES):
        y = ax0.transData.transform((0, i))[1] / fig.bbox.height
        fig.text(0.04, y, label, fontsize=10, color=t["ink"], va="center")
    footer(fig, t, FOOT + " · × = speedup over the previous stage")
    save(fig, "progression", theme)


# ---------------------------------------------------------------- 2. stall breakdown
STALLS = [   # (column, legend label) in categorical slot order; everything else folds into "Other"
    ("stall_short_scoreboard", "Waiting on shared memory"),
    ("stall_long_scoreboard", "Waiting on global / local memory"),
    ("stall_mio_throttle", "Shared-memory pipe full (MIO)"),
    ("stall_barrier", "Barrier (__syncthreads)"),
    ("stall_wait", "Fixed-latency dependency"),
    ("stall_not_selected", "Ready, another warp issued"),
    ("stall_selected", "Issuing"),
]
SHORT_NAMES = ["shared memory", "global / local memory", "MIO throttle", "barrier", "fixed latency",
               "not selected", "issuing"]


def stalls(rows, theme):
    t = THEMES[theme]
    fig = figure(t, 13, 6.6)
    header(fig, t, "Where each warp's cycles go",
           "Average cycles a warp spends per instruction, split by stall reason (the reasons sum to the total). "
           "4 heads, causal, best tile per stage.")
    items = [(t["series"][i], lab) for i, (_, lab) in enumerate(STALLS)] + [(t["other"], "Other")]
    legend_row(fig, t, items[:4], 0.04, 0.805)
    legend_row(fig, t, items[4:], 0.04, 0.762)

    stages = STAGES[3:]
    ax = fig.add_axes([0.255, 0.12, 0.68, 0.565])
    style_axis(ax, t)
    data = [best(rows, s, "B") for s, _ in stages]
    totals = [float(r["warp_latency_cycles"]) for r in data]
    xmax = max(totals) * 1.12
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(stages) - 0.4, -0.6)
    ax.set_yticks([])
    ax.set_xlabel("cycles per issued instruction", color=t["muted"], fontsize=9, loc="right", labelpad=2)
    fig.canvas.draw()
    h = min(0.62, 24 * PX / px_per_unit(ax)[1])
    for i, r in enumerate(data):
        segs = [float(r[c]) for c, _ in STALLS]
        segs.append(max(0.0, sum(float(r[k]) for k in r if k.startswith("stall_")) - sum(segs)))
        colors = t["series"][:len(STALLS)] + [t["other"]]
        x = 0
        last = max(j for j, w in enumerate(segs) if w > 0)
        pxx = px_per_unit(ax)[0]
        for j, (w, c) in enumerate(zip(segs, colors)):
            if w > 0:
                hbar(ax, i, x, x + w, h, c, round_end=(j == last), gap_after=(j != last))
                # selective direct label: only the dominant segment of a bar, and only if it fits with padding
                if w == max(segs) and w >= 3 and j < len(SHORT_NAMES):   # only the bottlenecks worth calling out
                    rgb = matplotlib.colors.to_rgb(c)
                    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                    for s in (f"{w:.1f} cycles · {SHORT_NAMES[j]}", f"{w:.1f}"):
                        txt = ax.text(x + 8 * PX / pxx, i, s, fontsize=9, va="center",
                                      color="#0b0b0b" if lum > 0.45 else "#ffffff")
                        if text_width(fig, txt) * fig.bbox.width + 16 * PX <= w * pxx:
                            break
                        txt.remove()
            x += w
        ax.text(x + xmax * 0.01, i, f"{totals[i]:.1f}", fontsize=10, color=t["ink"], va="center", fontweight="semibold")
        lab = f"{stages[i][1]}"
        y = ax.transData.transform((0, i))[1] / fig.bbox.height
        fig.text(0.04, y + 0.012, lab, fontsize=10, color=t["ink"], va="center")
        fig.text(0.04, y - 0.022, f"{r['br']}×{r['bc']} · {float(r['ncu_ms']):.2f} ms", fontsize=8.5,
                 color=t["muted"], va="center")
    footer(fig, t)
    save(fig, "stalls", theme)


# ---------------------------------------------------------------- 3. tile heatmaps
BINS = [(0.95, "≥ 95% of best"), (0.80, "80–95%"), (0.60, "60–80%"), (0.40, "40–60%"), (0.0, "< 40%")]


def tiles(rows, theme):
    t = THEMES[theme]
    fig = figure(t, 13, 8.4)
    header(fig, t, "The best tile shrinks as the kernel improves",
           "Speed of every B_r × B_c tile relative to that stage's best (outlined). 4 heads, causal; cells show ncu ms.",
           y=0.962)
    ramp = t["ramp"]   # low -> high speed
    legend_row(fig, t, [(ramp[4 - k], lab) for k, (_, lab) in enumerate(BINS)] + [(t["grid"], "not run (> 48 KB smem)")],
               0.04, 0.845)

    stages = STAGES[3:]
    brs, bcs = [16, 32, 64, 128], [16, 32, 64]
    cols, w, hgt = 3, 0.27, 0.265
    for k, (stage, label) in enumerate(stages):
        cx, cy = 0.075 + (k % cols) * 0.31, 0.485 - (k // cols) * 0.365
        ax = fig.add_axes([cx, cy, w, hgt])
        ax.set_facecolor(t["surface"])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_aspect("auto")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(range(3), [f"{b}" for b in bcs])
        ax.set_yticks(range(4), [f"{b}" for b in brs])
        ax.tick_params(length=0, colors=t["muted"], labelsize=9, pad=4)
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold", color=t["ink"], pad=8)
        if k % cols == 0:
            ax.set_ylabel("B_r", color=t["muted"], fontsize=9, rotation=0, labelpad=14, va="center")
        if k // cols == 1:
            ax.set_xlabel("B_c", color=t["muted"], fontsize=9, labelpad=4)
        cell = {(int(r["br"]), int(r["bc"])): float(r["ncu_ms"]) for r in rows
                if r["stage"] == stage and r["workload"] == "B"}
        b = min(cell.values())
        fig.canvas.draw()
        pxx, pxy = px_per_unit(ax)
        g = 1.5 * PX           # half the 3px visual gap between cells
        for yi, br in enumerate(brs):
            for xi, bc in enumerate(bcs):
                x0, y0 = xi - 0.5 + g / pxx, yi - 0.5 + g / pxy
                cw, ch = 1 - 2 * g / pxx, 1 - 2 * g / pxy
                ms = cell.get((br, bc))
                style = dict(boxstyle=f"round,pad=0,rounding_size={4 * PX / pxx}", mutation_aspect=pxx / pxy)
                if ms is None:
                    ax.add_patch(FancyBboxPatch((x0, y0), cw, ch, facecolor=t["grid"], linewidth=0, **style))
                    ax.text(xi, yi, "—", ha="center", va="center", fontsize=9, color=t["muted"])
                    continue
                rel = b / ms
                cls = next(i for i, (lo, _) in enumerate(BINS) if rel >= lo)
                fill = ramp[4 - cls]
                is_best = ms == b
                ax.add_patch(FancyBboxPatch((x0, y0), cw, ch, facecolor=fill, linewidth=0, **style))
                if is_best:   # ink ring with a surface-colored inner ring, visible on any fill
                    for inset, color, lw in ((1.0, t["ink"], 2.0), (2.8, t["surface"], 1.5)):
                        dx, dy = inset * PX / pxx, inset * PX / pxy
                        ax.add_patch(FancyBboxPatch((x0 + dx, y0 + dy), cw - 2 * dx, ch - 2 * dy, facecolor="none",
                                                    edgecolor=color, linewidth=lw * 72 / 100, **style))
                rgb = matplotlib.colors.to_rgb(fill)
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(xi, yi, f"{ms:.2f}", ha="center", va="center", fontsize=9.5,
                        fontweight="semibold" if is_best else "normal",
                        color="#0b0b0b" if lum > 0.45 else "#ffffff")
    footer(fig, t)
    save(fig, "tiles", theme)


if __name__ == "__main__":
    rows = load()
    for theme in THEMES:
        progression(rows, theme)
        stalls(rows, theme)
        tiles(rows, theme)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
