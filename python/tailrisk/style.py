"""
House style for the Portfolio Tail Risk Engine figures.

Two genres, one switch:

    style.apply()                 # "paper": journal look (default)
    style.apply(mode="report")    # "report": research-note look for slides / LinkedIn / README

Paper mode:  serif (Latin Modern), closed axes box, inward ticks, panel letters inside
             the plot, legend above the panels, NO title / subtitle / source inside the
             image (the LaTeX or Markdown caption carries them). Width 7.0 in or 3.4 in.
Report mode: sans (Inter), open axes, left-aligned finding title + subtitle, direct
             line labels, source line. Width 7.5 in or 3.6 in.

The same figure script works in both modes. Helpers that only make sense in one mode
(title_block, source_line, direct labels) quietly do nothing in the other.

Typical script:

    from tailrisk import style
    style.apply()                                   # or mode="report"
    fig, axes = style.figure(1, 2, height=2.4, sharey=True)

    # line / dot charts
    style.model_line(ax, x, y, "EWMA")
    style.value_grid(ax, "y")

    # heatmaps and rank grids
    im = ax.imshow(data, ...)
    style.heatmap_axes(ax)

    # charts with a categorical axis (years down the side)
    style.categorical_axis(ax, "y")

    style.panel_labels(axes, ["Quantile loss", "FZ0 loss"])
    style.model_legend(fig, axes)                    # paper: legend row above panels
    style.title_block(fig, title, subtitle)          # report only
    style.source_line(fig)                           # report only
    style.save(fig, "fig_name", outdir)

Figure scripts never set colors, fonts or sizes directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import patheffects as pe
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------------------
# Identity (identical in both modes, never changed per figure)
# ---------------------------------------------------------------------------
MODEL_ORDER = ["RG", "EWMA", "FHS"]
MODEL_IDS = {
    "RG": "rolling_gaussian_monte_carlo",
    "EWMA": "ewma_gaussian_monte_carlo",
    "FHS": "filtered_historical_simulation",
}
MODEL_ABBREVIATIONS = {
    model_id: abbreviation
    for abbreviation, model_id in MODEL_IDS.items()
}
MODEL_NAMES = {
    "RG": "Rolling Gaussian",
    "EWMA": "EWMA Gaussian",
    "FHS": "Filtered Historical Simulation",
}
# Okabe-Ito subset; validated for colorblind separation (worst adjacent dE 18.0)
MODEL_COLORS = {"RG": "#D55E00", "EWMA": "#0072B2", "FHS": "#009E73"}
MODEL_MARKERS = {"RG": "o", "EWMA": "s", "FHS": "^"}
MODEL_STYLES_BY_ID = {
    model_id: {
        "abbreviation": abbreviation,
        "label": MODEL_NAMES[abbreviation],
        "color": MODEL_COLORS[abbreviation],
        "marker": MODEL_MARKERS[abbreviation],
    }
    for abbreviation, model_id in MODEL_IDS.items()
}

CONF_LEVELS = [0.95, 0.975, 0.99]
CONF_LABELS = {0.95: "95%", 0.975: "97.5%", 0.99: "99%"}
NOMINAL = {0.95: 0.05, 0.975: 0.025, 0.99: 0.01}

STRESS_LABEL = "Feb–Apr 2020 (n = 62)"
STRESS_SPAN = ("2020-02-01", "2020-04-30")

# Neutrals
INK = "#000000"
INK_2 = "#333333"
INK_3 = "#555555"      # reference lines, secondary text
INK_4 = "#6b6b6b"      # lightest text allowed (rank 2/3 numbers, report subtitles)
RULE = "#e6e6e6"       # gridlines
BAND = "#eeeeee"       # binomial reference bands
EVENT = "#f0f0f0"      # stress-window shading
PAPER = "#ffffff"

EXCEPTION_RATE_PALETTE = {
    "negative": "#2166AC",
    "neutral": "#F7F7F7",
    "positive": "#B2182B",
    "missing": "#D9D9D9",
    "text": "#17212B",
}

PREDICTIVE_RANK_PALETTE = {
    "rank_1": "#244A64",
    "rank_2": PAPER,
    "rank_3": PAPER,
    "summary": PAPER,
    "stress_background": PAPER,
    "text": "#17212B",
    "secondary_text": "#5B6670",
    "tertiary_text": "#8A949C",
    "muted_text": "#5F6B76",
    "grid": "#D9E0E5",
    "divider": "#87939D",
}

PAPER_FONT_STACK = [
    "STIX Two Text",
    "STIXGeneral",
    "DejaVu Serif",
]
REPORT_FONT_STACK = [
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]

SOURCE_TEXT = "Source: Yahoo Finance adjusted closes; author's calculations."

# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------
_MODES = {
    "paper": dict(
        widths=(7.0, 3.4),
        font_family="serif",
        fonts=PAPER_FONT_STACK,
        mathtext="stix",
        size=8, size_tick=7.5, size_legend=7.5, size_panel=8,
        line_width=1.2, marker_size=4,
        axes_lw=0.6, closed_box=True, tick_dir="in",
    ),
    "report": dict(
        widths=(7.5, 3.6),
        font_family="sans-serif",
        fonts=REPORT_FONT_STACK,
        mathtext="dejavusans",
        size=8, size_tick=7.5, size_legend=7.5, size_panel=8.5,
        line_width=1.6, marker_size=5,
        axes_lw=0.6, closed_box=False, tick_dir="out",
    ),
}
MODE = "paper"


def _cfg():
    return _MODES[MODE]


def rc_params(mode: str = "paper") -> dict[str, object]:
    """Return house-style Matplotlib parameters without mutation."""

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)}")
    c = _MODES[mode]
    return {
        "font.family": c["font_family"],
        f"font.{c['font_family']}": c["fonts"],
        "mathtext.fontset": c["mathtext"],
        "font.size": c["size"],
        "text.color": INK,
        "axes.labelsize": c["size"],
        "axes.labelcolor": INK,
        "axes.titlesize": c["size_panel"],
        "axes.titlelocation": "left",
        "axes.edgecolor": INK if c["closed_box"] else INK_2,
        "axes.linewidth": c["axes_lw"],
        "axes.spines.top": c["closed_box"],
        "axes.spines.right": c["closed_box"],
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.facecolor": PAPER,
        "grid.color": RULE,
        "grid.linewidth": 0.4,
        "xtick.labelsize": c["size_tick"],
        "ytick.labelsize": c["size_tick"],
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": c["tick_dir"],
        "ytick.direction": c["tick_dir"],
        # mirrored ticks on top/right only in paper mode; heatmap_axes() removes them
        "xtick.top": c["closed_box"],
        "ytick.right": c["closed_box"],
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": c["line_width"],
        "lines.markersize": c["marker_size"],
        "legend.fontsize": c["size_legend"],
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 1.8,
        "image.interpolation": "nearest",
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": 300,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    }


def apply(mode: str = "paper") -> None:
    """Apply one genre globally for a standalone figure script.

    Production figure builders should prefer :func:`context` so style
    state cannot leak between figures or tests. Gridlines remain off
    globally; individual axes opt in through :func:`value_grid`.
    """

    global MODE
    MODE = mode
    mpl.rcdefaults()
    mpl.rcParams.update(rc_params(mode))


@contextmanager
def context(mode: str = "paper") -> Iterator[None]:
    """Apply a genre temporarily and restore prior style state."""

    global MODE
    previous_mode = MODE
    MODE = mode

    try:
        with mpl.rc_context(rc_params(mode)):
            yield
    finally:
        MODE = previous_mode


def model_style(model: str) -> dict[str, str]:
    """Return one model's style from an ID or abbreviation."""

    model_id = MODEL_IDS.get(model, model)

    try:
        return dict(MODEL_STYLES_BY_ID[model_id])
    except KeyError as error:
        raise ValueError(f"Unknown tail-risk model: {model}.") from error


def figure_widths(mode: str = "paper") -> tuple[float, float]:
    """Return the full- and single-column widths for one genre."""

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)}")

    return tuple(_MODES[mode]["widths"])


# ---------------------------------------------------------------------------
# Figure scaffold
# ---------------------------------------------------------------------------
def figure(nrows=1, ncols=1, height=2.6, full_width=True, **subplot_kw):
    """Figure at a house width for the current mode. Margins are set for the mode:
    paper leaves room for a legend row on top; report leaves room for a title block."""
    width = _cfg()["widths"][0 if full_width else 1]
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    axes = fig.subplots(
        nrows=nrows,
        ncols=ncols,
        **subplot_kw,
    )
    h = height * 72
    if MODE == "paper":
        top = 1 - 22 / h          # one legend row
        bottom = 30 / h           # tick labels + x label
    else:
        top = 1 - 62 / h          # title + one subtitle line + panel titles
        bottom = 44 / h           # x label + source line
    fig.subplots_adjust(left=0.08, right=0.985, top=top, bottom=bottom,
                        wspace=0.06 if subplot_kw.get("sharey") else 0.22, hspace=0.35)
    return fig, axes


def title_block(fig, title: str, subtitle: str | None = None) -> None:
    """Report mode only. In paper mode the caption carries the title, so this does nothing."""
    if MODE != "report":
        return
    h = fig.get_figheight() * 72
    fig.text(0.012, 1 - 8 / h, title, ha="left", va="top", fontsize=10.5,
             fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 1 - 25 / h, subtitle, ha="left", va="top", fontsize=8.5,
                 color=INK_4, linespacing=1.4)


def source_line(fig, text: str = SOURCE_TEXT) -> None:
    """Report mode only. In paper mode the source goes in the caption or data section."""
    if MODE != "report":
        return
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=7, color=INK_4)


def panel_labels(axes, names=None, start: str = "a") -> None:
    """Panel letter plus optional short name.

    paper:  "(a)  Quantile loss" inside the top-left corner, letter in bold.
    report: "(a)  Quantile loss" as a left-aligned bold axes title.

    For heatmaps in paper mode, pass inside=False through panel_label_above()
    so the label never sits on top of cells.
    """
    for i, ax in enumerate(_flat(axes)):
        letter = chr(ord(start) + i)
        name = names[i] if names else ""
        if MODE == "paper":
            ax.text(0.025, 0.955, rf"$\bf{{({letter})}}$  {name}".rstrip(),
                    transform=ax.transAxes, ha="left", va="top", fontsize=_cfg()["size_panel"],
                    zorder=6, bbox=dict(fc=PAPER, ec="none", pad=1.0))
        else:
            ax.set_title(f"({letter})  {name}".rstrip(), loc="left", fontweight="bold", pad=6)


def panel_label_above(ax, letter: str, name: str = "") -> None:
    """Label above the axes (for heatmaps, where an inside label would cover cells)."""
    if MODE == "paper":
        ax.set_title(rf"$\bf{{({letter})}}$  {name}".rstrip(), loc="left",
                     fontsize=_cfg()["size_panel"], pad=4)
    else:
        ax.set_title(f"({letter})  {name}".rstrip(), loc="left", fontweight="bold", pad=6)


# ---------------------------------------------------------------------------
# Axes treatments: one per chart family. These keep the global style from
# breaking heatmaps and categorical charts.
# ---------------------------------------------------------------------------
def value_grid(ax, axis: str = "y") -> None:
    """Light gridlines on the value axis of line, dot and interval charts."""
    ax.grid(True, axis=axis, color=RULE, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    if MODE == "report" and axis == "y":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)


def heatmap_axes(ax, shape: tuple[int, int] | None = None, separators: bool = True) -> None:
    """For imshow / pcolormesh / rank grids. Removes everything that draws on cells
    (gridlines, mirrored ticks, tick marks) and keeps a thin frame in paper mode.

    shape=(nrows, ncols) adds thin white separators between cells. Call it AFTER
    imshow and after setting the category tick labels.
    """
    import numpy as np
    ax.grid(False, which="both")
    ax.tick_params(which="both", length=0, top=False, right=False,
                   labeltop=False, labelright=False, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(MODE == "paper")
        sp.set_linewidth(0.6)
    if shape and separators:
        nr, nc = shape
        ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
        ax.grid(True, which="minor", color=PAPER, lw=1.0)
        ax.grid(False, which="major")
        ax.tick_params(which="minor", length=0)


def categorical_axis(ax, axis: str = "y") -> None:
    """Category labels (years, models) on one axis: no tick marks or mirrored ticks there."""
    if axis == "y":
        ax.tick_params(axis="y", which="both", length=0, right=False, labelright=False)
    else:
        ax.tick_params(axis="x", which="both", length=0, top=False, labeltop=False)


def percent_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
def model_line(ax, x, y, model: str, dy_pts: float = 0, **kw):
    """Line + markers in house identity. Report mode adds a direct label at the end;
    paper mode relies on model_legend()."""
    (ln,) = ax.plot(x, y, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                    markeredgecolor=PAPER, markeredgewidth=0.6, label=model,
                    zorder=3, clip_on=False, **kw)
    if MODE == "report":
        ax.annotate(model, (list(x)[-1], list(y)[-1]), xytext=(7, dy_pts),
                    textcoords="offset points", va="center", ha="left", fontsize=8,
                    fontweight="bold", color=MODEL_COLORS[model], annotation_clip=False,
                    zorder=4, path_effects=[pe.withStroke(linewidth=3, foreground=PAPER)])
    return ln


def model_points(ax, x, y, model: str, xerr=None, yerr=None, **kw):
    """Dot with optional interval (exception-rate interval charts)."""
    return ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=MODEL_MARKERS[model],
                       color=MODEL_COLORS[model], ms=_cfg()["marker_size"],
                       mec=PAPER, mew=0.5, elinewidth=0.9, capsize=1.8, capthick=0.9,
                       label=model, zorder=3, **kw)


def reference_line(ax, value, orient: str = "h", label: str | None = None) -> None:
    """Thin dashed reference (nominal rate, zero, even split).
    The label goes to the legend in paper mode and is drawn on the line in report mode."""
    kw = dict(color=INK_3, lw=0.7, ls=(0, (4, 3)), zorder=2)
    line = ax.axhline(value, **kw) if orient == "h" else ax.axvline(value, **kw)
    if label and MODE == "paper":
        line.set_label(label)
    elif label:
        if orient == "h":
            ax.annotate(label, (0.99, value), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="right",
                        va="bottom", fontsize=7, color=INK_3)
        else:
            ax.annotate(label, (value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -2), textcoords="offset points", ha="left",
                        va="top", fontsize=7, color=INK_3)


def reference_band(ax, lo, hi, orient: str = "v", label: str | None = None) -> None:
    span = ax.axvspan if orient == "v" else ax.axhspan
    span(lo, hi, color=BAND, lw=0, zorder=0, label=label if MODE == "paper" else None)


def stress_shading(ax, start=STRESS_SPAN[0], end=STRESS_SPAN[1]) -> None:
    """Shade Feb–Apr 2020 on a date axis."""
    import pandas as pd
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=EVENT, lw=0, zorder=0)


def model_legend(fig, axes, extra_first: bool = False) -> None:
    """Paper mode: one legend row centered above all panels, models first in house order.
    Report mode: nothing (direct labels do the job)."""
    if MODE != "paper":
        return
    handles, labels = {}, []
    for ax in _flat(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles and not l.startswith("_"):
                handles[l] = h
    models = [m for m in MODEL_ORDER if m in handles]
    others = [l for l in handles if l not in MODEL_ORDER]
    order = others + models if extra_first else models + others
    fig.legend([handles[l] for l in order], order, loc="upper center",
               ncol=len(order), frameon=False, bbox_to_anchor=(0.53, 1.0))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def normalize_svg_whitespace(svg_path: str | Path) -> Path:
    """Remove generator-only trailing spaces from an SVG file."""

    path = Path(svg_path)
    svg_text = path.read_text(encoding="utf-8")
    normalized_text = "\n".join(
        line.rstrip() for line in svg_text.splitlines()
    )
    path.write_text(f"{normalized_text}\n", encoding="utf-8")
    return path


def save(fig, name: str, outdir: str | Path = ".", formats=("pdf", "svg", "png")) -> list[Path]:
    """Save at fixed size. Never bbox_inches='tight' (it changes the width figure to figure)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    w = fig.get_figwidth()
    assert any(abs(w - ok) < 1e-6 for ok in _cfg()["widths"]), \
        f"figure width {w} in is not a {MODE}-mode house width {_cfg()['widths']}"
    paths = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)

        if ext.lower() == "svg":
            normalize_svg_whitespace(p)

        paths.append(p)
    return paths


def _flat(axes):
    try:
        return list(axes.flat)
    except AttributeError:
        return list(axes) if isinstance(axes, (list, tuple)) else [axes]
