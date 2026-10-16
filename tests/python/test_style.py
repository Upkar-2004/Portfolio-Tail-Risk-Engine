"""
Guards for the house style. Run with:
    pytest tests/python/test_style.py

These exist so that changing style.py (fonts, ticks, grids, margins) can never
silently break the heatmaps or make figures drift away from the house style.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tailrisk import style  # noqa: E402

MODES = ["paper", "report"]


def _visible_gridlines(ax, which="major"):
    lines = []
    for axis in (ax.xaxis, ax.yaxis):
        ticks = axis.get_major_ticks() if which == "major" else axis.get_minor_ticks()
        lines += [t.gridline for t in ticks if t.gridline.get_visible()]
    return lines


@pytest.fixture(autouse=True)
def _restore_matplotlib_style():
    original_parameters = matplotlib.rcParams.copy()
    original_mode = style.MODE
    yield
    plt.close("all")
    matplotlib.rcParams.update(original_parameters)
    style.MODE = original_mode


# --- global defaults -------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_grid_is_off_globally(mode):
    """A global grid draws over heatmap cells. It must stay off; value_grid() opts in."""
    style.apply(mode)
    assert matplotlib.rcParams["axes.grid"] is False


@pytest.mark.parametrize("mode", MODES)
def test_house_widths(mode):
    style.apply(mode)
    for full in (True, False):
        fig, _ = style.figure(1, 1, full_width=full)
        assert fig.get_figwidth() in style.figure_widths(mode)


def test_canonical_model_styles_match_abbreviations():
    """IDs and abbreviations resolve to the same visual identity."""

    for abbreviation, model_id in style.MODEL_IDS.items():
        assert style.model_style(abbreviation) == style.model_style(
            model_id
        )


def test_context_restores_mode_and_rc_parameters():
    """Local production styling cannot leak into later figures."""

    original_mode = style.MODE
    original_family = list(matplotlib.rcParams["font.family"])

    with style.context("report"):
        assert style.MODE == "report"
        assert matplotlib.rcParams["font.family"] == [
            "sans-serif"
        ]

    assert style.MODE == original_mode
    assert list(matplotlib.rcParams["font.family"]) == (
        original_family
    )


# --- heatmaps --------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_continuous_heatmap_is_clean(mode):
    """Exception-rate style heatmap: no major gridlines, no tick marks, no mirrored ticks."""
    style.apply(mode)
    fig, ax = style.figure(1, 1, height=3.0)
    data = np.random.default_rng(0).uniform(0, 0.1, size=(14, 3))
    im = ax.imshow(data, aspect="auto", cmap="cividis")
    ax.set_xticks(range(3), style.MODEL_ORDER)
    ax.set_yticks(range(14), [str(y) for y in range(2012, 2026)])
    style.heatmap_axes(ax, shape=(14, 3))
    fig.colorbar(im, ax=ax)
    fig.canvas.draw()

    assert _visible_gridlines(ax, "major") == []
    for t in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        assert not t.tick1line.get_visible() or t.tick1line.get_markersize() == 0
        assert not t.tick2line.get_visible() or t.tick2line.get_markersize() == 0
    # separators exist and are white
    minor = _visible_gridlines(ax, "minor")
    assert minor and all(matplotlib.colors.to_hex(g.get_color()) == "#ffffff" for g in minor)


@pytest.mark.parametrize("mode", MODES)
def test_line_chart_grid_opt_in(mode):
    style.apply(mode)
    fig, ax = style.figure(1, 1)
    style.model_line(ax, [0, 1, 2], [1, 2, 3], "EWMA")
    assert _visible_gridlines(ax) == []
    style.value_grid(ax, "y")
    fig.canvas.draw()
    assert _visible_gridlines(ax)


# --- mode-specific helpers -------------------------------------------------
def test_paper_mode_draws_no_title_or_source():
    style.apply("paper")
    fig, ax = style.figure(1, 1)
    n = len(fig.texts)
    style.title_block(fig, "Title", "Subtitle")
    style.source_line(fig)
    assert len(fig.texts) == n


def test_report_mode_draws_title_and_source():
    style.apply("report")
    fig, ax = style.figure(1, 1)
    style.title_block(fig, "Title", "Subtitle")
    style.source_line(fig)
    assert len(fig.texts) == 3


def test_save_rejects_wrong_width(tmp_path):
    style.apply("paper")
    fig = plt.figure(figsize=(6.0, 3.0))
    with pytest.raises(AssertionError):
        style.save(fig, "bad", tmp_path, formats=("png",))


