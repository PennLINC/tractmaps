# ------------------------------------------------------------------------------------------------
# --- Figure formatting utilities ---
# ------------------------------------------------------------------------------------------------

# This script contains utilities for formatting figures to match speficied InDesign panel dimensions
# This also involves enforcing consistent font size and family across all text artists in the figure.
# Additionally, it involves fitting labels to the canvas if necessary and saving the figure at the exact canvas size.

# Example usage:
# fig, ax = create_panel_figure(width_mm=50, height_mm=18, font_pt=6)
# [...] create plot here on ax
# enforce_fonts(fig, font_pt=6, font_family="Arial")
# fit_labels_to_canvas(fig, ax)
# save_figure(fig, "panel.svg")
# plt.close(fig)

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

from typing import Tuple, Dict, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

DEFAULT_DPI = 300
DEFAULT_FONT_PT = 6
DEFAULT_FONT = "Arial"

# Keep text as text in vector files
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

def mm_to_in(mm: float) -> float:
    return mm / 25.4

def create_panel_figure(
    width_mm: float,
    height_mm: float,
    margins_mm: Optional[Dict[str, float]] = None,
    font_pt: float = DEFAULT_FONT_PT,
    font_family: str = DEFAULT_FONT,
    dpi: int = DEFAULT_DPI,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create an exact-size canvas with fixed margins (in mm).
    """
    if margins_mm is None:
        margins_mm = {"left": 8, "right": 2, "top": 4, "bottom": 8}

    W_in = mm_to_in(width_mm)
    H_in = mm_to_in(height_mm)

    fig, ax = plt.subplots(figsize=(W_in, H_in), dpi=dpi)

    # margins in figure fraction
    l = mm_to_in(margins_mm["left"]) / W_in
    r = 1 - (mm_to_in(margins_mm["right"]) / W_in)
    b = mm_to_in(margins_mm["bottom"]) / H_in
    t = 1 - (mm_to_in(margins_mm["top"]) / H_in)
    fig.subplots_adjust(left=l, right=r, bottom=b, top=t)

    # light, consistent cosmetics
    ax.tick_params(pad=2, width=0.5, length=2, colors="black")
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("black")
    ax.set_facecolor("white")
    return fig, ax

def enforce_fonts(fig: plt.Figure, font_pt: float = DEFAULT_FONT_PT, font_family: str = DEFAULT_FONT) -> None:
    """
    HARD-SET font size/family on *all* text artists in the figure, including:
    axes labels, tick labels, titles, legends, and colorbar axes.
    This makes the result immune to Seaborn contexts and rcParams drift, which
    can happen when multiple plots are generated in the same script. 
    """
    fp = FontProperties(size=font_pt, family=font_family)

    # Axes: labels, ticks, titles
    for ax in fig.axes:
        # axis labels + title
        ax.xaxis.label.set_fontproperties(fp)
        ax.yaxis.label.set_fontproperties(fp)
        ax.title.set_fontproperties(fp)

        # tick labels
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(fp)

        # axis tick_params as safety net
        ax.tick_params(labelsize=font_pt)

        # legends (axes-level)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontproperties(fp)
            leg.set_title(leg.get_title().get_text())
            if leg.get_title() is not None:
                leg.get_title().set_fontproperties(fp)

    # figure-level legends (rare but possible)
    for leg in fig.legends:
        for txt in leg.get_texts():
            txt.set_fontproperties(fp)
        if leg.get_title() is not None:
            leg.get_title().set_fontproperties(fp)

    # any stray Text artists (annotations, suptitle, etc.)
    for txt in fig.findobj(Text):
        # skip if it already has properties (this is fine to overwrite)
        txt.set_fontproperties(fp)

def fit_labels_to_canvas(fig: plt.Figure, ax: plt.Axes, pad_mm: float = 1.5, max_iter: int = 8) -> None:
    """
    Iteratively shrink/shift the axes so the labels fit inside the fixed canvas.
    Works for single-axes panels (heatmaps, colorbar panels, etc.).
    """
    pad_x = mm_to_in(pad_mm) / fig.get_figwidth()
    pad_y = mm_to_in(pad_mm) / fig.get_figheight()

    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        pos = ax.get_position()

        left_over   = max(0.0, (pad_x - tight.x0))
        bottom_over = max(0.0, (pad_y - tight.y0))
        right_over  = max(0.0, (tight.x1 + pad_x - 1.0))
        top_over    = max(0.0, (tight.y1 + pad_y - 1.0))

        if left_over == bottom_over == right_over == top_over == 0.0:
            break

        new_left   = pos.x0 + left_over
        new_bottom = pos.y0 + bottom_over
        new_width  = pos.width  - (left_over + right_over)
        new_height = pos.height - (top_over  + bottom_over)
        if new_width <= 0.05 or new_height <= 0.05:
            break
        ax.set_position([new_left, new_bottom, new_width, new_height])

def save_figure(fig: plt.Figure, filename: str, dpi: int = DEFAULT_DPI) -> None:
    """Save at the exact canvas size (no autoshrink/expand)."""
    W_in, H_in = fig.get_figwidth(), fig.get_figheight()
    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches=Bbox.from_bounds(0, 0, W_in, H_in),
        pad_inches=0.0,
        facecolor="white",
        edgecolor="none",
    )