# ------------------------------------------------------------------------------------------------
# --- Figure formatting utilities ---
# ------------------------------------------------------------------------------------------------

# This script contains utilities for formatting figures at an exact physical size (mm) 
# with consistent styling.

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Tuple, Union
import matplotlib as mpl
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH

def _apply_rc(
    *,
    base_pt=7, label_pt=7, title_pt=7,
    font_family="sans-serif",
    sans_list: Union[list[str], str] = "Helvetica",
    axes_linewidth=0.8, line_width=1,
):
    # Convert sans_list to list if it's a string
    font_list = [sans_list] if isinstance(sans_list, str) else list(sans_list)
    
    mpl.rcParams.update({
        # Fonts / sizes
        "font.family": font_family,
        "font.sans-serif": font_list,
        "axes.titlesize": title_pt,
        "axes.labelsize": label_pt,
        "xtick.labelsize": base_pt,
        "ytick.labelsize": base_pt,
        "legend.fontsize": base_pt,

        # Strokes
        "axes.linewidth": axes_linewidth,
        "lines.linewidth": line_width,

        # Keep live text in PDF/SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # NO automatic layout that can push/resize things
        "figure.constrained_layout.use": False,
        "savefig.bbox": "standard",   # never 'tight'
        "savefig.pad_inches": 0.0,
    })

def setup_figure(
    width_mm: float,
    height_mm: float,
    *,
    base_pt: int = 7,
    label_pt: int = 7,
    title_pt: int = 7,
    font_family: str = "sans-serif",
    sans_list: Union[list[str], str] = "Helvetica",
    axes_linewidth: float = 0.8,
    line_width: float = 1,
    margins_mm: Tuple[float, float, float, float] = (18.0, 2.0, 18.0, 8.0),  # L, R, B, T
    nrows: int = 1,
    ncols: int = 1,
    sharex=False,
    sharey=False,
    axes_aspect: str | float = "auto",  # "auto", "equal", or a numeric aspect
    strict_overflow_check: bool = True,
):
    """
    Create a figure with exact physical size (no tight cropping), fixed margins,
    and optional strict check that content fits inside the page.

    This function creates a matplotlib figure with precise physical dimensions
    in millimeters. Unlike figures saved with `bbox_inches='tight'`, this ensures
    the saved figure matches the specified dimensions exactly. The function also
    applies consistent styling (fonts, line widths) and can create subplot grids.

    Parameters
    ----------
    width_mm : float
        Figure width in millimeters.
    height_mm : float
        Figure height in millimeters.
    base_pt : int, default=7
        Font size in points for tick labels, legend text, and other base elements.
    label_pt : int, default=7
        Font size in points for axis labels (xlabel, ylabel).
    title_pt : int, default=7
        Font size in points for axis titles and figure titles.
    font_family : str, default="sans-serif"
        Main font family. Common values: "sans-serif", "serif", "monospace".
    sans_list : list[str] or str, default="Helvetica"
        Preferred sans-serif font(s). Can be a single font name (str) or a list of fonts.
        If a list, matplotlib uses the first available font from the list.
        Used when font_family="sans-serif".
    axes_linewidth : float, default=0.8
        Line width in points for axes (spines, ticks, grid lines).
    line_width : float, default=1
        Default line width in points for plot lines.
    margins_mm : tuple[float, float, float, float], default=(18.0, 2.0, 18.0, 8.0)
        Margins in millimeters as (left, right, bottom, top). These define the
        space between the figure edges and the plot area.
    nrows : int, default=1
        Number of rows in the subplot grid.
    ncols : int, default=1
        Number of columns in the subplot grid.
    sharex : bool, default=False
        If True, subplots share x-axis. See matplotlib's subplots() for details.
    sharey : bool, default=False
        If True, subplots share y-axis. See matplotlib's subplots() for details.
    axes_aspect : str or float, default="auto"
        Aspect ratio control for axes:
        - "auto" (recommended): Prevents aspect from forcing layout expansion.
          Use this unless you specifically need a fixed aspect ratio.
        - "equal": Forces equal aspect ratio (x and y scales match).
        - float: Numeric aspect ratio (e.g., 1.0 for square, 2.0 for 2:1).
    strict_overflow_check : bool, default=True
        If True, checks whether content (labels, titles, etc.) exceeds the
        specified margins and prints a warning if it does. This helps catch
        cases where content might be cut off.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : matplotlib.axes.Axes or array of Axes
        Single axes object if nrows=ncols=1, or array of axes for subplots.

    Examples
    --------
    >>> # Single plot with default margins
    >>> fig, ax = setup_figure(width_mm=100, height_mm=80)
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> save_figure(fig, 'plot.svg')

    >>> # Subplot grid with custom margins
    >>> fig, axes = setup_figure(
    ...     width_mm=150, height_mm=100,
    ...     margins_mm=(10, 10, 10, 10),  # left, right, bottom, top
    ...     nrows=2, ncols=2
    ... )

    >>> # Figure with equal aspect ratio
    >>> fig, ax = setup_figure(
    ...     width_mm=80, height_mm=80,
    ...     axes_aspect="equal"
    ... )

    >>> # Figure with single font (string)
    >>> fig, ax = setup_figure(width_mm=100, height_mm=80, sans_list="Arial")

    >>> # Figure with font list
    >>> fig, ax = setup_figure(
    ...     width_mm=100, height_mm=80,
    ...     sans_list=["Helvetica", "Arial"]
    ... )

    Notes
    -----
    - The figure size is fixed and will not be cropped or resized during saving.
    - Use `strict_overflow_check=True` to detect when content might exceed margins.
    - For best results, use `axes_aspect="auto"` unless you specifically need
      a fixed aspect ratio, as fixed aspects can cause layout issues with margins.
    """
    _apply_rc(
        base_pt=base_pt, label_pt=label_pt, title_pt=title_pt,
        font_family=font_family, sans_list=sans_list,
        axes_linewidth=axes_linewidth, line_width=line_width,
    )

    fig = plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)))
    # Convert mm margins to fractions
    L, R, B, T = margins_mm
    left   = L / width_mm
    right  = 1.0 - (R / width_mm)
    bottom = B / height_mm
    top    = 1.0 - (T / height_mm)

    # Build the grid inside the fixed page
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, left=left, right=right, bottom=bottom, top=top)
    axes = gs.subplots(sharex=sharex, sharey=sharey)

    # Ensure aspect won’t fight your margins
    def _set_aspect(ax):
        if axes_aspect == "auto":
            ax.set_aspect("auto", adjustable="box")
        else:
            ax.set_aspect(axes_aspect, adjustable="box")

    if isinstance(axes, (list, tuple)):
        for ax in axes:
            _set_aspect(ax)
    else:
        try:
            # ndarray of axes
            for ax in axes.ravel():
                _set_aspect(ax)
        except Exception:
            _set_aspect(axes)

    # Optional: check that the “tight” bbox would still fit within the page
    if strict_overflow_check:
        fig.canvas.draw()  # ensure artists have sizes
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        page  = fig.bbox
        # allow a tiny epsilon
        eps = 0.5
        if (tight.x0 < page.x0 - eps or tight.y0 < page.y0 - eps or
            tight.x1 > page.x1 + eps or tight.y1 > page.y1 + eps):
            print("[figformat] WARNING: content exceeds margins/page. "
                  "Increase margins_mm or reduce text/object sizes; "
                  "avoid legends/colorbars outside axes.")

    return fig, axes


def save_figure(fig: mpl.figure.Figure, path: str) -> None:
    # No bbox_inches='tight' here; page size remains exactly as requested
    fig.savefig(path)