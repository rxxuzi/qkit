"""Utilities for saving matplotlib and plotly figures to ``out/charts/``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "out"
CHARTS_DIR = OUT_DIR / "charts"
REPORTS_DIR = OUT_DIR / "reports"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str, fmt: str = "auto", dpi: int = 150,
             timestamp: bool = False) -> Path:
    """Save a matplotlib or plotly figure to ``out/charts/``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure to save.
    name : str
        Base filename (without extension).
    fmt : str
        One of ``"png"``, ``"svg"``, ``"html"``, or ``"auto"``
        (matplotlib -> png, plotly -> html).
    dpi : int
        Resolution for matplotlib output.
    timestamp : bool
        Append a datetime suffix to *name*.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    if timestamp:
        name = f"{name}_{datetime.now():%Y%m%d_%H%M%S}"

    is_plotly = hasattr(fig, "to_html")
    if fmt == "auto":
        fmt = "html" if is_plotly else "png"

    path = CHARTS_DIR / f"{name}.{fmt}"

    if is_plotly:
        if fmt == "html":
            fig.write_html(str(path))
        else:
            fig.write_image(str(path), scale=2)
    else:
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return path
