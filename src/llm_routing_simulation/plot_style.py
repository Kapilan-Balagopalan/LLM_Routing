"""Central publication styling for the routing-study figures.

Edit the constants and label dictionaries in this module to restyle every
tuning figure, then regenerate a completed sweep with ``--plot-only``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PLOT_CONFIDENCE_LEVEL = 0.95
PUBLICATION_PNG_DPI = 400
PUBLICATION_FIGSIZE = (7.0, 4.25)
PUBLICATION_FIGSIZE_SHORT = (7.0, 3.8)
LEGEND_FONT_SIZE = 8.0

AXIS_LABELS = {
    "round": r"Round ($t$)",
    "cumulative_reference_regret": "Regret (excess cost)",
    "average_reference_regret": r"Average regret ($R_t/t$)",
    "routing_rate": "Strong-model routing rate",
    "accuracy": "Agreement with cached strong-model reference",
    "l01": r"$\ell_{01}$",
    "total_cost": "Total cost",
    "selected_multiplier": "Selected multiplier (log scale)",
    "squarecb_cost_difference": (
        "Linear SquareCB.PMSide cost - tree SquareCB.PMSide cost\n"
        "(positive favors tree)"
    ),
}

METHOD_LABELS = {
    "pgts": "PG-TS (Bayesian logistic)",
    "random": "Random",
}


def student_t_critical_value(
    sample_count: int,
    *,
    confidence_level: float = PLOT_CONFIDENCE_LEVEL,
) -> float:
    """Return the two-sided Student-t critical value for a sample mean."""
    if sample_count < 2:
        raise ValueError("Student-t intervals require at least two trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must be between zero and one")

    from scipy.stats import t as student_t

    return float(
        student_t.ppf(
            (1.0 + confidence_level) / 2.0,
            sample_count - 1,
        )
    )


def student_t_half_width(
    sample_standard_deviation: Any,
    sample_count: int,
    *,
    confidence_level: float = PLOT_CONFIDENCE_LEVEL,
) -> np.ndarray | float:
    """Return a two-sided Student-t interval half-width for a sample mean."""
    standard_deviation = np.asarray(sample_standard_deviation, dtype=np.float64)
    critical_value = student_t_critical_value(
        sample_count,
        confidence_level=confidence_level,
    )
    half_width = critical_value * standard_deviation / np.sqrt(sample_count)
    if half_width.ndim == 0:
        return float(half_width)
    return half_width


def publication_pyplot():
    """Return pyplot configured for compact conference-quality figures."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def save_publication_figure(figure: Any, png_path: Path) -> tuple[Path, Path]:
    """Save a high-resolution PNG and a vector PDF with the same stem."""
    if png_path.suffix.lower() != ".png":
        raise ValueError("Publication figure path must use a .png suffix")
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(
        png_path,
        dpi=PUBLICATION_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    return png_path, pdf_path
