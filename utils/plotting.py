"""
Shared plotting utilities for signal-not-noise.
Consistent style across all notebooks and the Streamlit app.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Repo-wide style ──────────────────────────────────────────────────────────

COLOURS = {
    "signal": "#2563EB",      # Blue — signal, important, primary
    "noise": "#EF4444",       # Red — noise, curse, danger
    "neutral": "#6B7280",     # Grey — axes, annotations
    "accent": "#F59E0B",      # Amber — highlights, callouts
    "success": "#10B981",     # Green — good outcomes, silver lining
    "bg": "#FAFAFA",          # Off-white background
    "grid": "#E5E7EB",        # Light grid
}

PALETTE = [
    COLOURS["signal"],
    COLOURS["noise"],
    COLOURS["accent"],
    COLOURS["success"],
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#84CC16",  # Lime
]


def apply_style():
    """Apply consistent matplotlib style across all notebooks."""
    mpl.rcParams.update({
        "figure.facecolor": COLOURS["bg"],
        "axes.facecolor": COLOURS["bg"],
        "axes.edgecolor": COLOURS["neutral"],
        "axes.labelcolor": COLOURS["neutral"],
        "axes.grid": True,
        "grid.color": COLOURS["grid"],
        "grid.alpha": 0.5,
        "text.color": "#1F2937",
        "xtick.color": COLOURS["neutral"],
        "ytick.color": COLOURS["neutral"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def plot_distance_distributions(dims_to_test, n_points=200, seed=42):
    """
    Show how distance distributions collapse as dimensions increase.
    Core visualisation for 01b (curse of dimensionality).
    """
    apply_style()
    rng = np.random.default_rng(seed)

    n_plots = len(dims_to_test)
    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes)

    for idx, d in enumerate(dims_to_test):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        points = rng.uniform(0, 1, size=(n_points, d))
        dists = np.linalg.norm(points[1:] - points[0], axis=1)

        # Normalise for visual comparison
        dists_norm = dists / np.sqrt(d)

        ax.hist(dists_norm, bins=30, alpha=0.7, color=COLOURS["noise"],
                edgecolor="white", linewidth=0.5)
        ax.set_title(f"d = {d}")

        contrast = (dists.max() - dists.min()) / (dists.min() + 1e-10)
        ax.text(0.95, 0.9, f"contrast: {contrast:.1f}",
                transform=ax.transAxes, fontsize=9, ha="right",
                color=COLOURS["neutral"])

    # Hide unused axes
    for idx in range(len(dims_to_test), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Distance Distributions Collapse in High Dimensions",
                 fontweight="bold")
    plt.tight_layout()
    return fig


def plot_explained_variance(explained_ratio, title="Explained Variance"):
    """Plot cumulative explained variance with elbow annotation."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    n = len(explained_ratio)
    cumulative = np.cumsum(explained_ratio)

    # Individual
    ax1.bar(range(1, n + 1), explained_ratio, color=COLOURS["signal"], alpha=0.7)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("Per Component")

    # Cumulative
    ax2.plot(range(1, n + 1), cumulative, "o-",
             color=COLOURS["signal"], markersize=5)
    ax2.axhline(y=0.95, color=COLOURS["accent"], linestyle="--",
                alpha=0.7, label="95% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title("Cumulative")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_2d_comparison(datasets, titles, figsize=None):
    """
    Side-by-side 2D scatter plots for comparing reduction methods.
    datasets: list of (X, y) tuples where X is (n, 2) and y is labels.
    """
    apply_style()
    n = len(datasets)
    if figsize is None:
        figsize = (5 * n, 4.5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (X, y), title in zip(axes, datasets, titles):
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10",
                             s=15, alpha=0.7, edgecolors="white",
                             linewidth=0.3)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig
