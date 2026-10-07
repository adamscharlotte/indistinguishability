"""Figure styling shared by every plot: one `PlotStyle` carries figure size, fonts, colours
and output format, so a plot is restyled for a talk or a paper without touching its code.

    style = get_style("talk", cmap_mean="magma", figsize=(12, 8))
    fig, ax = plot_score_distribution(df, style=style, output_path="figures")

Presets are starting points; any field can be overridden by keyword.
"""

import os
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PlotStyle:
    #: Figure size in inches. ``None`` lets size-adaptive plots (heatmaps) pick their own.
    figsize: tuple = (6.5, 4.5)
    dpi: int = 300
    #: seaborn context: "paper", "notebook", "talk" or "poster".
    context: str = "paper"
    font_scale: float = 1.0
    font_family: str = "sans-serif"
    title_size: float = 12
    label_size: float = 11
    tick_size: float = 9
    annot_size: float = 7
    #: Colour for single-series plots (histograms, bars, scatter points).
    color: str = "#397edc"
    #: Secondary colour (reference / cutoff lines, the second of two series).
    accent: str = "#d62728"
    #: Palette for categorical series (violins, KDE per group, ...).
    palette: str = "viridis"
    #: Colormap for heatmaps / hexbins coloured by a mean score.
    cmap_mean: str = "viridis"
    #: Colormap for heatmaps / hexbins coloured by a count.
    cmap_count: str = "Blues"
    #: File format for saved figures: "png", "pdf", "svg", ...
    format: str = "png"
    transparent: bool = False
    despine: bool = True
    show_title: bool = True
    #: Extra matplotlib rcParams applied by `apply`.
    rc: dict = field(default_factory=dict)

    def apply(self):
        """Set the seaborn context and rcParams globally for subsequent figures."""
        sns.set_theme(
            context=self.context,
            style="ticks",
            font=self.font_family,
            font_scale=self.font_scale,
            rc={
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "legend.fontsize": self.tick_size,
                "savefig.dpi": self.dpi,
                **self.rc,
            },
        )
        return self

    def save(self, fig, output_path, name):
        """Write `fig` to ``output_path/name.<format>`` and return the path."""
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"{name}.{self.format}")
        fig.savefig(
            path, dpi=self.dpi, bbox_inches="tight", transparent=self.transparent
        )
        return path

    def updated(self, **overrides):
        """Copy with some fields replaced; ``None`` values are ignored, so CLI defaults pass
        straight through."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


PRESETS = {
    # Single-column journal figure: small, dense, vector output, no titles (captions do that).
    "paper": PlotStyle(
        figsize=(3.5, 2.6),
        dpi=600,
        context="paper",
        title_size=9,
        label_size=8,
        tick_size=7,
        annot_size=5,
        format="pdf",
        show_title=False,
    ),
    # Slide: large fonts readable from the back of the room.
    "talk": PlotStyle(
        figsize=(10, 6.5),
        dpi=200,
        context="talk",
        title_size=20,
        label_size=18,
        tick_size=15,
        annot_size=11,
        format="png",
        show_title=True,
    ),
    # Interactive exploration.
    "notebook": PlotStyle(
        figsize=(8, 5),
        dpi=150,
        context="notebook",
        title_size=13,
        label_size=12,
        tick_size=10,
        annot_size=8,
        format="png",
    ),
}


def get_style(preset="notebook", **overrides):
    """A `PlotStyle` from `preset` with `overrides` applied (``None`` values ignored)."""
    if isinstance(preset, PlotStyle):
        return preset.updated(**overrides)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return PRESETS[preset].updated(**overrides)


def new_axes(style, ax=None, figsize=None):
    """Return ``(fig, ax)``: `ax`'s own figure when given, else a fresh figure at
    `figsize` (falling back to ``style.figsize``)."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize or style.figsize)
    return fig, ax
