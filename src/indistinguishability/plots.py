"""Plots of an enriched scores DataFrame (the output of `load_scores`).

Every function follows one convention::

    fig, ax = plot_x(df, style=None, ax=None, output_path=None, name=None, **options)

* `style` — a `PlotStyle` (or preset name); sizes, fonts, colours and format come from it.
* `ax` — draw into an existing axes (subplots, notebooks); a figure is created otherwise.
* `output_path` — directory to save into; nothing is written when it is ``None``.
* `name` — file stem; each plot has a sensible default.

Figures are returned, not closed, so they can be edited further. Close them yourself
(``plt.close(fig)``) when generating many in a loop.
"""

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from .rearrangements import add_swap_positions
from .sequence import add_aa_position
from .style import PlotStyle, get_style, new_axes
from .substitutions import (
    aa_grid_order,
    aa_tick_labels,
    annotate_substitutions,
    annotate_terminal_swap,
)

SCORE_LABEL = "Spectral similarity score"
LEV_LABEL = "Levenshtein distance"


# --------------------------------------------------------------------------- helpers


def _style(style):
    if style is None:
        return get_style("notebook")
    if isinstance(style, PlotStyle):
        return style
    return get_style(style)


def _require(df, *columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"this plot needs column(s) {missing}, which the loaded data does not have"
        )


def _finish(fig, ax, style, output_path, name, title=None, despine=True):
    """Apply title/despine and save. Returns ``(fig, ax)``."""
    if title and style.show_title:
        ax.set_title(title, fontsize=style.title_size)
    if despine and style.despine:
        sns.despine(ax=ax)
    if output_path is not None:
        print(f"saved {style.save(fig, output_path, name)}")
    return fig, ax


def _heatmap_figsize(style, shape, figsize):
    """Heatmaps grow with their grid unless a figsize is forced."""
    if figsize is not None:
        return figsize
    n_rows, n_cols = shape
    w, h = style.figsize
    return (max(w, n_cols * 0.6 + 2), max(h, n_rows * 0.4 + 1.5))


def _cap(series, max_value):
    """Clip an integer series at `max_value`, labelling the top bin "≥max". Returns the
    string series and its numeric sort order."""
    s = series.astype(int)
    if max_value is not None:
        s = s.clip(upper=max_value)
    labels = s.astype(str)
    if max_value is not None:
        labels = labels.where(s < max_value, f"≥{max_value}")
    order = sorted(labels.unique(), key=lambda x: int(str(x).lstrip("≥")))
    return labels, order


def _compact_int(x):
    """1234 -> "1.2k", 2_000_000 -> "2M"; blank for empty cells."""
    if pd.isna(x):
        return ""
    x = int(x)
    for cutoff, suffix in ((1_000_000, "M"), (1_000, "k")):
        if abs(x) >= cutoff:
            scaled = x / cutoff
            return f"{scaled:.0f}{suffix}" if scaled >= 10 else f"{scaled:.1f}{suffix}"
    return f"{x}"


def _position_labels(index, aa):
    return [f"No {aa}" if v == -1 else v for v in index]


def _mean_count_heatmap(
    df,
    index,
    columns,
    style,
    ax,
    figsize,
    vmin,
    vmax,
    annot,
    xlabel,
    ylabel,
    cbar_label,
    value_col="score",
):
    """Heatmap coloured by mean `value_col`, each cell annotated with its pair count."""
    mean = df.pivot_table(values=value_col, index=index, columns=columns, aggfunc="mean")
    count = df.pivot_table(
        values=value_col, index=index, columns=columns, aggfunc="count"
    )
    fig, ax = new_axes(style, ax, _heatmap_figsize(style, mean.shape, figsize))
    sns.heatmap(
        mean,
        mask=mean.isna(),
        annot=count.map(_compact_int) if annot else False,
        fmt="",
        annot_kws={"fontsize": style.annot_size},
        cmap=style.cmap_mean,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    ax.figure.axes[-1].yaxis.label.set_size(style.label_size)
    ax.set_xlabel(xlabel, fontsize=style.label_size)
    ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    ax.tick_params(axis="y", rotation=0)
    return fig, ax, mean


# --------------------------------------------------------------------------- distributions


def plot_score_distribution(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="score_distribution",
    score_col="score",
    xlim=(0, 1),
    bins=20,
    kde=True,
    hue=None,
    log=False,
    threshold=None,
    title=None,
    figsize=None,
):
    """Histogram of `score_col` on **fixed** bin edges (`xlim` pins the axis, the bin
    edges and the KDE clip, so separate figures stay comparable). `hue` splits by a column
    (e.g. ``"charge"``); `threshold` draws a vertical cutoff line."""
    s = _style(style)
    _require(df, score_col, *([hue] if hue else []))
    fig, ax = new_axes(s, ax, figsize)
    sns.histplot(
        data=df,
        x=score_col,
        hue=hue,
        bins=bins,
        binrange=xlim,
        kde=kde,
        kde_kws={"clip": xlim} if (kde and xlim) else None,
        color=None if hue else s.color,
        palette=s.palette if hue else None,
        element="step" if hue else "bars",
        log_scale=(False, log),
        ax=ax,
    )
    if xlim:
        ax.set_xlim(*xlim)
    if threshold is not None:
        ax.axvline(threshold, color=s.accent, linestyle="--", linewidth=1)
    ax.set_xlabel(SCORE_LABEL if score_col == "score" else score_col)
    ax.set_ylabel("Number of pairs")
    return _finish(fig, ax, s, output_path, name, title or "Score distribution")


def plot_levenshtein_distribution(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    lev_col="peptide_levenshtein_O",
    max_distance=None,
    annot=True,
    title=None,
    figsize=None,
):
    """Bar chart of the number of pairs per Levenshtein distance; distances above
    `max_distance` are pooled into one "≥max" bar."""
    s = _style(style)
    _require(df, lev_col)
    labels, order = _cap(df[lev_col], max_distance)
    counts = labels.value_counts().reindex(order)
    fig, ax = new_axes(s, ax, figsize)
    pos = np.arange(len(order))
    ax.bar(pos, counts.values, color=s.color, edgecolor="white", linewidth=0.5)
    if annot:
        for i, n in enumerate(counts.values):
            ax.text(
                i,
                n,
                _compact_int(n),
                ha="center",
                va="bottom",
                fontsize=s.annot_size,
                color="grey",
            )
    ax.set_xticks(pos)
    ax.set_xticklabels(order)
    ax.set_xlabel(LEV_LABEL)
    ax.set_ylabel("Number of pairs")
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name or f"{lev_col}_distribution",
        title or "Pairs per Levenshtein distance",
    )


def plot_irt_diff_histogram(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="irt_diff_histogram",
    col="irt_diff",
    bins=500,
    xlim=(0, 50),
    cutoff=5,
    log=False,
    title=None,
    figsize=None,
):
    """Histogram of the absolute iRT difference within a pair (or any other diff column
    via `col`), with a vertical `cutoff` line (``None`` to hide). Prints the fraction of
    pairs below the cutoff."""
    s = _style(style)
    _require(df, col)
    fig, ax = new_axes(s, ax, figsize)
    values = df[col].dropna()
    ax.hist(values, bins=bins, color=s.color, log=log)
    if cutoff is not None:
        ax.axvline(cutoff, color=s.accent, linestyle="--", linewidth=1)
        frac = (values <= cutoff).mean() if len(values) else float("nan")
        print(f"{frac:.1%} of {len(values)} pairs have {col} <= {cutoff}")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel({"irt_diff": "iRT difference", "relative_ccs_diff": "Relative CCS difference"}.get(col, col))
    ax.set_ylabel("Number of pairs")
    return _finish(fig, ax, s, output_path, name, title or f"{col} distribution")


def plot_peptide_length(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="peptide_length",
    input_peptides=None,
    peptide_cols=("peptide_1_O", "peptide_2_O"),
    label_input="Input peptides",
    label_pairs="Indistinguishable peptides",
    title=None,
    figsize=None,
):
    """Normalised length distribution of the unique peptides appearing in `df`, against
    `input_peptides` (any iterable of sequences) when given."""
    s = _style(style)
    _require(df, *peptide_cols)
    pair_peptides = set().union(*(set(df[c]) for c in peptide_cols))
    sets = [(label_pairs, [len(p) for p in pair_peptides], s.color)]
    if input_peptides is not None:
        sets.insert(0, (label_input, [len(str(p)) for p in input_peptides], s.accent))
    all_lengths = [n for _, lengths, _ in sets for n in lengths]
    bins = np.arange(min(all_lengths), max(all_lengths) + 2)
    fig, ax = new_axes(s, ax, figsize)
    for label, lengths, color in sets:
        ax.hist(
            lengths,
            bins=bins,
            label=label,
            histtype="step",
            linewidth=2,
            density=True,
            color=color,
        )
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Fraction of peptides")
    ax.legend(frameon=False)
    return _finish(fig, ax, s, output_path, name, title or "Peptide length distribution")


# --------------------------------------------------------------------------- score vs feature


def plot_levenshtein_violin(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    lev_col="peptide_levenshtein_O",
    score_col="score",
    max_distance=None,
    show_n=True,
    title=None,
    figsize=None,
):
    """Score distribution per Levenshtein distance: violins with a thin median/IQR box and
    the number of pairs under each violin."""
    s = _style(style)
    _require(df, lev_col, score_col)
    plot_df = df[[score_col]].copy()
    plot_df[lev_col], order = _cap(df[lev_col], max_distance)
    fig, ax = new_axes(s, ax, figsize)
    sns.violinplot(
        data=plot_df,
        x=lev_col,
        y=score_col,
        order=order,
        cut=0,
        density_norm="width",
        linewidth=0.6,
        hue=lev_col,
        palette=s.palette,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x=lev_col,
        y=score_col,
        order=order,
        width=0.1,
        showcaps=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.6),
        whiskerprops=dict(linewidth=0),
        medianprops=dict(color="black", linewidth=1),
        flierprops=dict(marker=""),
        ax=ax,
    )
    if show_n:
        counts = plot_df.groupby(lev_col).size()
        y0 = ax.get_ylim()[0]
        for i, val in enumerate(order):
            ax.text(
                i,
                y0,
                f"n={_compact_int(counts.get(val, 0))}",
                ha="center",
                va="bottom",
                fontsize=s.annot_size,
                color="grey",
            )
    ax.set_xlabel(LEV_LABEL)
    ax.set_ylabel(SCORE_LABEL if score_col == "score" else score_col)
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name or f"{lev_col}_vs_{score_col}",
        title or "Score by Levenshtein distance",
    )


def plot_feature_hexbin(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    x="irt_diff",
    y="mz_diff",
    color_by="score",
    gridsize=60,
    extent=None,
    title=None,
    figsize=None,
):
    """Hexbin over two numeric columns. ``color_by="score"`` colours each hexagon by the
    mean score; ``color_by="count"`` by the number of pairs (log scale)."""
    s = _style(style)
    _require(df, x, y, *([color_by] if color_by != "count" else []))
    fig, ax = new_axes(s, ax, figsize)
    if color_by == "count":
        hb = ax.hexbin(
            df[x],
            df[y],
            gridsize=gridsize,
            cmap=s.cmap_count,
            mincnt=1,
            bins="log",
            extent=extent,
        )
        label = "Number of pairs"
    else:
        hb = ax.hexbin(
            df[x],
            df[y],
            C=df[color_by],
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap=s.cmap_mean,
            extent=extent,
        )
        label = f"Mean {SCORE_LABEL.lower() if color_by == 'score' else color_by}"
    fig.colorbar(hb, ax=ax, label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return _finish(
        fig, ax, s, output_path, name or f"{x}_vs_{y}_{color_by}", title or f"{x} vs {y}"
    )


def plot_irt_ccs_hexbin(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="irt_vs_ccs_diff",
    gridsize=50,
    extent=(0, 50, 0, 0.3),
    title=None,
    figsize=None,
):
    """Pair counts over iRT difference × relative CCS difference. Needs CCS in the input
    (``peptides.tsv`` with a ``ccs`` column loaded with ``include_ccs=True``, or ``ccs 1`` /
    ``ccs 2`` columns in the scores TSV)."""
    s = _style(style)
    _require(df, "irt_diff", "relative_ccs_diff")
    fig, ax = new_axes(s, ax, figsize)
    hb = ax.hexbin(
        df["irt_diff"],
        df["relative_ccs_diff"],
        gridsize=gridsize,
        cmap=s.cmap_count,
        mincnt=1,
        extent=extent,
    )
    fig.colorbar(hb, ax=ax, label="Number of pairs")
    ax.set_xlabel("iRT difference")
    ax.set_ylabel("Relative CCS difference")
    return _finish(fig, ax, s, output_path, name, title or "iRT vs CCS difference")


def plot_binned_mean(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    x="irt_diff",
    y="length_1",
    bin_width=0.2,
    xlim=(0, 10),
    title=None,
    figsize=None,
):
    """Bar chart of the mean of `y` per `bin_width`-wide bin of `x` within `xlim`
    (e.g. mean peptide length or Levenshtein distance per iRT-difference bin)."""
    s = _style(style)
    _require(df, x, y)
    edges = np.arange(xlim[0], xlim[1] + bin_width, bin_width)
    binned = pd.cut(df[x], bins=edges, include_lowest=True)
    means = df.groupby(binned, observed=False)[y].mean()
    fig, ax = new_axes(s, ax, figsize)
    ax.bar(
        edges[:-1],
        means.values,
        width=bin_width,
        align="edge",
        color=s.color,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlim(*xlim)
    ax.set_xlabel(x)
    ax.set_ylabel(f"Mean {y}")
    return _finish(
        fig, ax, s, output_path, name or f"mean_{y}_per_{x}", title or f"Mean {y} per {x} bin"
    )


def plot_score_by_combo(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="score_by_p_and_distance",
    score_col="score",
    p_col="first_p_1",
    lev_col="peptide_levenshtein_O",
    title=None,
    figsize=None,
):
    """Box plots of the score for every (proline position from C-term, Levenshtein
    distance) combination present in the data, ordered by position then distance."""
    s = _style(style)
    _require(df, score_col, p_col, lev_col)
    plot_df = df[[score_col]].copy()
    plot_df["combo"] = "p" + df[p_col].astype(str) + " d" + df[lev_col].astype(str)
    order = [
        f"p{p} d{d}"
        for p in sorted(df[p_col].unique())
        for d in sorted(df[lev_col].unique())
        if f"p{p} d{d}" in set(plot_df["combo"])
    ]
    w, h = s.figsize
    fig, ax = new_axes(s, ax, figsize or (max(w, len(order) * 0.35), h))
    sns.boxplot(
        data=plot_df,
        x="combo",
        y=score_col,
        order=order,
        color=s.color,
        fliersize=1,
        linewidth=0.8,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("Proline position from C-terminus (p) and Levenshtein distance (d)")
    ax.set_ylabel(SCORE_LABEL if score_col == "score" else score_col)
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name,
        title or "Score by proline position and Levenshtein distance",
    )


# --------------------------------------------------------------------------- heatmaps


def plot_heatmap(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    rows="length_1",
    cols="peptide_levenshtein_O",
    aggfunc="mean",
    value_col="score",
    vmin=None,
    vmax=None,
    annot=True,
    row_label=None,
    col_label=None,
    title=None,
    figsize=None,
):
    """Heatmap of `value_col` aggregated over two categorical columns.
    ``aggfunc="mean"`` colours by mean score (annotated with the mean);
    ``aggfunc="count"`` by the number of pairs (log colour scale). A row value of -1 in
    a proline-position column is labelled "No P"."""
    s = _style(style)
    _require(df, rows, cols, value_col)
    pivot = df.pivot_table(values=value_col, index=rows, columns=cols, aggfunc=aggfunc)
    is_count = aggfunc == "count"
    fig, ax = new_axes(s, ax, _heatmap_figsize(s, pivot.shape, figsize))
    if is_count:
        annot_values = pivot.map(_compact_int)
        cbar_label = "Number of pairs"
    else:
        annot_values = pivot.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        cbar_label = f"{aggfunc.capitalize()} {SCORE_LABEL.lower() if value_col == 'score' else value_col}"
    sns.heatmap(
        pivot,
        mask=pivot.isna(),
        annot=annot_values if annot else False,
        fmt="",
        annot_kws={"fontsize": s.annot_size},
        cmap=s.cmap_count if is_count else s.cmap_mean,
        norm=LogNorm(vmin=vmin, vmax=vmax) if is_count else None,
        vmin=None if is_count else vmin,
        vmax=None if is_count else vmax,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    if rows.startswith("first_p") or rows.startswith("aa_pos_P"):
        ax.set_yticklabels(_position_labels(pivot.index, "P"))
    default_labels = {
        "length_1": "Peptide length",
        "peptide_levenshtein_O": LEV_LABEL,
        "peptide_levenshtein_O_IL": f"{LEV_LABEL} (I = L)",
        "first_p_1": "Proline position from C-terminus",
        "charge": "Precursor charge",
    }
    ax.set_xlabel(col_label or default_labels.get(cols, cols), fontsize=s.label_size)
    ax.set_ylabel(row_label or default_labels.get(rows, rows), fontsize=s.label_size)
    ax.tick_params(axis="both", labelsize=s.tick_size)
    ax.tick_params(axis="y", rotation=0)
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name or f"{rows}_{cols}_{aggfunc}_heatmap",
        title or f"{cbar_label} by {rows} and {cols}",
        despine=False,
    )


def plot_aa_position_heatmap(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    aa="P",
    terminus="N",
    lev_col="peptide_levenshtein_O",
    vmin=None,
    vmax=None,
    annot=True,
    title=None,
    figsize=None,
):
    """Mean score (colour) and pair count (annotation) over the position of residue `aa`
    nearest to `terminus` × Levenshtein distance. See `add_aa_position` for how the pair
    position is defined; -1 ("No X") means neither peptide contains `aa`."""
    s = _style(style)
    _require(df, "score", lev_col, "peptide_1_O", "peptide_2_O")
    pos_col = f"aa_pos_{aa}_{terminus}"
    plot_df = add_aa_position(df, aa, terminus, column=pos_col)
    label_aa = "M(ox)" if aa == "O" else aa
    fig, ax, mean = _mean_count_heatmap(
        plot_df,
        pos_col,
        lev_col,
        s,
        ax,
        figsize,
        vmin,
        vmax,
        annot,
        xlabel=LEV_LABEL,
        ylabel=f"{label_aa} position from the {terminus}-terminus",
        cbar_label=f"Mean {SCORE_LABEL.lower()}",
    )
    ax.set_yticklabels(_position_labels(mean.index, label_aa), rotation=0)
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name or f"aa_position_heatmap_{aa}_{terminus}",
        title or f"Score by {label_aa} position and Levenshtein distance",
        despine=False,
    )


def plot_aa_position_violin(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    aa="P",
    terminus="C",
    title=None,
    figsize=None,
):
    """Score distribution per position of residue `aa` nearest to `terminus` (default:
    proline from the C-terminus); -1 is shown as "No X"."""
    s = _style(style)
    _require(df, "score", "peptide_1_O", "peptide_2_O")
    pos_col = f"aa_pos_{aa}_{terminus}"
    plot_df = add_aa_position(df, aa, terminus, column=pos_col)
    order = sorted(plot_df[pos_col].unique())
    label_aa = "M(ox)" if aa == "O" else aa
    w, h = s.figsize
    fig, ax = new_axes(s, ax, figsize or (max(w, len(order) * 0.6), h))
    sns.violinplot(
        data=plot_df,
        x=pos_col,
        y="score",
        order=order,
        cut=0,
        inner="quartile",
        linewidth=0.6,
        hue=pos_col,
        palette=s.palette,
        legend=False,
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(_position_labels(order, label_aa))
    ax.set_xlabel(f"{label_aa} position from the {terminus}-terminus")
    ax.set_ylabel(SCORE_LABEL)
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name or f"aa_position_violin_{aa}_{terminus}",
        title or f"Score by {label_aa} position",
    )


def plot_aa_swap_heatmap(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    aggfunc="mean",
    value_col="score",
    collapse_il=True,
    vmin=None,
    vmax=None,
    annot=True,
    show_groups=True,
    title=None,
    figsize=None,
):
    """Lower-triangle residue × residue heatmap of first-two-residue swaps ("XY..." vs
    "YX..."). Runs `annotate_terminal_swap` itself when the swap columns are missing.
    ``aggfunc="mean"`` colours by mean `value_col`, ``"count"`` by pair count (log)."""
    s = _style(style)
    if "swap_aa_row" not in df.columns:
        _require(df, "peptide_1_O", "peptide_2_O")
        df = annotate_terminal_swap(df, collapse_il=collapse_il)
    _require(df, value_col)
    order, groups = aa_grid_order(collapse_il)
    pairs = df[df["swap_aa_row"].notna() & df["swap_aa_col"].notna()]
    if pairs.empty:
        raise ValueError("no first-two-residue swap pairs in the data")
    unknown = sorted((set(pairs["swap_aa_row"]) | set(pairs["swap_aa_col"])) - set(order))
    if unknown:
        raise ValueError(
            f"residues {unknown} are not in the collapse_il={collapse_il} grid order; "
            "pass the same collapse_il to annotate_terminal_swap and to this function"
        )
    pivot = pairs.pivot_table(
        values=value_col, index="swap_aa_row", columns="swap_aa_col", aggfunc=aggfunc
    ).reindex(index=order, columns=order)
    # The first residue can never be a row and the last never a column, so those lines are
    # always empty. Dropping them shifts every row up by one, hence the mask starts at k=1.
    pivot = pivot.iloc[1:, :-1]
    mask = pivot.isna() | np.triu(np.ones(pivot.shape, dtype=bool), k=1)
    is_count = aggfunc == "count"
    if is_count:
        cbar_label = "Number of pairs"
        annot_values = pivot.map(_compact_int)
    else:
        cbar_label = f"Mean {SCORE_LABEL.lower() if value_col == 'score' else value_col}"
        annot_values = pivot.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    fig, ax = new_axes(s, ax, figsize or (max(s.figsize[0], 8), max(s.figsize[1], 7)))
    sns.heatmap(
        pivot,
        mask=mask,
        annot=annot_values if annot else False,
        fmt="",
        annot_kws={"fontsize": s.annot_size},
        cmap=s.cmap_count if is_count else s.cmap_mean,
        norm=LogNorm(vmin=vmin, vmax=vmax) if is_count else None,
        vmin=None if is_count else vmin,
        vmax=None if is_count else vmax,
        linewidths=0.4,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"label": cbar_label, "shrink": 0.6},
    )
    ax.set_xticklabels(aa_tick_labels(pivot.columns, collapse_il), rotation=0)
    ax.set_yticklabels(aa_tick_labels(pivot.index, collapse_il), rotation=0)
    ax.set_xlabel("Swapped amino acid")
    ax.set_ylabel("Swapped amino acid")
    if show_groups:
        # Group separators + names along the top. The trim above cost the row axis one
        # residue and the column axis another, so each axis keeps its own running edge.
        row_edge = col_edge = 0
        for group_name, residues in groups:
            row_edge += sum(aa in pivot.index for aa in residues)
            start, col_edge = col_edge, col_edge + sum(
                aa in pivot.columns for aa in residues
            )
            ax.axhline(row_edge, color="grey", linewidth=1.0)
            ax.axvline(col_edge, color="grey", linewidth=1.0)
            if col_edge > start:
                ax.text(
                    (start + col_edge) / 2,
                    -0.4,
                    group_name,
                    ha="center",
                    va="bottom",
                    fontsize=s.tick_size,
                    color="grey",
                )
    if title and s.show_title:
        ax.set_title(title, fontsize=s.title_size, pad=30 if show_groups else 6)
    return _finish(
        fig, ax, s, output_path, name or f"aa_swap_{aggfunc}_heatmap", despine=False
    )


def plot_swap_position_heatmap(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="swap_position_heatmap",
    vmin=None,
    vmax=None,
    annot=True,
    title=None,
    figsize=None,
):
    """Mean score (colour) and pair count (annotation) over the two positions involved in
    a two-residue rearrangement (from the ``aa_substitution`` labels, N-terminal 1-based)."""
    s = _style(style)
    _require(df, "score", "aa_substitution")
    pairs = add_swap_positions(df)
    if pairs.empty:
        raise ValueError("no two-position rearrangements in the data")
    fig, ax, _ = _mean_count_heatmap(
        pairs,
        "pos_b",
        "pos_a",
        s,
        ax,
        figsize,
        vmin,
        vmax,
        annot,
        xlabel="First swapped position",
        ylabel="Second swapped position",
        cbar_label=f"Mean {SCORE_LABEL.lower()}",
    )
    return _finish(
        fig, ax, s, output_path, name, title or "Score by swapped positions", despine=False
    )


def plot_ox_sub_distance(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    sub_type=None,
    kind="heatmap",
    bin_width=0.05,
    vmax=None,
    title=None,
    figsize=None,
):
    """How far an oxidation change sits from the substitution, against the score.
    Only pairs with exactly one real substitution *and* an oxidation change (or an I/L
    swap plus an oxidation change) have a distance. `sub_type` (e.g. ``"A/S"``) restricts
    to one substitution. ``kind="heatmap"``: pair counts per distance × score bin;
    ``kind="scatter"``: one point per pair."""
    s = _style(style)
    _require(df, "score", "peptide_1_O", "peptide_2_O")
    if "ox_sub_distance" not in df.columns:
        df = annotate_substitutions(df)
    plot_df = df[df["ox_sub_distance"].notna()]
    if sub_type is not None:
        plot_df = plot_df[plot_df["sub_type"] == sub_type]
    if plot_df.empty:
        raise ValueError(
            f"no pairs with an oxidation-substitution distance"
            + (f" for sub_type={sub_type!r}" if sub_type else "")
        )
    plot_df = plot_df.assign(ox_sub_distance=plot_df["ox_sub_distance"].astype(int))
    suffix = f"_{sub_type.replace('/', '_')}" if sub_type else ""
    label = f" ({sub_type})" if sub_type else ""
    if kind == "scatter":
        fig, ax = new_axes(s, ax, figsize)
        sns.stripplot(
            data=plot_df,
            x="ox_sub_distance",
            y="score",
            color=s.color,
            size=3,
            alpha=0.6,
            jitter=0.2,
            ax=ax,
        )
        ax.set_xlabel("Distance between oxidation and substitution")
        ax.set_ylabel(SCORE_LABEL)
        return _finish(
            fig,
            ax,
            s,
            output_path,
            name or f"ox_sub_distance_scatter{suffix}",
            title or f"Score by oxidation–substitution distance{label}",
        )
    edges = np.round(np.arange(0, 1 + bin_width, bin_width), 6)
    score_bin = pd.cut(plot_df["score"], bins=edges, include_lowest=True)
    counts = (
        plot_df.groupby([plot_df["ox_sub_distance"], score_bin], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index(ascending=False)
    )
    counts.columns = [f"{c.left:.2f}–{c.right:.2f}" for c in counts.columns]
    fig, ax = new_axes(s, ax, _heatmap_figsize(s, counts.shape, figsize))
    sns.heatmap(
        counts,
        cmap=s.cmap_count,
        annot=True,
        fmt="d",
        annot_kws={"fontsize": s.annot_size},
        linewidths=0.5,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Number of pairs"},
        ax=ax,
    )
    ax.set_xlabel(SCORE_LABEL)
    ax.set_ylabel("Distance between oxidation and substitution")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return _finish(
        fig,
        ax,
        s,
        output_path,
        name or f"ox_sub_distance_heatmap{suffix}",
        title or f"Oxidation–substitution distance vs score{label}",
        despine=False,
    )


# --------------------------------------------------------------------------- composition


def plot_swap_aa_vs_input(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name="swap_aa_vs_input",
    input_peptides=None,
    label_k=1.0,
    title=None,
    figsize=None,
):
    """Each residue's frequency among the residues of two-position rearrangements (y)
    against its frequency in the input (x), with a y = x reference line. Residues whose
    residual exceeds `label_k` standard deviations are labelled. `input_peptides`
    defaults to the unique peptides in `df` (I merged into L throughout)."""
    s = _style(style)
    _require(df, "aa_substitution")
    swaps = add_swap_positions(df)
    if swaps.empty:
        raise ValueError("no two-position rearrangements in the data")
    swap_counter = Counter("".join(swaps["swap_aas"].str.replace("I", "L")))
    if input_peptides is None:
        _require(df, "peptide_1_O", "peptide_2_O")
        input_peptides = set(df["peptide_1_O"]) | set(df["peptide_2_O"])
    input_counter = Counter(
        "".join(str(p) for p in input_peptides).replace("I", "L")
    )
    residues = sorted(set(swap_counter) | set(input_counter))
    x = np.array([input_counter[a] / sum(input_counter.values()) for a in residues])
    y = np.array([swap_counter[a] / sum(swap_counter.values()) for a in residues])
    resid = y - x
    outlier = (
        np.abs(resid) > label_k * resid.std()
        if resid.std() > 0
        else np.zeros_like(resid, dtype=bool)
    )
    fig, ax = new_axes(s, ax, figsize or (min(s.figsize),) * 2)
    lim = max(x.max(), y.max()) * 1.1
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, zorder=0)
    ax.scatter(x, y, color=s.color, s=30, zorder=1)
    for a, xi, yi, is_out in zip(residues, x, y, outlier):
        if is_out:
            ax.annotate(
                {"L": "I/L", "O": "M(ox)"}.get(a, a),
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=s.tick_size,
                fontweight="bold",
            )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Frequency in input")
    ax.set_ylabel("Frequency among swapped residues")
    return _finish(
        fig, ax, s, output_path, name, title or "Swapped residues vs input composition"
    )
