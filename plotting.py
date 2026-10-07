# Different kinds of plots
"""Make plots of indistinguishable-peptide results, one (or several) at a time.

Input is either a verbose ``scores.tsv`` or ``peptides.tsv`` + ``scores.npy``:

    python plotting.py --list
    python plotting.py --scores scores.tsv --plot score_distribution --out figures
    python plotting.py --peptides peptides.tsv --npy scores.npy \\
        --plot aa_swap_heatmap --opt aggfunc=count \\
        --preset talk --figsize 12 9 --cmap-count magma --format pdf --out figures
    python plotting.py --scores scores.tsv --threshold 0.7 --remove MO_IL --plot all

Plot-specific options go through ``--opt key=value`` (repeatable); ``--list`` shows them.

From a notebook, the same registry is importable:

    from plotting import PLOTS, make_plot
    from indistinguishability import load_scores, get_style
    df = load_scores(scores_tsv="scores.tsv")
    fig, ax = make_plot("heatmap_mean", df, style=get_style("paper"), rows="first_p_1")
"""

import argparse
import inspect
import sys

import matplotlib
import matplotlib.pyplot as plt

import indistinguishability as ind


def _plot_mirror(
    df,
    *,
    style=None,
    ax=None,
    output_path=None,
    name=None,
    pair=None,
    charge=None,
    top=1,
    ce=30,
    title=None,
    figsize=None,
):
    """Koina mirror plot. ``pair="PEP1,PEP2"`` plots that pair (at `charge`, default 2);
    otherwise the `top` highest-scoring pairs in the data are plotted, each at its own
    charge. Needs the ``spectra`` extra and network access to Koina."""
    if pair is not None:
        p1, p2 = [p.strip() for p in str(pair).split(",")]
        rows = [(p1, p2, int(charge or 2), None)]
    else:
        missing = [
            c for c in ("peptide_1", "peptide_2", "score") if c not in df.columns
        ]
        if missing:
            raise KeyError(f"mirror needs column(s) {missing}")
        best = df.nlargest(int(top), "score")
        charges = best["charge_1"] if "charge_1" in best else [charge or 2] * len(best)
        rows = [
            (a, b, int(z), sc)
            for a, b, z, sc in zip(
                best["peptide_1"], best["peptide_2"], charges, best["score"]
            )
        ]
    out = None
    for i, (p1, p2, z, sc) in enumerate(rows):
        plot_title = title or (
            f"{p1} vs {p2} ({z}+)" + (f", score {sc:.3f}" if sc is not None else "")
        )
        stem = name or f"mirror_{p1}_{p2}_z{z}"
        stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)
        out = ind.plot_mirror(
            p1,
            p2,
            charge=z,
            ce=ce,
            style=style,
            ax=ax if len(rows) == 1 else None,
            output_path=output_path,
            name=stem,
            title=plot_title,
            figsize=figsize,
        )
        if len(rows) > 1 and output_path is not None:
            matplotlib.pyplot.close(out[0])
    return out


def _partial(func, **fixed):
    def wrapper(df, **kwargs):
        return func(df, **{**fixed, **kwargs})

    wrapper.__doc__ = func.__doc__
    wrapper.__wrapped__ = func
    wrapper.fixed = fixed
    return wrapper


#: Registry: CLI name -> callable(df, *, style, ax, output_path, name, **options).
PLOTS = {
    # distributions
    "score_distribution": ind.plot_score_distribution,
    "levenshtein_distribution": ind.plot_levenshtein_distribution,
    "irt_diff_histogram": ind.plot_irt_diff_histogram,
    "peptide_length": ind.plot_peptide_length,
    # score vs features
    "levenshtein_violin": ind.plot_levenshtein_violin,
    "feature_hexbin": ind.plot_feature_hexbin,
    "binned_mean": ind.plot_binned_mean,
    "score_by_combo": ind.plot_score_by_combo,
    "irt_ccs_hexbin": ind.plot_irt_ccs_hexbin,
    # heatmaps
    "heatmap_mean": _partial(ind.plot_heatmap, aggfunc="mean"),
    "heatmap_count": _partial(ind.plot_heatmap, aggfunc="count"),
    "aa_position_heatmap": ind.plot_aa_position_heatmap,
    "aa_position_violin": ind.plot_aa_position_violin,
    "aa_swap_heatmap": ind.plot_aa_swap_heatmap,
    "swap_position_heatmap": ind.plot_swap_position_heatmap,
    "ox_sub_distance": ind.plot_ox_sub_distance,
    "swap_aa_vs_input": ind.plot_swap_aa_vs_input,
    # spectra (Koina; not part of "all")
    "mirror": _plot_mirror,
}

#: Plots skipped by ``--plot all``: they need the network.
NOT_IN_ALL = {"mirror"}

_COMMON = {"df", "style", "ax", "output_path", "name", "title", "figsize"}


def plot_options(key):
    """Plot-specific keyword options and their defaults, for ``--list``."""
    func = PLOTS[key]
    fixed = getattr(func, "fixed", {})
    sig = inspect.signature(getattr(func, "__wrapped__", func))
    return {
        n: p.default
        for n, p in sig.parameters.items()
        if n not in _COMMON and n not in fixed and p.kind == p.KEYWORD_ONLY
    }


def make_plot(key, df, **kwargs):
    """Call plot `key` from `PLOTS` on `df`; returns ``(fig, ax)``."""
    if key not in PLOTS:
        raise KeyError(f"unknown plot {key!r}; choose from {sorted(PLOTS)}")
    return PLOTS[key](df, **kwargs)


def _parse_value(text):
    """``--opt`` values: numbers, booleans, None and comma-separated tuples are converted;
    anything else stays a string."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none":
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if "," in text:
        parts = [_parse_value(p) for p in text.split(",")]
        if all(isinstance(p, (int, float)) for p in parts):
            return tuple(parts)
    return text


def _parse_opts(pairs):
    opts = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--opt expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        opts[key.strip()] = _parse_value(value.strip())
    return opts


def _print_list():
    print("Available plots (pass with --plot; options with --opt key=value):\n")
    for key, func in PLOTS.items():
        doc = (inspect.getdoc(func) or "").split("\n\n")[0].replace("\n", " ")
        print(f"  {key}{' (not in all)' if key in NOT_IN_ALL else ''}")
        print(f"      {doc}")
        opts = plot_options(key)
        if opts:
            print("      options: " + ", ".join(f"{k}={v!r}" for k, v in opts.items()))
        print()
    print("Style presets: " + ", ".join(ind.PRESETS))


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list to see every plot and its options.",
    )
    src = p.add_argument_group("input (either --scores, or --peptides + --npy)")
    src.add_argument(
        "--scores", help="verbose scores TSV (peptide 1, peptide 2, score, ...)"
    )
    src.add_argument("--peptides", help="peptides.tsv for a compact scores.npy")
    src.add_argument("--npy", help="compact (i, j, score) scores.npy")
    src.add_argument(
        "--filtered",
        action="store_true",
        help="peptides.tsv is a filtered subset: look i/j up by row label",
    )
    src.add_argument(
        "--ccs", action="store_true", help="also load the 'ccs' column of peptides.tsv"
    )
    src.add_argument("--cache", help="pickle of the loaded data; reused when it exists")
    src.add_argument(
        "--input-peptides",
        help="newline-delimited input peptide list (peptide_length, swap_aa_vs_input)",
    )
    flt = p.add_argument_group("filters (applied before plotting)")
    flt.add_argument(
        "--threshold", type=float, help="keep pairs with score >= THRESHOLD (e.g. 0.7)"
    )
    flt.add_argument(
        "--remove",
        choices=["MO_IL", "MO", "IL"],
        help="drop oxidation-only (MO), I/L-identical (IL) pairs, or both",
    )
    flt.add_argument("--charge", type=int, help="keep one precursor charge")
    sel = p.add_argument_group("plots")
    sel.add_argument(
        "--plot",
        action="append",
        metavar="NAME",
        help="plot to make; repeat for several, or 'all'",
    )
    sel.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        help="plot-specific option, e.g. aggfunc=count, aa=M, xlim=0,30",
    )
    sel.add_argument("--name", help="output file stem (single plot only)")
    sel.add_argument("--title", help="figure title")
    sel.add_argument("--list", action="store_true", help="list plots and their options")
    sel.add_argument("--show", action="store_true", help="open the figures in a window")
    sty = p.add_argument_group("style")
    sty.add_argument(
        "--preset",
        default="notebook",
        choices=list(ind.PRESETS),
        help="paper, talk or notebook (default)",
    )
    sty.add_argument(
        "--out", default="figures", help="output directory (default: figures)"
    )
    sty.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"))
    sty.add_argument("--dpi", type=int)
    sty.add_argument("--format", help="png, pdf, svg, ...")
    sty.add_argument("--color", help="main colour for single-series plots")
    sty.add_argument("--accent", help="colour for cutoff lines / second series")
    sty.add_argument("--palette", help="seaborn palette for categorical series")
    sty.add_argument("--cmap", help="colormap for mean-score heatmaps/hexbins")
    sty.add_argument("--cmap-count", help="colormap for count heatmaps/hexbins")
    sty.add_argument("--font-scale", type=float)
    sty.add_argument("--font-family")
    sty.add_argument("--no-title", action="store_true", help="omit figure titles")
    sty.add_argument(
        "--transparent", action="store_true", help="transparent background"
    )
    return p


def load_from_args(args):
    df = ind.load_scores(
        scores_tsv=args.scores,
        peptides=args.peptides,
        npy=args.npy,
        filtered=args.filtered,
        include_ccs=args.ccs,
        cache=args.cache,
    )
    n = len(df)
    if args.threshold is not None:
        df = df[df["score"] >= args.threshold]
    if args.remove:
        df = {"MO_IL": ind.remove_MO_IL, "MO": ind.remove_MO, "IL": ind.remove_IL}[
            args.remove
        ](df)
    if args.charge is not None:
        if "charge" not in df.columns:
            raise SystemExit("--charge given but the data has no charge column")
        df = df[df["charge"] == args.charge]
    print(f"{len(df)} of {n} pairs after filtering")
    return df


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list:
        _print_list()
        return
    if not args.plot:
        build_parser().error("give at least one --plot NAME (or --list)")
    keys = (
        [k for k in PLOTS if k not in NOT_IN_ALL] if "all" in args.plot else args.plot
    )
    unknown = [k for k in keys if k not in PLOTS]
    if unknown:
        build_parser().error(f"unknown plot(s) {unknown}; see --list")
    if not args.show:
        matplotlib.use("Agg")
    style = ind.get_style(
        args.preset,
        figsize=tuple(args.figsize) if args.figsize else None,
        dpi=args.dpi,
        format=args.format,
        color=args.color,
        accent=args.accent,
        palette=args.palette,
        cmap_mean=args.cmap,
        cmap_count=args.cmap_count,
        font_scale=args.font_scale,
        font_family=args.font_family,
        show_title=False if args.no_title else None,
        transparent=True if args.transparent else None,
    ).apply()

    df = load_from_args(args)
    opts = _parse_opts(args.opt)
    input_peptides = None
    if args.input_peptides:
        with open(args.input_peptides) as fh:
            input_peptides = [line.strip() for line in fh if line.strip()]

    failed = []
    for key in keys:
        accepted = set(plot_options(key)) | {"title"}
        kwargs = {k: v for k, v in opts.items() if k in accepted}
        ignored = sorted(set(opts) - accepted)
        if ignored and len(keys) == 1:
            print(f"warning: {key} ignores option(s) {ignored}", file=sys.stderr)
        if input_peptides is not None and "input_peptides" in accepted:
            kwargs["input_peptides"] = input_peptides
        if args.title:
            kwargs["title"] = args.title
        if args.name and len(keys) == 1:
            kwargs["name"] = args.name
        try:
            fig, _ = make_plot(key, df, style=style, output_path=args.out, **kwargs)
        except (KeyError, ValueError, ImportError) as err:
            print(f"skipped {key}: {err}", file=sys.stderr)
            failed.append(key)
            continue
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()
    if failed and len(keys) == 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
