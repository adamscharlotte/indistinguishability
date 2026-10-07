"""Load + enrich similarity-tool outputs into a scores DataFrame, plus filtering helpers.

Two input shapes are supported:
  * ``combine_pep_score`` — compact ``(i, j, score)`` numpy indices + a peptides TSV.
  * ``read_results`` — the verbose TSV ("new tool") with ``peptide 1``/``peptide 2`` columns.
Both add the same enrichment columns (``_O`` sequences, Levenshtein, lengths, proline
position, AA substitution classification).
"""

import os

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .sequence import first_p_from_right, get_aa_changes

#: A *variable* modification must keep a placeholder character: whether it is present is
#: exactly what distinguishes one peptide of a pair from the other, so oxidised Met becomes
#: the single char ``O`` and stays visible to Levenshtein / ``get_aa_changes``.
VARIABLE_MOD_REPLACEMENTS = {"M[UNIMOD:35]": "O"}

#: A *fixed* modification is information-free — every eligible residue carries it — so it is
#: stripped back to the bare residue instead. Leaving ``C[UNIMOD:4]`` in place would break
#: four things at once: ``length_{1,2}`` would count 12 string characters for one residue,
#: both Levenshtein columns would be measured over the mod text, ``annotate_substitutions()``
#: would mark every cysteine pair un-``analysable`` on its residual-``[`` rule, and the
#: ``O``->``M`` join key used by ``get_peptide_type`` / ``annotate_pair_characteristics``
#: would never match an unmodified peptide table.
FIXED_MOD_REPLACEMENTS = {"C[UNIMOD:4]": "C"}


def normalise_mods(sequences):
    """Rewrite ProForma modification tags in a peptide Series to the project's ``_O`` form.

    Variable mods collapse to a placeholder character, fixed mods are stripped — see
    ``VARIABLE_MOD_REPLACEMENTS`` / ``FIXED_MOD_REPLACEMENTS`` for why the two are treated
    differently. Run over a peptide column *before* any length, distance or join work.
    """
    out = sequences
    for tag, replacement in {
        **FIXED_MOD_REPLACEMENTS,
        **VARIABLE_MOD_REPLACEMENTS,
    }.items():
        out = out.str.replace(tag, replacement, regex=False)
    return out


def _enrich(scores_df):
    """Add ``peptide_{1,2}_O``, Levenshtein distances, lengths, proline
    position, and ``aa_substitution`` classification to a scores frame that
    already has ``peptide_1``/``peptide_2`` columns.

    Shared by ``combine_pep_score`` and ``combine_pep_scores_wide`` — this is
    per-row Python work (``Levenshtein.distance``, ``get_aa_changes``), so it
    should only ever be called on a row count small enough for that to be
    fast (the whole point of ``combine_pep_scores_wide`` filtering first).
    """
    scores_df["peptide_1_O"] = normalise_mods(scores_df["peptide_1"])
    scores_df["peptide_2_O"] = normalise_mods(scores_df["peptide_2"])
    scores_df["peptide_levenshtein_O"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"],
            scores_df["peptide_2_O"],
        )
    ]
    scores_df["peptide_levenshtein_O_IL"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"].str.replace("I", "L"),
            scores_df["peptide_2_O"].str.replace("I", "L"),
        )
    ]
    scores_df["length_1"] = scores_df["peptide_1_O"].str.len()
    scores_df["length_2"] = scores_df["peptide_2_O"].str.len()
    scores_df["first_p_1"] = scores_df["peptide_1_O"].apply(first_p_from_right)
    scores_df["first_p_2"] = scores_df["peptide_2_O"].apply(first_p_from_right)
    scores_df["aa_substitution"] = [
        get_aa_changes(a, b)
        for a, b in zip(scores_df["peptide_1_O"], scores_df["peptide_2_O"])
    ]
    return scores_df


def combine_pep_score(path_np, path_peptide, include_ccs=False):
    """Combine peptide scores from a numpy file with peptide information from a TSV file.
    Numpy file contains the index of the peptides.

    Parameters
    ----------
    include_ccs : bool, default False
        If True, also include ccs_1 / ccs_2 (and ccs_diff) columns,
        pulled from the "ccs" column in the peptide TSV.
    """
    scores = np.load(path_np)
    peptides = pd.read_csv(path_peptide, sep="\t")
    scores_df = pd.DataFrame(scores, columns=["i", "j", "score"])

    cols = ["peptide_sequences", "irt", "m/z", "precursor_charges"]
    out_names = ["peptide", "irt", "m/z", "charge"]
    if include_ccs:
        cols = cols + ["ccs"]
        out_names = out_names + ["ccs"]

    pep_vals = peptides[cols].to_numpy()
    i_idx = scores_df["i"].to_numpy(dtype=int)
    j_idx = scores_df["j"].to_numpy(dtype=int)

    for name, vals in zip(out_names, pep_vals[i_idx].T):
        scores_df[f"{name}_1"] = vals
    for name, vals in zip(out_names, pep_vals[j_idx].T):
        scores_df[f"{name}_2"] = vals

    scores_df["irt_diff"] = (scores_df["irt_1"] - scores_df["irt_2"]).abs()
    scores_df["mz_diff"] = (scores_df["m/z_1"] - scores_df["m/z_2"]).abs()
    scores_df["charge_diff"] = (scores_df["charge_1"] - scores_df["charge_2"]).abs()
    if include_ccs:
        scores_df["ccs_diff"] = (scores_df["ccs_1"] - scores_df["ccs_2"]).abs()

    scores_df = scores_df.drop_duplicates()
    scores_df = _enrich(scores_df)
    return scores_df


def combine_pep_scores_wide(model_paths, peptide_path, score_threshold=0.7):
    """Compare many similarity-tool score files that were run over the SAME
    candidate ``(i, j)`` pairs (e.g. the ``similarity`` package's
    ``slurm/02_band_compare.slurm`` output across intensity models/instruments).

    Loads each model's compact ``(i, j, score)`` ``.npy`` and aligns them on
    a composite integer key ``i * npeptides + j`` — not row position (parallel
    scoring workers do not guarantee row order across separate runs) and not
    a string key (far too memory-heavy at hundreds of millions to billions of
    rows) — into one numeric-only wide table. From that table:

    * the pairwise R^2 between every pair of models is reported over ALL
      aligned pairs (cheap: no peptide/string enrichment at this stage);
    * the subset of pairs where ``score_threshold`` is met by AT LEAST ONE
      model is enriched (Levenshtein, ``aa_substitution``, ...) and returned
      — a pair only one model calls indistinguishable is not lost, and
      enrichment only ever runs on this much smaller filtered subset, since
      it is per-row Python work that does not scale to the full candidate set.

    Parameters
    ----------
    model_paths : dict[str, str or Path]
        Maps a model/instrument label (e.g. ``"Prosit_2020_intensity_HCD__QE"``)
        to that model's ``scores.npy`` path.
    peptide_path : str or Path
        Path to any ONE of the per-model ``peptides.tsv`` files — identical
        across models, since it is built once from ``input.txt`` independent
        of which intensity model/instrument was used.
    score_threshold : float, default 0.7
        The project's indistinguishability cutoff (see ``CLAUDE.md``).

    Returns
    -------
    r_squared : pandas.DataFrame
        Model x model pairwise R^2 matrix over all aligned pairs.
    filtered : pandas.DataFrame
        One row per pair where any model's score >= ``score_threshold``, with
        a ``score_<model>`` column per model (``NaN`` where that model did
        not flag or did not report the pair) plus the usual enrichment
        columns.
    """
    peptides = pd.read_csv(peptide_path, sep="\t")
    npeptides = len(peptides)

    score_columns = {}
    counts = {}
    for model, npy_path in model_paths.items():
        scores = np.load(npy_path)
        counts[model] = len(scores)
        key = scores["i"].astype(np.int64) * npeptides + scores["j"].astype(np.int64)
        score_columns[f"score_{model}"] = pd.Series(
            scores["score"].astype(np.float32), index=key
        )

    print(
        "Raw pair counts per model (should be similar -- the candidate set is the same for all):"
    )
    print(pd.Series(counts, name="n_pairs").to_string())

    wide = pd.concat(score_columns, axis=1)

    r_squared = wide.corr() ** 2
    print("\nPairwise R^2 between models (over all aligned pairs):")
    print(r_squared.to_string())

    above_threshold = wide >= score_threshold
    print(f"\nPairs with score >= {score_threshold}, per model:")
    print(above_threshold.sum().to_string())
    n_models_flagging = above_threshold.sum(axis=1)
    print(f"\nOf {(n_models_flagging > 0).sum()} pairs flagged by at least one model:")
    print(f"  flagged by exactly 1 model: {(n_models_flagging == 1).sum()}")
    print(f"  flagged by >1 model: {(n_models_flagging > 1).sum()}")

    filtered = wide.loc[n_models_flagging > 0].rename_axis("key").reset_index()
    filtered["i"], filtered["j"] = divmod(filtered["key"].to_numpy(), npeptides)
    filtered = filtered.drop(columns="key")

    cols = ["peptide_sequences", "irt", "m/z", "precursor_charges"]
    out_names = ["peptide", "irt", "m/z", "charge"]
    pep_vals = peptides[cols].to_numpy()
    i_idx = filtered["i"].to_numpy(dtype=int)
    j_idx = filtered["j"].to_numpy(dtype=int)
    for name, vals in zip(out_names, pep_vals[i_idx].T):
        filtered[f"{name}_1"] = vals
    for name, vals in zip(out_names, pep_vals[j_idx].T):
        filtered[f"{name}_2"] = vals
    filtered["irt_diff"] = (filtered["irt_1"] - filtered["irt_2"]).abs()
    filtered["mz_diff"] = (filtered["m/z_1"] - filtered["m/z_2"]).abs()
    filtered["charge_diff"] = (filtered["charge_1"] - filtered["charge_2"]).abs()
    filtered = filtered.drop_duplicates()
    filtered = _enrich(filtered)

    return r_squared, filtered


def combine_filtered_pep_score_add_I(path_np, path_peptide):
    """Combine peptide scores from a numpy file with peptide information from a TSV file.
    Numpy file contains the index of the peptides."""
    scores = np.load(path_np)
    # This peptides table is a *filtered* subset: its "Unnamed: 0" column holds the
    # original peptide row-numbers that the .npy i/j indices refer to. We must look up
    # by that label (via the index), NOT by position — positions no longer line up
    # once rows have been filtered out.
    peptides = pd.read_csv(path_peptide, sep="\t")
    if "Unnamed: 0" in peptides.columns:
        peptides = peptides.set_index("Unnamed: 0")
        peptides.index.name = None
    scores_df = pd.DataFrame(scores, columns=["i", "j", "score"])
    scores_df["i"] = scores_df["i"].astype(int)
    scores_df["j"] = scores_df["j"].astype(int)
    # Keep only pairs whose peptides survived the filter (labels still present).
    present = peptides.index
    n_before = len(scores_df)
    scores_df = scores_df[
        scores_df["i"].isin(present) & scores_df["j"].isin(present)
    ].reset_index(drop=True)
    n_dropped = n_before - len(scores_df)
    if n_dropped:
        print(
            f"Dropped {n_dropped} of {n_before} pairs "
            f"({n_dropped / n_before:.1%}) referencing filtered-out peptides"
        )
    cols = ["peptide_sequences", "irt", "m/z", "precursor_charges", "total intensity"]
    (
        scores_df["peptide_1"],
        scores_df["irt_1"],
        scores_df["m/z_1"],
        scores_df["charge_1"],
        scores_df["total intensity_1"],
    ) = (
        peptides.loc[scores_df["i"], cols].to_numpy().T
    )
    (
        scores_df["peptide_2"],
        scores_df["irt_2"],
        scores_df["m/z_2"],
        scores_df["charge_2"],
        scores_df["total intensity_2"],
    ) = (
        peptides.loc[scores_df["j"], cols].to_numpy().T
    )
    scores_df["irt_diff"] = (scores_df["irt_1"] - scores_df["irt_2"]).abs()
    scores_df["mz_diff"] = (scores_df["m/z_1"] - scores_df["m/z_2"]).abs()
    scores_df["charge_diff"] = (scores_df["charge_1"] - scores_df["charge_2"]).abs()
    scores_df = scores_df.drop_duplicates()
    scores_df["peptide_1_O"] = normalise_mods(scores_df["peptide_1"])
    scores_df["peptide_2_O"] = normalise_mods(scores_df["peptide_2"])
    scores_df["peptide_levenshtein_O"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"],
            scores_df["peptide_2_O"],
        )
    ]
    scores_df["peptide_levenshtein_O_IL"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"].str.replace("I", "L"),
            scores_df["peptide_2_O"].str.replace("I", "L"),
        )
    ]
    scores_df["length_1"] = scores_df["peptide_1_O"].str.len()
    scores_df["length_2"] = scores_df["peptide_2_O"].str.len()
    scores_df["first_p_1"] = scores_df["peptide_1_O"].apply(first_p_from_right)
    scores_df["first_p_2"] = scores_df["peptide_2_O"].apply(first_p_from_right)
    scores_df["aa_substitution"] = [
        get_aa_changes(a, b)
        for a, b in zip(scores_df["peptide_1_O"], scores_df["peptide_2_O"])
    ]
    return scores_df


def combine_filtered_pep_score(path_np, path_peptide):
    """Combine peptide scores from a numpy file with peptide information from a TSV file.
    Numpy file contains the index of the peptides."""
    scores = np.load(path_np)
    # This peptides table is a *filtered* subset: its "Unnamed: 0" column holds the
    # original peptide row-numbers that the .npy i/j indices refer to. We must look up
    # by that label (via the index), NOT by position — positions no longer line up
    # once rows have been filtered out.
    peptides = pd.read_csv(path_peptide, sep="\t")
    if "Unnamed: 0" in peptides.columns:
        peptides = peptides.set_index("Unnamed: 0")
        peptides.index.name = None
    scores_df = pd.DataFrame(scores, columns=["i", "j", "score"])
    scores_df["i"] = scores_df["i"].astype(int)
    scores_df["j"] = scores_df["j"].astype(int)
    # Keep only pairs whose peptides survived the filter (labels still present).
    present = peptides.index
    n_before = len(scores_df)
    scores_df = scores_df[
        scores_df["i"].isin(present) & scores_df["j"].isin(present)
    ].reset_index(drop=True)
    n_dropped = n_before - len(scores_df)
    if n_dropped:
        print(
            f"Dropped {n_dropped} of {n_before} pairs "
            f"({n_dropped / n_before:.1%}) referencing filtered-out peptides"
        )
    cols = ["peptide_sequences", "irt", "m/z", "precursor_charges"]
    (
        scores_df["peptide_1"],
        scores_df["irt_1"],
        scores_df["m/z_1"],
        scores_df["charge_1"],
    ) = (
        peptides.loc[scores_df["i"], cols].to_numpy().T
    )
    (
        scores_df["peptide_2"],
        scores_df["irt_2"],
        scores_df["m/z_2"],
        scores_df["charge_2"],
    ) = (
        peptides.loc[scores_df["j"], cols].to_numpy().T
    )
    scores_df["irt_diff"] = (scores_df["irt_1"] - scores_df["irt_2"]).abs()
    scores_df["mz_diff"] = (scores_df["m/z_1"] - scores_df["m/z_2"]).abs()
    scores_df["charge_diff"] = (scores_df["charge_1"] - scores_df["charge_2"]).abs()
    scores_df = scores_df.drop_duplicates()
    scores_df["peptide_1_O"] = normalise_mods(scores_df["peptide_1"])
    scores_df["peptide_2_O"] = normalise_mods(scores_df["peptide_2"])
    scores_df["peptide_levenshtein_O"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"],
            scores_df["peptide_2_O"],
        )
    ]
    scores_df["peptide_levenshtein_O_IL"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"].str.replace("I", "L"),
            scores_df["peptide_2_O"].str.replace("I", "L"),
        )
    ]
    scores_df["length_1"] = scores_df["peptide_1_O"].str.len()
    scores_df["length_2"] = scores_df["peptide_2_O"].str.len()
    scores_df["first_p_1"] = scores_df["peptide_1_O"].apply(first_p_from_right)
    scores_df["first_p_2"] = scores_df["peptide_2_O"].apply(first_p_from_right)
    scores_df["aa_substitution"] = [
        get_aa_changes(a, b)
        for a, b in zip(scores_df["peptide_1_O"], scores_df["peptide_2_O"])
    ]
    return scores_df


def read_results(path_results):
    scores_df = pd.read_csv(path_results, sep="\t")
    # The precursor columns are optional: some result TSVs carry only the two peptides
    # and the score, so each diff is added only when both of its source columns exist.
    for diff_col, src in (
        ("irt_diff", "iRT"),
        ("mz_diff", "m/z"),
        ("charge_diff", "charge"),
    ):
        if f"{src} 1" in scores_df and f"{src} 2" in scores_df:
            scores_df[diff_col] = (scores_df[f"{src} 1"] - scores_df[f"{src} 2"]).abs()
    scores_df = scores_df.drop_duplicates()
    scores_df["peptide_1_O"] = normalise_mods(scores_df["peptide 1"])
    scores_df["peptide_2_O"] = normalise_mods(scores_df["peptide 2"])
    scores_df["peptide_levenshtein_O"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"],
            scores_df["peptide_2_O"],
        )
    ]
    scores_df["peptide_levenshtein_O_IL"] = [
        Levenshtein.distance(a, b)
        for a, b in zip(
            scores_df["peptide_1_O"].str.replace("I", "L"),
            scores_df["peptide_2_O"].str.replace("I", "L"),
        )
    ]
    scores_df["length_1"] = scores_df["peptide_1_O"].str.len()
    scores_df["length_2"] = scores_df["peptide_2_O"].str.len()
    scores_df["first_p_1"] = scores_df["peptide_1_O"].apply(first_p_from_right)
    scores_df["first_p_2"] = scores_df["peptide_2_O"].apply(first_p_from_right)
    scores_df["aa_substitution"] = [
        get_aa_changes(a, b)
        for a, b in zip(scores_df["peptide_1_O"], scores_df["peptide_2_O"])
    ]
    return scores_df


def remove_MO_IL(scores_df):
    filtered_scores_df = scores_df[
        ~scores_df["aa_substitution"].apply(
            lambda x: (
                any("MO" in s or "identical" in s for s in x)
                if isinstance(x, list)
                else any(term in str(x) for term in ["MO", "identical"])
            )
        )
    ]
    return filtered_scores_df


def remove_MO(scores_df):
    filtered_scores_df = scores_df[
        ~scores_df["aa_substitution"].apply(
            lambda x: (
                any("MO" in s for s in x)
                if isinstance(x, list)
                else any(term in str(x) for term in ["MO"])
            )
        )
    ]
    return filtered_scores_df


def remove_IL(scores_df):
    filtered_scores_df = scores_df[
        ~scores_df["aa_substitution"].apply(
            lambda x: (
                any("identical" in s for s in x)
                if isinstance(x, list)
                else any(term in str(x) for term in ["identical"])
            )
        )
    ]
    return filtered_scores_df


def lookup_peptide_rows(
    path_peptide,
    indices,
    cols=("peptide_sequences", "precursor_charges", "irt", "m/z"),
    cache_path=None,
    chunksize=5_000_000,
):
    """Pull specific rows out of a peptides TSV that is too large to load whole.

    `indices` are 0-based row numbers in the peptide table — exactly what the ``i``/``j``
    columns of a compact scores file refer to. The table is streamed in chunks and only the
    wanted rows are kept, so peak memory is one chunk instead of the whole file (the 50M
    ``peptides.tsv`` is 7.3 GB, which the other loaders' plain ``read_csv`` cannot handle).
    Returns a DataFrame indexed by row number.

    `cache_path` makes the scan a one-off: the extracted rows are written there as a TSV and
    re-read on later calls, and a rescan is triggered only if the cache is missing indices.
    """
    wanted = {int(i) for i in indices}
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, sep="\t", index_col="index")
        missing = wanted - set(cached.index)
        if not missing:
            return cached.loc[sorted(wanted)]
        print(
            f"{len(missing)} of {len(wanted)} indices are absent from {cache_path}; "
            "rescanning the peptide table"
        )
    frames, start = [], 0
    for chunk in pd.read_csv(
        path_peptide, sep="\t", usecols=list(cols), chunksize=chunksize
    ):
        stop = start + len(chunk)
        hit = sorted(w for w in wanted if start <= w < stop)
        if hit:
            block = chunk.iloc[[w - start for w in hit]].copy()
            block.index = hit
            frames.append(block)
        start = stop
    out = pd.concat(frames) if frames else pd.DataFrame(columns=list(cols))
    out.index.name = "index"
    n_found = len(out)
    if n_found < len(wanted):
        print(
            f"Only {n_found} of {len(wanted)} indices found; the peptide table has "
            f"{start} rows"
        )
    if cache_path:
        out.to_csv(cache_path, sep="\t")
    return out


def annotate_charge(scores_df, index_table, charge_col="precursor_charges"):
    """Add ``charge_1``/``charge_2`` and a single ``charge`` column to a scores frame whose
    ``i``/``j`` are peptide row numbers, by looking them up in `index_table` (the output of
    `lookup_peptide_rows`).

    Candidate pairs are generated inside an m/z window, so both precursors of a pair carry
    the same charge — that is asserted here rather than assumed, and ``charge`` is the shared
    value. Splitting on it matters: the *same* peptide pair is scored once per charge state
    with a different score each time, so a "number of pairs" count taken over all rows counts
    precursors, not peptide pairs.
    """
    out = scores_df.copy()
    charges = index_table[charge_col]
    out["charge_1"] = charges.reindex(out["i"].astype(int)).to_numpy()
    out["charge_2"] = charges.reindex(out["j"].astype(int)).to_numpy()
    unresolved = out["charge_1"].isna() | out["charge_2"].isna()
    if unresolved.any():
        raise KeyError(
            f"{int(unresolved.sum())} of {len(out)} pairs reference peptide rows that are "
            "not in index_table — re-run lookup_peptide_rows over all i/j values"
        )
    mismatch = out["charge_1"] != out["charge_2"]
    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} of {len(out)} pairs pair two different precursor "
            "charges; a single 'charge' column would be meaningless for those"
        )
    out["charge"] = out["charge_1"].astype(int)
    return out


#: Verbose-TSV column -> compact-loader column. Applied by `load_scores` so the two input
#: shapes end up with one schema and every plot can rely on the same names.
VERBOSE_RENAMES = {
    f"{src} {side}": f"{dst}_{side}"
    for src, dst in (
        ("peptide", "peptide"),
        ("iRT", "irt"),
        ("m/z", "m/z"),
        ("charge", "charge"),
        ("ccs", "ccs"),
        ("CCS", "ccs"),
    )
    for side in (1, 2)
}


def _add_derived(scores_df):
    """Columns both input shapes should carry once loaded: a single ``charge`` column and,
    when CCS is present, ``ccs_diff`` / ``relative_ccs_diff``."""
    if "charge_1" in scores_df and "charge_2" in scores_df:
        # Candidate pairs share a charge, so ``charge`` is the shared value; a pair that
        # somehow mixes charges gets NaN instead of silently taking one side's value.
        same = scores_df["charge_1"] == scores_df["charge_2"]
        scores_df["charge"] = scores_df["charge_1"].where(same)
    elif "charge_1" in scores_df:
        scores_df["charge"] = scores_df["charge_1"]
    if "ccs_1" in scores_df and "ccs_2" in scores_df:
        scores_df["ccs_diff"] = (scores_df["ccs_1"] - scores_df["ccs_2"]).abs()
        scores_df["relative_ccs_diff"] = scores_df["ccs_diff"] / scores_df[
            ["ccs_1", "ccs_2"]
        ].max(axis=1)
    return scores_df


def load_scores(
    scores_tsv=None,
    peptides=None,
    npy=None,
    filtered=False,
    include_ccs=False,
    cache=None,
):
    """Load a similarity-tool output into the enriched scores schema used by every plot.

    Pass **either** `scores_tsv` (the verbose ``peptide 1`` / ``peptide 2`` / ``score`` TSV,
    via `read_results`) **or** `peptides` + `npy` (``peptides.tsv`` + compact ``(i, j,
    score)`` array, via `combine_pep_score`). The verbose columns are renamed to the compact
    loader's names (``peptide_1``, ``irt_1``, ``m/z_1``, ``charge_1``, ...), so the result
    has the same columns whichever input was used.

    `filtered=True` uses `combine_filtered_pep_score`, which looks peptides up by their
    original row label: required when ``peptides.tsv`` is a filtered subset, where
    positional lookup silently mislabels every pair.

    `cache` is a pickle path: read when it exists, written after loading otherwise.
    Enrichment is per-row Python work, so this saves minutes on large files. Delete the
    cache when the inputs change; it is not invalidated automatically.
    """
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    if scores_tsv is not None:
        if peptides is not None or npy is not None:
            raise ValueError("pass either scores_tsv or peptides + npy, not both")
        scores_df = read_results(scores_tsv).rename(columns=VERBOSE_RENAMES)
    elif peptides is not None and npy is not None:
        if filtered:
            if include_ccs:
                raise ValueError("include_ccs is not supported with filtered=True")
            scores_df = combine_filtered_pep_score(npy, peptides)
        else:
            scores_df = combine_pep_score(npy, peptides, include_ccs=include_ccs)
    else:
        raise ValueError("pass scores_tsv, or both peptides and npy")
    scores_df = _add_derived(scores_df.reset_index(drop=True))
    if cache:
        scores_df.to_pickle(cache)
    return scores_df
