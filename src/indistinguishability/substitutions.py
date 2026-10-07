"""Single amino-acid substitution analysis on the ``peptide_1_O``/``peptide_2_O``
columns (M[UNIMOD:35] already encoded as the single char "O"). At an aligned
position a mismatch is {M,O} -> oxidation change, {I,L} -> isobaric (ignored
as a real substitution, queryable as "I/L"), anything else -> real substitution.
"""

import numpy as np
import pandas as pd

_OX_PAIR = frozenset({"M", "O"})
_IL_PAIR = frozenset({"I", "L"})

# Amino acids grouped by physicochemical property, for the axes of the swap heatmaps.
# "O" is oxidized Met (M[UNIMOD:35]); I is merged into L because the two are isobaric.
AA_GROUPS = [
    ("Aliphatic", ["G", "A", "V", "L", "M", "O"]),
    ("Aromatic", ["F", "W", "Y"]),
    ("Polar", ["S", "T", "C", "N", "Q", "P"]),
    ("Positive", ["K", "R", "H"]),
    ("Negative", ["D", "E"]),
]
AA_ORDER = [aa for _, group in AA_GROUPS for aa in group]
AA_LABELS = {"O": "M(ox)"}


def aa_grid_order(collapse_il=True):
    """Residues in `AA_GROUPS` order, as a flat list and as (name, residues) blocks.
    With `collapse_il=False`, I is restored as its own entry just before L."""
    if collapse_il:
        return list(AA_ORDER), [(name, list(g)) for name, g in AA_GROUPS]
    groups = [
        (name, [aa for res in g for aa in (["I", "L"] if res == "L" else [res])])
        for name, g in AA_GROUPS
    ]
    return [aa for _, g in groups for aa in g], groups


def aa_tick_labels(order, collapse_il=True):
    """Display labels for the grid axes: O -> M(ox) always, L -> I/L only when the two
    isobaric residues share a row/column."""
    return [
        "I/L" if (aa == "L" and collapse_il) else AA_LABELS.get(aa, aa) for aa in order
    ]


def _classify_pair(p1_O, p2_O):
    """Classify one peptide pair from its aligned `_O` sequences. Only equal-length,
    bracket-free pairs are analysable (positional alignment is meaningless otherwise,
    and a residual "[" means an unhandled modification is still in the string)."""
    if len(p1_O) != len(p2_O) or "[" in p1_O or "[" in p2_O:
        return {"analysable": False, "n_sub": 0, "n_ox": 0, "n_il": 0, "sub_type": None,
                "sub_aa_1": None, "sub_aa_2": None, "has_ox_change": False,
                "sub_pos": [], "ox_pos": [], "il_pos": [], "ox_sub_distance": np.nan}
    real_subs, ox_pos, il_pos = [], [], []
    for pos, (a, b) in enumerate(zip(p1_O, p2_O), start=1):
        if a == b:
            continue
        pair = frozenset({a, b})
        if pair == _OX_PAIR:
            ox_pos.append(pos)
        elif pair == _IL_PAIR:
            il_pos.append(pos)
        else:
            real_subs.append((pos, a, b))
    n_sub = len(real_subs)
    if n_sub == 1:
        _, a1, b1 = real_subs[0]
        sub_type = "/".join(sorted((a1, b1)))
    else:
        a1 = b1 = sub_type = None
    if n_sub == 1 and ox_pos:
        ox_sub_distance = min(abs(real_subs[0][0] - o) for o in ox_pos)
    elif n_sub == 0 and il_pos and ox_pos:
        ox_sub_distance = min(abs(s - o) for s in il_pos for o in ox_pos)
    else:
        ox_sub_distance = np.nan
    return {"analysable": True, "n_sub": n_sub, "n_ox": len(ox_pos), "n_il": len(il_pos),
            "sub_type": sub_type, "sub_aa_1": a1, "sub_aa_2": b1,
            "has_ox_change": len(ox_pos) > 0, "sub_pos": [p for p, _, _ in real_subs],
            "ox_pos": ox_pos, "il_pos": il_pos, "ox_sub_distance": ox_sub_distance}


def annotate_substitutions(scores_df):
    """Return a copy of `scores_df` with per-pair substitution metrics added: analysable,
    n_sub, n_ox, n_il, sub_type, sub_aa_1/2, has_ox_change, sub_pos, ox_pos, il_pos,
    ox_sub_distance. Needs `peptide_1_O`/`peptide_2_O` (from combine_pep_score/read_results)."""
    metrics = pd.DataFrame(
        [_classify_pair(a, b) for a, b in zip(scores_df["peptide_1_O"], scores_df["peptide_2_O"])],
        index=scores_df.index,
    )
    return pd.concat([scores_df.copy(), metrics], axis=1)


def annotate_terminal_swap(scores_df, collapse_il=True, verbose=True):
    """Flag pairs whose only difference is the first two residues having swapped places
    ("XY..." vs "YX...") and label each with the unordered residue pair involved.

    Adds `is_swap` plus `swap_aa_row` / `swap_aa_col`: the two swapped residues ordered by
    their position in the grid order (row = later), so every pair lands in the lower triangle
    of a residue x residue grid.

    `collapse_il` switches the whole comparison, not just the axes. With it on, sequences are
    I->L-normalized first: I and L share a row/column, an I/L difference elsewhere in the
    peptide does not break a swap, and an I<->L swap of the first two residues is not a swap
    at all (the peptides are isobaric, hence the same peptide). With it off nothing is
    normalized: I and L get their own rows/columns and an I/L difference anywhere breaks the
    swap, so `is_swap` is strictly smaller than in the collapsed case.

    `swap_aa_row` is NaN for non-swap pairs and for residues outside the grid order, so
    filter on `swap_aa_row.notna()` before plotting.
    """
    p1 = scores_df["peptide_1_O"].astype(str)
    p2 = scores_df["peptide_2_O"].astype(str)
    if collapse_il:
        n1, n2 = p1.str.replace("I", "L"), p2.str.replace("I", "L")
    else:
        n1, n2 = p1, p2
    aa_1, aa_2 = n1.str[0], n1.str[1]
    is_swap = (
        ~p1.str.contains("[", regex=False)
        & ~p2.str.contains("[", regex=False)
        & (n1.str.len() == n2.str.len())
        & (n1.str.len() >= 2)
        & (aa_1 == n2.str[1])
        & (aa_2 == n2.str[0])
        & (aa_1 != aa_2)
        & (n1.str[2:] == n2.str[2:])
    )
    x, y = aa_1.where(is_swap), aa_2.where(is_swap)
    order, _ = aa_grid_order(collapse_il)
    rank = {aa: i for i, aa in enumerate(order)}
    later_first = x.map(rank) >= y.map(rank)  # False for unknown residues (NaN rank)
    row = pd.Series(np.where(later_first, x, y), index=scores_df.index).where(x.notna())
    col = pd.Series(np.where(later_first, y, x), index=scores_df.index).where(x.notna())
    off_grid = x.notna() & (~row.isin(order) | ~col.isin(order))  # residue not in the grid
    row, col = row.mask(off_grid), col.mask(off_grid)
    if verbose:
        n_equal = int((~is_swap & (n1 == n2)).sum())
        print(
            f"{len(scores_df)} pairs (collapse_il={collapse_il}): {int(is_swap.sum())} "
            f"first-two-residue swaps, {int((~is_swap).sum())} not a swap (of which "
            f"{n_equal} are the same peptide under this comparison), {int(off_grid.sum())} "
            f"with a residue outside the grid, {int(row.notna().sum())} plottable"
        )
    out = scores_df.copy()
    out["is_swap"] = is_swap
    out["swap_aa_row"] = row
    out["swap_aa_col"] = col
    return out


def ox_coverage_report(swap_df, verbose=True):
    """Check whether each Met-containing swap pair also has its oxidised twin in the data.

    Every Met-containing peptide is present in both its unmodified and its M[ox] form, so a
    swap pair carrying M ought to appear a second time with every M replaced by O — the
    ``X/M`` and ``X/M(ox)`` cells of the swap heatmap should therefore hold equal counts.
    Where they don't, the cause is upstream: the similarity search returned one form of the
    pair and not the other, even though both peptides exist at the same charge and identical
    m/z. This function measures that gap so the heatmap is not read as prevalence.

    Only pairs that are "pure" one way — containing M but no O, or O but no M — can be
    matched unambiguously, so mixed pairs (an oxidation sitting on a different Met in each
    peptide) are excluded. Pairs are canonicalised with ``tuple(sorted(...))``, so direction
    does not matter, and charge states are collapsed: the unit here is the peptide pair.

    Returns the orphaned pairs as a DataFrame (``peptide_1_O``, ``peptide_2_O``,
    ``swap_aa_row``, ``swap_aa_col``, ``direction``, ``expected_1``, ``expected_2``).
    """
    swaps = swap_df[swap_df["swap_aa_row"].notna() & swap_df["swap_aa_col"].notna()]
    seen = {
        tuple(sorted((a, b)))
        for a, b in zip(swaps["peptide_1_O"], swaps["peptide_2_O"])
    }
    rows, counts = [], {}
    for pair, grp in swaps.groupby(
        swaps[["peptide_1_O", "peptide_2_O"]].apply(
            lambda r: tuple(sorted(r)), axis=1
        )
    ):
        both = pair[0] + pair[1]
        has_m, has_o = "M" in both, "O" in both
        if has_m == has_o:  # neither Met at all, or a mixed M+O pair
            continue
        frm, to, direction = ("M", "O", "M->ox") if has_m else ("O", "M", "ox->M")
        expected = tuple(sorted((pair[0].replace(frm, to), pair[1].replace(frm, to))))
        first = grp.iloc[0]
        row, col = first["swap_aa_row"], first["swap_aa_col"]
        other = col if row in ("M", "O") else row  # the residue Met was swapped with
        key = (other, direction)
        matched = expected in seen
        counts[key] = counts.get(key, [0, 0])
        counts[key][0] += 1
        counts[key][1] += int(matched)
        if not matched:
            rows.append(
                {
                    "peptide_1_O": pair[0],
                    "peptide_2_O": pair[1],
                    "swap_aa_row": row,
                    "swap_aa_col": col,
                    "direction": direction,
                    "expected_1": expected[0],
                    "expected_2": expected[1],
                }
            )
    orphans = pd.DataFrame(
        rows,
        columns=["peptide_1_O", "peptide_2_O", "swap_aa_row", "swap_aa_col",
                 "direction", "expected_1", "expected_2"],
    )
    if verbose:
        residues = sorted({r for r, _ in counts}, key=lambda a: AA_ORDER.index(a)
                          if a in AA_ORDER else len(AA_ORDER))
        print("Met/M(ox) coverage of the swap pairs (unit: peptide pairs, charge collapsed)")
        print(f"{'X':>3}  {'X/M pairs':>10} {'ox twin':>8}  {'X/M(ox) pairs':>14} {'M twin':>7}")
        for aa in residues:
            m_tot, m_ok = counts.get((aa, "M->ox"), [0, 0])
            o_tot, o_ok = counts.get((aa, "ox->M"), [0, 0])
            print(f"{aa:>3}  {m_tot:>10} {m_ok:>8}  {o_tot:>14} {o_ok:>7}")
        tot_m = sum(v[0] for k, v in counts.items() if k[1] == "M->ox")
        ok_m = sum(v[1] for k, v in counts.items() if k[1] == "M->ox")
        tot_o = sum(v[0] for k, v in counts.items() if k[1] == "ox->M")
        ok_o = sum(v[1] for k, v in counts.items() if k[1] == "ox->M")
        print(
            f"\n{tot_m} M-only swap pairs, {tot_m - ok_m} with no oxidised twin in the data"
        )
        print(
            f"{tot_o} M(ox)-only swap pairs, {tot_o - ok_o} with no unmodified twin"
        )
        print(
            "Missing twins are pairs the similarity search never returned, not pairs that "
            "cannot exist — check a few in the peptide table before reading the counts as "
            "prevalence."
        )
    return orphans


def _sub_type_mask(annotated_df, sub_type):
    """Mask of analysable pairs for a substitution type. "I/L" selects pairs whose only real
    difference is I/L; any other value selects pairs with exactly that one real substitution.
    M<->O oxidation changes are allowed alongside in both cases."""
    base = annotated_df["analysable"]
    if sub_type == "I/L":
        return base & (annotated_df["n_sub"] == 0) & (annotated_df["n_il"] >= 1)
    return base & (annotated_df["n_sub"] == 1) & (annotated_df["sub_type"] == sub_type)


def substitution_summary(annotated_df, threshold=0.7):
    """Table of candidate vs indistinguishable pairs per substitution type (counting unit is
    pairs). One row per single-substitution type, plus an "I/L" row (I/L-only pairs) and a
    "multi" row (>=2 real substitutions) so nothing is dropped. Returns the DataFrame."""
    ana = annotated_df[annotated_df["analysable"]]
    n_skipped = len(annotated_df) - len(ana)
    def _row(sub_df):
        n_pairs = len(sub_df)
        n_indist = int((sub_df["score"] > threshold).sum())
        n_with_ox = int(sub_df["has_ox_change"].sum())
        return {"n_pairs": n_pairs, "n_indist": n_indist,
                "frac_indist": n_indist / n_pairs if n_pairs else np.nan,
                "n_with_ox": n_with_ox, "n_pure": n_pairs - n_with_ox,
                "mean_score": sub_df["score"].mean(), "median_score": sub_df["score"].median()}
    rows = {st: _row(grp) for st, grp in ana[ana["n_sub"] == 1].groupby("sub_type")}
    il_df = ana[(ana["n_sub"] == 0) & (ana["n_il"] >= 1)]
    if len(il_df):
        rows["I/L"] = _row(il_df)
    multi_df = ana[ana["n_sub"] >= 2]
    if len(multi_df):
        rows["multi"] = _row(multi_df)
    summary = pd.DataFrame(rows).T.sort_values("n_pairs", ascending=False)
    summary[["n_pairs", "n_indist", "n_with_ox", "n_pure"]] = summary[
        ["n_pairs", "n_indist", "n_with_ox", "n_pure"]].astype(int)
    print(f"{len(ana)} analysable pairs ({n_skipped} skipped: unequal length or unhandled mod)")
    print(summary)
    return summary


def substitution_report(annotated_df, sub_type, threshold=0.7):
    """Focused printout for one substitution type ("A/S", "I/L", ...): total vs
    indistinguishable pairs, the pure vs +oxidation split, and the distribution of residue
    distances between the substitution site and the M[ox] site. Returns the filtered df."""
    sub_df = annotated_df[_sub_type_mask(annotated_df, sub_type)]
    n_pairs = len(sub_df)
    indist = sub_df[sub_df["score"] > threshold]
    print(f"=== {sub_type} substitution ===")
    print(f"{n_pairs} candidate pairs, {len(indist)} indistinguishable (score > {threshold})")
    if n_pairs == 0:
        return sub_df
    with_ox = sub_df[sub_df["has_ox_change"]]
    pure = sub_df[~sub_df["has_ox_change"]]
    print(f"  pure (no M[ox] change): {len(pure)} pairs, "
          f"{int((pure['score'] > threshold).sum())} indistinguishable")
    print(f"  with M[ox] change:      {len(with_ox)} pairs, "
          f"{int((with_ox['score'] > threshold).sum())} indistinguishable")
    if len(with_ox):
        print("  residues between substitution site and M[ox] site (all with-ox pairs):")
        print(with_ox["ox_sub_distance"].value_counts().sort_index().to_string())
        ox_indist = with_ox[with_ox["score"] > threshold]
        if len(ox_indist):
            print("  ...and among the indistinguishable with-ox pairs:")
            print(ox_indist["ox_sub_distance"].value_counts().sort_index().to_string())
    return sub_df
