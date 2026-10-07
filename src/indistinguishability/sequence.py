"""Sequence-level helpers: proline position and amino-acid change classification."""

from collections import Counter


def first_p_from_right(s):
    for i, char in enumerate(reversed(s), start=1):
        if char == "P":
            return i
    return -1  # if no 'P' found


def aa_composition_diff(pep1, pep2):
    """Return AA composition difference between two peptides."""
    c1, c2 = Counter(pep1), Counter(pep2)
    lost = "".join(sorted((c1 - c2).elements()))
    gained = "".join(sorted((c2 - c1).elements()))
    if not lost and not gained:
        return None
    return f"{lost}->{gained}"


def aa_changes(pep1, pep2):
    mismatches = [(i + 1, a, b) for i, (a, b) in enumerate(zip(pep1, pep2)) if a != b]
    if not mismatches:
        return "identical"
    used = set()
    results = []
    for i, (pos1, a, b) in enumerate(mismatches):
        if i in used:
            continue
        for j, (pos2, c, d) in enumerate(mismatches):
            if j <= i or j in used:
                continue
            if a == d and b == c:
                label = "".join(sorted([a, b]))
                results.append(f"{label} {pos1} {pos2}")
                used.update([i, j])
                break
    return ", ".join(results) if results else "identical"


def get_aa_changes(pep1, pep2):
    """Compare two peptides and return what changed between them.

    Three shapes come out, and callers branch on them:
      * ``"identical"`` (str) — equal once I and L are collapsed,
      * ``"complex (<lost>-><gained>)"`` (str) — the compositions differ,
      * ``["rearrangement <aas> at pos <p,q,...>"]`` (list) — same composition, so the
        residues have only moved.

    The rearrangement label lists *every* mismatch position in one string and takes `aas`
    from peptide 1's residues at those positions (sorted, deduplicated). It does not try to
    pair mismatches up into individual transpositions — `annotate_terminal_swap` in
    `substitutions.py` does that properly, positionally, for the first two residues.

    Note the normalisation direction: L->I here, while `loading.py` and
    `annotate_terminal_swap` normalise I->L. Both collapse the isobaric pair, so consumers of
    this column re-normalise (see `plot_swap_aa_vs_input`); don't flip one to match the other
    without checking every caller.

    `remove_MO` keys off the substring ``"MO"``, which only appears here when peptide 1
    carries both an unmodified and an oxidised Met among its mismatch positions — i.e. when
    the oxidation has moved between two methionines.
    """
    np1 = pep1.replace("L", "I")
    np2 = pep2.replace("L", "I")
    # Equal sorted residues == no normalised composition difference; much cheaper than
    # building Counters, and this runs once per pair on tens of millions of pairs.
    if sorted(np1) != sorted(np2):
        comp_diff = aa_composition_diff(pep1, pep2)
        return f"complex ({comp_diff})"
    # Same composition -> the residues moved rather than changed.
    mismatches = [(i + 1, a, b) for i, (a, b) in enumerate(zip(np1, np2)) if a != b]
    if not mismatches:
        return "identical"
    positions = ",".join(str(p) for p, _, _ in mismatches)
    aas = "".join(sorted(set(a for _, a, _ in mismatches)))
    return [f"rearrangement {aas} at pos {positions}"]


def first_aa_position(s, aa, terminus="N"):
    """1-based position of the first `aa` in `s`, counted from the N- or C-terminus;
    -1 if absent. ``first_aa_position(s, "P", "C")`` equals ``first_p_from_right(s)``."""
    if terminus not in ("N", "C"):
        raise ValueError(f"terminus must be 'N' or 'C', not {terminus!r}")
    idx = (s if terminus == "N" else s[::-1]).find(aa)
    return idx + 1 if idx >= 0 else -1


def add_aa_position(df, aa, terminus="N", column=None):
    """Return a copy of `df` with ``aa_pos_{aa}_{terminus}`` (or `column`): the nearest
    occurrence of residue `aa` to `terminus` across *both* peptides of the pair (from
    ``peptide_{1,2}_O``, so oxidised Met is ``O``).

    The pair value is the smaller of the two per-peptide positions among the peptides that
    contain `aa`, and -1 only when neither does. (The `find_aa_position` this replaces took a
    plain minimum, so a residue missing from just one peptide collapsed the pair to -1.)
    """
    column = column or f"aa_pos_{aa}_{terminus}"
    out = df.copy()
    pos = [
        [first_aa_position(str(p), aa, terminus) for p in out[f"peptide_{side}_O"]]
        for side in ("1", "2")
    ]
    out[column] = [
        min((v for v in (a, b) if v > 0), default=-1) for a, b in zip(*pos)
    ]
    return out
