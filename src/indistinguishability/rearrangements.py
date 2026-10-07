"""Parsing of the ``rearrangement <aas> at pos <p,q,...>`` labels that `get_aa_changes`
writes into the ``aa_substitution`` column, plus position-based selection on top of them.

`get_aa_changes` reports *every* mismatch position of a same-composition pair in one label,
so "a swap at positions 1 and 2" and "a swap at 1,2 plus an oxidation moving between
positions 5 and 6" both mention ``pos 1,2``. Selecting the former with a substring test
therefore also pulls in the latter — use `is_position_rearrangement`, which compares the
parsed position list exactly.
"""

import re

_REARRANGEMENT_RE = re.compile(
    r"rearrangement\s+([A-Za-z]+)\s+at\s+pos\s+([\d,\s]+)"
)


def parse_rearrangement(label):
    """``"[rearrangement FI at pos 1,2]"`` -> ``{'aas': 'FI', 'positions': [1, 2],
    'n_positions': 2}``. Returns None for non-rearrangement labels (e.g. ``"complex
    (AS->GT)"``, ``"identical"``).

    Accepts the list form `get_aa_changes` returns as well as a plain string, since the
    column holds both.
    """
    m = _REARRANGEMENT_RE.search(str(label))
    if not m:
        return None
    positions = [int(p) for p in re.findall(r"\d+", m.group(2))]
    return {"aas": m.group(1), "positions": positions, "n_positions": len(positions)}


def is_position_rearrangement(label, positions=(1, 2)):
    """True when `label` is a rearrangement at *exactly* `positions` and nothing else.

    This is the strict replacement for ``"pos 1,2" in label``: that substring test also
    matches ``"rearrangement APMO at pos 1,2,5,6"``, a first-two-residue swap that
    additionally moves an oxidation between two methionines. Those pairs are not
    position-1,2 swaps and are rejected downstream anyway, so letting them into the dataset
    only makes the row count disagree with the analysis.
    """
    parsed = parse_rearrangement(label)
    return parsed is not None and parsed["positions"] == list(positions)


def add_position_columns(df, column="aa_substitution"):
    """Add ``positions`` / ``n_positions`` parsed out of `column`. Both are None for rows
    whose label is not a rearrangement."""
    df = df.copy()
    parsed = df[column].apply(parse_rearrangement)
    df["positions"] = parsed.apply(lambda d: d["positions"] if d else None)
    df["n_positions"] = parsed.apply(lambda d: d["n_positions"] if d else None)
    return df


def add_swap_positions(df, column="aa_substitution"):
    """Return only the two-position rearrangements of `df`, with ``pos_a`` < ``pos_b`` the
    two positions that swapped (1-based, from the N-terminus) and ``swap_aas`` the residues
    involved. Rows with any other label (complex, identical, 3+ positions) are dropped."""
    parsed = df[column].apply(parse_rearrangement)
    keep = parsed.apply(lambda d: d is not None and d["n_positions"] == 2)
    out = df[keep].copy()
    out["pos_a"] = parsed[keep].apply(lambda d: min(d["positions"]))
    out["pos_b"] = parsed[keep].apply(lambda d: max(d["positions"]))
    out["swap_aas"] = parsed[keep].apply(lambda d: d["aas"])
    return out
