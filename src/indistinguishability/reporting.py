"""Textual summaries printed from an enriched scores DataFrame."""

from .loading import remove_MO_IL, remove_IL, remove_MO


def reporting_values(scores_df):
    scores_df["contains_O_1"] = scores_df["peptide_1_O"].str.contains("O")
    scores_df["contains_O_2"] = scores_df["peptide_2_O"].str.contains("O")
    O_mismatch_df = scores_df[
        scores_df["contains_O_1"] != scores_df["contains_O_2"]
    ]
    print(f"{len(scores_df)} total number of pairs")
    print(f"In {len(O_mismatch_df)} of the pairs one of the two contains an oxidation")
    length_mismatch = scores_df[scores_df["length_1"] != scores_df["length_2"]]
    print(f"{len(length_mismatch)} pairs between peptides with different lengths")
    charge_mismatch = scores_df[scores_df["charge_diff"] != 0]
    print(f"{len(charge_mismatch)} pairs between peptides with different charges")
    unique_peptides = set(scores_df["peptide_1_O"]).union(set(scores_df["peptide_2_O"]))
    O_count = sum(1 for p in unique_peptides if "O" in p)
    print(
        f"{len(unique_peptides)} unique peptides, of which {O_count} contain an oxidation"
    )
    print(f"Top AA substitutions are:")
    print(f"{scores_df['aa_substitution'].value_counts().head(10)}")


def indistinguishable_peptides_report(scores_df, input_peptides_df):
    filtered_scores_df = remove_MO_IL(scores_df)
    no_IL_scores_df = remove_IL(scores_df)
    no_MO_scores_df = remove_MO(scores_df)
    indistinguishable_peptides = set(scores_df["peptide_1_O"]).union(set(scores_df["peptide_2_O"]))
    unique_indistinguishable_peptides = set(filtered_scores_df["peptide_1_O"]).union(set(filtered_scores_df["peptide_2_O"]))
    no_IL_peptides = set(no_IL_scores_df["peptide_1_O"]).union(set(no_IL_scores_df["peptide_2_O"]))
    no_MO_peptides = set(no_MO_scores_df["peptide_1_O"]).union(set(no_MO_scores_df["peptide_2_O"]))
    input_peptides = set(input_peptides_df[0])
    print(f"{len(input_peptides)} input peptide")
    print(f"{len(indistinguishable_peptides)} indistinguishable peptide")
    print(f"{len(no_IL_peptides)} indistinguishable peptide without  I/L")
    print(f"{len(no_MO_peptides)} indistinguishable peptide without  M <-> O")
    print(f"{len(unique_indistinguishable_peptides)} indistinguishable peptide without  M <-> O and I/L")
