"""Indistinguishable-peptides loading and plotting toolkit.

    import indistinguishability as ind
    df = ind.load_scores(scores_tsv="scores.tsv")
    # or: df = ind.load_scores(peptides="peptides.tsv", npy="scores.npy")
    style = ind.get_style("talk", cmap_mean="magma")
    fig, ax = ind.plot_aa_swap_heatmap(df, style=style, output_path="figures")
"""

from .sequence import (
    first_p_from_right,
    first_aa_position,
    add_aa_position,
    aa_composition_diff,
    aa_changes,
    get_aa_changes,
)
from .rearrangements import (
    parse_rearrangement,
    is_position_rearrangement,
    add_position_columns,
    add_swap_positions,
)
from .substitutions import (
    annotate_substitutions,
    annotate_terminal_swap,
    substitution_summary,
    substitution_report,
    ox_coverage_report,
    aa_grid_order,
    aa_tick_labels,
    AA_GROUPS,
    AA_ORDER,
    AA_LABELS,
)
from .loading import (
    normalise_mods,
    VARIABLE_MOD_REPLACEMENTS,
    FIXED_MOD_REPLACEMENTS,
    VERBOSE_RENAMES,
    load_scores,
    combine_pep_score,
    combine_pep_scores_wide,
    combine_filtered_pep_score,
    combine_filtered_pep_score_add_I,
    read_results,
    lookup_peptide_rows,
    annotate_charge,
    remove_MO_IL,
    remove_MO,
    remove_IL,
)
from .reporting import (
    reporting_values,
    indistinguishable_peptides_report,
)
from .style import PlotStyle, PRESETS, get_style
from .plots import (
    plot_score_distribution,
    plot_levenshtein_distribution,
    plot_irt_diff_histogram,
    plot_peptide_length,
    plot_levenshtein_violin,
    plot_feature_hexbin,
    plot_irt_ccs_hexbin,
    plot_binned_mean,
    plot_score_by_combo,
    plot_heatmap,
    plot_aa_position_heatmap,
    plot_aa_position_violin,
    plot_aa_swap_heatmap,
    plot_swap_position_heatmap,
    plot_ox_sub_distance,
    plot_swap_aa_vs_input,
)
from .spectra import (
    get_model,
    predict_spectrum,
    koina_to_spectrum_utils,
    plot_mirror,
)
