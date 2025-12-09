# ------------------------------------------------------------------------------------------------
#### Sensitivity Analysis: Functional Diversity vs S-A Axis Range by Tract Type ####
# ------------------------------------------------------------------------------------------------
# This script plots Gini coefficients against S-A range for tracts
# separately for projection and association tracts
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# Set up directories
root_dir = '/Users/joelleba/PennLINC/tractmaps'
results_dir = f'{root_dir}/results/tract_functional_diversity/gini_coefficients' 
output_dir = f'{root_dir}/results/tract_functional_diversity/gini_sa_axis_figures/sensitivity'

# make output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# get custom colormaps
_, _, _, _, _, bppy_cmap = tm_utils.make_colormaps()

# Set connection threshold
connection_threshold = 0.5

# set fontsize for all plots
plt.rcParams.update({'font.size': 18})

print("Using Gini coefficients for tract type sensitivity analysis")

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load tract names
tract_types = pd.read_excel(f'{root_dir}/data/derivatives/tract_names/abbreviations.xlsx')

# Load Gini coefficients
tract_diversity_score = 'Gini_Coefficient'
tract_diversity_score_label = 'Gini coefficient'
diversity_score_path = os.path.join(results_dir, 'tract_gini_term_scores.csv') 
diversity_score_df = pd.read_csv(diversity_score_path)
diversity_score_df.rename(columns={'Tract': 'Tract_Long_Name'}, inplace=True)

# Load S-A ranges
sa_ranges_path = f'{root_dir}/data/derivatives/tracts/tracts_sa_axis/tract_sa_axis_ranges_thresh{int(connection_threshold * 100)}.csv'
sa_ranges_df = pd.read_csv(sa_ranges_path)

# ------------------------------------------------------------------------------------------------
# --- Merge data ---
# ------------------------------------------------------------------------------------------------

# Merge 
diversity_score_df = diversity_score_df.merge(
    tract_types[['Tract_Long_Name', 'Tract', 'Type']],
    on='Tract_Long_Name', how='left'
)

# Merge tract names, gini coefficients and S-A ranges
diversity_score_df = diversity_score_df.merge(
    sa_ranges_df[['Tract', 'SA_Range']],
    on='Tract', how='left'
)

# ------------------------------------------------------------------------------------------------
# --- Get tract subgroups by type ---
# ------------------------------------------------------------------------------------------------

# Get projection and association tracts
projection_tracts = diversity_score_df[diversity_score_df['Type'] == 'Projection']['Tract'].tolist()
association_tracts = diversity_score_df[diversity_score_df['Type'] == 'Association']['Tract'].tolist()

print(f"Found {len(projection_tracts)} projection tracts and {len(association_tracts)} association tracts")

# ------------------------------------------------------------------------------------------------
# --- Plot by tract type ---
# ------------------------------------------------------------------------------------------------

# Plot projection tracts
# Prepare data for projection tracts
proj_plot_df = diversity_score_df[diversity_score_df['Tract'].isin(projection_tracts)].copy()
proj_plot_df = proj_plot_df.dropna(subset=['SA_Range', tract_diversity_score])

if not proj_plot_df.empty:
    x_data_proj = proj_plot_df['SA_Range'].values
    y_data_proj = proj_plot_df[tract_diversity_score].values
    tract_labels_proj = [tract.replace('_', ' ') for tract in proj_plot_df['Tract'].values]

    # Calculate correlation for projection tracts
    corr_result_proj = tm_utils.perm_corr_test(x_data_proj, y_data_proj, n_permutations=10000, method='spearman', 
                                              alternative='two-sided', random_state=42)
    r_value_proj = corr_result_proj['observed_corr']
    p_value_proj = corr_result_proj['p_value']

    # Create output path for projection tracts
    output_path_proj = f'{output_dir}/{tract_diversity_score.lower()}_vs_sa_range_projection.svg'

    # Plot projection tracts using tm_utils function
    tm_utils.plot_correlation(
        x=x_data_proj, 
        y=y_data_proj,
        corr_value=r_value_proj,
        p_value=p_value_proj,
        x_label='S-A Range',
        y_label=tract_diversity_score_label,
        color_scheme=bppy_cmap,
        reverse_colormap=False,
        colorbar='same_plot',
        colorbar_label=tract_diversity_score_label,
        color_by='y',
        point_labels=tract_labels_proj,
        text_box_position='top_right',
        figure_size=(7, 8),
        point_size=100,
        point_alpha=0.8,
        regression_line=True,
        title='Projection Tracts',
        output_path=output_path_proj,
        dpi=300
    )

# Plot association tracts
# Prepare data for association tracts
assoc_plot_df = diversity_score_df[diversity_score_df['Tract'].isin(association_tracts)].copy()
assoc_plot_df = assoc_plot_df.dropna(subset=['SA_Range', tract_diversity_score])

if not assoc_plot_df.empty:
    x_data_assoc = assoc_plot_df['SA_Range'].values
    y_data_assoc = assoc_plot_df[tract_diversity_score].values
    tract_labels_assoc = [tract.replace('_', ' ') for tract in assoc_plot_df['Tract'].values]

    # Calculate correlation for association tracts
    corr_result_assoc = tm_utils.perm_corr_test(x_data_assoc, y_data_assoc, n_permutations=10000, method='spearman', 
                                               alternative='two-sided', random_state=42)
    r_value_assoc = corr_result_assoc['observed_corr']
    p_value_assoc = corr_result_assoc['p_value']

    # Create output path for association tracts
    output_path_assoc = f'{output_dir}/{tract_diversity_score.lower()}_vs_sa_range_association.svg'

    # Plot association tracts using tm_utils function
    tm_utils.plot_correlation(
        x=x_data_assoc, 
        y=y_data_assoc,
        corr_value=r_value_assoc,
        p_value=p_value_assoc,
        x_label='S-A Range',
        y_label=tract_diversity_score_label,
        color_scheme=bppy_cmap,
        reverse_colormap=False,
        colorbar='same_plot',
        colorbar_label=tract_diversity_score_label,
        color_by='y',
        point_labels=tract_labels_assoc,
        text_box_position='top_right',
        figure_size=(7, 8),
        point_size=100,
        point_alpha=0.8,
        regression_line=True,
        title='Association Tracts',
        output_path=output_path_assoc,
        dpi=300
    )

# Save correlation results to CSV
results_summary = pd.DataFrame({
    'analysis': ['projection_tracts_gini', 'association_tracts_gini'],
    'observed_correlation': [r_value_proj, r_value_assoc],
    'p_value': [p_value_proj, p_value_assoc],
    'n_samples': [corr_result_proj['n_samples'], corr_result_assoc['n_samples']],
    'n_permutations': [corr_result_proj['n_permutations'], corr_result_assoc['n_permutations']],
    'method': [corr_result_proj['method'], corr_result_assoc['method']],
    'alternative': [corr_result_proj['alternative'], corr_result_assoc['alternative']],
    'score_type': [tract_diversity_score, tract_diversity_score],
    'n_tracts': [len(projection_tracts), len(association_tracts)]
})
results_summary.to_csv(f'{output_dir}/correlation_results_gini_tract_types.csv', index=False)

print(f"Generated plots for projection ({len(projection_tracts)} tracts) and association ({len(association_tracts)} tracts) tract types with {tract_diversity_score_label.lower()}")
print(f"Saved correlation results to: correlation_results_gini_tract_types.csv")
