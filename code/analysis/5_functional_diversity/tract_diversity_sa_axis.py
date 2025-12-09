# ------------------------------------------------------------------------------------------------
#### Plot Functional Diversity vs S-A Axis Range ####
# ------------------------------------------------------------------------------------------------

# This script plots Gini coefficients against S-A range for tracts
# Uses pre-computed S-A ranges from prep_sa_axis.py and Gini coefficients from tract_gini_coefficients.py

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# Set up directories
root_dir = '/Users/joelleba/PennLINC/tractmaps'
results_dir = f'{root_dir}/results/tract_functional_diversity/gini_coefficients' 
output_dir = f'{root_dir}/results/tract_functional_diversity/gini_sa_axis_figures'

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
# --- Plot full tract set with Gini coefficients ---
# ------------------------------------------------------------------------------------------------

# Plot all tracts together
all_tracts = diversity_score_df['Tract'].tolist()
plot_df = diversity_score_df[diversity_score_df['Tract'].isin(all_tracts)].copy()
plot_df = plot_df.dropna(subset=['SA_Range', tract_diversity_score])

if plot_df.empty:
    print("No valid tracts found for analysis")
    exit(1)

# Prepare data for plotting
x_data = plot_df['SA_Range'].values
y_data = plot_df[tract_diversity_score].values
tract_labels = [tract.replace('_', ' ') for tract in plot_df['Tract'].values]

# Calculate correlation
corr_result = tm_utils.perm_corr_test(x_data, y_data, n_permutations=10000, method='spearman', 
                                      alternative='two-sided', random_state=42)
r_value = corr_result['observed_corr']
p_value = corr_result['p_value']

# Save correlation results to CSV
results_summary = pd.DataFrame({
    'analysis': ['full_tract_set_gini_coefficient'],
    'observed_correlation': [r_value],
    'p_value': [p_value],
    'n_samples': [corr_result['n_samples']],
    'n_permutations': [corr_result['n_permutations']],
    'method': [corr_result['method']],
    'alternative': [corr_result['alternative']],
    'score_type': [tract_diversity_score]
})
results_summary.to_csv(f'{output_dir}/correlation_results_gini_coefficient.csv', index=False)

# Create output path
output_path = f'{output_dir}/{tract_diversity_score.lower()}_vs_sa_range.svg'

# Plot using tm_utils function
tm_utils.plot_correlation(
    x=x_data, 
    y=y_data,
    corr_value=r_value,
    p_value=p_value,
    x_label='S-A range',
    y_label=tract_diversity_score_label,
    color_scheme=bppy_cmap,
    reverse_colormap=False,
    colorbar='separate_figure',
    colorbar_label=tract_diversity_score_label,
    color_by='y',
    point_labels=tract_labels,
    text_box_position='top_right',
    point_size=30,
    point_alpha=0.8,
    regression_line=True,
    output_path=output_path,
    dpi=300,
    figure_size_mm=(70, 60)
)

print(f"Generated plot for full tract set ({len(all_tracts)} tracts) with {tract_diversity_score_label.lower()}")
print(f"Saved correlation results to: correlation_results_gini_coefficient.csv")