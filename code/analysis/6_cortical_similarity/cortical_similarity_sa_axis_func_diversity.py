# ------------------------------------------------------------------------------------------------
# --- Cortical similarity plotting ---
# ------------------------------------------------------------------------------------------------
# This script examines the relationship between tract mean cortical similarity vs S-A range and Gini coefficient. It also visualizes two example tracts.

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
from utils.tract_visualizer import TractVisualizer

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# root directory
root_dir = '/Users/joelleba/PennLINC/tractmaps'
data_dir = f'{root_dir}/data/derivatives/tracts/tracts_cortical_similarity/'
output_dir = f'{root_dir}/results/cortical_similarity/'

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")


# set parameters
connection_threshold = 0.5

# get custom colormaps
warm_cmap, _, _, _, _, bppy_cmap = tm_utils.make_colormaps()


# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load tract information and S-A ranges
tract_types = pd.read_excel(f'{root_dir}/data/derivatives/tract_names/abbreviations.xlsx')

# Load S-A ranges
sa_ranges_df = pd.read_csv(f'{root_dir}/data/derivatives/tracts/tracts_sa_axis/tract_sa_axis_ranges_thresh{int(connection_threshold * 100)}.csv')

# Load Gini coefficients
gini_df = pd.read_csv(f'{root_dir}/results/tract_functional_diversity/gini_coefficients/tract_gini_term_scores.csv').drop('Hemisphere', axis=1)

# Map gini_df long tract names to short tract names
gini_df['Tract'] = gini_df['Tract'].map(tract_types.set_index('Tract_Long_Name')['Tract'])

# Load mean cortical similarities
mean_similarities_df = pd.read_csv(f'{data_dir}/tracts_mean_cortical_similarity.csv')

# Convert to dictionary
means = dict(zip(mean_similarities_df['Tract'], mean_similarities_df['Mean_Cortical_Similarity']))


# ------------------------------------------------------------------------------------------------
# --- Create dataframe with tract information, S-A ranges, Gini coefficients, and mean similarities ---
# ------------------------------------------------------------------------------------------------

tract_data = []
for tract in means.keys():
    # Get tract info
    tract_info = tract_types[tract_types['Tract'] == tract]
    if not tract_info.empty:
        tract_data.append({
            'Tract': tract,
            'Type': tract_info.iloc[0]['Type'],
            'Hemisphere': tract_info.iloc[0]['Hemisphere'],
            'Mean_Similarity': means[tract]
        })

tract_df = pd.DataFrame(tract_data)

# Merge with S-A ranges
tract_df = tract_df.merge(sa_ranges_df[['Tract', 'SA_Range']], on='Tract', how='left')

# Merge with Gini coefficients (both use short tract names)
tract_df = tract_df.merge(gini_df[['Tract', 'Gini_Coefficient']], on='Tract', how='left')


# ------------------------------------------------------------------------------------------------
# --- Plot cortical similarity analyses ---
# ------------------------------------------------------------------------------------------------

print("Creating mean cortical similarity vs S-A range plot...")

# create output directory for cortical similarity analyses
output_correlations_dir = f'{output_dir}/correlations'
os.makedirs(output_correlations_dir, exist_ok=True)

# Get all tracts
all_tracts = tract_df['Tract'].unique().tolist()

# Plot 1: Mean cortical similarity vs S-A range
# Prepare data for SA range analysis
sa_plot_df = tract_df[tract_df['Tract'].isin(all_tracts)].copy()
sa_plot_df = sa_plot_df.dropna(subset=['SA_Range', 'Mean_Similarity'])

if not sa_plot_df.empty:
    x_data_sa = sa_plot_df['SA_Range'].values
    y_data_sa = sa_plot_df['Mean_Similarity'].values
    tract_labels_sa = [tract.replace('_', ' ') for tract in sa_plot_df['Tract'].values]

    # Calculate correlation for SA range analysis
    corr_result_sa = tm_utils.perm_corr_test(x_data_sa, y_data_sa, n_permutations=10000, method='spearman', 
                                            alternative='two-sided', random_state=42)
    r_value_sa = corr_result_sa['observed_corr']
    p_value_sa = corr_result_sa['p_value']


    # Create output path for SA range analysis
    output_path_sa = f'{output_correlations_dir}/mean_cortical_similarity_vs_sa_range.svg'

    # Plot SA range analysis using tm_utils function
    tm_utils.plot_correlation(
        x=x_data_sa, 
        y=y_data_sa,
        corr_value=r_value_sa,
        p_value=p_value_sa,
        x_label='S-A range',
        y_label='Mean cortical similarity',
        color_scheme=bppy_cmap,
        reverse_colormap=False,
        colorbar='separate_figure',
        colorbar_label='Mean cortical similarity',
        colorbar_filename='mean_cortical_similarity.svg',
        color_by='y',
        point_labels=tract_labels_sa,
        text_box_position='top_right',
        point_size=30,
        point_alpha=0.8,
        regression_line=True,
        colorbar_tick_interval=0.2,
        output_path=output_path_sa,
        dpi=300,
        figure_size_mm=(85, 80)
    )

print("Creating mean cortical similarity vs Gini coefficient plot...")

# Plot 2: Mean cortical similarity vs Gini coefficient
# Prepare data for Gini coefficient analysis
gini_plot_df = tract_df[tract_df['Tract'].isin(all_tracts)].copy()
gini_plot_df = gini_plot_df.dropna(subset=['Gini_Coefficient', 'Mean_Similarity'])

if not gini_plot_df.empty:
    x_data_gini = gini_plot_df['Gini_Coefficient'].values
    y_data_gini = gini_plot_df['Mean_Similarity'].values
    tract_labels_gini = [tract.replace('_', ' ') for tract in gini_plot_df['Tract'].values]

    # Calculate correlation for Gini coefficient analysis
    corr_result_gini = tm_utils.perm_corr_test(x_data_gini, y_data_gini, n_permutations=10000, method='spearman', 
                                              alternative='two-sided', random_state=42)
    r_value_gini = corr_result_gini['observed_corr']
    p_value_gini = corr_result_gini['p_value']

    # Create output path for Gini coefficient analysis
    output_path_gini = f'{output_correlations_dir}/mean_cortical_similarity_vs_gini.svg'

    # Plot Gini coefficient analysis using tm_utils function
    tm_utils.plot_correlation(
        x=x_data_gini, 
        y=y_data_gini,
        corr_value=r_value_gini,
        p_value=p_value_gini,
        x_label='Gini coefficient',
        y_label='Mean cortical similarity',
        color_scheme=bppy_cmap,
        reverse_colormap=False,
        colorbar='separate_figure',
        colorbar_label='Mean cortical similarity',
        colorbar_filename='mean_cortical_similarity.svg',
        color_by='y',
        point_labels=tract_labels_gini,
        text_box_position='top_left',
        point_size=30,
        point_alpha=0.8,
        regression_line=True,
        colorbar_tick_interval=0.2,
        output_path=output_path_gini,
        dpi=300,
        figure_size_mm=(85, 80)
    )

# Save correlation results to CSV
results_summary = pd.DataFrame({
    'analysis': ['cortical_similarity_vs_sa_range', 'cortical_similarity_vs_gini'],
    'observed_correlation': [r_value_sa, r_value_gini],
    'p_value': [p_value_sa, p_value_gini],
    'n_samples': [corr_result_sa['n_samples'], corr_result_gini['n_samples']],
    'n_permutations': [corr_result_sa['n_permutations'], corr_result_gini['n_permutations']],
    'method': [corr_result_sa['method'], corr_result_gini['method']],
    'alternative': [corr_result_sa['alternative'], corr_result_gini['alternative']],
    'x_variable': ['SA_Range', 'Gini_Coefficient'],
    'y_variable': ['Mean_Cortical_Similarity', 'Mean_Cortical_Similarity']
})
results_summary.to_csv(f'{output_correlations_dir}/correlation_results_cortical_similarity.csv', index=False)

# Create tract visualizations using TractVisualizer for two example tracts
tract_output_dir = os.path.join(output_dir, 'tract_visualizations')
example_tracts = ['VOF_left', 'IFOF_left']
viz = TractVisualizer(root_dir=root_dir,
                    output_dir=tract_output_dir)
viz.visualize_tracts(tract_list=example_tracts, 
                    single_color='#626bda', 
                    plot_mode='iterative'
                    )
                    
print(f"Generated plots for full tract set ({len(all_tracts)} tracts) with cortical similarity analyses")
print(f"Saved correlation results to: correlation_results_cortical_similarity.csv")
print(f"\nAll plots saved to: {output_dir}")
print(f"Results directory: {data_dir}")
