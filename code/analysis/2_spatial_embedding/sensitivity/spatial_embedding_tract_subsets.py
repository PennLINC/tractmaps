#------------------------------------------------------------------------------------------------
# --- Sensitivity Analysis: Spatial embedding by tract type ---
#------------------------------------------------------------------------------------------------

# This script plots the relationship between the S-A range of tracts and their mean euclidean distance,
# separately for projection and association tracts.

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

# Load packages
import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# ------------------------------------------------------------------------------------------------
# --- Set inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# Set paths
root = '/Users/joelleba/PennLINC/tractmaps'
data_root = f'{root}/data/derivatives/tracts'
tracts_dir = f'{root}/data/derivatives/tracts'
euc_dist_dir = f'{root}/data/derivatives/tracts/tracts_distances'

# create results directory if it doesn't yet exist
results_dir = f'{root}/results/spatial_embedding/sensitivity'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Folder '{results_dir}' created.")
else:
    print(f"Folder '{results_dir}' already exists.")

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Create threshold suffix for filenames
thresh_suffix = f'_thresh{int(tract_threshold * 100)}'

# set fontsize for all plots
plt.rcParams.update({'font.size': 18})

# ------------------------------------------------------------------------------------------------
# --- Load tract distances ---
# ------------------------------------------------------------------------------------------------

# Load pre-computed Euclidean tract distances
tract_means = pd.read_csv(f'{euc_dist_dir}/tract_euclidean_distances_means.csv')
tract_dist = pd.read_csv(f'{euc_dist_dir}/tract_euclidean_distances_pairwise.csv')
distance_column = 'Mean_Euclidean_Distance'
distance_label = 'Euclidean distance'
distance_filename = 'euclidean'

print(f"Using Euclidean distances for tract subset analysis")

# get custom colormaps
warm_cmap, tract_cmap, categ_warm, cool_warm_cmap, categ_cool_warm, fds_cmap = tm_utils.make_colormaps()

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load data
tractdata = pd.read_csv(f'{data_root}/tracts_probabilities/tracts_probabilities.csv')
sa_data = pd.read_csv(f'{data_root}/tracts_sa_axis/tract_sa_axis_regionwise{thresh_suffix}.csv')
sa_ranges = pd.read_csv(f'{data_root}/tracts_sa_axis/tract_sa_axis_ranges{thresh_suffix}.csv')
sa_axis = pd.read_csv(f'{root}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv')

# ------------------------------------------------------------------------------------------------
# --- Get sorted tracts by type ---
# ------------------------------------------------------------------------------------------------

# Get sorted tracts for each hemisphere and type
projection_tracts = tract_means[tract_means['Type'] == 'Projection'].sort_values(distance_column, ascending=False)['Tract'].tolist()
association_tracts = tract_means[tract_means['Type'] == 'Association'].sort_values(distance_column, ascending=False)['Tract'].tolist()

print(f"Found {len(projection_tracts)} projection tracts and {len(association_tracts)} association tracts")

# ------------------------------------------------------------------------------------------------
# --- Plot by tract type ---
# ------------------------------------------------------------------------------------------------

# Plot projection tracts
# Prepare data for projection tracts
tract_sa_ranges_proj = []
tract_dist_means_proj = []
tract_labels_proj = []

# Use pre-computed distance means and get S-A range for each projection tract
for tract in projection_tracts:
    # Check if tract exists in pre-computed data
    if tract not in tract_means['Tract'].values:
        print(f"Warning: Tract {tract} not found in pre-computed data, skipping...")
        continue
        
    # Get pre-computed S-A range for this tract
    tract_sa_range = sa_ranges[sa_ranges['Tract'] == tract]['SA_Range'].iloc[0]
    
    # Skip if S-A range is NaN (no regions connected)
    if pd.isna(tract_sa_range):
        continue
    
    # Get pre-computed mean distance for this tract
    tract_distance_mean = tract_means[tract_means['Tract'] == tract][distance_column].iloc[0]
        
    tract_sa_ranges_proj.append(tract_sa_range)
    tract_dist_means_proj.append(tract_distance_mean)
    tract_labels_proj.append(tract.replace('_', ' '))  # Format names for display

# Convert to numpy arrays
x_data_proj = np.array(tract_sa_ranges_proj)
y_data_proj = np.array(tract_dist_means_proj)

# Calculate correlation for projection tracts
corr_result_proj = tm_utils.perm_corr_test(x_data_proj, y_data_proj, n_permutations=10000, method='spearman', 
                                          alternative='two-sided', random_state=42)
r_value_proj = corr_result_proj['observed_corr']
p_value_proj = corr_result_proj['p_value']

# Create output path for projection tracts
output_path_proj = f'{results_dir}/sa_range_vs_{distance_filename}_distance_projection.svg'

# Plot projection tracts using tm_utils function
tm_utils.plot_correlation(
    x=x_data_proj, 
    y=y_data_proj,
    corr_value=r_value_proj,
    p_value=p_value_proj,
    x_label='S-A Range',
    y_label=f'Mean {distance_label}',
    color_scheme=fds_cmap,
    reverse_colormap=True,
    colorbar='same_plot',
    colorbar_label=f'Mean {distance_label}',
    color_by='y',
    point_labels=tract_labels_proj,
    text_box_position='top_left',
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
tract_sa_ranges_assoc = []
tract_dist_means_assoc = []
tract_labels_assoc = []

# Use pre-computed distance means and get S-A range for each association tract
for tract in association_tracts:
    # Check if tract exists in pre-computed data
    if tract not in tract_means['Tract'].values:
        print(f"Warning: Tract {tract} not found in pre-computed data, skipping...")
        continue
        
    # Get pre-computed S-A range for this tract
    tract_sa_range = sa_ranges[sa_ranges['Tract'] == tract]['SA_Range'].iloc[0]
    
    # Skip if S-A range is NaN (no regions connected)
    if pd.isna(tract_sa_range):
        continue
    
    # Get pre-computed mean distance for this tract
    tract_distance_mean = tract_means[tract_means['Tract'] == tract][distance_column].iloc[0]
        
    tract_sa_ranges_assoc.append(tract_sa_range)
    tract_dist_means_assoc.append(tract_distance_mean)
    tract_labels_assoc.append(tract.replace('_', ' '))  # Format names for display

# Convert to numpy arrays
x_data_assoc = np.array(tract_sa_ranges_assoc)
y_data_assoc = np.array(tract_dist_means_assoc)

# Calculate correlation for association tracts
corr_result_assoc = tm_utils.perm_corr_test(x_data_assoc, y_data_assoc, n_permutations=10000, method='spearman', 
                                           alternative='two-sided', random_state=42)
r_value_assoc = corr_result_assoc['observed_corr']
p_value_assoc = corr_result_assoc['p_value']

# Create output path for association tracts
output_path_assoc = f'{results_dir}/sa_range_vs_{distance_filename}_distance_association.svg'

# Plot association tracts using tm_utils function
tm_utils.plot_correlation(
    x=x_data_assoc, 
    y=y_data_assoc,
    corr_value=r_value_assoc,
    p_value=p_value_assoc,
    x_label='S-A Range',
    y_label=f'Mean {distance_label}',
    color_scheme=fds_cmap,
    reverse_colormap=True,
    colorbar='same_plot',
    colorbar_label=f'Mean {distance_label}',
    color_by='y',
    point_labels=tract_labels_assoc,
    text_box_position='top_left',
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
    'analysis': ['projection_tracts_euclidean', 'association_tracts_euclidean'],
    'observed_correlation': [r_value_proj, r_value_assoc],
    'p_value': [p_value_proj, p_value_assoc],
    'n_samples': [corr_result_proj['n_samples'], corr_result_assoc['n_samples']],
    'n_permutations': [corr_result_proj['n_permutations'], corr_result_assoc['n_permutations']],
    'method': [corr_result_proj['method'], corr_result_assoc['method']],
    'alternative': [corr_result_proj['alternative'], corr_result_assoc['alternative']],
    'distance_type': [distance_filename, distance_filename],
    'n_tracts': [len(projection_tracts), len(association_tracts)]
})
results_summary.to_csv(f'{results_dir}/correlation_results_tract_subsets_{distance_filename}.csv', index=False)

print(f"Generated plots for projection ({len(projection_tracts)} tracts) and association ({len(association_tracts)} tracts) tract subsets with {distance_label.lower()}")
print(f"Saved correlation results to: correlation_results_tract_subsets_{distance_filename}.csv")
