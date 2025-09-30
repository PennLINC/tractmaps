#------------------------------------------------------------------------------------------------
# --- The spatial and hierarchical embedding of tracts ---
#------------------------------------------------------------------------------------------------

# This script plots the relationship between the S-A range of a tract and its mean euclidean distance.

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
results_dir = f'{root}/results/spatial_embedding/'
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
distance_label = 'Euclidean Distance'
distance_filename = 'euclidean'

print(f"Using Euclidean distances")

# get custom colormaps
warm_cmap, tract_cmap, categ_warm, cool_warm_cmap, categ_cool_warm, bppy_cmap = tm_utils.make_colormaps()

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load data
tractdata = pd.read_csv(f'{data_root}/tracts_probabilities/tracts_probabilities.csv')
sa_data = pd.read_csv(f'{data_root}/tracts_sa_axis/tract_sa_axis_regionwise{thresh_suffix}.csv')
sa_ranges = pd.read_csv(f'{data_root}/tracts_sa_axis/tract_sa_axis_ranges{thresh_suffix}.csv')
sa_axis = pd.read_csv(f'{root}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv')



# ------------------------------------------------------------------------------------------------
# --- Plot full tract set with Euclidean distances ---
# ------------------------------------------------------------------------------------------------

# Plot all tracts together
all_tracts = tractdata.filter(regex='left|right').columns.tolist()
# Filter to only include tracts that exist in the pre-computed data
available_tracts = tract_means['Tract'].tolist()
all_tracts = [tract for tract in all_tracts if tract in available_tracts]

# Prepare data for plotting
tract_sa_ranges = []
tract_dist_means = []
tract_labels = []

# Use pre-computed distance means and get S-A range for each tract
for tract in all_tracts:
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
        
    tract_sa_ranges.append(tract_sa_range)
    tract_dist_means.append(tract_distance_mean)
    tract_labels.append(tract.replace('_', ' '))

# Convert to numpy arrays
x_data = np.array(tract_sa_ranges)
y_data = np.array(tract_dist_means)

# Calculate correlation
corr_result = tm_utils.perm_corr_test(x_data, y_data, n_permutations=10000, method='spearman', 
                                      alternative='two-sided', random_state=42)
r_value = corr_result['observed_corr']
p_value = corr_result['p_value']

# Save correlation results to CSV
results_summary = pd.DataFrame({
    'analysis': ['full_tract_set_euclidean'],
    'observed_correlation': [r_value],
    'p_value': [p_value],
    'n_samples': [corr_result['n_samples']],
    'n_permutations': [corr_result['n_permutations']],
    'method': [corr_result['method']],
    'alternative': [corr_result['alternative']],
    'distance_type': [distance_filename]
})
results_summary.to_csv(f'{results_dir}/correlation_results_{distance_filename}.csv', index=False)

# Create output path
output_path = f'{results_dir}/sa_range_vs_{distance_filename}_distance.svg'

# Plot using tm_utils function
tm_utils.plot_correlation(
    x=x_data, 
    y=y_data,
    corr_value=r_value,
    p_value=p_value,
    x_label='S-A Range',
    y_label=f'Mean {distance_label}',
    color_scheme=bppy_cmap,
    reverse_colormap=True,
    colorbar='same_plot',
    colorbar_label=f'Mean {distance_label}',
    color_by='y',
    point_labels=tract_labels,
    text_box_position='top_left',
    figure_size=(7, 8),
    point_size=100,
    point_alpha=0.8,
    regression_line=True,
    output_path=output_path,
    dpi=300
)

print(f"Generated plot for full tract set ({len(all_tracts)} tracts) with {distance_label.lower()}")
print(f"Saved correlation results to: correlation_results_{distance_filename}.csv")
 
