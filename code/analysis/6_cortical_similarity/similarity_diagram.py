# ------------------------------------------------------------------------------------------------
# --- Cortical similarity diagram ---
# ------------------------------------------------------------------------------------------------
# Creates example plots showing:
# 1. Three example cortical properties as heatmaps (first 6 regions)
# 2. Full cortical similarity matrix
# 3. Tract subsetting examples
# 4. Connected-only examples
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
from utils.figure_formatting import setup_figure, save_figure
plt.switch_backend('Agg')

# ------------------------------------------------------------------------------------------------
# --- Set up paths ---
# ------------------------------------------------------------------------------------------------

root_dir = '/Users/joelleba/PennLINC/tractmaps'
data_dir = f'{root_dir}/data/derivatives/tracts/tracts_cortical_similarity/'
results_dir = f'{root_dir}/results/cortical_similarity/'

# Create output directory
output_dir = f'{results_dir}/diagram'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# get custom colormaps
warm_cmap, _, _, cool_warm_cmap, _, _ = tm_utils.make_colormaps()

# Tracts to plot (left hemisphere)
tracts_for_maps = ['IFOF_left', 'VOF_left']

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load cortical properties data
data = pd.read_csv(f'{root_dir}/data/derivatives/cortical_annotations/glasser/glasser_tracts_corticalmaps.csv')

# Select cortical properties
properties = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6',
             'genes_pc1', 'myelin'] + [col for col in data.columns if col.startswith('pet_')]
cortical_properties = data[properties]

# Load cortical similarity matrix
cortical_similarity = np.load(f'{data_dir}/cortical_similarity.npy')

# Load tract probabilities for subsetting example
tracts_csv = f'{root_dir}/data/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv'
tracts_df = pd.read_csv(tracts_csv)

print(f"Number of cortical properties: {len(properties)}")
print(f"Cortical similarity matrix shape (left hemisphere): {cortical_similarity.shape}")

# ------------------------------------------------------------------------------------------------
# --- Plot example cortical properties (first 6 regions) ---
# ------------------------------------------------------------------------------------------------

# Select 3 example properties
example_properties = ['myelin', 'layer1', 'genes_pc1']

# Create combined matrix with 3 columns
combined_matrix = np.zeros((6, 3))
for i, prop in enumerate(example_properties):
    # Get property values for first 6 regions
    prop_values = cortical_properties[prop].values[:6]
    combined_matrix[:, i] = prop_values

fig, ax = setup_figure(width_mm=23, height_mm=45, margins_mm=(0, 0, 0, 0))
sns.heatmap(
    combined_matrix,
    annot=False,
    cmap=cool_warm_cmap,
    linecolor='white', linewidths=1,
    square=True,
    xticklabels=False, yticklabels=False,
    cbar=False,
    ax=ax
)
save_figure(fig, f"{output_dir}/cortical_properties_combined.svg")
plt.close(fig)
print(f"Saved: {output_dir}/cortical_properties_combined.svg")

# ------------------------------------------------------------------------------------------------
# --- Plot cortical similarity matrix ---
# ------------------------------------------------------------------------------------------------

fig, ax = setup_figure(width_mm=40, height_mm=40, margins_mm=(0, 0, 0, 0))
sns.heatmap(
    cortical_similarity, 
    cmap=cool_warm_cmap, 
    center=0, 
    vmin=-1, vmax=1, 
    xticklabels=False, yticklabels=False,
    linecolor=None, linewidths=0,
    square=True,
    cbar=False,
    ax=ax
)
save_figure(fig, f'{output_dir}/cortical_similarity_matrix.svg')
plt.close(fig)
print(f"Saved: {output_dir}/cortical_similarity_matrix.svg")

# ------------------------------------------------------------------------------------------------
# --- Create cortical similarity colorbar ---
# ------------------------------------------------------------------------------------------------

fig_cbar, ax_cbar = setup_figure(width_mm=35, height_mm=13, margins_mm=(2, 2, 8, 2))
sm = plt.cm.ScalarMappable(cmap=cool_warm_cmap)
sm.set_clim(-1, 1)
sm.set_array([])
cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_label('Cortical similarity', labelpad=4)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.ax.tick_params(width=0.5, length=2)
cbar.outline.set_visible(False)

out_cbar = f"{output_dir}/cortical_similarity_colorbar.svg"
save_figure(fig_cbar, out_cbar)
plt.close(fig_cbar)
print(f"Saved: {out_cbar}")

# ------------------------------------------------------------------------------------------------
# --- Create upper triangle matrix for just the connected regions of two example tracts ---
# ------------------------------------------------------------------------------------------------

# Connectivity threshold
conn_thresh = 0.5

# Create masks once for efficiency
n_regions = cortical_similarity.shape[0]
# upper_mask = np.triu(np.ones((n_regions, n_regions), dtype=bool), k=1)
# lower_mask = ~upper_mask

for tract in tracts_for_maps:
    # Get regions connected to this tract
    connected = (tracts_df[tract].values >= conn_thresh)
    print(f"Number of regions connected to {tract}: {np.sum(connected)}")

   
    # Get indices of connected regions
    connected_indices = np.where(connected)[0]
    n_connected = len(connected_indices)
    
    if n_connected > 1:
        # Extract similarity values for connected regions only
        connected_similarity = cortical_similarity[np.ix_(connected_indices, connected_indices)]
        
        # Create upper triangle mask for the smaller matrix (excluding the diagonal)
        connected_upper_mask = np.triu(np.ones((n_connected, n_connected), dtype=bool), k=1)
        connected_lower_mask = ~connected_upper_mask
        
        fig, ax = setup_figure(width_mm=16, height_mm=16, margins_mm=(0, 0, 0, 0))
        sns.heatmap(
            connected_similarity,
            mask=connected_lower_mask,
            cmap=cool_warm_cmap,
            vmin=-1, vmax=1, # full range of similarity values
            square=True,
            xticklabels=False, yticklabels=False,
            cbar=False,
            linewidths=0,
            linecolor=None,
            ax=ax
        )
        out_path_connected = f"{output_dir}/region_similarity_upper_{tract}.svg"
        save_figure(fig, out_path_connected)
        plt.close(fig)
        print(f"Saved: {out_path_connected}")
    else:
        print(f"Tract {tract} has {n_connected} connected regions - skipping connected-only plot")

print(f"All plots saved to: {output_dir}")
