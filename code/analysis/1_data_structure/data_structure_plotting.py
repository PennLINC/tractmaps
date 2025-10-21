# ------------------------------------------------------------------------------------------------
# --- Figure 1: Plot tract probabilities, neurosynth terms, and cortical properties ---
# ------------------------------------------------------------------------------------------------

# This script does the following:
# 1. Load tract probabilities, neurosynth terms, and cortical properties
# 2. Plot them as heatmaps
# 3. Plot them on brain surfaces
# 4. Plot example tracts in glass brain

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from traits.api import true
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
from utils.tract_visualizer import TractVisualizer
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import matplotlib.patches as patches
import re
import random

# ------------------------------------------------------------------------------------------------
# --- Set inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# set root path
root = '/Users/joelleba/PennLINC/tractmaps'

# create folder to store plots if doesn't yet exist
results_dir = os.path.join(root, 'results/data_structure/')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Folder '{results_dir}' created.")
else:
    print(f"Folder '{results_dir}' already exists.")

# set all plot font sizes to 16, and x and y labels to 14
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

# get custom colormaps
warm_cmap, tract_cmap, categ_warm, cool_warm_cmap, categ_cool_warm, bppy_cmap = tm_utils.make_colormaps()

# ------------------------------------------------------------------------------------------------
# --- Helper functions ---
# ------------------------------------------------------------------------------------------------

# Helper to prettify labels (e.g., 'genes_pc1' -> 'Genes PC1')
def pretty_label(s: str) -> str:
    s = s.replace('_', ' ')
    s = s.title()
    s = re.sub(r'\bPc(\d+)\b', r'PC\1', s)
    return s

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# load glasser labels for plotting on brain surface
lhlabels = f'{root}/data/derivatives/glasser_parcellation/HCP_MMP_L.label.gii'
rhlabels = f'{root}/data/derivatives/glasser_parcellation/HCP_MMP_R.label.gii'

# load tract probabilities
tracts = pd.read_csv(f'{root}/data/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv')

# reorder columns alphabetically
# tracts = tracts.reindex(sorted(tracts.columns), axis=1)

# load cognitive terms
nsdata = pd.read_csv(f'{root}/data/derivatives/neurosynth_annotations/glasser/glasser_tracts_neurosynth_125terms.csv')
nsdata = nsdata.iloc[:, 59:]

# load cortical properties
cortical_properties = pd.read_csv(f'{root}/data/derivatives/cortical_annotations/glasser/glasser_tracts_corticalmaps.csv')
cortical_properties = cortical_properties.iloc[:, 59:]
# Reorder columns to place 'genes_pc1' in the middle (for plotting)
if 'genes_pc1' in cortical_properties.columns:
    cols_cp = list(cortical_properties.columns)
    cols_cp.remove('genes_pc1')
    insert_idx = len(cols_cp) // 2
    cols_cp.insert(insert_idx, 'genes_pc1')
    cortical_properties = cortical_properties[cols_cp]

# define tracts, terms, and cortical properties to plot
tracts_to_plot = ['AF', 'UF']
tract_connection_threshold = 0 # showing full tract probabilities
tract_longnames = ['arcuate', 'uncinate']
terms_to_plot = ['language', 'emotion']
properties_to_plot = ['myelin', 'genes_pc1']

# ------------------------------------------------------------------------------------------------
# --- Make colorbars ---
# ------------------------------------------------------------------------------------------------

# # show cool_warm_cmap as colorbar
# fig, ax = plt.subplots(figsize=(8, 1))
# fig.subplots_adjust(bottom=0.5)
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap=cool_warm_cmap),
# 					cax=ax, orientation='horizontal')
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 1))
# fig.subplots_adjust(bottom=0.5)
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap=warm_cmap),
# 					cax=ax, orientation='horizontal')
# plt.show()

# ------------------------------------------------------------------------------------------------
# --- Make tract probabilities heatmap ---
# ------------------------------------------------------------------------------------------------
print('Making tract probabilities heatmap...')

# Create single heatmap with both hemispheres
fig, ax = plt.subplots(1, 1, figsize=(4, 5))

# drop columns that are not tracts from the tracts dataframe
tracts_df = tracts.filter(regex='left|right')

# Plot heatmap with NA handling
g = sns.heatmap(tracts_df, cmap=tract_cmap, vmin=0, vmax=1, 
                ax=ax, cbar=False, xticklabels=True, yticklabels=False)

# Set NAs to grey
g.set_facecolor('lightgrey')

# Set subdivided labels (replacing main labels)
ax.xaxis.set_label_position('top')

# Add subdivided labels
# Find the midpoint between left and right tracts
left_tracts = [col for col in tracts_df.columns if '_left' in col]
right_tracts = [col for col in tracts_df.columns if '_right' in col]
left_tract_count = len(left_tracts)
right_tract_count = len(right_tracts)

# Add "Left tracts" and "Right tracts" labels (on top)
if left_tract_count > 0 and right_tract_count > 0:
    # Left tracts label
    ax.text(left_tract_count/2, -30, 'Left tracts', 
            ha='center', va='top')
    # Right tracts label  
    ax.text(left_tract_count + right_tract_count/2, -30, 'Right tracts', 
            ha='center', va='top')

# Add "Left regions" and "Right regions" labels (on left)
# Find the midpoint between left and right regions
left_regions = tracts[tracts['parcel_name'].str.contains('L_')]
right_regions = tracts[tracts['parcel_name'].str.contains('R_')]
left_region_count = len(left_regions)
right_region_count = len(right_regions)

if left_region_count > 0 and right_region_count > 0:
    # Left regions label
    ax.text(-1.5, left_region_count/2, 'Left regions', 
            ha='right', va='center', rotation=90)
    # Right regions label
    ax.text(-1.5, left_region_count + right_region_count/2, 'Right regions', 
            ha='right', va='center', rotation=90)

# Add rectangles and labels for selected tracts
xtick_labels = [''] * len(tracts_df.columns)
for tract in tracts_to_plot:
    # Left hemisphere
    if f'{tract}_left' in tracts_df.columns:
        col_idx = list(tracts_df.columns).index(f'{tract}_left')
        rect = patches.Rectangle(
            (col_idx, 0), 1, tracts_df.shape[0],
            linewidth=1, edgecolor='grey', facecolor='none'
        )
        ax.add_patch(rect)
        xtick_labels[col_idx] = f'Left {tract}'
    
    # Right hemisphere
    if f'{tract}_right' in tracts_df.columns:
        col_idx = list(tracts_df.columns).index(f'{tract}_right')
        rect = patches.Rectangle(
            (col_idx, 0), 1, tracts_df.shape[0],
            linewidth=1, edgecolor='grey', facecolor='none'
        )
        ax.add_patch(rect)
        xtick_labels[col_idx] = f'Right {tract}'

ax.tick_params(axis='x', bottom=False)
ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

# Create figure for NA legend
from matplotlib.patches import Patch
legend_fig, legend_ax = plt.subplots(figsize=(0.4, 0.5))
legend_elements = [Patch(facecolor='lightgrey', label='NA', edgecolor='none')]
legend_ax.legend(handles=legend_elements, loc='center', fontsize=10, frameon=False)
legend_ax.axis('off')
legend_fig.tight_layout()
legend_fig.savefig(f'{results_dir}/na_legend.svg', dpi=300, bbox_inches='tight')
plt.close(legend_fig)

# add a shared colorbar 
# fig.colorbar(axes[0].collections[0], ax=axes, location='bottom', shrink=0.4, orientation='horizontal', pad=0.3)
# cbar = axes[0].collections[0].colorbar
# cbar.set_ticks(np.round(cbar.get_ticks(), 1))
# cbar.set_label('Connection probability', fontsize=14, labelpad=20)
# cbar.ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.savefig(f'{results_dir}/heatmap_tracts_to_regions.svg', dpi=300)
plt.show()

# create a separate figure for the colorbar
fig_cbar, ax_cbar = plt.subplots(figsize=(6, 1.5))
sm = plt.cm.ScalarMappable(cmap=tract_cmap)
sm.set_array([])
cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_ticks([0, 0.5, 1])
cbar.set_label('Connection probability', labelpad=15)
cbar.ax.xaxis.set_label_position('top')
cbar.outline.set_visible(False)
plt.tight_layout()
plt.savefig(f'{results_dir}/colorbar_tracts_to_regions.svg', dpi=300)
plt.show()




# ------------------------------------------------------------------------------------------------
# --- Make neurosynth heatmap ---
# ------------------------------------------------------------------------------------------------

print('Making neurosynth terms heatmap...')

# heatmap
fig, ax = plt.subplots(figsize=(5, 5))
abs_max = np.max(np.abs(nsdata)) 
sns.heatmap(nsdata, cmap=cool_warm_cmap, center=0, cbar=False, xticklabels=True, yticklabels=False, ax=ax, 
            vmin=np.percentile(-abs_max, 2.5), vmax=np.percentile(abs_max, 97.5))
ax.set_xlabel('Terms', labelpad=10)
ax.set_ylabel('Regions', labelpad=10)
ax.xaxis.set_label_position('top') 

# add rectangle and show xtick labels only for selected tracts
xtick_labels = [''] * len(nsdata.columns)
for term in terms_to_plot:
    if term in nsdata.columns:
        col_idx = list(nsdata.columns).index(term)
        rect = patches.Rectangle(
            (col_idx, 0), 1, nsdata.shape[0],
            linewidth=1, edgecolor='grey', facecolor='none'
        )
        ax.add_patch(rect) 
        xtick_labels[col_idx] = term 

# set xticks and xtick labels
ax.tick_params(axis='x', bottom=False) # to remove xtick marks
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([pretty_label(x) if x else '' for x in xtick_labels], rotation=45, ha='right')

# save heatmap
plt.tight_layout()
plt.savefig(f'{results_dir}/heatmap_neurosynth_terms.svg', dpi=300)
plt.show()

# create a separate figure for the colorbar
fig_cbar, ax_cbar = plt.subplots(figsize=(6, 1.5))
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
sm = plt.cm.ScalarMappable(cmap=cool_warm_cmap, norm=norm)
cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_ticks([round(np.percentile(-abs_max, 2.5), 0), 0, round(np.percentile(abs_max, 97.5), 0)])
cbar.set_ticklabels(['low', '0', 'high'])
cbar.set_label('z-scores', labelpad=15)
cbar.ax.xaxis.set_label_position('top')
cbar.outline.set_visible(False)
plt.tight_layout()
plt.savefig(f'{results_dir}/colorbar_neurosynth_terms.svg', dpi=300)
plt.show()

# ------------------------------------------------------------------------------------------------
# --- Make cortical properties heatmap ---
# ------------------------------------------------------------------------------------------------

print('Making cortical properties heatmap...')

# heatmap
fig, ax = plt.subplots(figsize=(4, 5)) # fig width, height
abs_max = np.max(np.abs(cortical_properties)) 
sns.heatmap(cortical_properties, cmap=cool_warm_cmap, center=0, cbar=False, xticklabels=True, yticklabels=False, ax=ax, 
            vmin=np.percentile(-abs_max, 2.5), vmax=np.percentile(abs_max, 97.5))
ax.set_xlabel('Cortical properties', labelpad=10)
ax.set_ylabel('Regions', labelpad=10)
ax.xaxis.set_label_position('top') 

# add rectangle and show xtick labels only for selected tracts
xtick_labels = [''] * len(cortical_properties.columns)
for property in properties_to_plot:
    if property in cortical_properties.columns:
        col_idx = list(cortical_properties.columns).index(property)
        rect = patches.Rectangle(
            (col_idx, 0), 1, cortical_properties.shape[0],
            linewidth=1, edgecolor='grey', facecolor='none'
        )
        ax.add_patch(rect) 
        xtick_labels[col_idx] = property 

# set xticks and xtick labels
# ax.set_xticks(np.arange(len(cortical_properties.columns)) + 0.5) # to keep xtick marks
ax.tick_params(axis='x', bottom=False) # to remove xtick marks
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([pretty_label(x) if x else '' for x in xtick_labels], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{results_dir}/heatmap_cortical_properties.svg', dpi=300)
plt.show()

# create a separate figure for the colorbar
fig_cbar, ax_cbar = plt.subplots(figsize=(6, 1.5))
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
sm = plt.cm.ScalarMappable(cmap=cool_warm_cmap, norm=norm)
cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_ticks([round(np.percentile(-abs_max, 2.5), 0), 0, round(np.percentile(abs_max, 97.5), 0)])
cbar.set_ticklabels(['low', '0', 'high'])
cbar.set_label('z-scores', labelpad=15)
cbar.ax.xaxis.set_label_position('top')
cbar.outline.set_visible(False)
plt.tight_layout()
plt.savefig(f'{results_dir}/colorbar_cortical_properties.svg', dpi=300)
plt.show()

# ------------------------------------------------------------------------------------------------
# --- Make tract probabilities brain surface plots ---
# ------------------------------------------------------------------------------------------------

print('Making tract probabilities brain surface plots...')

for tract in tracts_to_plot:
    left_name = f'{tract}_left'
    right_name = f'{tract}_right'
    left_tract = tracts[left_name].iloc[:180]
    right_tract = tracts[right_name].iloc[180:]
    toplot = pd.concat([left_tract, right_tract], axis=0).reset_index(drop=True)
    tm_utils.conte69_plot_grid(data = toplot, 
                            lhlabel = lhlabels, 
                            rhlabel = rhlabels, 
                            vmin = np.nanmin(toplot), 
                            vmax = np.nanmax(toplot),
                            surf = 'inflated',
                            customcmap=tract_cmap,
                            tractdata = tracts,
                            tracts = [left_name, right_name], 
                            connection_threshold=tract_connection_threshold,
                            shared_colorbar=False,
                            title = f'Cortical endpoints',
                            outpath=f'{results_dir}/brain_{tract}.svg'
                            )

# ------------------------------------------------------------------------------------------------
# --- Make neurosynth brain surface plots ---
# ------------------------------------------------------------------------------------------------

print('Making neurosynth terms brain surface plots...')

for term in terms_to_plot:
    toplot = nsdata[term]
    toplot_min = toplot.min()
    toplot_max = toplot.max()
    abs_max = max(abs(toplot_min), abs(toplot_max))
    tm_utils.conte69_plot_grid(data = toplot, 
                            lhlabel = lhlabels, 
                            rhlabel = rhlabels, 
                            vmin = np.percentile(-abs_max, 2.5), # np.min(toplot),
                            vmax = np.percentile(abs_max, 97.5), # np.max(toplot),
                            surf = 'inflated',
                            customcmap=cool_warm_cmap,
                            shared_colorbar=False,
                            title=pretty_label(term),
                            outpath=f'{results_dir}/brain_neurosynth_{term}.svg'
                            )
    
# ------------------------------------------------------------------------------------------------
# --- Make cortical properties brain surface plots ---
# ------------------------------------------------------------------------------------------------

print('Making cortical properties brain surface plots...')

for property in properties_to_plot:
    toplot = cortical_properties[property]
    toplot_min = toplot.min()
    toplot_max = toplot.max()
    abs_max = max(abs(toplot_min), abs(toplot_max))
    tm_utils.conte69_plot_grid(data = toplot, 
                            lhlabel = lhlabels, 
                            rhlabel = rhlabels, 
                            vmin = np.percentile(-abs_max, 2.5), # np.nanmin(toplot)
                            vmax = np.percentile(abs_max, 97.5), # np.nanmax(toplot),
                            surf = 'inflated',
                            customcmap=cool_warm_cmap,
                            shared_colorbar=False,
                            title = pretty_label(property),
                            outpath=f'{results_dir}/brain_{property}.svg'
                            )

# ------------------------------------------------------------------------------------------------
# --- Plot Glasser 360 regions on brain surface with distinct colors ---
# ------------------------------------------------------------------------------------------------

print('Making Glasser 360 regions brain surface plot...')

# Create a custom colormap using only Set3 colors but with randomized assignment

# Get Set3 colors (12 distinct colors)
set3_colors = plt.cm.Set3(np.linspace(0, 1, 12))[:, :3]  # Extract RGB values

# Create a randomized colormap by repeating and shuffling Set3 colors
# We need 256 colors total, so repeat Set3 colors ~21 times and shuffle
repeated_colors = []
for i in range(22):  # 22 * 12 = 264 colors
    repeated_colors.extend(set3_colors)

# Shuffle the colors to randomize their order
random.shuffle(repeated_colors)

# Take exactly 256 colors
colors_rgb = np.array(repeated_colors[:256])

region_cmap = ListedColormap(colors_rgb)

# Create region data in the same format as other brain plots
# We need to create data that matches the brain surface vertices
# Use the same approach as the tract plots - create data for left and right hemispheres

# Create region data for left hemisphere (regions 1-180)
left_regions = pd.Series(np.arange(1, 181), index=tracts.iloc[:180].index)
# Create region data for right hemisphere (regions 181-360) 
right_regions = pd.Series(np.arange(181, 361), index=tracts.iloc[180:].index)
# Combine left and right hemisphere data
region_data = pd.concat([left_regions, right_regions], axis=0).reset_index(drop=True)

# Use the tm_utils function to plot on brain surface
tm_utils.conte69_plot_grid(data=region_data, 
                        lhlabel=lhlabels, 
                        rhlabel=rhlabels, 
                        vmin=1, 
                        vmax=360,
                        surf='inflated',
                        customcmap=region_cmap,
                        shared_colorbar=False,
                        title='HCP-MMP parcellation',
                        outpath=f'{results_dir}/brain_glasser360_regions.svg'
                        )

# Also create a legend showing the color scheme
fig_legend, ax_legend = plt.subplots(figsize=(15, 10))
# Create a sample of colors to show in legend
sample_colors = region_cmap(np.linspace(0, 1, 20))  # Show 20 sample colors
for i, color in enumerate(sample_colors):
    ax_legend.add_patch(plt.Rectangle((i*0.05, 0), 0.04, 1, facecolor=color, edgecolor='black', linewidth=0.5))

ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.set_title('Sample of Glasser 360 Region Colors (Set3)', fontsize=16)
ax_legend.set_xlabel('Region Index', fontsize=14)
ax_legend.set_ylabel('Color Intensity', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_dir}/glasser360_color_legend.svg', dpi=300)
plt.show()

# ------------------------------------------------------------------------------------------------
# --- Plot example tracts in glass brain ---
# ------------------------------------------------------------------------------------------------

example_tracts = ['AF_left', 'UF_left']
viz = TractVisualizer(root_dir=root, 
                      output_dir=results_dir)
viz.visualize_tracts(tract_list=example_tracts, 
                    single_color='#626bda', 
                    plot_mode='iterative')


# ------------------------------------------------------------------------------------------------
# --- Print output locations ---
# ------------------------------------------------------------------------------------------------

print(f"Files saved to: {results_dir}")
