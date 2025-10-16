# ------------------------------------------------------------------------------------------------
# --- Plot PLS diagram ---
# ------------------------------------------------------------------------------------------------
# This script generates the PLS explanatory diagram.


# ------------------------------------------------------------------------------------------------
# --- Import packages ---
# ------------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
from netneurotools import datasets

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# get custom colormaps
warm_cmap, tract_cmap, _, cool_warm_cmap, _, _ = tm_utils.make_colormaps()

# set fontsize for all plots
plt.rcParams.update({'font.size': 20})

# create results directory if it doesn't yet exist
root = '/Users/joelleba/PennLINC/tractmaps'
results_dir = f'{root}/results/pls_terms_tracts/pls_diagram'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Folder '{results_dir}' created.")
else:
    print(f"Folder '{results_dir}' already exists.")


# ------------------------------------------------------------------------------------------------
# --- Generate explanation diagram ---
# ------------------------------------------------------------------------------------------------


# 1. Generate Fake Data for Regions x Terms and Regions x Tracts
np.random.seed(34)
regions, terms, tracts = 4, 10, 6
regions_terms = np.random.rand(regions, terms)
regions_tracts = np.random.rand(regions, tracts)
term_weights = np.random.randn(terms)
tract_weights = np.random.randn(tracts)

# regions x terms matrix
plt.figure(figsize=(5, 5))
sns.heatmap(regions_terms, annot=False, cmap=cool_warm_cmap, 
			linecolor='white', linewidths=2, square=True,
			xticklabels=False, yticklabels=False,
			cbar=True, cbar_kws={'orientation':'horizontal', 'label':'z-scores', 'ticks':[]})
plt.xlabel('cognitive terms')
plt.ylabel('regions')
plt.tight_layout()
plt.savefig(f'{results_dir}/terms_matrix.svg')
plt.show()

# # regions x tracts matrix
plt.figure(figsize=(4, 5))
sns.heatmap(regions_tracts, annot=False, cmap=tract_cmap, 
			linecolor='white', linewidths=2, square=True,
			xticklabels=False, yticklabels=False,
			cbar=True, cbar_kws={'orientation':'horizontal', 'label':'connection probability', 'ticks':[]})
plt.xlabel('tracts')
plt.ylabel('regions')
plt.tight_layout()
plt.savefig(f'{results_dir}/tracts_matrix.svg')
plt.show()

# correlation matrix
plt.figure(figsize=(4, 6))
corrs = np.corrcoef(regions_terms.T, regions_tracts.T)[:terms, terms:]
sns.heatmap(corrs, annot=False, cmap=cool_warm_cmap, center=0,
			linecolor='white', linewidths=2, square=True,
			xticklabels=False, yticklabels=False,
			cbar=True, cbar_kws={'orientation':'horizontal', 'label':'correlation', 'ticks':[]})
plt.xlabel('tracts')
plt.ylabel('terms')
plt.tight_layout()
plt.savefig(f'{results_dir}/correlations.svg')
plt.show()

# term and tract weights
plt.figure(figsize=(3, 3))
plt.bar(np.arange(terms), term_weights, color='darkgray')
plt.xticks([])
plt.yticks([])
plt.axhline(0, color='black', lw=0.75)
plt.xlabel('cognitive terms')
plt.ylabel('weights')
sns.despine()
plt.tight_layout()
plt.savefig(f'{results_dir}/weights_terms.svg')
plt.show()

# plot tract weights (make some negative)
plt.figure(figsize=(2.5, 3))
plt.bar(np.arange(tracts), tract_weights, color='darkgray')
plt.xticks([])
plt.yticks([])
plt.axhline(0, color='black', lw=0.75)
plt.xlabel('tracts')
plt.ylabel('weights')
sns.despine()
plt.tight_layout()
plt.savefig(f'{results_dir}/weights_tracts.svg')
plt.show()


# create fake data of size 360
np.random.seed(123)
regions = 360
term_scores = np.random.rand(regions)
tract_scores = np.random.rand(regions)
labels = datasets.fetch_mmpall(version = 'fslr32k')
rhlabels, lhlabels = labels[0], labels[1] # inverted order so that reg ids 1-180 correspond to lh and 181-360 to rh, for consistency with our data

# term scores
tm_utils.conte69_plot_grid(data = term_scores, 
                        lhlabel = lhlabels, 
                        rhlabel = rhlabels, 
                        outpath = f'{results_dir}/term_scores.svg', 
                        vmin = np.nanmin(term_scores), 
                        vmax = np.nanmax(term_scores),
                        surf = 'inflated',
                        customcmap = cool_warm_cmap,
						title = 'term scores',
                        shared_colorbar = False
)

# tract scores
tm_utils.conte69_plot_grid(data = tract_scores, 
                        lhlabel = lhlabels, 
                        rhlabel = rhlabels, 
                        outpath = f'{results_dir}/tract_scores.svg',
                        vmin = np.nanmin(tract_scores), 
                        vmax = np.nanmax(tract_scores),
                        surf = 'inflated',
                        customcmap = cool_warm_cmap,
						title = 'tract scores',
                        shared_colorbar = False
)


def plot_loadings(loadings, errors, names, filename, figsize, top_n = 20, title=None, barheight=0.8):
    """
    Plots the top N positive and negative loadings with error bars, ensuring both
    sides are aligned and saved to a file.

    Parameters:
    loadings (np.ndarray): Array of loadings.
    errors (np.ndarray): Array of errors corresponding to the loadings.
    names (np.ndarray): Array of names corresponding to the loadings.
    title (str): Title of the plot.
    filename (str): Path to save the plot image.
    top_n (int): Number of top values to display for both negative and positive loadings.

    Returns:
    Saves the plot as an svg file. 
    """
        
    sorted_idx = np.argsort(abs(loadings))[::-1]  # Sort by absolute value descending
    sorted_loadings = loadings[sorted_idx]
    sorted_errors = errors[sorted_idx]
    sorted_names = names[sorted_idx]

    # Split into negative and positive parts and keep only top N
    negative_loadings = sorted_loadings[sorted_loadings < 0][:top_n]
    positive_loadings = sorted_loadings[sorted_loadings >= 0][:top_n]
    negative_errors = sorted_errors[sorted_loadings < 0][:top_n]
    positive_errors = sorted_errors[sorted_loadings >= 0][:top_n]
    negative_names = sorted_names[sorted_loadings < 0][:top_n]
    positive_names = sorted_names[sorted_loadings >= 0][:top_n]

    # Ensure both sides are aligned
    max_length = max(len(negative_loadings), len(positive_loadings))
    negative_loadings = np.pad(negative_loadings, (0, max_length - len(negative_loadings)), 'constant', constant_values = np.nan)
    positive_loadings = np.pad(positive_loadings, (0, max_length - len(positive_loadings)), 'constant', constant_values = np.nan)
    negative_errors = np.pad(negative_errors, (0, max_length - len(negative_errors)), 'constant', constant_values = np.nan)
    positive_errors = np.pad(positive_errors, (0, max_length - len(positive_errors)), 'constant', constant_values = np.nan)
    negative_names = np.pad(negative_names, (0, max_length - len(negative_names)), 'constant', constant_values = '')
    positive_names = np.pad(positive_names, (0, max_length - len(positive_names)), 'constant', constant_values = '')

    fig, ax = plt.subplots(figsize = figsize)
    
    # Plotting the negative loadings
    ax.barh(np.arange(len(negative_loadings)), negative_loadings, xerr = negative_errors, 
            color = '#636BD8', ecolor = 'gray', alpha = 0.8, height=barheight)
    
    # Creating a secondary y-axis for the positive loadings
    ax_right = ax.twinx()
    ax_right.barh(np.arange(len(positive_loadings)), positive_loadings, xerr = positive_errors, 
                  color = '#D53D69', ecolor = 'gray', alpha = 0.8, height=barheight)

    # remove x and y-ticks
    ax.set_yticks([]) 
    ax.set_xticks([]) 
    ax_right.set_yticks([]) 
    ax_right.set_xticks([]) 
    ax.axvline(0, color = 'black', linewidth = 0.7)
    ax.set_xlabel('loadings')
    ax.set_title(title, fontsize = 20)

    # hide all spines (axis lines) except for the bottom one
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    # invert y-axis for proper ordering
    ax.invert_yaxis()
    ax_right.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(filename)


np.random.seed(123)

# term loadings
term_loadings = np.random.randn(terms)
errors = np.random.rand(terms)
plot_loadings(term_loadings, errors, np.array(['' for i in range(20)]), f'{results_dir}/term_loadings.svg', 
			  figsize = (3, 4), top_n = 20, 
			  title = 'cognitive terms')
plt.show()

# tract loadings
tract_loadings = np.random.randn(tracts)
errors = np.random.rand(tracts)
plot_loadings(tract_loadings, errors, np.array(['' for i in range(20)]), f'{results_dir}/tract_loadings.svg', 
			  figsize = (3, 3), top_n = 20, 
			  title = 'tracts', barheight=0.6)
plt.show()