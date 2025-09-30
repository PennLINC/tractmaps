# ------------------------------------------------------------------------------------------------
# --- PLS plotting ---
# ------------------------------------------------------------------------------------------------
# Inputs: PLS results from pls_terms_tracts.py
# Outputs: Plots of PLS result figures.
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
import pyls
from scipy.stats import zscore
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import scipy

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

root_dir = '/Users/joelleba/PennLINC/tractmaps'
results_dir = f'{root_dir}/results/pls_terms_tracts'
output_dir = f'{results_dir}/pls_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    print(f'Output directory {output_dir} already exists.')
 
# set parameters
nspins = 10000
lv = 0 # plot the first latent variable

# get custom colormaps
warm_cmap, _, _, cool_warm_cmap, _, _ = tm_utils.make_colormaps()

# set fontsize for all plots
plt.rcParams.update({'font.size': 18})

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# load tract probabilities and cognitive term scores
data = pd.read_csv(f'{root_dir}/data/derivatives/neurosynth_annotations/glasser/glasser_tracts_neurosynth_125terms.csv')
tractdata = data.filter(regex = 'left|right') 
nsdata = data.iloc[:, 59:]

# load region labels for brain surface plotting
lhlabels = f'{root_dir}/data/derivatives/glasser_parcellation/HCP_MMP_L.label.gii'
rhlabels = f'{root_dir}/data/derivatives/glasser_parcellation/HCP_MMP_R.label.gii'

# load PLS results
pls_result = pyls.load_results(f'{results_dir}/pls_result.hdf5')

# load train test results
train = np.load(f'{results_dir}/pls_glasser360_train_corrs.npy')
test = np.load(f'{results_dir}/pls_glasser360_test_corrs.npy')
testnullres = np.load(f'{results_dir}/pls_glasser360_null_corrs.npy')

# load test scores and correlations
test_scores = np.load(f'{results_dir}/pls_glasser360_test_scores.npy')
test = np.load(f'{results_dir}/pls_glasser360_test_corrs.npy')

# ------------------------------------------------------------------------------------------------
# --- Percent covariance explained by latent variables ---
# ------------------------------------------------------------------------------------------------

# NOTE: `pls_result.permres.permsingval comes from locally editing
# pyls.base.py run_pls(self, X, Y) to return the `d_perm` variable.
# Also note that for this edit to work, the structures.PLSPermResult
# object needs to be updated to allow the `d_perm` to be returned.
# Code located at: 
# /opt/anaconda3/envs/tractmaps/lib/python3.8/site-packages/pyls-0.0.1-py3.8.egg/pyls/base.py
# /opt/anaconda3/envs/tractmaps/lib/python3.8/site-packages/pyls-0.0.1-py3.8.egg/pyls/structures.py

cv = pls_result.singvals**2 / np.sum(pls_result.singvals**2)
null_singvals = pls_result.permres.perm_singval
cv_spins = null_singvals**2 / sum(null_singvals**2)


p = (1 + sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1 + nspins) # p-val from Hansen paper

# print variance explained and pvalue
print(f'PLS \n \
LV 1 var exp = {round(cv[0]*100, 1)}%, pspin = {round(pls_result.permres.pvals[0], 6)} \n \
LV 2 var exp = {round(cv[1]*100, 1)}%, pspin = {round(pls_result.permres.pvals[1], 6)} \n \
LV 3 var exp = {round(cv[2]*100, 1)}%, pspin = {round(pls_result.permres.pvals[2], 6)} \n \
LV 4 var exp = {round(cv[2]*100, 1)}%, pspin = {round(pls_result.permres.pvals[3], 6)} \n')

# subset to the first 10 LVs for visualization
cv = cv[:10]
cv_spins = cv_spins[:10, :]

# make significance mask using p < 0.05
mask = np.zeros_like(cv)
mask[pls_result.permres.pvals[:10] < 0.05] = 1

# plot variance explained
plt.ion()
plt.figure(figsize=(4, 4))
plt.boxplot(x = cv_spins.T * 100, positions = range(len(cv)), # a box is drawn for each column of x
                boxprops = dict(alpha = 0.5, color = 'gray'), whiskerprops = dict(color = 'grey'), medianprops=dict(color='gray'),
                showcaps = False, showfliers = False, zorder = 1)
# scatter plot of variance explained where significant LVs are colored using mask, non-significant ones are grey
for i in range(len(cv)):
    color = 'palevioletred' if mask[i] == 1 else 'darkgrey'
    plt.scatter(i, cv[i] * 100, s = 40, color=color, zorder=2)

plt.xlabel('latent variable')
plt.ylabel('covariance explained (%)')
sns.despine(trim = False)
plt.gca().set_xticks([])
plt.yticks(np.arange(0, 50, 10))
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/scatter_pls_var_exp.svg')


# ------------------------------------------------------------------------------------------------
# --- cross-validation: boxplot of train, test, and null test score correlations; 
# scatter plot of X and Y scores from the median test set ---
# ------------------------------------------------------------------------------------------------

# compute pvalue
p = (1 + np.sum(testnullres > np.mean(test))) / (1 + len(testnullres))

# print mean test score correlation
print(f'PLS \n \
Mean train score correlation = {round(np.mean(train), 2)} \n \
Mean test score correlation = {round(np.mean(test), 2)} \n \
Mean null test score correlation = {round(np.mean(testnullres), 2)} \n \
p = {round(p, 4)} \n')

# plot train and test score correlation distribution
fig, ax = plt.subplots(figsize=(2.5, 4))
boxplot = sns.boxplot(data = [train, test, testnullres], ax = ax, palette = ['palevioletred', 'darkgrey', 'lightgrey'], fill = False, 
                      width = 0.6, showcaps = False, showfliers = False)
sns.despine()
ax.set_xticklabels(['train', 'test', 'null'])
ax.text(0.1, 0.1, f'r = {round(np.mean(test), 2)}\np = {round(p, 4)}', transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', boxstyle='round,pad=0.8'))

ax.set_ylabel('score correlation', fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=16)
plt.savefig(f'{output_dir}/pls_glasser360_crossvalidation.svg',
            bbox_inches = 'tight', dpi = 300,
            transparent = True)
    

# scatter plot of X and Y scores from the median test set

# get the median index and correlation based on the correlation values from test
median_index = np.argsort(test)[len(test) // 2]

# extract the X and Y scores corresponding to the median correlation
median_x_scores, median_y_scores = test_scores[median_index]
print(median_x_scores.shape, median_y_scores.shape) # should be ~90 regions, as 25% of 360 regions are used for test = 90 regions

plt.ion()
plt.figure(figsize=(5, 4))
sns.regplot(x=median_x_scores, y=median_y_scores, scatter=False, color='darkgrey')
plt.scatter(median_x_scores, median_y_scores, alpha=0.8, color='#636BD8', linewidths=0)
r = scipy.stats.spearmanr(median_x_scores, median_y_scores)
sns.despine()
plt.xlabel('median test cog. term scores', fontsize=18) 
plt.ylabel('median test tract scores', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.text(0.05, 0.95, f'r = {round(r.statistic, 3)}', ha='left', va='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', boxstyle='round,pad=0.8'))
plt.tight_layout()
plt.savefig(f'{output_dir}/scatter_scores.svg')
plt.show() 

# ------------------------------------------------------------------------------------------------
# --- Plot PLS scores on the cortical surface ---
# ------------------------------------------------------------------------------------------------

# define inputs for plotting

for lv in range(1):
    
    # cognitive term scores
    term_scores = pls_result["x_scores"][:, lv] 
    tm_utils.conte69_plot_grid(data = term_scores, 
                        lhlabel = lhlabels, 
                        rhlabel = rhlabels, 
                        outpath = f'{output_dir}/LV{lv}_term_scores_glasser360.svg',
                        vmin = np.nanmin(term_scores), 
                        vmax = np.nanmax(term_scores),
                        surf = 'inflated',
                        customcmap = cool_warm_cmap,
                        shared_colorbar = True
                        )
    
    # tract scores
    tract_scores = pls_result["y_scores"][:, lv] 
    tm_utils.conte69_plot_grid(data = tract_scores, 
                        lhlabel = lhlabels, 
                        rhlabel = rhlabels, 
                        outpath = f'{output_dir}/LV{lv}_tract_scores_glasser360.svg',
                        vmin = np.nanmin(tract_scores), 
                        vmax = np.nanmax(tract_scores),
                        surf = 'inflated',
                        customcmap = cool_warm_cmap,
                        shared_colorbar = True
                        )


# ------------------------------------------------------------------------------------------------
# --- Plot PLS loadings ---

# The following is being plotted: 
# * positive and negative loadings share the same y axis to avoid having super long bar plots

# function to plot loadings
def plot_loadings(loadings, errors, names, filename, figsize, top_n = 20, title=None):
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
    colors = ['#636BD8' if val < 0 else 'lightcoral' for val in loadings[sorted_idx]]
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
    ax.barh(np.arange(len(negative_loadings)), negative_loadings, xerr = negative_errors, color = '#636BD8', ecolor = 'gray', alpha = 0.8)
    
    # Creating a secondary y-axis for the positive loadings
    ax_right = ax.twinx()
    ax_right.barh(np.arange(len(positive_loadings)), positive_loadings, xerr = positive_errors, color = '#D53D69', ecolor = 'gray', alpha = 0.8)

    # Set y-ticks and labels
    ax.set_yticks(np.arange(len(negative_names))) # negative side
    ax.set_yticklabels(negative_names, fontsize = 16) 
    ax_right.set_yticks(np.arange(len(positive_names))) # positive side
    ax_right.set_yticklabels(positive_names, fontsize = 16)
    ax_right.set_ylim(ax.get_ylim()) # align the left and right y-axis ticks 
    ax.axvline(0, color = 'black', linewidth = 0.7)
    ax.set_xlabel('loadings', fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.tick_params(axis='x', labelsize=18)
    ax_right.tick_params(axis='x', labelsize=18)
    
    # Hide all spines (axis lines) except for the bottom one
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


# plot cognitive terms and tracts loadings
# Get merged tract names (remove _left and _right suffixes)
left_tracts = tractdata.filter(regex='left')
right_tracts = tractdata.filter(regex='right')
left_tract_names = left_tracts.columns
base_tract_names = []
for name in left_tract_names:
    base_name = name.replace('_left', '')
    base_tract_names.append(base_name)

term_names = nsdata.columns

# Merge left and right data for each tract to get the data used in PLS
merged_tract_data = pd.DataFrame()
for i, base_name in enumerate(base_tract_names):
    left_col = left_tract_names[i]
    right_col = right_tracts.columns[i]
    
    # Get left and right data for this tract
    left_data = left_tracts.iloc[:180, :][left_col].values
    right_data = right_tracts.iloc[180:, :][right_col].values
    
    # Concatenate left and right (360 regions total)
    merged_data = np.concatenate([left_data, right_data])
    
    # Add to merged dataframe
    merged_tract_data[base_name] = merged_data

X = zscore(nsdata).values 
Y = zscore(merged_tract_data).values

for lv in range(1):  # plot the first LV

    # plot X loadings
    xload = pyls.behavioral_pls(Y, X, n_boot = 10000, n_perm = 0, test_split = 0)
    err_x = (xload["bootres"]["y_loadings_ci"][:, lv, 1] - xload["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
    relidx_x = (abs(xload["y_loadings"][:, lv]) - err_x) > 0
    plot_loadings(
        loadings = xload["y_loadings"][relidx_x, lv],
        errors = err_x[relidx_x],
        names = term_names[relidx_x],
        filename = f'{output_dir}/lv{lv}_bar_pls_cogloads.svg',
        figsize = (8, 9),
        top_n = 20
    )

    # plot Y loadings
    err_y = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1] - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
    relidx_y = (abs(pls_result["y_loadings"][:, lv]) - err_y) > 0
    plot_loadings(
        loadings = pls_result["y_loadings"][relidx_y, lv],
        errors = err_y[relidx_y],
        names = np.array(base_tract_names)[relidx_y],
        filename = f'{output_dir}/lv{lv}_bar_pls_tractloads.svg',
        figsize = (6, 7),
        top_n = 26 # all tracts 
    )

plt.show() 