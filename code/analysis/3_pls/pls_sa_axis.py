# ------------------------------------------------------------------------------------------------
# --- Association between PLS scores and SA axis ---
# ------------------------------------------------------------------------------------------------

# This script tests the association between PLS scores and SA axis.
# Inputs: PLS results from pls_terms_tracts.py
# Outputs: Plots of PLS scores and SA axis.
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
import pyls
from statsmodels.stats.multitest import multipletests
from neuromaps.stats import compare_images
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# root directory
root_dir = '/Users/joelleba/PennLINC/tractmaps'
results_dir = f'{root_dir}/results/pls_terms_tracts'
output_dir = f'{results_dir}/sa_axis/'

# create output directory if it doesn't yet exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# set parameters
nspins = 10000
lv = 0  # plot the first latent variable

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# load tract probabilities and cognitive term scores
data = pd.read_csv(f'{root_dir}/data/derivatives/neurosynth_annotations/glasser/glasser_tracts_neurosynth_125terms.csv')
tractdata = data.filter(regex = 'left|right') 
nsdata = data.iloc[:, 59:] 

# load SA axis data
sa_df = pd.read_csv(f'{root_dir}/data/derivatives/glasser_parcellation/glasser_sa_axis_zscores.csv').drop('regionID', axis=1)

# load nulls
spins = tm_utils.load_data(f'{root_dir}/data/derivatives/nulls/indices_{nspins}spins.pickle') # generated using data_prep/compute_nulls.py

# load PLS results
pls_result = pyls.load_results(f'{results_dir}/pls_result.hdf5')

# load PLS scores
# cognitive term scores
term_scores = pls_result["x_scores"][:, lv] 
# tract scores
tract_scores = pls_result["y_scores"][:, lv] 

# mean of term and tract scores
mean_scores = np.mean([term_scores, tract_scores], axis = 0)


# ------------------------------------------------------------------------------------------------
# --- Spin testing of the association between PLS scores and SA axis ---
# ------------------------------------------------------------------------------------------------

def compute_spin_test_corrs(X_data, Y_data, X_names, spins, corr_metric):
    """
    Computes correlations with spin-based permutation testing and FDR correction.
    
    Parameters:
    -----------
    - X_data (ndarray): Numpy array with shape (regions, features).
    - Y_data (ndarray): Numpy array with shape (regions, PCs).
    - X_names (list of str): List of feature names corresponding to the columns of X_data.
    - spins (dict or ndarray): Either a dictionary of shuffled X_data (with each key corresponding to 
                               a name in X_names and value being a matrix of shuffled X_data (regions, n_spins)) 
                               or a single numpy array of shape (regions, n_spins) for all features,
                               which contains indices to shuffle the original X_data.
    - corr_metric (str): which correlation metric to use. Options are 'pearsonr' or 'spearmanr'.
    
    Returns:
    --------
    - corrs (ndarray): Array of correlation coefficients with shape (features, PCs).
    - pvals_corrected (ndarray): FDR-corrected p-values with shape (features, PCs).
    - nulls_array (ndarray): Null distributions with shape (features, PCs, n_spins).
    - significant_mask (ndarray): Boolean array where True indicates significant correlations.
    """

    # determine number of X and Y variables
    n_xvar = X_data.shape[1]
    n_yvar = Y_data.shape[1]
    
    # determine number of spins
    if isinstance(spins, dict):
        n_spins = next(iter(spins.values())).shape[1]
    elif isinstance(spins, np.ndarray):
        n_spins = spins.shape[1]
    else:
        raise ValueError("spins must be either a dictionary or a numpy array.")

    # initialize result arrays
    corrs = np.zeros((n_xvar, n_yvar))
    pvals = np.zeros((n_xvar, n_yvar))
    nulls_array = np.zeros((n_xvar, n_yvar, n_spins))
    significant_mask = np.zeros((n_xvar, n_yvar), dtype=bool)

    for i in tqdm(range(n_xvar), desc="Computing correlations"): # progress bar
        xvar_name = X_names[i] 
    
        if isinstance(spins, dict):
            # use the pre-shuffled X_data directly from the dictionary
            spin_matrix = spins[xvar_name]
        else:
            # shuffle X_data using indices from spins
            spin_matrix = np.zeros((X_data.shape[0], n_spins))
            for spin in range(n_spins):
                spin_matrix[:, spin] = X_data[spins[:, spin], i]

        for j in range(n_yvar):
            # compute correlation with spin-testing
            r, p, nulls = compare_images(
                X_data[:, i],
                Y_data[:, j],
                metric=corr_metric,
                nulls=spin_matrix,
                return_nulls=True,
                ignore_zero=False
            )
            corrs[i, j] = r
            pvals[i, j] = p
            nulls_array[i, j, :] = nulls

    # FDR correction and creating the significance mask
    pvals_corrected = np.zeros_like(pvals)
    for j in range(n_yvar):
        _, pvals_corrected[:, j], _, _ = multipletests(pvals[:, j], alpha=0.05, method='fdr_bh')
        significant_mask[:, j] = pvals_corrected[:, j] < 0.05

    # count significant correlations for each yvar
    for j in range(n_yvar):
        print(f'{np.sum(significant_mask[:, j]):>4} correlation(s) survive FDR-correction for var{j+1}')


    return corrs, pvals, pvals_corrected, nulls_array, significant_mask


# --- Run spin testing for tract scores ---


# define X and Y data for tract scores
X_data = sa_df.values
Y_data = tract_scores.reshape(-1, 1)

# define X and Y names
X_names = sa_df.columns
Y_names = ['tract_scores']

# run spin testing for tract scores
corrs_tract, pvals_tract, pvals_corrected_tract, nulls_array_tract, significant_mask_tract = compute_spin_test_corrs(X_data, Y_data, X_names, spins, 'spearmanr')

# save results as dataframe for tract scores
df_results_tract = pd.DataFrame({
	'annot_names': np.repeat(X_names, len(Y_names)),
	'score_names': np.tile(Y_names, len(X_names)),
	'correlation': corrs_tract.flatten(),  
	'pval': pvals_tract.flatten(),
	'fdr_pval': pvals_corrected_tract.flatten(),
	'significant': significant_mask_tract.flatten() 
})
df_results_tract.to_csv(f'{output_dir}/tract_scores_sa_axis_spin_testing.csv', index=False)

# save nulls as npy for tract scores
np.save(f'{output_dir}/tract_scores_sa_axis_nulls_array.npy', nulls_array_tract)

# print the correlation coefficient and fdr pvalue for tract scores
corr_tract = df_results_tract.loc[(df_results_tract['annot_names'] == 'sa_axis') & (df_results_tract['score_names'] == 'tract_scores'), 'correlation'].values[0]
pval_tract = df_results_tract.loc[(df_results_tract['annot_names'] == 'sa_axis') & (df_results_tract['score_names'] == 'tract_scores'), 'fdr_pval'].values[0]
print(f'SA axis - tract scores correlation: {corr_tract:.4f}, p-value: {pval_tract:.4f}')

# --- Run spin testing for term scores ---

# define X and Y data for term scores
Y_data = term_scores.reshape(-1, 1)

# define X and Y names
Y_names = ['term_scores']

# run spin testing for term scores
corrs_term, pvals_term, pvals_corrected_term, nulls_array_term, significant_mask_term = compute_spin_test_corrs(X_data, Y_data, X_names, spins, 'spearmanr')

# save results as dataframe for term scores
df_results_term = pd.DataFrame({
	'annot_names': np.repeat(X_names, len(Y_names)),
	'score_names': np.tile(Y_names, len(X_names)),
	'correlation': corrs_term.flatten(),  
	'pval': pvals_term.flatten(),
	'fdr_pval': pvals_corrected_term.flatten(),
	'significant': significant_mask_term.flatten() 
})
df_results_term.to_csv(f'{output_dir}/term_scores_sa_axis_spin_testing.csv', index=False)

# save nulls as npy for term scores
np.save(f'{output_dir}/term_scores_sa_axis_nulls_array.npy', nulls_array_term)

# print the correlation coefficient and fdr pvalue for term scores
corr_term = df_results_term.loc[(df_results_term['annot_names'] == 'sa_axis') & (df_results_term['score_names'] == 'term_scores'), 'correlation'].values[0]
pval_term = df_results_term.loc[(df_results_term['annot_names'] == 'sa_axis') & (df_results_term['score_names'] == 'term_scores'), 'fdr_pval'].values[0]
print(f'SA axis - term scores correlation: {corr_term:.4f}, p-value: {pval_term:.4f}')

print(f"Results saved to {output_dir}") 