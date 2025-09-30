# ------------------------------------------------------------------------------------------------
# --- Partial least squares: cognitive terms and tracts ---
# ------------------------------------------------------------------------------------------------
# Here, PLS is done to capture relationships between two matrices: 
# 
# * brain regions x cognitive terms (360 regions)
# * brain regions x tracts (360 regions, jointly from left and right hemisphere tracts)
# 
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
from scipy.stats import zscore, spearmanr
from joblib import Parallel, delayed
from mapalign.align import iterative_alignment
from scipy.spatial.distance import squareform, pdist


# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# root directory
root_dir = '/Users/joelleba/PennLINC/tractmaps'

# create results directory if it doesn't yet exist
results_dir = f'{root_dir}/results/pls_terms_tracts'
summary_dir = f'{results_dir}/summary/'
for dir in [results_dir, summary_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Folder '{dir}' created.")
    else:
        print(f"Folder '{dir}' already exists.")

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load data
data = pd.read_csv(f'{root_dir}/data/derivatives/neurosynth_annotations/glasser/glasser_tracts_neurosynth_125terms.csv')
nsdata = data.iloc[:, 59:]  # cognitive terms (360 regions x 125 terms)
tractdata = data.filter(regex = 'left|right')  # tract data with left/right columns

# Load nulls
nspins = 10000
spins = tm_utils.load_data(f'{root_dir}/data/derivatives/nulls/indices_{nspins}spins.pickle') # generated using data_prep/compute_nulls.py

# load Glasser 360 parcellation coordinates for cross-validation
coor = pd.read_csv(f'{root_dir}/data/derivatives/glasser_parcellation/glasser_coords.csv')
coor = np.asarray(coor)[:, 3:].astype(float) # keep only the coordinates (discard region labels)

# ------------------------------------------------------------------------------------------------
# --- Merge left and right tract data ---
# ------------------------------------------------------------------------------------------------

print("Merging left and right hemisphere tract data...")

# Get left and right tract data
left_tracts = tractdata.filter(regex='left')
right_tracts = tractdata.filter(regex='right')

# Create merged tract names (remove _left and _right suffixes)
left_tract_names = left_tracts.columns
right_tract_names = right_tracts.columns

# Extract base tract names (remove _left and _right)
base_tract_names = []
for name in left_tract_names:
    base_name = name.replace('_left', '')
    base_tract_names.append(base_name)

print(f"Number of unique tracts: {len(base_tract_names)}")
print(f"Base tract names: {base_tract_names}")

# Check the structure of the data
print(f"left_tracts shape: {left_tracts.shape}")
print(f"right_tracts shape: {right_tracts.shape}")
print(f"nsdata shape: {nsdata.shape}")

# The tractdata already contains both left and right hemispheres
# We need to extract the left and right parts correctly
left_tract_data = left_tracts.iloc[:180, :]  # First 180 regions (left hemisphere)
right_tract_data = right_tracts.iloc[180:, :]  # Last 180 regions (right hemisphere)

# Merge left and right data for each tract
merged_tract_data = pd.DataFrame()

for i, base_name in enumerate(base_tract_names):
    left_col = left_tract_names[i]
    right_col = right_tract_names[i]
    
    # Get left and right data for this tract
    left_data = left_tract_data[left_col].values
    right_data = right_tract_data[right_col].values
    
    # Concatenate left and right (360 regions total)
    merged_data = np.concatenate([left_data, right_data])
    
    # Add to merged dataframe
    merged_tract_data[base_name] = merged_data

print(f"Merged tract data shape: {merged_tract_data.shape}")
print(f"nsdata shape: {nsdata.shape}")

# ------------------------------------------------------------------------------------------------
# --- Run PLS ---
# ------------------------------------------------------------------------------------------------

print('Running PLS...')

X = zscore(nsdata).values  # cognitive terms (360 regions x 124 terms) - convert to numpy array
Y = zscore(merged_tract_data).values  # merged tracts (360 regions x 26 tracts) - convert to numpy array

# Run PLS
pls_result = pyls.behavioral_pls(X, Y, n_boot = nspins, n_perm = nspins, permsamples = spins,
                             test_split = 0, seed = 1234, n_proc = 6)
pyls.save_results(f'{results_dir}/pls_result.hdf5', pls_result)


# ------------------------------------------------------------------------------------------------
# --- Summarize PLS results ---
# ------------------------------------------------------------------------------------------------

np.set_printoptions(suppress = True)

pls_result = pyls.load_results(f'{results_dir}/pls_result.hdf5')

print(f'*** PLS results *** \n')
pvals = pls_result.permres.pvals
varexp = pls_result.varexp * 100  # converting to percentage

# Count significant LVs
significant_count = np.sum(pvals < 0.05)
print(f'{significant_count} latent variable(s)\n')

# Create and print summary table
lv_summary = pd.DataFrame({
    'LV': np.arange(1, len(pvals) + 1),
    'Variance Explained (%)': np.round(varexp, 1),
    'p-value': np.round(pvals, 4)
})
print(lv_summary.to_string(index=False))
lv_summary.to_csv(f'{summary_dir}/lv_summary.csv', index = False)

# ------------------------------------------------------------------------------------------------
# --- Cross-validation ---
# ------------------------------------------------------------------------------------------------

## Cross-validation functions from Golia Shafiei's super cool MEG paper! https://github.com/netneurolab/shafiei_megdynamics/blob/main/code/analysis/scpt_pls.py ###
def spinCV(iSpin, orig_x_wei, orig_y_wei, spins, X, Y, coords, trainpct):
    '''
    Perform spin-based cross-validation for PLS analysis.
    
    This function computes the mean test correlation for a specific spin iteration,
    using distance-based cross-validation. Each node is used to create training and
    test datasets based on proximity, and a Partial Least Squares (PLS) model is trained
    to evaluate the correlation between test set scores for X and Y.

    Parameters
    ----------
    iSpin : int
        Index of the current spin permutation.
    orig_x_wei : list of arrays
        Original X weights from the PLS model for each node.
    orig_y_wei : list of arrays
        Original Y weights from the PLS model for each node.
    spins : (n, nspins) array_like
        Spin-test permutations.
    X : (n, p1) array_like
        Input data matrix. `n` is the number of brain regions.
    Y : (n, p2) array_like
        Input data matrix. `n` is the number of brain regions.
    coords : (n, 3) array_like
        Region (x,y,z) coordinates. `n` is the number of brain regions.
    trainpct : float
        Percent observations in train set. 0 < trainpct < 1.

    Returns
    -------
    testnull : (spins.shape[1], ) array
        Mean test set correlation for each spin permutation.
    error_list : list of tuples
        List of spin and node indices where the PLS analysis encountered errors.
    '''

    nnodes = len(coords)
    P = squareform(pdist(coords, metric="euclidean"))
    testnull = np.zeros((spins.shape[1], ))
    t = np.zeros((nnodes, ))
    Y_null = Y[spins[:, iSpin], :]
    lv = 0
    error_list = []
    
    for node in range(nnodes):
        # distance from the current node to all others
        distances = P[node, :]
        idx = np.argsort(distances) # sort indices by ascending distance

        # split data into train and test based on distance
        train_idx = idx[:int(np.floor(trainpct * nnodes))]
        test_idx = idx[int(np.floor(trainpct * nnodes)):]
        
        Xtrain = X[train_idx, :]
        Xtest = X[test_idx, :]
        Ytrain = Y_null[train_idx, :]
        Ytest = Y_null[test_idx, :]
      
        # pls analysis
        # train_result = pyls.behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        
        # added a try to account for a small number of cases where the PLS errors out due to the presence of NaNs or Inf, likely during SVD
        try:
            train_result = pyls.behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        except ValueError as e:
            print(f"Error during PLS analysis at spin {iSpin}, node {node}: {e}")
            error_list.append((iSpin, node))
            continue
            
        # extract weights from PLS analysis
        null_x_wei = train_result['x_weights']
        null_y_wei = train_result['y_weights']

        # align the weights to the original weights for both X and Y
        temp = [orig_x_wei[node], null_x_wei]
        realigned_null_x_wei, _ = iterative_alignment(temp, n_iters=1)

        temp = [orig_y_wei[node], null_y_wei]
        realigned_null_y_wei, _ = iterative_alignment(temp, n_iters=1)

        # compute Spearman correlation between the X and Y scores for the test set
        t[node], _ = spearmanr(Xtest @ realigned_null_x_wei[1][:, lv],
                                           Ytest @ realigned_null_y_wei[1][:, lv])

    # get the mean test correlation for this spin
    testnull[iSpin] = np.mean(t)

    return testnull, error_list


def pls_cv_distance_dependent_par(X, Y, coords, trainpct=0.75, lv=0, 
                                  testnull=False, spins=None, nspins=1000):
    """
    Distance-dependent cross validation.

    This function uses a distance-based cross-validation approach where nodes are split into training and test sets
    based on their proximity to each other. The PLS model is trained on the training set, and scores are computed
    by projecting the test set using the trained latent variable weights from the PLS model. The correlations between
    the X and Y test scores are then calculated to evaluate model performance. 
    If `testnull=True`, the function also performs spin-based permutation testing to assess the significance of 
    the observed correlations.

    Parameters
    ----------
    X : (n, p1) array_like
        Input data matrix. `n` is the number of brain regions.
    Y : (n, p2) array_like
        Input data matrix. `n` is the number of brain regions.
    coords : (n, 3) array_like
        Region (x,y,z) coordinates. `n` is the number of brain regions.
    trainpct : float
        Percent observations in train set. 0 < trainpct < 1.
        Default = 0.75.
    lv : int
        Index of latent variable to cross-validate. Default = 0.
    testnull : Boolean
        Whether to calculate and return null mean test-set correlation.
    spins : (n, nspins) array_like
        Spin-test permutations. Required if testnull=True.
    nspins : int
        Number of spin-test permutations. Only used if testnull=True
    Returns
    -------
    train : (nplit, ) array
        Training set correlation between X and Y scores.
    test : (nsplit, ) array
        Test set correlation between X and Y scores.
    testnullres: (nspins, ) array
        Null correlations between X and Y scores.
        
    """
    
    nnodes = len(coords)
    train = np.zeros((nnodes, ))
    test = np.zeros((nnodes, ))

    orig_x_wei = []
    orig_y_wei = []

    P = squareform(pdist(coords, metric='euclidean'))
    
    error_list = []
    test_scores = [] # to store correlations of X and Y scores for all test sets

    for k in range(nnodes):

        # distance from the current node to all others
        distances = P[k, :]
        idx = np.argsort(distances) # sort indices by ascending distance

        # split data based on distance
        train_idx = idx[:int(np.floor(trainpct * nnodes))]
        test_idx = idx[int(np.floor(trainpct * nnodes)):]

        # assign training and testing for X and Y
        Xtrain = X[train_idx, :]
        Xtest = X[test_idx, :]
        Ytrain = Y[train_idx, :]
        Ytest = Y[test_idx, :]
        
        # pls analysis
        train_result = pyls.behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)

        # get training set Spearman correlation between X and Y scores
        train[k], _ = spearmanr(train_result['x_scores'][:, lv],
                               train_result['y_scores'][:, lv])
        
        # project weights onto the test set and compute Spearman correlation between predicted scores
        test_score_x = Xtest @ train_result['x_weights'][:, lv]  # Project Xtest onto the train X weights
        test_score_y = Ytest @ train_result['y_weights'][:, lv]  # Project Ytest onto the train Y weights
        test[k], _ = spearmanr(test_score_x, test_score_y)

        # store original weights for use in spin-based cross-validation
        orig_x_wei.append(train_result['x_weights'])
        orig_y_wei.append(train_result['y_weights'])

        # append the X and Y scores for the test set to the list
        test_scores.append((test_score_x, test_score_y))

    # if testnull=True, get distribution of mean null test-set correlations.
    if testnull:
        print('Running null test-set correlations, will take time')
        results = Parallel(n_jobs = 6)(delayed(spinCV)(i, orig_x_wei, orig_y_wei, spins, X, Y, coords, trainpct) for i in range(nspins))
        testnullres = np.array([results[i][0] for i in range(nspins)]) # first get testnullres matrix
        testnullres = np.array([testnullres[i][i] for i in range(nspins)]) # tract diagonal containing mean correlations
        error_lists = [results[i][1] for i in range(nspins)]
        error_list = [item for sublist in error_lists for item in sublist] # list of spin & node combinations resultng in PLS errors
    else:
        testnullres = None
        error_list = []

    total_spins_nodes = nspins * nnodes
    error_proportion = len(error_list) / total_spins_nodes

    if testnull:
        return train, test, testnullres, test_scores, error_list, error_proportion
    else:
        return train, test, error_list, test_scores, error_proportion
    

# run cross-validation
print('Running cross-validation...')

# run cross-validation
train, test, testnullres, test_scores, error_list, error_proportion = pls_cv_distance_dependent_par(X, Y, coords = coor, trainpct = 0.75,
                                                                                       lv = 0, testnull = True, spins = spins, nspins = 1000)

np.save(f'{results_dir}/pls_glasser360_train_corrs.npy', train)
np.save(f'{results_dir}/pls_glasser360_test_corrs.npy', test)
np.save(f'{results_dir}/pls_glasser360_null_corrs.npy', testnullres)
np.save(f'{results_dir}/pls_glasser360_test_scores.npy', test_scores)

print(f"Results saved to {results_dir}")