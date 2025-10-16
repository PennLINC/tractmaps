# ------------------------------------------------------------------------------------------------
# --- Tract S-A Range vs Cortical Similarity Correlation Testing with Null Models ---
# This script tests the significance of correlation between tract S-A range and tract mean cortical similarity
# using rewiring null models. 
# Note: run the network_rewiring_nulls.py script first to generate the null datasets. 
# Also, this script will take a while to run!
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import sys
import pickle
from scipy.stats import pearsonr, spearmanr, kendalltau
import contextlib
import io

sys.path.append('/Users/joelleba/PennLINC/tractmaps/code')
from data_prep.prep_sa_axis import calculate_tract_sa_ranges
from utils.matrix_to_tracts import extract_tract_means_from_matrix
from data_prep.network_rewiring_nulls import plot_correlation_null_distribution

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and load data ---
# ------------------------------------------------------------------------------------------------
root = '/Users/joelleba/PennLINC/tractmaps'
data_root = f'{root}/data/derivatives/'
tracts_dir = f'{root}/data/derivatives/tracts'
nulls_dir = f'{data_root}/nulls/degree_preserving_nulls'

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Set correlation parameters
correlation_type = 'spearman'
n_nulls = 10000
progress_bar = True
plot_results = False

# Load tract data
tract_connection_path = f'{tracts_dir}/tracts_probabilities/tracts_probabilities.csv'
tractdata = pd.read_csv(tract_connection_path)
tractdata = tractdata.filter(regex='left|right')
print(f"Loaded tract connection data for {len(tractdata)} regions, {len(tractdata.filter(regex='left|right').columns)} tracts")

# Load null datasets
null_pickle_path = f'{nulls_dir}/null_tractdata_{n_nulls}nulls.pkl'
if os.path.exists(null_pickle_path):
    with open(null_pickle_path, 'rb') as f:
        null_datasets = pickle.load(f)
    print(f"Loaded {len(null_datasets)} null datasets from: {null_pickle_path}")
else:
    raise FileNotFoundError(f"Null datasets not found at: {null_pickle_path}. Please run region_label_nulls.py first.")

# Create results directory
results_dir = f'{root}/results/cortical_similarity/sensitivity'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Load S-A axis data
sa_path = f'{root}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv'
sa_data = pd.read_csv(sa_path)
sa_values = sa_data['sa_axis'].values
print(f"Loaded S-A axis data for {len(sa_values)} regions")

# Load cortical similarity matrix
cortical_similarity_path = f'{root}/data/derivatives/tracts/tracts_cortical_similarity/cortical_similarity.npy'
cortical_similarity_matrix = np.load(cortical_similarity_path)
print(f"Loaded cortical similarity matrix: {cortical_similarity_matrix.shape}")

# ------------------------------------------------------------------------------------------------
# --- Main Analysis ---
# ------------------------------------------------------------------------------------------------

print("Testing significance of S-A range vs cortical similarity correlation...")

# Limit number of nulls if specified
if n_nulls is not None:
    null_datasets = null_datasets[:n_nulls]

print(f"Testing correlation significance using {len(null_datasets)} null datasets...")

# Compute empirical tract S-A ranges (X-axis)
print("Computing empirical tract S-A ranges...")
sa_ranges_df = calculate_tract_sa_ranges(tractdata, sa_values, tract_threshold)
empirical_x_df = pd.DataFrame({
    'tract_name': sa_ranges_df['Tract'],
    'sa_range': sa_ranges_df['SA_Range']
})

# Compute empirical tract mean cortical similarity (Y-axis)
print("Computing empirical tract mean cortical similarity...")
_, cortical_similarity_means_df = extract_tract_means_from_matrix(
    tractdata=tractdata,
    region_matrix=cortical_similarity_matrix,
    tract_threshold=tract_threshold,
    matrix_name='Cortical_Similarity',
    tract_names=None
)
empirical_y_df = pd.DataFrame({
    'tract_name': cortical_similarity_means_df['Tract'],
    'cortical_similarity_mean': cortical_similarity_means_df['Mean_Cortical_Similarity']
})

# Merge empirical data
empirical_df = pd.merge(empirical_x_df, empirical_y_df, on='tract_name')

# Remove NaN values for correlation
empirical_clean = empirical_df.dropna(subset=['sa_range', 'cortical_similarity_mean'])

# Compute empirical correlation
if correlation_type == 'pearson':
    empirical_corr, _ = pearsonr(empirical_clean['sa_range'], empirical_clean['cortical_similarity_mean'])
elif correlation_type == 'spearman':
    empirical_corr, _ = spearmanr(empirical_clean['sa_range'], empirical_clean['cortical_similarity_mean'])
elif correlation_type == 'kendall':
    empirical_corr, _ = kendalltau(empirical_clean['sa_range'], empirical_clean['cortical_similarity_mean'])
else:
    raise ValueError("correlation_type must be 'pearson', 'spearman', or 'kendall'")

print(f"Empirical {correlation_type} correlation: {empirical_corr:.3f}")

# Compute null correlations
print("Computing null correlations...")
null_correlations = []
null_dfs = []

if progress_bar:
    iterator = tqdm(null_datasets, desc="Computing null correlations")
else:
    iterator = null_datasets

for null_dataset in iterator:
    # For cortical similarity, we use the original similarity matrix with null tract data
    # Suppress prints from extract_tract_means_from_matrix to avoid interfering with progress bar
    with contextlib.redirect_stdout(io.StringIO()):
        _, null_y_means_df = extract_tract_means_from_matrix(
            tractdata=null_dataset,
            region_matrix=cortical_similarity_matrix,
            tract_threshold=tract_threshold,
            matrix_name='Cortical_Similarity',
            tract_names=None
        )
    
    null_y_df = pd.DataFrame({
        'tract_name': null_y_means_df['Tract'],
        'cortical_similarity_mean': null_y_means_df['Mean_Cortical_Similarity']
    })
    
    # Merge empirical S-A ranges with null cortical similarity
    null_df = pd.merge(empirical_x_df, null_y_df, on='tract_name')
    null_dfs.append(null_df)
    
    # Remove NaN values for correlation
    null_clean = null_df.dropna(subset=['sa_range', 'cortical_similarity_mean'])
    
    if len(null_clean) >= 2:
        if correlation_type == 'pearson':
            null_corr, _ = pearsonr(null_clean['sa_range'], null_clean['cortical_similarity_mean'])
        elif correlation_type == 'spearman':
            null_corr, _ = spearmanr(null_clean['sa_range'], null_clean['cortical_similarity_mean'])
        elif correlation_type == 'kendall':
            null_corr, _ = kendalltau(null_clean['sa_range'], null_clean['cortical_similarity_mean'])
        null_correlations.append(null_corr)
    else:
        null_correlations.append(np.nan)

null_correlations = np.array(null_correlations)

# Compute two-tailed test p-value as the proportion of nulls with abs deviation from mean > empirical abs deviation from mean
valid_nulls = ~np.isnan(null_correlations)
if np.sum(valid_nulls) > 0:
    null_correlations_clean = null_correlations[valid_nulls]
    null_mean = np.nanmean(null_correlations_clean)
    
    # Count how many nulls have abs deviation from mean > empirical abs deviation from mean
    empirical_deviation = abs(empirical_corr - null_mean)
    null_deviations = abs(null_correlations_clean - null_mean)
    
    # Add 1 to numerator and denominator to include empirical value itself
    p_value = (1 + np.sum(null_deviations > empirical_deviation)) / (len(null_correlations_clean) + 1)
else:
    p_value = np.nan

print(f"P-value: {p_value:.4f}")
print(f"Mean null correlation: {np.mean(null_correlations[valid_nulls]):.3f}")
print(f"Std null correlation: {np.std(null_correlations[valid_nulls]):.3f}")

# Create plot if requested
if plot_results:
    plot_correlation_null_distribution(
        empirical_correlation=empirical_corr,
        null_correlations=null_correlations,
        correlation_type=correlation_type,
        outpath=f'{results_dir}/sa_range_cortical_similarity_correlation_distribution.png'
    )

# ------------------------------------------------------------------------------------------------
# --- Save Results ---
# ------------------------------------------------------------------------------------------------

print(f"\nCorrelation test results:")
print(f"Empirical correlation: {empirical_corr:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Mean null correlation: {np.mean(null_correlations[~np.isnan(null_correlations)]):.3f}")
print(f"Std null correlation: {np.std(null_correlations[~np.isnan(null_correlations)]):.3f}")

# Save results
results_df = pd.DataFrame({
    'empirical_correlation': [empirical_corr],
    'p_value': [p_value],
    'mean_null_correlation': [np.mean(null_correlations[~np.isnan(null_correlations)])],
    'std_null_correlation': [np.std(null_correlations[~np.isnan(null_correlations)])],
    'n_nulls': [len(null_correlations)],
    'correlation_type': [correlation_type]
})

results_df.to_csv(f'{results_dir}/sa_range_cortical_similarity_correlation_rewiring_null.csv', index=False)
print(f"Saved correlation test results to: {results_dir}/sa_range_cortical_similarity_correlation_rewiring_null.csv")

print("S-A range vs cortical similarity correlation testing complete!")
