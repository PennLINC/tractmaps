# ------------------------------------------------------------------------------------------------
# --- Tract S-A Range vs Gini Coefficient Correlation Testing with Null Models ---
# This script tests the significance of correlation between tract S-A range and tract Gini coefficient
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
import tempfile
import importlib.util

sys.path.append('/Users/joelleba/PennLINC/tractmaps/code')
from data_prep.prep_sa_axis import calculate_tract_sa_ranges
from data_prep.tract_rewiring_nulls import plot_correlation_null_distribution

# Import functions from numbered directories using importlib
gini_spec = importlib.util.spec_from_file_location("tract_gini_coefficients", 
    "/Users/joelleba/PennLINC/tractmaps/code/analysis/4_functional_diversity/tract_gini_coefficients.py")
gini_module = importlib.util.module_from_spec(gini_spec)
gini_spec.loader.exec_module(gini_module)
calculate_tract_gini_coefficients = gini_module.calculate_tract_gini_coefficients

term_contrib_spec = importlib.util.spec_from_file_location("tract_term_contributions", 
    "/Users/joelleba/PennLINC/tractmaps/code/analysis/2_functional_decoding/tract_term_contributions.py")
term_contrib_module = importlib.util.module_from_spec(term_contrib_spec)
term_contrib_spec.loader.exec_module(term_contrib_module)
calculate_tract_term_contributions = term_contrib_module.calculate_tract_term_contributions

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
results_dir = f'{root}/results/tract_functional_diversity/sensitivity'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Load S-A axis data
sa_path = f'{root}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv'
sa_data = pd.read_csv(sa_path)
sa_values = sa_data['sa_axis'].values
print(f"Loaded S-A axis data for {len(sa_values)} regions")

# Load the full data file (needed for creating null datasets with terms)
data_file_path = os.path.join(root, 'data/derivatives/neurosynth_annotations/glasser/glasser_tracts_neurosynth_125terms.csv')
full_data = pd.read_csv(data_file_path, index_col=0)
print(f"Loaded full dataset with {full_data.shape[0]} regions and {full_data.shape[1]} columns")

# Load tract name mapping
tract_names_df = pd.read_excel(os.path.join(root, 'data/raw/tract_names/abbreviations.xlsx'))
tract_name_mapping = dict(zip(tract_names_df['Tract'], tract_names_df['Tract_Long_Name']))

# ------------------------------------------------------------------------------------------------
# --- Main Analysis ---
# ------------------------------------------------------------------------------------------------

print("Testing significance of S-A range vs Gini coefficient correlation...")

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

# Compute empirical tract Gini coefficients (Y-axis)
print("Computing empirical tract Gini coefficients...")
# Use the existing calculate_tract_term_contributions function for consistency
with contextlib.redirect_stdout(io.StringIO()):
    empirical_term_contrib_df = calculate_tract_term_contributions(
        data_file_path=data_file_path,
        terms_start_col=58,
        tract_regex='left|right',
        connection_thresh=tract_threshold
    )

# Calculate Gini coefficients from empirical term contributions
with contextlib.redirect_stdout(io.StringIO()):
    _, gini_df, _, _ = calculate_tract_gini_coefficients(empirical_term_contrib_df, tract_name_mapping)

empirical_y_df = pd.DataFrame({
    'tract_name': gini_df['Tract'],
    'gini_coefficient': gini_df['Gini_Coefficient']
})

# Merge empirical data
empirical_df = pd.merge(empirical_x_df, empirical_y_df, on='tract_name')

# Remove NaN values for correlation
empirical_clean = empirical_df.dropna(subset=['sa_range', 'gini_coefficient'])

# Compute empirical correlation
if correlation_type == 'pearson':
    empirical_corr, _ = pearsonr(empirical_clean['sa_range'], empirical_clean['gini_coefficient'])
elif correlation_type == 'spearman':
    empirical_corr, _ = spearmanr(empirical_clean['sa_range'], empirical_clean['gini_coefficient'])
elif correlation_type == 'kendall':
    empirical_corr, _ = kendalltau(empirical_clean['sa_range'], empirical_clean['gini_coefficient'])
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

for i, null_dataset in enumerate(iterator):
    try:
        # Create a temporary dataset by replacing tract columns in full_data with null tract data
        # Keep the original non-tract columns (including terms) and replace only tract columns
        null_full_data = full_data.copy()
        
        # Replace tract columns with null dataset values
        tract_columns = [col for col in null_dataset.columns if col.endswith('_left') or col.endswith('_right')]
        for col in tract_columns:
            if col in null_full_data.columns:
                null_full_data[col] = null_dataset[col]
        
        # Create temporary file for the null dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            null_full_data.to_csv(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Calculate term contributions for null dataset using the existing function
            with contextlib.redirect_stdout(io.StringIO()):
                null_term_contrib_df = calculate_tract_term_contributions(
                    data_file_path=temp_file_path,
                    terms_start_col=58,
                    tract_regex='left|right',
                    connection_thresh=tract_threshold
                )
            
            # Calculate Gini coefficients for null tract data
            with contextlib.redirect_stdout(io.StringIO()):
                _, null_gini_df, _, _ = calculate_tract_gini_coefficients(null_term_contrib_df, tract_name_mapping)
            
            null_y_df = pd.DataFrame({
                'tract_name': null_gini_df['Tract'],
                'gini_coefficient': null_gini_df['Gini_Coefficient']
            })
            
            # Merge empirical S-A ranges with null Gini coefficients
            null_df = pd.merge(empirical_x_df, null_y_df, on='tract_name')
            null_dfs.append(null_df)
            
            # Remove NaN values for correlation
            null_clean = null_df.dropna(subset=['sa_range', 'gini_coefficient'])
            
            if len(null_clean) >= 2:
                if correlation_type == 'pearson':
                    null_corr, _ = pearsonr(null_clean['sa_range'], null_clean['gini_coefficient'])
                elif correlation_type == 'spearman':
                    null_corr, _ = spearmanr(null_clean['sa_range'], null_clean['gini_coefficient'])
                elif correlation_type == 'kendall':
                    null_corr, _ = kendalltau(null_clean['sa_range'], null_clean['gini_coefficient'])
                null_correlations.append(null_corr)
            else:
                null_correlations.append(np.nan)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Error processing null dataset {i}: {e}")
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
        title='S-A Range vs Gini Coefficient',
        outpath=f'{results_dir}/sa_range_gini_coefficient_correlation_distribution.png'
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

results_df.to_csv(f'{results_dir}/sa_range_gini_correlation_results_rewiring_null.csv', index=False)
print(f"Saved correlation test results to: {results_dir}/sa_range_gini_correlation_results_rewiring_null.csv")

print("S-A range vs Gini coefficient correlation testing complete!")
