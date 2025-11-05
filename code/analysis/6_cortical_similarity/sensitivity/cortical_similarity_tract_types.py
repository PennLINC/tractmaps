# ------------------------------------------------------------------------------------------------
#### Sensitivity Analysis: Cortical Similarity by Tract Type ####
# ------------------------------------------------------------------------------------------------
# This script plots cortical similarity analyses for projection and association tracts separately
# Includes both S-A range and Gini coefficient relationships
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
import pickle

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# root directory
root_dir = '/Users/joelleba/PennLINC/tractmaps'
data_dir = f'{root_dir}/data/derivatives/tracts/tracts_cortical_similarity/'
results_dir = f'{root_dir}/results/cortical_similarity/'
output_dir = f'{results_dir}/sensitivity'

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# set parameters
connection_threshold = 0.5

# get custom colormaps
warm_cmap, _, _, _, _, bppy_cmap = tm_utils.make_colormaps()

# set fontsize for all plots
plt.rcParams.update({'font.size': 18})

print("Using cortical similarity for tract type sensitivity analysis")

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load tract information and S-A ranges
tract_types = pd.read_excel(f'{root_dir}/data/raw/tract_names/abbreviations.xlsx')

# Load S-A ranges
sa_ranges_df = pd.read_csv(f'{root_dir}/data/derivatives/tracts/tracts_sa_axis/tract_sa_axis_ranges_thresh{int(connection_threshold * 100)}.csv')

# Load S-A axis ranks for regions 
sa_axis_df = pd.read_csv(f'{root_dir}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv')

# Load Gini coefficients
gini_df = pd.read_csv(f'{root_dir}/results/tract_functional_diversity/gini_coefficients/tract_gini_term_scores.csv').drop('Hemisphere', axis=1)

# Map gini_df long tract names to short tract names
gini_df['Tract'] = gini_df['Tract'].map(tract_types.set_index('Tract_Long_Name')['Tract'])

# Load mean cortical similarities
mean_similarities_df = pd.read_csv(f'{data_dir}/tracts_mean_cortical_similarity.csv')

# Convert to dictionary for compatibility with existing code
means = dict(zip(mean_similarities_df['Tract'], mean_similarities_df['Mean_Cortical_Similarity']))

# ------------------------------------------------------------------------------------------------
# --- Create dataframe with tract information, S-A ranges, Gini coefficients, and mean similarities ---
# ------------------------------------------------------------------------------------------------

tract_data = []
for tract in means.keys():
    # Get tract info
    tract_info = tract_types[tract_types['Tract'] == tract]
    if not tract_info.empty:
        tract_data.append({
            'Tract': tract,
            'Type': tract_info.iloc[0]['Type'],
            'Hemisphere': tract_info.iloc[0]['Hemisphere'],
            'Mean_Similarity': means[tract]
        })

tract_df = pd.DataFrame(tract_data)

# Merge with S-A ranges
tract_df = tract_df.merge(sa_ranges_df[['Tract', 'SA_Range']], on='Tract', how='left')

# Merge with Gini coefficients (both use short tract names)
tract_df = tract_df.merge(gini_df[['Tract', 'Gini_Coefficient']], on='Tract', how='left')

# ------------------------------------------------------------------------------------------------
# --- Get tract subgroups by type ---
# ------------------------------------------------------------------------------------------------

# Get projection and association tracts
projection_tracts = tract_df[tract_df['Type'] == 'Projection']['Tract'].tolist()
association_tracts = tract_df[tract_df['Type'] == 'Association']['Tract'].tolist()

print(f"Found {len(projection_tracts)} projection tracts and {len(association_tracts)} association tracts")

# ------------------------------------------------------------------------------------------------
# --- Analyze projection tracts ---
# ------------------------------------------------------------------------------------------------

print("Analyzing projection tracts...")

# Projection tracts: Mean cortical similarity vs S-A range
proj_sa_df = tract_df[tract_df['Tract'].isin(projection_tracts)].copy()
proj_sa_df = proj_sa_df.dropna(subset=['SA_Range', 'Mean_Similarity'])

if not proj_sa_df.empty:
    x_data_proj_sa = proj_sa_df['SA_Range'].values
    y_data_proj_sa = proj_sa_df['Mean_Similarity'].values
    tract_labels_proj_sa = [tract.replace('_', ' ') for tract in proj_sa_df['Tract'].values]

    # Calculate correlation
    corr_result_proj_sa = tm_utils.perm_corr_test(x_data_proj_sa, y_data_proj_sa, n_permutations=10000, method='spearman', 
                                                 alternative='two-sided', random_state=42)
    r_value_proj_sa = corr_result_proj_sa['observed_corr']
    p_value_proj_sa = corr_result_proj_sa['p_value']

# Projection tracts: Mean cortical similarity vs Gini coefficient
proj_gini_df = tract_df[tract_df['Tract'].isin(projection_tracts)].copy()
proj_gini_df = proj_gini_df.dropna(subset=['Gini_Coefficient', 'Mean_Similarity'])

if not proj_gini_df.empty:
    x_data_proj_gini = proj_gini_df['Gini_Coefficient'].values
    y_data_proj_gini = proj_gini_df['Mean_Similarity'].values
    tract_labels_proj_gini = [tract.replace('_', ' ') for tract in proj_gini_df['Tract'].values]

    # Calculate correlation
    corr_result_proj_gini = tm_utils.perm_corr_test(x_data_proj_gini, y_data_proj_gini, n_permutations=10000, method='spearman', 
                                                   alternative='two-sided', random_state=42)
    r_value_proj_gini = corr_result_proj_gini['observed_corr']
    p_value_proj_gini = corr_result_proj_gini['p_value']

# ------------------------------------------------------------------------------------------------
# --- Analyze association tracts ---
# ------------------------------------------------------------------------------------------------

print("Analyzing association tracts...")

# Association tracts: Mean cortical similarity vs S-A range
assoc_sa_df = tract_df[tract_df['Tract'].isin(association_tracts)].copy()
assoc_sa_df = assoc_sa_df.dropna(subset=['SA_Range', 'Mean_Similarity'])

if not assoc_sa_df.empty:
    x_data_assoc_sa = assoc_sa_df['SA_Range'].values
    y_data_assoc_sa = assoc_sa_df['Mean_Similarity'].values
    tract_labels_assoc_sa = [tract.replace('_', ' ') for tract in assoc_sa_df['Tract'].values]

    # Calculate correlation
    corr_result_assoc_sa = tm_utils.perm_corr_test(x_data_assoc_sa, y_data_assoc_sa, n_permutations=10000, method='spearman', 
                                                  alternative='two-sided', random_state=42)
    r_value_assoc_sa = corr_result_assoc_sa['observed_corr']
    p_value_assoc_sa = corr_result_assoc_sa['p_value']


# Association tracts: Mean cortical similarity vs Gini coefficient
assoc_gini_df = tract_df[tract_df['Tract'].isin(association_tracts)].copy()
assoc_gini_df = assoc_gini_df.dropna(subset=['Gini_Coefficient', 'Mean_Similarity'])

if not assoc_gini_df.empty:
    x_data_assoc_gini = assoc_gini_df['Gini_Coefficient'].values
    y_data_assoc_gini = assoc_gini_df['Mean_Similarity'].values
    tract_labels_assoc_gini = [tract.replace('_', ' ') for tract in assoc_gini_df['Tract'].values]

    # Calculate correlation
    corr_result_assoc_gini = tm_utils.perm_corr_test(x_data_assoc_gini, y_data_assoc_gini, n_permutations=10000, method='spearman', 
                                                    alternative='two-sided', random_state=42)
    r_value_assoc_gini = corr_result_assoc_gini['observed_corr']
    p_value_assoc_gini = corr_result_assoc_gini['p_value']

# ------------------------------------------------------------------------------------------------
# --- Save correlation results to CSV ---
# ------------------------------------------------------------------------------------------------

# Save correlation results to CSV
results_summary = pd.DataFrame({
    'analysis': ['projection_tracts_cortical_sim_vs_sa', 'projection_tracts_cortical_sim_vs_gini', 
                'association_tracts_cortical_sim_vs_sa', 'association_tracts_cortical_sim_vs_gini'],
    'observed_correlation': [r_value_proj_sa, r_value_proj_gini, r_value_assoc_sa, r_value_assoc_gini],
    'p_value': [p_value_proj_sa, p_value_proj_gini, p_value_assoc_sa, p_value_assoc_gini],
    'n_samples': [corr_result_proj_sa['n_samples'], corr_result_proj_gini['n_samples'], 
                 corr_result_assoc_sa['n_samples'], corr_result_assoc_gini['n_samples']],
    'n_permutations': [corr_result_proj_sa['n_permutations'], corr_result_proj_gini['n_permutations'],
                      corr_result_assoc_sa['n_permutations'], corr_result_assoc_gini['n_permutations']],
    'method': [corr_result_proj_sa['method'], corr_result_proj_gini['method'],
              corr_result_assoc_sa['method'], corr_result_assoc_gini['method']],
    'alternative': [corr_result_proj_sa['alternative'], corr_result_proj_gini['alternative'],
                   corr_result_assoc_sa['alternative'], corr_result_assoc_gini['alternative']],
    'x_variable': ['SA_Range', 'Gini_Coefficient', 'SA_Range', 'Gini_Coefficient'],
    'y_variable': ['Mean_Cortical_Similarity', 'Mean_Cortical_Similarity', 
                  'Mean_Cortical_Similarity', 'Mean_Cortical_Similarity'],
    'n_tracts': [len(projection_tracts), len(projection_tracts), len(association_tracts), len(association_tracts)]
})
results_summary.to_csv(f'{output_dir}/correlation_results_cortical_similarity_tract_types.csv', index=False)

print(f"Generated plots for projection ({len(projection_tracts)} tracts) and association ({len(association_tracts)} tracts) tract types with cortical similarity analyses")
print(f"Saved correlation results to: correlation_results_cortical_similarity_tract_types.csv")
