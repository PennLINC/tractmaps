# ------------------------------------------------------------------------------------------------
# --- Calculate term contributions for each tract ---
# ------------------------------------------------------------------------------------------------

# This script calculates the term contributions for each tract and saves them in tract_term_contributions.csv
# as a terms x tracts matrix containing the mean of normalized z-scores (across connected regions) per tract.

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import abagen

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------


# Set up directories
root_dir = '/Users/joelleba/PennLINC/tractmaps'

# output directory
decoding_results_dir = os.path.join(root_dir, 'results/tract_functional_decoding/decoding_results')
if not os.path.exists(decoding_results_dir):
    os.makedirs(decoding_results_dir)
    print(f"Folder '{decoding_results_dir}' created.")
else:
    print(f"Folder '{decoding_results_dir}' already exists.")

# Set data file path
data_file_path = os.path.join(root_dir, 'data/derivatives/neurosynth_annotations/glasser/glasser_tracts_neurosynth_125terms.csv')

# ------------------------------------------------------------------------------------------------
# --- Function to calculate tract term contributions ---
# ------------------------------------------------------------------------------------------------

def calculate_tract_term_contributions(data_file_path, terms_start_col=58, tract_regex='left|right', connection_thresh=0.5):
    """
    Calculate term contributions for each tract.
    
    Parameters:
    -----------
    data_file_path : str
        Path to the CSV file containing tract and term data
    terms_start_col : int
        Column index where term data begins (default: 59)
    tract_regex : str
        Regex pattern to filter tract columns (default: 'left|right')
    connection_thresh : float
        Connection threshold for determining connected regions (default: 0.5)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with terms as index and tracts as columns containing term contributions
    """
    
    print("Loading data...")
    data = pd.read_csv(data_file_path, index_col=0)
    nsdata = data.iloc[:, terms_start_col:]  # terms data (regions x terms)
    tractdata = data.filter(regex=tract_regex)  # tract connection data (regions x tracts)

    print(f"Loaded data for {len(tractdata.columns)} tracts and {len(nsdata.columns)} terms")

    print("Extracting term contributions for each tract...")

    # Store location of all negative values in the data; this will be used as a mask later to set those cells to 0
    neg_mask = nsdata < 0

    # Store location of raw z-scores < 1.64 (non-significant at p < 0.05, one-tailed)
    nonsig_mask = nsdata < 1.64

    # Non-linear normalization (scaled robust sigmoid) of terms (columns) to mitigate the effect of outliers (very high term z-scores)
    print("Normalizing expression data...")
    norm_nsdata = abagen.normalize_expression(nsdata, norm='scaled_robust_sigmoid')

    # Apply neg_mask to set all negative values to 0
    norm_nsdata = norm_nsdata.mask(neg_mask, 0)

    # Apply nonsig_mask to set all non-significant values (raw z < 1.64) to 0
    norm_nsdata = norm_nsdata.mask(nonsig_mask, 0)

    # Create dictionary to store tract-specific term contributions
    all_tracts_data = {}

    # For each tract, calculate term contributions based on connected regions
    for tract in tractdata.columns:
        # Subset regions connected to the tract based on the threshold
        connected_regions = np.where(tractdata[tract] >= connection_thresh)[0]

        if len(connected_regions) == 0:
            print(f'Tract: {tract} has no connected regions above the threshold.')
            continue
        
        # Select terms associated with the connected regions
        connected_terms = norm_nsdata.iloc[connected_regions]

        # Calculate mean contribution: mean of positive values across regions for each term (column)
        positive_term_means = connected_terms.clip(lower=0).mean(axis=0)
        
        # Store tract data
        all_tracts_data[tract] = positive_term_means

    # Create DataFrame with terms as index and tracts as columns
    tract_df = pd.DataFrame(all_tracts_data)
    tract_df.index.name = 'Term'
    
    print(f"Calculated term contributions for {tract_df.shape[1]} tracts and {tract_df.shape[0]} terms")
    print("Term contribution calculation complete!")
    
    return tract_df

# ------------------------------------------------------------------------------------------------
# --- Main execution (when script is run directly) ---
# ------------------------------------------------------------------------------------------------

# Calculate tract term contributions
tract_df = calculate_tract_term_contributions(
    data_file_path=data_file_path,
    terms_start_col=58, # first term (action)
    tract_regex='left|right',
    connection_thresh=0.5
)

# Save to CSV
print("Saving tract-term contributions...")
tract_df.to_csv(os.path.join(decoding_results_dir, 'tract_term_contributions.csv'))
print(f"Saved tract-term contributions matrix to: {os.path.join(decoding_results_dir, 'tract_term_contributions.csv')}")
print(f"Matrix shape: {tract_df.shape} (terms x tracts)")
print(f"Number of tracts: {tract_df.shape[1]}")
