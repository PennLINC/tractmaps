# --- Tract Euclidean and Geodesic Distances Preparation ---
# Inputs: tract connection data, coordinates, tract types
# Outputs: tract distance data with mean Euclidean and geodesic distances
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import sys
from scipy.spatial.distance import squareform, pdist

# Add utils to path for importing custom functions
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.matrix_to_tracts import extract_tract_means_from_matrix

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

root = '/Users/joelleba/PennLINC/tractmaps/data'
abbrev_dir = f'{root}/raw/tract_names'
derivatives_root = f'{root}/derivatives'

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Create results directory if it doesn't yet exist
euc_dist_dir = f'{derivatives_root}/tracts/tracts_distances'
if not os.path.exists(euc_dist_dir):
    os.makedirs(euc_dist_dir)
    print(f"Folder '{euc_dist_dir}' created.")
else:
    print(f"Folder '{euc_dist_dir}' already exists.")

# ------------------------------------------------------------------------------------------------
# --- Load all data ---
# ------------------------------------------------------------------------------------------------

# Load coordinates
coords_path = f'{derivatives_root}/glasser_parcellation/glasser_coords.csv'
coords = pd.read_csv(coords_path)
print(f"Loaded coordinates for {len(coords)} regions")

# Load tract names
tract_names = pd.read_excel(f'{abbrev_dir}/abbreviations.xlsx')[['Tract', 'Abbreviation', 'Tract_Long_Name', 'new_qsirecon_tract_names', 'Hemisphere', 'Type']]
tract_names.rename(columns={'new_qsirecon_tract_names': 'bundle_name'}, inplace=True)

# Load tract connection data
tract_connection_path = f'{derivatives_root}/tracts/tracts_probabilities/tracts_probabilities.csv'
tractdata = pd.read_csv(tract_connection_path)
print(f"Loaded tract connection data for {len(tractdata)} tracts")


# ------------------------------------------------------------------------------------------------
# --- Calculate euclidean distances between all region pairs ---
# ------------------------------------------------------------------------------------------------

print("Calculating euclidean distances...")
eu_dist = squareform(pdist(coords[['x-cog', 'y-cog', 'z-cog']], metric='euclidean'))

print(f"Euclidean distance matrix shape: {eu_dist.shape}")

# Save euclidean distance matrix
np.save(f'{euc_dist_dir}/euclidean_distances.npy', eu_dist)
print(f"Saved euclidean distance matrix to: {euc_dist_dir}/euclidean_distances.npy")

# ------------------------------------------------------------------------------------------------
# --- Calculate tract euclidean distances using generic function ---
# ------------------------------------------------------------------------------------------------

# Extract tract euclidean distances using the generic matrix function
tract_euc_dist_data, tract_euc_means_df = extract_tract_means_from_matrix(
    tractdata=tractdata,
    region_matrix=eu_dist,
    tract_threshold=tract_threshold,
    matrix_name='Euclidean_Distance',
    tract_names=tract_names
)

# Save detailed tract distance data with region information
tract_euc_dist_data.to_csv(f'{euc_dist_dir}/tract_euclidean_distances_pairwise.csv', index=False)
print(f"Saved detailed tract Euclidean distances to: {euc_dist_dir}/tract_euclidean_distances_pairwise.csv")

# Save tract mean euclidean distance data
tract_euc_means_df.to_csv(f'{euc_dist_dir}/tract_euclidean_distances_means.csv', index=False)
print(f"Saved tract mean Euclidean distances to: {euc_dist_dir}/tract_euclidean_distances_means.csv")

print(f"Completed tract euclidean distance analysis!")
print(f"Results saved to: {euc_dist_dir}")