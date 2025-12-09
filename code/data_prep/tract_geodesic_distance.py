########################################################
### Geodesic Distance Computation using brainsmash ###
########################################################

# This script computes geodesic distances between brain regions connected by tracts
# using the FSLR32k midthickness surface and Glasser parcellation via brainsmash

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from neuromaps.datasets import fetch_fslr
import brainsmash.workbench as wb

# ------------------------------------------------------------------------------------------------
# --- Set up inputs ---
root = '/Users/joelleba/PennLINC/tractmaps/data'
abbrev_dir = f'{root}/derivatives/tract_names'
derivatives_root = f'{root}/derivatives'

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Create results directory if it doesn't yet exist
geo_dist_dir = f'{derivatives_root}/tracts/tracts_distances'
if not os.path.exists(geo_dist_dir):
    os.makedirs(geo_dist_dir)
    print(f"Folder '{geo_dist_dir}' created.")
else:
    print(f"Folder '{geo_dist_dir}' already exists.")

# ------------------------------------------------------------------------------------------------
# --- Load FSLR32k surface ---
print("Loading FSLR32k surface...")
surfaces = fetch_fslr()
lh_surface, rh_surface = surfaces['midthickness']  # Use midthickness surface
print(f"Left hemisphere surface: {lh_surface}")
print(f"Right hemisphere surface: {rh_surface}")

# ------------------------------------------------------------------------------------------------
# --- Load Glasser parcellation ---
# Note: computed using prep_glasser_labels.sh (calls connectome workbench)
print("\nLoading Glasser parcellation...")
lh_glasser_path = f'{derivatives_root}/glasser_parcellation/HCP_MMP_L.dlabel.nii'
rh_glasser_path = f'{derivatives_root}/glasser_parcellation/HCP_MMP_R.dlabel.nii'
print(f"Left hemisphere Glasser parcellation: {lh_glasser_path}")
print(f"Right hemisphere Glasser parcellation: {rh_glasser_path}")

# ------------------------------------------------------------------------------------------------
# --- Load tract data ---
print("\nLoading tract data...")
tract_connection_path = f'{derivatives_root}/tracts/tracts_probabilities/tracts_probabilities.csv'
tractdata = pd.read_csv(tract_connection_path)
print(f"Loaded tract connection data for {len(tractdata)} regions, {len(tractdata.filter(regex='left|right').columns)} tracts")

# Load tract names
tract_names = pd.read_excel(f'{abbrev_dir}/abbreviations.xlsx')[['Tract', 'Abbreviation', 'Tract_Long_Name', 'new_qsirecon_tract_names', 'Hemisphere', 'Type']]
tract_names.rename(columns={'new_qsirecon_tract_names': 'bundle_name'}, inplace=True)

# ------------------------------------------------------------------------------------------------
# --- Compute geodesic distances using brainsmash ---
print("\nComputing geodesic distances using brainsmash...")

# Compute geodesic distances for left hemisphere in glasser parcellation (takes about 10 minutes)
print("Computing left hemisphere geodesic distances...")
lh_distance_file = f'{geo_dist_dir}/L_geodesic_distances.txt'
wb.cortex(
    surface=lh_surface,
    outfile=lh_distance_file,
    dlabel=lh_glasser_path,
    use_wb=True,
    verbose=True,
    n_jobs=10
)

# Compute geodesic distances for right hemisphere in glasser parcellation (takes about 10 minutes)
print("Computing right hemisphere geodesic distances...")
rh_distance_file = f'{geo_dist_dir}/R_geodesic_distances.txt'
wb.cortex(
    surface=rh_surface,
    outfile=rh_distance_file,
    dlabel=rh_glasser_path,
    use_wb=True,
    verbose=True,
    n_jobs=10 
)

# Load parcellated distance matrices
print("Loading parcellated distance matrices...")
lh_geo_dist = np.loadtxt(lh_distance_file)
rh_geo_dist = np.loadtxt(rh_distance_file)

print(f"Left hemisphere distance matrix shape: {lh_geo_dist.shape}")
print(f"Right hemisphere distance matrix shape: {rh_geo_dist.shape}")

# ------------------------------------------------------------------------------------------------
# --- Calculate tract-wise geodesic distances ---
print("\nCalculating tract-wise geodesic distances...")

# Prepare data for all tracts
tract_geo_dist_data = pd.DataFrame()

for tract in tractdata.filter(regex='left|right').columns:
    # Get regions connected by this tract with a probability >= tract_threshold
    connected_regions = tractdata[tractdata[tract] >= tract_threshold].index
    
    if len(connected_regions) > 1:
        distances = []
        idx1_list = []
        idx2_list = []
        parcel_name_1_list = []
        parcel_name_2_list = []
        
        for i in range(len(connected_regions)):
            for j in range(i+1, len(connected_regions)):
                region1 = connected_regions[i]
                region2 = connected_regions[j]
                
                # Get distance from appropriate hemisphere matrix
                if region1 < 180 and region2 < 180:  # Left hemisphere
                    idx1 = region1
                    idx2 = region2
                    dist = lh_geo_dist[idx1, idx2]
                elif region1 >= 180 and region2 >= 180:  # Right hemisphere
                    idx1 = region1 - 180
                    idx2 = region2 - 180
                    dist = rh_geo_dist[idx1, idx2]
                else:
                    # Skip inter-hemispheric connections for now
                    continue
                
                distances.append(dist)
                idx1_list.append(tractdata.loc[region1, 'regionID'])
                idx2_list.append(tractdata.loc[region2, 'regionID'])
                parcel_name_1_list.append(tractdata.loc[region1, 'parcel_name'])
                parcel_name_2_list.append(tractdata.loc[region2, 'parcel_name'])
        
        # Add geodesic distance data
        if distances:
            temp_dist_df = pd.DataFrame({
                'Tract': tract,
                'Distance': distances,
                'idx1': idx1_list,
                'idx2': idx2_list,
                'parcel_name_1': parcel_name_1_list,
                'parcel_name_2': parcel_name_2_list
            })
            tract_geo_dist_data = pd.concat([tract_geo_dist_data, temp_dist_df])
    elif len(connected_regions) == 1:
        # For tracts with only 1 connected region, assign distance to 0 (fornix)
        region1 = connected_regions[0]
        temp_dist_df = pd.DataFrame({
            'Tract': tract,
            'Distance': [0.0],
            'idx1': [tractdata.loc[region1, 'regionID']],
            'idx2': [tractdata.loc[region1, 'regionID']],  # Same region for self-distance
            'parcel_name_1': [tractdata.loc[region1, 'parcel_name']],
            'parcel_name_2': [tractdata.loc[region1, 'parcel_name']]  # Same parcel for self-distance
        })
        tract_geo_dist_data = pd.concat([tract_geo_dist_data, temp_dist_df])

print(f"Computed geodesic distances for {len(tract_geo_dist_data)} region pairs")

# ------------------------------------------------------------------------------------------------
# --- Save results ---
print("\nSaving results...")

# Save detailed tract distance data with region information
tract_geo_dist_data.to_csv(f'{geo_dist_dir}/tract_geodesic_distances_pairwise.csv', index=False)
print(f"Saved detailed tract geodesic distances to: {geo_dist_dir}/tract_geodesic_distances_pairwise.csv")

# ------------------------------------------------------------------------------------------------
# --- Calculate mean geodesic distances for each tract ---
print("\nCalculating mean geodesic distances for each tract...")

# Calculate mean geodesic distances for each tract
tract_geo_means = tract_geo_dist_data.groupby('Tract')['Distance'].mean()

# Create a DataFrame with tract info for easier filtering
tract_geo_means_df = pd.DataFrame({
    'Tract': tract_geo_means.index,
    'Mean_Geodesic_Distance': tract_geo_means.values
})

# Merge tract info with tract names
tract_geo_means_df = tract_geo_means_df.merge(tract_names, on='Tract', how='left')

# Reorder columns
tract_geo_means_df = tract_geo_means_df[['Tract', 'Abbreviation', 'Tract_Long_Name', 'bundle_name', 'Hemisphere', 'Type', 'Mean_Geodesic_Distance']]

print(f"Calculated mean geodesic distances for {len(tract_geo_means_df)} tracts")

# Save tract mean geodesic distance data
output_path = f'{geo_dist_dir}/tract_geodesic_distances_means.csv'
tract_geo_means_df.to_csv(output_path, index=False)
print(f"Saved tract geodesic distances to: {output_path}")

