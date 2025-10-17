# ------------------------------------------------------------------------------------------------
# --- Prepare the S-A axis data ---
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
from neuromaps.datasets import fetch_annotation
from data_prep import tm_parcellate


# ------------------------------------------------------------------------------------------------
# --- Set inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# Set root
root = '/Users/joelleba/PennLINC/tractmaps/data'

# create output folder if it doesn't exist
output_folder = f'{root}/derivatives/tracts/tracts_sa_axis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' created.")
else:
    print(f"Folder '{output_folder}' already exists.")


# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# S-A axis was obtained from Val's Neuron review paper: https://github.com/PennLINC/S-A_ArchetypalAxis/tree/main/Glasser360_MMP
sa_axis = pd.read_csv(f'{root}/raw/sa_axis/glasser/Sensorimotor_Association_Axis_AverageRanks.csv')

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Load tract data
tractdata = pd.read_csv(f'{root}/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv')

# ------------------------------------------------------------------------------------------------
# --- Reformat region labels of S-A axis data ---
# ------------------------------------------------------------------------------------------------

sa_axis.rename(columns={'final.rank': 'sa_axis', 'region': 'parcel_name'}, inplace=True)
sa_axis.drop(columns=['brains.average.rank'], inplace=True)
# Duplicate rows and add L_ and R_ prefixes to parcel names
left_sa = sa_axis.copy()
right_sa = sa_axis.copy()
left_sa['parcel_name'] = 'L_' + left_sa['parcel_name'] 
right_sa['parcel_name'] = 'R_' + right_sa['parcel_name']
sa_axis = pd.concat([left_sa, right_sa], ignore_index=True)
sa_axis.to_csv(f'{root}/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv', index=False)


# ------------------------------------------------------------------------------------------------
# --- Compute S-A axis values for tracts ---
# ------------------------------------------------------------------------------------------------

# Get sa_axis values for each region
region_sa = sa_axis['sa_axis']

# Prepare S-A axis data
sa_data = pd.DataFrame()

for tract in tractdata.filter(regex='left|right').columns:
    # Get regions connected by this tract with a probability >= tract_threshold
    connected_regions = tractdata[tractdata[tract] >= tract_threshold].index
    if len(connected_regions) >= 1: # at least 1 region connected
        # Add S-A axis data
        temp_sa_df = pd.DataFrame({
            'SA_Axis': region_sa.iloc[connected_regions].values,
            'Tract': tract
        })
        sa_data = pd.concat([sa_data, temp_sa_df])

# Save to CSV
output_path = f'{output_folder}/tract_sa_axis_regionwise_thresh{int(tract_threshold * 100)}.csv'
sa_data.to_csv(output_path, index=False)

print(f"\nTract S-A axis values calculated for {len(sa_data['Tract'].unique())} tracts")
print(f"Results saved to: {output_path}")


# ------------------------------------------------------------------------------------------------
# --- Function to calculate tract S-A axis ranges ---
# ------------------------------------------------------------------------------------------------

def calculate_tract_sa_ranges(tractdata, sa_axis_values, tract_threshold=0.5):
    """
    Calculate S-A axis ranges for all tracts based on their connected regions.
    
    Parameters
    ----------
    tractdata : pandas.DataFrame
        Tract connectivity data with tract columns ending in '_left' or '_right'
    sa_axis_values : numpy.ndarray or pandas.Series
        S-A axis values for each region (indexed by region order)
    tract_threshold : float, optional
        Threshold for considering a region connected to a tract. Default: 0.5
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: Tract, SA_Range, N_Regions, Hemisphere, Abbreviation
    """
    # Get S-A ranks as numpy array for efficient indexing
    sa_rank = np.asarray(sa_axis_values)
    
    # Initialize list to store tract S-A range data
    tract_sa_ranges = []
    
    # Calculate S-A range for each tract
    for tract in tractdata.filter(regex='left|right').columns:
        # Get regions connected to this tract with probability >= tract_threshold
        tract_regions = np.where(tractdata[tract] >= tract_threshold)[0]
        
        if len(tract_regions) > 0:
            # Calculate S-A range for connected regions
            sa_range = np.max(sa_rank[tract_regions]) - np.min(sa_rank[tract_regions])

            # Get number of regions connected to this tract
            n_regions = len(tract_regions)
            
            tract_sa_ranges.append({
                'Tract': tract,
                'SA_Range': sa_range,
                'N_Regions': n_regions
            })
        else:
            # If no regions connected above threshold, set all values to NaN
            tract_sa_ranges.append({
                'Tract': tract,
                'SA_Range': np.nan,
                'N_Regions': 0
            })

    # Convert to DataFrame
    tract_sa_ranges_df = pd.DataFrame(tract_sa_ranges)

    # Add hemisphere and tract type information
    tract_sa_ranges_df['Hemisphere'] = tract_sa_ranges_df['Tract'].str.extract(r'_(left|right)$')
    tract_sa_ranges_df['Abbreviation'] = tract_sa_ranges_df['Tract'].str.replace(r'_(left|right)$', '', regex=True)
    
    return tract_sa_ranges_df

# ------------------------------------------------------------------------------------------------
# --- Calculate tract S-A axis ranges ---
# ------------------------------------------------------------------------------------------------

print("Calculating S-A ranges for tracts...")
tract_sa_ranges_df = calculate_tract_sa_ranges(tractdata, sa_axis['sa_axis'].values, tract_threshold)

# Save to CSV
output_path = f'{output_folder}/tract_sa_axis_ranges_thresh{int(tract_threshold * 100)}.csv'
tract_sa_ranges_df.to_csv(output_path, index=False)

# Print summary statistics
print(f"\nTract S-A axis ranges calculated for {len(tract_sa_ranges_df)} tracts")
print(f"Results saved to: {output_path}")

