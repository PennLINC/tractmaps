# --- Generic Tract Matrix Extractor ---
# Function to extract pairwise values from any region-region matrix for tract-connected regions
# Inputs: tract connection data, region-region matrix, optional tract names, parameters
# Outputs: detailed pairwise data, tract mean values DataFrame
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np


def extract_tract_means_from_matrix(tractdata, region_matrix, tract_threshold=0.5, 
                               matrix_name='matrix_value', tract_names=None):
    """
    Extract pairwise values from a region-region matrix for tract-connected regions.
    
    This generic function works with any symmetric region-region matrix (e.g., euclidean distances, 
    cortical similarity, functional connectivity, etc.) and extracts pairwise values between 
    regions connected to each tract.
    
    Parameters
    ----------
    tractdata : pandas.DataFrame
        DataFrame containing tract data with columns for each tract ending in '_left' or '_right'
        Rows should be regions, with regionID and parcel_name columns if available
    region_matrix : numpy.ndarray
        Symmetric matrix of values between all region pairs (regions x regions)
    tract_threshold : float, optional
        Threshold for considering a region connected to a tract. Default: 0.5
    matrix_name : str, optional
        Name for the matrix values in output (e.g., 'Distance', 'Similarity'). Default: 'matrix_value'
    tract_names : pandas.DataFrame, optional
        DataFrame with tract metadata (Tract, Abbreviation, etc.) for merging. Default: None
        
    Returns
    -------
    tuple
        Tuple of (detailed_pairwise_df, tract_means_df)
        - detailed_pairwise_df: DataFrame with columns [Tract, matrix_name, idx1, idx2, parcel_name_1, parcel_name_2]
        - tract_means_df: DataFrame with columns [Tract, Mean_{matrix_name}] plus optional tract metadata
    """
    
    print(f"Extracting tract-wise {matrix_name} values...")
    
    # Validate inputs
    if region_matrix.shape[0] != region_matrix.shape[1]:
        raise ValueError("region_matrix must be square (symmetric)")
    if region_matrix.shape[0] != len(tractdata):
        raise ValueError("region_matrix dimensions must match number of regions in tractdata")
    
    # Initialize list to store detailed pairwise data
    detailed_data = []
    
    # Get tract columns (those ending with _left or _right)
    tract_columns = [col for col in tractdata.columns if col.endswith('_left') or col.endswith('_right')]
    
    # Check if tractdata has regionID and parcel_name columns
    has_region_info = 'regionID' in tractdata.columns and 'parcel_name' in tractdata.columns
    
    # Process each tract
    for tract in tract_columns:
        # Get regions connected to this tract with probability >= tract_threshold
        connected_regions = tractdata[tractdata[tract] >= tract_threshold].index.tolist()
        
        if len(connected_regions) > 1:
            # Get all pairwise values between connected regions (upper triangle only)
            for i in range(len(connected_regions)):
                for j in range(i+1, len(connected_regions)):
                    idx1 = connected_regions[i]
                    idx2 = connected_regions[j]
                    matrix_value = region_matrix[idx1, idx2]
                    
                    # Create row data
                    row_data = {
                        'Tract': tract,
                        matrix_name: matrix_value,
                        'idx1': idx1,
                        'idx2': idx2
                    }
                    
                    # Add region information if available
                    if has_region_info:
                        row_data['regionID_1'] = tractdata.loc[idx1, 'regionID']
                        row_data['regionID_2'] = tractdata.loc[idx2, 'regionID']
                        row_data['parcel_name_1'] = tractdata.loc[idx1, 'parcel_name']
                        row_data['parcel_name_2'] = tractdata.loc[idx2, 'parcel_name']
                    
                    detailed_data.append(row_data)
                    
        elif len(connected_regions) == 1:
            # For tracts with only 1 connected region, assign self-value (typically 0 for distances, 1 for similarities)
            idx1 = connected_regions[0]
            
            # For distances, self-distance is 0; for similarities, self-similarity is typically 1
            # Use the diagonal value from the matrix
            matrix_value = region_matrix[idx1, idx1] if not np.isnan(region_matrix[idx1, idx1]) else 0.0
            
            row_data = {
                'Tract': tract,
                matrix_name: matrix_value,
                'idx1': idx1,
                'idx2': idx1  # Same region for self-value
            }
            
            # Add region information if available
            if has_region_info:
                row_data['regionID_1'] = tractdata.loc[idx1, 'regionID']
                row_data['regionID_2'] = tractdata.loc[idx1, 'regionID']
                row_data['parcel_name_1'] = tractdata.loc[idx1, 'parcel_name']
                row_data['parcel_name_2'] = tractdata.loc[idx1, 'parcel_name']
                
            detailed_data.append(row_data)
    
    # Convert to DataFrame
    detailed_pairwise_df = pd.DataFrame(detailed_data)
    
    # Calculate mean values for each tract
    if len(detailed_pairwise_df) > 0:
        tract_means = detailed_pairwise_df.groupby('Tract')[matrix_name].mean().reset_index()
        tract_means.rename(columns={matrix_name: f'Mean_{matrix_name}'}, inplace=True)
        
        # Merge with tract names if provided
        if tract_names is not None:
            tract_means = tract_means.merge(tract_names, on='Tract', how='left')
            # Reorder columns to put tract info first
            name_cols = [col for col in tract_names.columns if col != 'Tract']
            tract_means = tract_means[['Tract'] + name_cols + [f'Mean_{matrix_name}']]
        
        # Sort tracts from high to low based on mean values
        tract_means = tract_means.sort_values(f'Mean_{matrix_name}', ascending=False).reset_index(drop=True)
        
        print(f"Calculated mean {matrix_name} for {len(tract_means)} tracts")
    else:
        print("Warning: No tract connections found above threshold")
        tract_means = pd.DataFrame()
    
    return detailed_pairwise_df, tract_means
