# ------------------------------------------------------------------------------------------------
# --- Cortical similarity: biological properties of cortical regions connected to tracts ---
# ------------------------------------------------------------------------------------------------

# Inputs: biological properties of cortical regions
# Outputs: cortical similarity between regions, tract-wise biological similarity
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import sys
import pickle

# Add utils to path for importing custom functions
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.matrix_to_tracts import extract_tract_means_from_matrix

# ------------------------------------------------------------------------------------------------
# --- set up inputs ---
# ------------------------------------------------------------------------------------------------

root_dir = '/Users/joelleba/PennLINC/tractmaps'


# create results directory if it doesn't yet exist
output_dir = f'{root_dir}/data/derivatives/tracts/tracts_cortical_similarity/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# set tract connection threshold
tract_threshold = 0.5


# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

data = pd.read_csv(f'{root_dir}/data/derivatives/cortical_annotations/glasser/glasser_tracts_corticalmaps.csv')
tractdata = data.filter(regex = 'left|right')

# Load tract names
abbrev_dir = f'{root_dir}/data/raw/tract_names'
tract_names = pd.read_excel(f'{abbrev_dir}/abbreviations.xlsx')[['Tract', 'Abbreviation', 'Tract_Long_Name', 'new_qsirecon_tract_names', 'Hemisphere', 'Type']]
tract_names.rename(columns={'new_qsirecon_tract_names': 'bundle_name'}, inplace=True) 

# select cortical properties
properties = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6',
             'genes_pc1', 'myelin'] + [col for col in data.columns if col.startswith('pet_')]
cortical_properties = data[properties]

# load sa_axis
sa_axis = pd.read_csv(f'{root_dir}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv')

# print number of cortical properties
print(f"Number of cortical properties: {len(properties)}")


# ------------------------------------------------------------------------------------------------
# --- Compute cortical similarity across all regions ---
# ------------------------------------------------------------------------------------------------

# generate cortical similarity matrix
cortical_similarity = np.corrcoef(zscore(cortical_properties))

# save cortical similarity matrix
np.save(f'{output_dir}/cortical_similarity.npy', cortical_similarity)


# ------------------------------------------------------------------------------------------------
# --- Calculate tract cortical similarities using generic function ---
# ------------------------------------------------------------------------------------------------

# Extract tract cortical similarities using the generic matrix function
tract_similarities_data, tract_similarities_means_df = extract_tract_means_from_matrix(
    tractdata=tractdata,
    region_matrix=cortical_similarity,
    tract_threshold=tract_threshold,
    matrix_name='Cortical_Similarity',
    tract_names=tract_names
)

# Save detailed tract similarities data with region information
tract_similarities_data.to_csv(f'{output_dir}/tract_cortical_similarities_pairwise.csv', index=False)
print(f"Saved detailed tract cortical similarities to: {output_dir}/tract_cortical_similarities_pairwise.csv")

# Save tract mean cortical similarity data
tract_similarities_means_df.to_csv(f'{output_dir}/tracts_mean_cortical_similarity.csv', index=False)
print(f"Saved tract mean cortical similarities to: {output_dir}/tracts_mean_cortical_similarity.csv")

print(f"Results saved to: {output_dir}")