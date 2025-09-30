# ------------------------------------------------------------------------------------------------
# --- Prepare Glasser region coordinates ---
# ------------------------------------------------------------------------------------------------

# This script does the following:
# 1. Load Glasser region coordinates from hcp_utils
# 2. Add additional columns containing more information about the Glasser parcels
# 3. Save the dataframe to csv for further analyses

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import hcp_utils as hcp

# ------------------------------------------------------------------------------------------------
# --- Set inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# set root path
root = '/Users/joelleba/PennLINC/tractmaps'

# define output folder
output_folder = f'{root}/data/derivatives/glasser_parcellation'

# make output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' created.")
else:
    print(f"Folder '{output_folder}' already exists.")

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# get Glasser region IDs and corresponding region labels (transform dictionary into dataframe)
glasser_ids = pd.DataFrame({'regionName': list(hcp.mmp.labels.values()), 
                            'regionID': list(hcp.mmp.labels.keys())})

# load tract-to-regions data
tracts_regs = pd.read_csv(f'{root}/data/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv')

# ------------------------------------------------------------------------------------------------
# --- Prep coordinates dataframe ---
# ------------------------------------------------------------------------------------------------

# keep only cortical regions (361-380 are subcortical in hcp_utils)
glasser_ids = glasser_ids[glasser_ids['regionID'] <= 360] 

# add long region labels (full description based on supplemental files of the Glasser 2016 paper and made available at https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/atlases.html#how-to-download-and-install-additional-fsl-atlases)
glasser_longnames = pd.read_csv(f'{root}/data/raw/glasser_parcellation/HCP-MMP1_UniqueRegionList.csv')

# dropping region IDs here as they are non-continuous (goes from L 180 to R 201)
glasser_longnames.drop(columns = ['Cortex_ID', 'volmm', 'regionID', 'LR'], inplace = True) 

# place the hemisphere (L_ or R_) as a prefix, not a suffix in region labels (because tract-to-regions data has hem as prefix)
glasser_longnames['regionName'] = glasser_longnames['regionName'].apply(lambda x: f"{x[-1]}_{x[:-2]}")

# merge the Glasser regionIDs, labels, and long region names
glasser_ids = pd.merge(glasser_ids, glasser_longnames, on = 'regionName', how = 'left')

# merge with the tract-to-regions data
tracts_regs_coords = pd.merge(tracts_regs[['parcel_name']], glasser_ids, left_on = 'parcel_name', right_on = 'regionName', how = 'left')
tracts_regs_coords.drop(columns = ['regionName'], inplace = True)

# reorganize a bit
tracts_regs_coords.sort_values(by = 'regionID', inplace = True)
region_col = tracts_regs_coords.pop('regionID')
tracts_regs_coords.insert(0, 'regionID', region_col)
tracts_regs_coords.drop(columns = ['regionLongName', 'regionIdLabel', 'region', 'cortex'], inplace = True)

# ------------------------------------------------------------------------------------------------
# --- Save dataframe ---
# ------------------------------------------------------------------------------------------------

# save to csv
tracts_regs_coords.to_csv(f'{output_folder}/glasser_coords.csv', header = True, index = False)
print(f'Glasser coordinates df dimensions: {tracts_regs_coords.shape}')
print(f'Saved to: {output_folder}/glasser_coords.csv')