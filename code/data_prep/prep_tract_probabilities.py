# ------------------------------------------------------------------------------------------------
# --- Prepare tract-to-region probabilities csv ---
# ------------------------------------------------------------------------------------------------

# This script does the following:

# 1. Load probabilistic tract-to-region maps publicly released by Yeh 2022, Nature Methods (https://www.nature.com/articles/s41467-022-32595-4). This map is used to determine, for each white matter tract, which brain regions it is most likely connected to. 

# 2. Add a few additional columns containing more information about the Glasser parcels. The dataframe is then saved to csv for further analyses.

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import hcp_utils as hcp

# ------------------------------------------------------------------------------------------------
# --- Set inputs and outputs ---
# ------------------------------------------------------------------------------------------------

root = '/Users/joelleba/PennLINC/tractmaps'
output_folder = f'{root}/data/derivatives/tracts/tracts_probabilities'

# make output folder if it doesn't exist
os.makedirs(output_folder, exist_ok = True)

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# load Glasser tract-to-region matrix
tracts_regs = pd.read_excel(f'{root}/data/raw/tracts_probabilities/Yeh_2022_fig5_Glasser.xlsx', skiprows = 1, usecols = range(0, 53))

# load Glasser region IDs and corresponding region labels from hcp_utils (transform dictionary into dataframe)
glasser_ids = pd.DataFrame({'regionName': list(hcp.mmp.labels.values()), 
                            'regionID': list(hcp.mmp.labels.keys())})

# load long region labels (full description based on supplemental files of the Glasser 2016 paper and made available at https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/atlases.html#how-to-download-and-install-additional-fsl-atlases)
glasser_longnames = pd.read_csv(f'{root}/data/raw/glasser_parcellation/HCP-MMP1_UniqueRegionList.csv')


# ------------------------------------------------------------------------------------------------
# --- Rename and reorganize columns in tract-to-region dataframe ---
# ------------------------------------------------------------------------------------------------

# rename columns (tract names)
columns = {}
for colidx, colname in enumerate(tracts_regs.columns):
    if colidx == 0:
        columns[colname] = f'parcel_name'
    if 1 <= colidx <= 26:
        columns[colname] = f'{colname}_left'
    elif colidx >= 27:
        columns[colname] = colname[:colname.index(".")] + "_right"
        
tracts_regs.rename(columns = columns, inplace = True)

# separate into L and R dataframes, rename parcel.name values
lh_tracts_regs = tracts_regs.filter(regex='^(parcel.name|.*_left$)').copy()
rh_tracts_regs = tracts_regs.filter(regex='^(parcel.name|.*_right$)').copy()
lh_tracts_regs['LR'] = 'L'
lh_tracts_regs['parcel_name'] = lh_tracts_regs['LR'] + '_' + lh_tracts_regs['parcel_name'].astype(str)
rh_tracts_regs['LR'] = 'R'
rh_tracts_regs['parcel_name'] = rh_tracts_regs['LR'] + '_' + rh_tracts_regs['parcel_name'].astype(str) 

# aggregate the two dataframes back again now that L and R regions are assigned
tracts_regs = pd.concat([lh_tracts_regs, rh_tracts_regs])
print(f'Tract-to-region dataframe dimensions: {tracts_regs.shape}')


# ------------------------------------------------------------------------------------------------
# --- Add regional information to tract-to-region dataframe ---
# ------------------------------------------------------------------------------------------------

# keep only cortical regions (361-380 are subcortical in hcp_utils)
glasser_ids = glasser_ids[glasser_ids['regionID'] <= 360] 

# dropping region IDs here as they are non-continuous (goes from L 180 to R 201)
glasser_longnames.drop(columns = ['Cortex_ID', 'x-cog', 'y-cog', 'z-cog', 'volmm', 'Lobe', 'regionID', 'LR'], 
                       inplace = True) 

# place the hemisphere (L_ or R_) as a prefix, not a suffix in region labels (because tract-to-regions data has hem as prefix)
glasser_longnames['regionName'] = glasser_longnames['regionName'].apply(lambda x: f"{x[-1]}_{x[:-2]}")

# merge the Glasser regionIDs, labels, and long region names
glasser_ids = pd.merge(glasser_ids, glasser_longnames, on = 'regionName', how = 'left')

# merge with the tract-to-regions data
tracts_regs_ids = pd.merge(tracts_regs, glasser_ids, left_on = 'parcel_name', right_on = 'regionName', how = 'left')
tracts_regs_ids.drop(columns = ['regionName'], inplace = True)

# reorganize a bit
tracts_regs_ids.sort_values(by = 'regionID', inplace = True)
region_col = tracts_regs_ids.pop('regionID')
tracts_regs_ids.insert(0, 'regionID', region_col)

# save to csv
tracts_regs_ids.to_csv(f'{output_folder}/tracts_probabilities.csv', header = True, index = False)
print(f'Saved to: {output_folder}/tracts_probabilities.csv')
print(f'Tract-to-region dataframe (with additional region label columns) dimensions: {tracts_regs_ids.shape}')


# ------------------------------------------------------------------------------------------------