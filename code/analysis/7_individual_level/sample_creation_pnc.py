# ------------------------------------------------------------------------------------------------
# --- Sample Creation for Individual-level Analysis ---
# ------------------------------------------------------------------------------------------------

# Script to create a final sample of participants with complete data for analysis. Requires first running group_level_tract_scalars_pnc.R 
# and group_level_qc_measures_pnc.R to create the group-level measures.
# Inputs: PNC demographics, health data, T1 QA data, diffusion QC data, and participant lists
# Outputs: Final sample of participants with complete data for analysis
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd

# ------------------------------------------------------------------------------------------------
# --- set up inputs ---
# ------------------------------------------------------------------------------------------------

cubic_root = '/Volumes/tractmaps/data/PNC/behavioral_data'
root = '/Users/joelleba/PennLINC/tractmaps'
data_root = f'{root}/data/derivatives/individual_level_pnc'
output_dir = f'{data_root}/final_sample'

# Create results directory if it doesn't yet exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created.")
else:
    print(f"Folder '{output_dir}' already exists.")

# ------------------------------------------------------------------------------------------------
# --- load data ---
# ------------------------------------------------------------------------------------------------
# Note: participants, demographics, and bblid_scanid, health_exclude_data, and t1_qa, dwi_qc are files that Audrey shared for the sample construction code of her paper: https://github.com/PennLINC/luo_wm_dev/blob/main/code/sample_construction/construct_initial_sample/PNC_InitialSampleSelection.Rmd

# Initial participant list (N = 1368 with non-variant diffusion MRI data)
non_variant_participants = pd.read_csv(f'{cubic_root}/PNC_NonVariantDWI_participantlist.csv')
non_variant_participants.columns = ['rbcid']
print(f"Initial participant list: N = {len(non_variant_participants)}") # N = 1368

# Demographics data
demographics = pd.read_csv(f'{cubic_root}/n1601_demographics_go1_20161212.csv')
print(f"Demographics data: N = {len(demographics)}") # N = 1629

# BBLID to scan ID mapping
bblid_scanid = pd.read_csv(f'{cubic_root}/bblid_scanid_sub.csv')
print(f"BBLID to scan ID mapping: N = {len(bblid_scanid)}") # N = 1629

# Health exclusion data
health_exclude_data = pd.read_csv(f'{cubic_root}/n1601_health_20170421.csv')
print(f"Health exclusion data: N = {len(health_exclude_data)}") # N = 1601

# T1 QA data
t1_qa = pd.read_csv(f'{cubic_root}/n1601_t1QaData_20170306.csv')
print(f"T1 QA data: N = {len(t1_qa)}") # N = 1601

# Diffusion QC data
dwi_qc = pd.read_csv(f'{cubic_root}/PNC_DWI_QCmetrics.csv')
print(f"Diffusion QC data: N = {len(dwi_qc)}") # N = 1406

# Scalar measures of tracts
# Script to great a group-level csv: group_level_tract_scalars.R
tracts_fa = pd.read_csv(f'{data_root}/cleaned/pnc_tracts_fa.csv')

# Cognition data
cognition = pd.read_csv(f'{cubic_root}/n1601_cnb_factor_scores_tymoore_20151006.csv')
print(f"Cognition data: N = {len(cognition)}") # N = 1601

# ------------------------------------------------------------------------------------------------
# --- construct initial sample ---
# ------------------------------------------------------------------------------------------------

# Print initial sample size
initial_sample_size = len(demographics)
print(f"Initial sample size: N = {initial_sample_size}") # N = 1629

# Merge demographics with bblid_scanid mapping
demographics = demographics.merge(bblid_scanid, on='scanid', how='left')
print(f"Number of NAs in rbcid: {demographics['rbcid'].isna().sum()}") # check for NAs in rbcid: 0 (good)
demographics = demographics[['bblid', 'rbcid', 'ageAtScan1', 'sex', 'race']] 
demographics['rbcid'] = 'sub-' + demographics['rbcid'].astype(str) 
demographics['age'] = demographics['ageAtScan1'] / 12 # convert age from months to years

# Filter to only participants with non-variant diffusion MRI data
demographics = demographics[demographics['rbcid'].isin(non_variant_participants['rbcid'])]
print(f"After filtering to non-variant DWI participants: N = {len(demographics)}") # N = 1368
print(f"Number of participants excluded due to variant diffusion MRI data: {initial_sample_size - len(demographics)}") # N = 261 

# ------------------------------------------------------------------------------------------------
# --- health history exclusion ---
# ------------------------------------------------------------------------------------------------

# Create list of participants to exclude based on health criteria
health_exclude_data['rbcid'] = 'sub-' + health_exclude_data['bblid'].astype(str) # bblid in health_exclude_data is the same as rbcid in demographics, so I'm renaming it to rbcid
health_exclude_subs = health_exclude_data[health_exclude_data['healthExcludev2'] == 1]['rbcid'].tolist()

# Count participants before health exclusion
n_before_health = len(demographics)

# Remove health exclude participants
demographics = demographics[~demographics['rbcid'].isin(health_exclude_subs)]
print(f"After health exclusion: N = {len(demographics)}") # N = 1250
print(f"Number of participants excluded due to health history: {n_before_health - len(demographics)}") # N = 118

# ------------------------------------------------------------------------------------------------
# --- T1 quality exclusion ---
# ------------------------------------------------------------------------------------------------

# Merge T1 QA data with bblid_scanid mapping
t1_qa = t1_qa.merge(bblid_scanid, on='scanid', how='left')
t1_qa = t1_qa[['rbcid', 't1Exclude']]
t1_qa['rbcid'] = 'sub-' + t1_qa['rbcid'].astype(str)

# Get participants to exclude based on T1 QA
t1_quality_exclude = t1_qa[t1_qa['t1Exclude'] == 1]['rbcid'].tolist()

# Count participants before T1 quality exclusion
n_before_t1 = len(demographics)

# Remove T1 quality exclude participants
demographics = demographics[~demographics['rbcid'].isin(t1_quality_exclude)]
print(f"After T1 quality exclusion: N = {len(demographics)}") # N = 1225
print(f"Number of participants excluded: {n_before_t1 - len(demographics)}") # N = 25

# ------------------------------------------------------------------------------------------------
# --- diffusion acquisition exclusion ---
# ------------------------------------------------------------------------------------------------

# Filter diffusion QC data to only participants with non-variant diffusion data
dwi_qc = dwi_qc[dwi_qc['subject_id'].isin(non_variant_participants['rbcid'])]

# Get participants to exclude for missing a diffusion MRI run, resulting in 35 diffusion directions instead of 71
acquisition_exclude = dwi_qc[dwi_qc['raw_num_directions'] != 71]['subject_id'].tolist()

# Count participants before acquisition exclusion
n_before_acquisition = len(demographics)

# Remove acquisition exclude participants
demographics = demographics[~demographics['rbcid'].isin(acquisition_exclude)]
print(f"After diffusion acquisition exclusion: N = {len(demographics)}") # N = 1215
print(f"Number of participants excluded: {n_before_acquisition - len(demographics)}") # N = 10

# ------------------------------------------------------------------------------------------------
# --- diffusion quality exclusion ---
# ------------------------------------------------------------------------------------------------

# Get participants to exclude based on processed diffusion MRI neighborhood correlation
quality_exclude = dwi_qc[dwi_qc['t1_neighbor_corr'] < 0.9]['subject_id'].tolist() # threshold determined by Sydnor et al. 2025 Nat Neuro thalamocortical paper: https://doi.org/10.1038/s41593-025-01991-6

# Count participants before quality exclusion
n_before_quality = len(demographics)

# Remove quality exclude participants
demographics = demographics[~demographics['rbcid'].isin(quality_exclude)]
print(f"After diffusion quality exclusion: N = {len(demographics)}") # N = 1158
print(f"Number of participants excluded: {n_before_quality - len(demographics)}") # N = 57

# ------------------------------------------------------------------------------------------------
# --- diffusion scan head motion exclusion ---
# ------------------------------------------------------------------------------------------------

# Get participants to exclude based on high in-scanner head motion
motion_exclude = dwi_qc[dwi_qc['mean_fd'] > 1]['subject_id'].tolist()

# Count participants before motion exclusion
n_before_motion = len(demographics)

# Remove motion exclude participants
demographics = demographics[~demographics['rbcid'].isin(motion_exclude)]
print(f"After motion exclusion: N = {len(demographics)}") # N = 1145
print(f"Number of participants excluded: {n_before_motion - len(demographics)}") # N = 13

# ------------------------------------------------------------------------------------------------
# --- create final sample dataframe ---
# ------------------------------------------------------------------------------------------------

# Recode race variable
race_mapping = {
    1: 'White',
    2: 'Black',
    3: 'US_India_Alaska',
    4: 'Asian',
    5: 'More_than_one_race',
    6: 'Hawaiian_Pacific',
    9: 'Unknown_Unreported'
}
demographics['race'] = demographics['race'].map(race_mapping)

# Recode sex variable
sex_mapping = {1: 'M', 2: 'F'}
demographics['sex'] = demographics['sex'].map(sex_mapping)

# Select final columns
demographics = demographics[['rbcid', 'age', 'sex', 'race']]

# Create QC metrics dataframe for final sample
dwi_qc_final = dwi_qc[['subject_id', 'mean_fd', 't1_neighbor_corr']]
dwi_qc_final = dwi_qc_final.rename(columns={'subject_id': 'rbcid'})
final_sample = demographics.merge(dwi_qc_final, on='rbcid', how='left')


# ------------------------------------------------------------------------------------------------
# --- add cognition data ---
# ------------------------------------------------------------------------------------------------

# Merge demographics with bblid_scanid mapping
cognition = cognition.merge(bblid_scanid, on=['bblid', 'scanid'], how='left')
print(f"Number of NAs in rbcid: {cognition['rbcid'].isna().sum()}") # check for NAs in rbcid: 0 (good)
cognition['rbcid'] = 'sub-' + cognition['rbcid'].astype(str)

# select final columns
cognition = cognition[['rbcid', 'F3_Executive_Efficiency']]

# Merge cognition data with demographics
final_cognition_sample = final_sample.merge(cognition, on='rbcid', how='left')

# remove rows with NAs in any of the cognitive variables (N = 3)
final_cognition_sample = final_cognition_sample.dropna(subset=['F3_Executive_Efficiency'])

# Count participants before and after cognition merge
n_before_cognition = len(final_sample)
n_after_cognition = len(final_cognition_sample)
print(f"Final sample with cognition data: N = {n_after_cognition}") # N = 1142
print(f"Number of participants excluded due to missing cognition data: {n_before_cognition - n_after_cognition}") # N = 3

# ------------------------------------------------------------------------------------------------
# --- add tract scalar measures ---
# ------------------------------------------------------------------------------------------------

# rename subject_id to rbcid in tracts_fa and tracts_md
tracts_fa = tracts_fa.rename(columns={'subject_id': 'rbcid'})

# Merge tract scalar measures with demographics
final_sample_fa = final_sample.merge(tracts_fa, on='rbcid', how='left')

# Merge tract scalar measures with cognition
final_sample_fa_cognition = final_cognition_sample.merge(tracts_fa, on='rbcid', how='left')

# Count participants before and after tract scalar measures merge
print(f"Final sample: N = {len(final_sample_fa)}") # N = 1145
print(f"Final cognition sample: N = {len(final_sample_fa_cognition)}") # N = 1142

# ------------------------------------------------------------------------------------------------
# --- descriptive statistics for final sample ---
# ------------------------------------------------------------------------------------------------

# Mean age, sd age
print(f"Mean age: {final_sample_fa['age'].mean():.1f} years") # 15.3 years
print(f"SD age: {final_sample_fa['age'].std():.1f} years") # 3.5 years
print(f"Age range: {final_sample_fa['age'].min():.1f} - {final_sample_fa['age'].max():.1f} years") #  8.2 - 23.0 years
print(f"Sex distribution: {final_sample_fa['sex'].value_counts().to_dict()}") # 'F': 608, 'M': 537
print(f"Race distribution: {final_sample_fa['race'].value_counts().to_dict()}") # 'White': 525, 'Black': 486, 'More_than_one_race': 120, 'Asian': 10, 'US_India_Alaska': 3, 'Hawaiian_Pacific': 1
print(f"Mean fd: {final_sample_fa['mean_fd'].mean():.1f}") # 0.3
print(f"SD fd: {final_sample_fa['mean_fd'].std():.1f}") # 0.2
print(f"T1 neighbor correlation: {final_sample_fa['t1_neighbor_corr'].mean():.1f}") # 0.9
print(f"SD t1 neighbor correlation: {final_sample_fa['t1_neighbor_corr'].std():.2f}") # 0.01

# ------------------------------------------------------------------------------------------------
# --- save final sample files ---
# ------------------------------------------------------------------------------------------------

# Save full sample
final_sample_fa.to_csv(f'{output_dir}/pnc_final_sample_fa.csv', index=False)
print(f"\nFinal full sample created: N = {len(final_sample_fa)}") # N = 1145
print(f"Files saved to: {output_dir}")

# Save cognition sample
final_sample_fa_cognition.to_csv(f'{output_dir}/pnc_final_cognition_sample_fa.csv', index=False)
print(f"\nFinal cognition sample created: N = {len(final_sample_fa_cognition)}") # N = 1142
print(f"Files saved to: {output_dir}")

print(f"Files saved to: {output_dir}")