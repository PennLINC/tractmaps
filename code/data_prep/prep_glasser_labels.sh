#!/bin/bash

########################################################################################
# Script to prepare glasser label.gii and dlabel.nii for left and right hemispheres
########################################################################################

# usage: bash prep_glasser_labels.sh

cd /Users/joelleba/PennLINC/tractmaps/data/raw/glasser_parcellation/
output_folder='/Users/joelleba/PennLINC/tractmaps/data/derivatives/glasser_parcellation'

# create output folder if it doesn't exist
mkdir -p $output_folder

# get left hemisphere gifti
wb_command -cifti-separate Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii COLUMN -label CORTEX_LEFT $output_folder/HCP_MMP_L.label.gii

# get right hemisphere gifti
wb_command -cifti-separate Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii COLUMN -label CORTEX_RIGHT $output_folder/HCP_MMP_R.label.gii

# print the region labels
cd $output_folder
echo "Current region labels are stored as:"
wb_command -file-information HCP_MMP_L.label.gii | grep -A 10 "Label table"
# region labels are stored as right first: 1-180, then left: 181-360. 

# remap the the region indices to be consistent with the rest of the data (left: 1-180, right: 181-360) using prep_gifti_labels.sh
echo "Remapping region indices to be consistent with the rest of the data..."
root='/Users/joelleba/PennLINC/tractmaps/code/data_prep/'
bash $root/remap_labels.sh HCP_MMP_L.label.gii
bash $root/remap_labels.sh HCP_MMP_R.label.gii

# convert GIFTI files to CIFTI files
wb_command -cifti-create-label HCP_MMP_L.dlabel.nii -left-label HCP_MMP_L.label.gii
wb_command -cifti-create-label HCP_MMP_R.dlabel.nii -right-label HCP_MMP_R.label.gii