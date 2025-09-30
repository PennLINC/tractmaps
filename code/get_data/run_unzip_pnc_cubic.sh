#!/bin/bash
########################################################
### Extract PNC dMRI scalar measures in tracts ###
########################################################


# This script calls unzip_files.sh to extract the dMRI scalar measures in tracts for all PNC subjects.

input_dir=/cbica/projects/pennlinc_rbc/datasets/LINC_PNC/derivatives/QSIRECON-1-1-0_BUNDLE-STATS_zipped
output_dir=/cbica/projects/tractmaps/data/PNC/QSIRECON-1-1-0_BUNDLE-STATS/individual_tract_scalar_measures/


# file with scalar measures in tracts
file_pattern="qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-PNC1/dwi/sub-*_ses-PNC1_*space-ACPC_bundles-DSIStudio_scalarstats.tsv"

# run script
bash unzip_files.sh \
	${input_dir} \
	${output_dir} \
	pnc_subject_list.txt \
	${file_pattern}


