#!/bin/bash
########################################################
### Extract HCP-YA dMRI scalar measures (tracts) ###
########################################################
# Run on CUBIC after get_subjects_list_hcpya.sh.

input_dir=/cbica/projects/pennlinc_hcpya/data/qsirecon/DANGER
sub_list="hcpya_subject_list.txt"

# 1. DSIStudio scalarstats -> scalar_stats
output_dir_scalar=/cbica/projects/tractmaps/data/HCPYA/QSIRECON-1-0-0rc2_BUNDLE-STATS/scalar_stats
file_pattern_scalar="qsirecon-1-0-0rc2/derivatives/qsirecon-DSIStudio/sub-*/dwi/sub-*_space-T1w_bundles-DSIStudio_scalarstats.tsv"
bash unzip_files.sh \
	${input_dir} \
	${output_dir_scalar} \
	${sub_list} \
	${file_pattern_scalar}

echo "HCP-YA extraction complete."

