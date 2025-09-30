#!/bin/bash
########################################################
### Get subject list from PNC dMRI scalar measures ###
########################################################


# This script lists the subjects in the input directory and saves the subject list to a file.

input_dir=/cbica/projects/pennlinc_rbc/datasets/LINC_PNC/derivatives/QSIRECON-1-1-0_BUNDLE-STATS_zipped

# List subjects in input directory
echo "Listing subjects in input directory:"
ls -1 ${input_dir}/sub-*_ses-PNC1_qsirecon-1-1-0.zip | wc -l

# Save subject list to file (extract only subject ID)
ls -1 ${input_dir}/sub-*_ses-PNC1_qsirecon-1-1-0.zip | sed 's/.*\///' | sed 's/_ses-PNC1_qsirecon-1-1-0.zip//' > pnc_subject_list.txt
echo "Subject list saved to pnc_subject_list.txt"



