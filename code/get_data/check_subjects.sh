#!/bin/bash
########################################################
### Check subjects in PNC dMRI scalar measures ###
########################################################


# This script checks the subjects in the input directory and saves the subject list to a file.
pnc_subject_list=/cbica/projects/tractmaps/code/get_data/pnc_subject_list.txt
dmri_dir=/cbica/projects/tractmaps/data/PNC/QSIRECON-1-1-0_BUNDLE-STATS/individual_tract_scalar_measures/

# Create output directory
OUTPUT_DIR="/cbica/projects/tractmaps/code/get_data/folder_contents"
mkdir -p "$OUTPUT_DIR"

# Extract subject IDs from dmri_dir files (remove path and extension)
# Handle both formats: simple and with additional words between ses-PNC1 and space
ls -1 ${dmri_dir}/sub-*_ses-PNC1_*space-ACPC_bundles-DSIStudio_scalarstats.tsv | sed 's/.*\///' | sed 's/_ses-PNC1_.*_space-ACPC_bundles-DSIStudio_scalarstats.tsv//' | sed 's/_ses-PNC1_space-ACPC_bundles-DSIStudio_scalarstats.tsv//' > ${OUTPUT_DIR}/dmri_dir_subjects.txt

# Find subjects in list but not in dmri_dir
echo "Subjects in subject list but missing from dmri_dir:"
comm -23 <(sort ${pnc_subject_list}) <(sort ${OUTPUT_DIR}/dmri_dir_subjects.txt) > ${OUTPUT_DIR}/missing_subjects.txt
missing_count=$(wc -l < ${OUTPUT_DIR}/missing_subjects.txt)
echo "N = ${missing_count} subjects missing from dmri_dir"
if [ ${missing_count} -gt 0 ]; then
    echo "Missing subject IDs:"
    cat ${OUTPUT_DIR}/missing_subjects.txt
fi