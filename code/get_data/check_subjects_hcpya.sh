#!/bin/bash
########################################################
### Check HCP-YA subjects: list vs extracted TSVs ###
########################################################
# Expect 0 missing if unzip completed successfully.
# get_subjects_list_hcpya.sh stores IDs without "sub-"; TSVs are named sub-<id>_space-T1w_...
# Normalize both sides (strip leading sub-) so comm matches the same participant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
subject_list="${SCRIPT_DIR}/hcpya_subject_list.txt"
scalar_dir=/cbica/projects/tractmaps/data/HCPYA/QSIRECON-1-0-0rc2_BUNDLE-STATS/scalar_stats
OUTPUT_DIR="${SCRIPT_DIR}/folder_contents_hcpya"
mkdir -p "$OUTPUT_DIR"

# Basename -> subject ID; strip sub- to align with hcpya_subject_list.txt (zip stems, no sub-)
tsv_to_subject_id() {
	sed 's/.*\///' | sed 's/_space-T1w_bundles-DSIStudio_scalarstats.tsv//' | sed 's/^sub-//'
}

# Subjects that have DSIStudio scalar_stats TSV output
ls -1 ${scalar_dir}/sub-*_space-T1w_bundles-DSIStudio_scalarstats.tsv 2>/dev/null | tsv_to_subject_id | sort -u > "${OUTPUT_DIR}/dmri_dir_subjects_hcpya_scalar.txt"

normalize_ids() {
	sort | sed 's/^sub-//'
}

echo "Subjects in subject list but missing from scalar_stats:"
comm -23 <(normalize_ids < "${subject_list}") <(sort "${OUTPUT_DIR}/dmri_dir_subjects_hcpya_scalar.txt") > "${OUTPUT_DIR}/missing_subjects_hcpya_scalar.txt"
missing_scalar=$(wc -l < "${OUTPUT_DIR}/missing_subjects_hcpya_scalar.txt")
echo "N = ${missing_scalar} subjects missing from scalar_stats"
if [ "${missing_scalar}" -gt 0 ]; then
  cat "${OUTPUT_DIR}/missing_subjects_hcpya_scalar.txt"
fi
