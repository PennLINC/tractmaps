#!/bin/bash
########################################################
### Check HBN subjects: list vs outputs (run_unzip_hbn_cubic.sh paths) ###
########################################################
# Compares hbn_subject_list.txt to files under scalar_stats, noddi, and QSIPREP qc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
subject_list="${SCRIPT_DIR}/hbn_subject_list.txt"

# Same as run_unzip_hbn_cubic.sh
out_scalars=/cbica/projects/tractmaps/data/HBN/QSIRECON-1-1-1_BUNDLE-STATS/scalar_stats
out_noddi=/cbica/projects/tractmaps/data/HBN/QSIRECON-1-1-1_BUNDLE-STATS/noddi
out_qc=/cbica/projects/tractmaps/data/HBN/QSIPREP-1-0-1/qc

OUTPUT_DIR="${SCRIPT_DIR}/folder_contents_hbn"
mkdir -p "$OUTPUT_DIR"

# Basenames → subject id (strip path and _ses-* suffix)
_subs_from_glob() {
  local dir="$1"
  local glob="$2"
  local out="$3"
  ls -1 "$dir"/$glob 2>/dev/null | sed 's#.*/##; s#_ses-.*##' | sort -u > "$out"
}

_subs_from_glob "$out_scalars" '*_space-ACPC_bundles-DSIStudio_scalarstats.tsv' \
  "${OUTPUT_DIR}/dmri_dir_subjects_hbn_scalar.txt"
_subs_from_glob "$out_noddi" '*_space-ACPC_bundles-DSIStudio_scalarstats.tsv' \
  "${OUTPUT_DIR}/dmri_dir_subjects_hbn_noddi.txt"
_subs_from_glob "$out_qc" '*_space-ACPC_desc-image_qc.tsv' \
  "${OUTPUT_DIR}/dmri_dir_subjects_hbn_qc.txt"

_report_missing() {
  local label="$1"
  local found_file="$2"
  local out_missing="$3"
  echo ""
  echo "Subjects in list but missing from ${label}:"
  comm -23 <(sort "$subject_list") <(sort "$found_file") > "$out_missing"
  local n
  n=$(wc -l < "$out_missing" | tr -d ' ')
  echo "N = ${n}"
  if [ "$n" -gt 0 ]; then
    cat "$out_missing"
  fi
}

_report_missing "scalar_stats (${out_scalars})" \
  "${OUTPUT_DIR}/dmri_dir_subjects_hbn_scalar.txt" \
  "${OUTPUT_DIR}/missing_subjects_hbn_scalar.txt"
_report_missing "noddi (${out_noddi})" \
  "${OUTPUT_DIR}/dmri_dir_subjects_hbn_noddi.txt" \
  "${OUTPUT_DIR}/missing_subjects_hbn_noddi.txt"
_report_missing "qc (${out_qc})" \
  "${OUTPUT_DIR}/dmri_dir_subjects_hbn_qc.txt" \
  "${OUTPUT_DIR}/missing_subjects_hbn_qc.txt"
