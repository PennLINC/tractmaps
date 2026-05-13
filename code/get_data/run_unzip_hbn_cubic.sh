#!/bin/bash
########################################################
### Extract HBN dMRI scalar measures in tracts ###
########################################################

# This script copies the dMRI scalar and NODDI measures in tracts for all HBN subjects.

sub_list="hbn_subject_list.txt"
input_base=/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/derivatives/HBN_QSIRECON-1-1-1_scalarstats/qsirecon/derivatives
out_scalars=/cbica/projects/tractmaps/data/HBN/QSIRECON-1-1-1_BUNDLE-STATS/scalar_stats
out_noddi=/cbica/projects/tractmaps/data/HBN/QSIRECON-1-1-1_BUNDLE-STATS/noddi
file_pattern='*_space-ACPC_bundles-DSIStudio_scalarstats.tsv'

mkdir -p "$out_scalars" "$out_noddi"

# Copy DSIStudio scalarstats and wmNODDI scalarstats to output directories
(
  shopt -s nullglob
  while read -r s; do
    [ -z "$s" ] && continue
    for f in "$input_base/qsirecon-DSIStudio/$s/ses-1/dwi/"$file_pattern; do cp -f "$f" "$out_scalars/"; done
    for f in "$input_base/qsirecon-wmNODDI/$s/ses-1/dwi/"$file_pattern; do cp -f "$f" "$out_noddi/"; done
  done < "$sub_list"
)

# Copy QC measures to output directory
input_dir=/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/derivatives/QSIPREP-1-0-1_zipped
output_dir_qc=/cbica/projects/tractmaps/data/HBN/QSIPREP-1-0-1/qc
file_pattern_qc="qsiprep/sub-*/ses*/dwi/sub-*space-ACPC_desc-image_qc.tsv"

mkdir -p "$output_dir_qc"
bash unzip_files.sh \
	${input_dir} \
	${output_dir_qc} \
	${sub_list} \
	${file_pattern_qc}

echo "HBN extraction complete."