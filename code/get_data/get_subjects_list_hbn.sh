#!/bin/bash
########################################################
### Get subject list from HBN QSIRECON (DSIStudio) ###
########################################################

input_dir=/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/derivatives/HBN_QSIRECON-1-1-1_scalarstats/qsirecon/derivatives/qsirecon-DSIStudio

out_file=hbn_subject_list.txt

echo "Listing subject in input directory:"

# Only immediate child dirs named sub-*; output is basename only (one sub-ID per line)
find "$input_dir" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print |
  sed 's|.*/||' |
  LC_ALL=C sort -u > "$out_file"

count=$(wc -l < "$out_file" | tr -d ' ')
echo "Found $count unique subject ID(s)."
echo "Subject list saved to $out_file"
