#!/bin/bash
# -----------------------------------------------------------------------------
# remap_labels.sh
# -----------------------------------------------------------------------------
# This script remaps the region indices of a gifti label file to be consistent 
# with the rest of the data (original indices are right first: 1-180, then left: 181-360). 
# The script will remap the indices to be *left* first: 1-180, then *right*: 181-360.
# -----------------------------------------------------------------------------

# === USAGE ===
# bash remap_labels.sh input.label.gii

CIFTI_IN=$1
ROOT=
BASE=$(basename "$CIFTI_IN" .label.gii)
REMAP_FILE="${ROOT}${BASE}_key_remap.txt"
CIFTI_OUT=$CIFTI_IN # same name as input

# 1. Export the label table for inspection
wb_command -label-export-table "$CIFTI_IN" "${ROOT}${BASE}_labels.txt"

# 2. Auto-generate a remapping table (heuristic: left = 1–180, right = 181–360)
# You may need to edit this logic if your data differs!
awk '
BEGIN { FS=" "; OFS=" " }
NR % 2 == 1 { label = $0; next }  # odd lines: label names
NR % 2 == 0 {
  key = $1;
  side = substr(label, 1, 1);  # L or R from label name
  newkey = key;
  if (side == "R" && key <= 180) {
    newkey = key + 180;
  } else if (side == "L" && key > 180) {
    newkey = key - 180;
  }
  if (key != newkey) {
    print key, newkey;
  }
}
' "${ROOT}${BASE}_labels.txt" > "$REMAP_FILE"

# 3. Apply key remapping
wb_command -label-modify-keys "$CIFTI_IN" "$REMAP_FILE" "$CIFTI_OUT"

echo "Done. New file: $CIFTI_OUT"

# check that the remapping worked
echo "New region labels are stored as:"
wb_command -file-information "$CIFTI_OUT" | grep -A 10 "Label table"

# remove the labels.txt file and the remapping file
rm "${ROOT}${BASE}_labels.txt"
rm "$REMAP_FILE"