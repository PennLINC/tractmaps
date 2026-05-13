#!/bin/bash
########################################################
### Get subject list from HCP-YA dMRI scalar measures ###
########################################################


# HCP-YA qsirecon zips (adjust path and zip pattern if DANGER naming differs)
input_dir=/cbica/projects/pennlinc_hcpya/data/qsirecon/DANGER

# List and count
echo "Listing subjects in input directory:"
ls -1 ${input_dir}/*.zip 2>/dev/null | wc -l

# Save subject list (strip path, .zip, and any _qsirecon-* suffix to get sub-XXXXX)
ls -1 ${input_dir}/*.zip 2>/dev/null | sed 's/.*\///' | sed 's/\.zip//' | sed 's/_qsirecon-.*//' > hcpya_subject_list.txt
echo "Subject list saved to hcpya_subject_list.txt"



