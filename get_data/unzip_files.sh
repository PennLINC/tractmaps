#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 <input_dir> <output_dir> [sub_list|file_pattern] [file_pattern]"
    echo ""
    echo "Extract all .zip files from the specified input directory to the output directory using 7z."
    echo "Options:"
    echo "  sub_list      Path to a text file with subject IDs (one per line)."
    echo "  file_pattern  A pattern to extract specific files from each archive (using 7z's -r flag)."
    echo ""
    echo "Examples:"
    echo "  # Extract all zip files and overwrite existing files:"
    echo "  $0 /path/to/input /path/to/output"
    echo ""
    echo "  # Extract only files matching subjects in the list and overwrite existing files:"
    echo "  $0 /path/to/input /path/to/output subject_list.txt"
    echo ""
    echo "  # Extract only files matching the given pattern and overwrite existing files:"
    echo "  $0 /path/to/input /path/to/output \"*/sub-*/anat/*_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz\""
    echo ""
    echo "  # Use both subject list and file pattern (overwrite existing files):"
    echo "  $0 /path/to/input /path/to/output subject_list.txt \"*/sub-*/anat/*_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz\""
    exit 0
}

# Check if help is requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
fi

# Check if the correct number of arguments is provided (minimum 2, maximum 4)
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Error: Invalid number of arguments."
    show_help
fi

# Assign input and output directory variables
input_dir="$1"
output_dir="$2"
sub_list=""
file_pattern=""

# Determine if the third argument is a subject list file or a file pattern.
if [ "$#" -ge 3 ]; then
    if [ -f "$3" ]; then
        sub_list="$3"
    else
        file_pattern="$3"
    fi
fi

# If four arguments are provided, assume the third is sub_list and the fourth is file_pattern.
if [ "$#" -eq 4 ]; then
    sub_list="$3"
    file_pattern="$4"
fi

# Validate input directory
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' does not exist."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$output_dir"

# Define extraction command based on whether a file pattern is provided.
if [ -n "$file_pattern" ]; then
    extract_cmd() {
        local zip_file="$1"
        echo "Extracting (with file pattern): $zip_file"
        7z e "$zip_file" -aoa -o"$output_dir" -r "$file_pattern"
    }
else
    extract_cmd() {
        local zip_file="$1"
        echo "Extracting: $zip_file"
        7z x "$zip_file" -aoa -o"$output_dir"
    }
fi

# Extraction using subject list if provided
if [ -n "$sub_list" ]; then
    if [ ! -f "$sub_list" ]; then
        echo "Error: Subject list file '$sub_list' not found."
        exit 1
    fi
    while read -r subid; do
        # Skip empty lines
        [ -z "$subid" ] && continue
        found=false
        for zip_file in "$input_dir"/*${subid}*.zip; do
            if [ -f "$zip_file" ]; then
                extract_cmd "$zip_file"
                found=true
            fi
        done
        if [ "$found" == false ]; then
            echo "Warning: No zip file found for subject: $subid"
        fi
    done < "$sub_list"
else
    # Extraction without a subject list (i.e. all zip files)
    found=false
    for zip_file in "$input_dir"/*.zip; do
        if [ -f "$zip_file" ]; then
            extract_cmd "$zip_file"
            found=true
        fi
    done
    if [ "$found" == false ]; then
        echo "Warning: No zip files found in $input_dir"
    fi
fi

echo "Extraction complete."

