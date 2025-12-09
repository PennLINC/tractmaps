# ------------------------------------------------------------------------------
### Generate group-level tract scalar data ####
# ------------------------------------------------------------------------------

# Get the data first on cubic using: 
# - /cbic/projects/tractmaps/code/get_data/get_subjects_list.sh
# - /cbic/projects/tractmaps/code/get_data/unzip_files.sh
# - /cbic/projects/tractmaps/code/get_data/run_unzip_pnc_cubic.sh

# scalar measures of tracts are from RBC on cubic (07/2025): /cbica/projects/pennlinc_rbc/datasets/LINC_PNC/derivatives/
# QSIRECON-1-1-0_BUNDLE-STATS_zipped
# These files were pulled: qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-PNC1/dwi/sub-*_ses-PNC1_space-ACPC_bundles-DSIStudio_scalarstats.tsv
# Individual-level files are at: /cbica/projects/tractmaps/data/PNC/QSIRECON-1-1-0_BUNDLE-STATS/individual_tract_scalar_measures/

# Note: this script requires access to the cubic project (mounted locally) 
# ------------------------------------------------------------------------------

library(tidyverse)


# Define root directory and output directory
root_dir <- "/Volumes/tractmaps/data/PNC/QSIRECON-1-1-0_BUNDLE-STATS/individual_tract_scalar_measures"
output_dir <- "/Users/joelleba/PennLINC/tractmaps/data/derivatives/individual_level_pnc/cleaned"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


### Helper function to read and process individual TSV files ####

read_tract_scalar_file <- function(file_path) {
  # Read the TSV file
  data <- read_tsv(file_path, show_col_types = FALSE)
  
  # Extract subject ID from filename
  subject_id <- str_extract(basename(file_path), "sub-[0-9]+")
  
  # Add subject ID column
  data <- data %>%
    mutate(subject_id = subject_id)
  
  return(data)
}

# ------------------------------------------------------------------------------
### Combine all tract scalar data ####
# ------------------------------------------------------------------------------

# Check that root_dir exists
if (!dir.exists(root_dir)) {
  stop("Root directory does not exist. Need to mount the cubic tractmaps dir onto the local machine.")
}

# Get list of all TSV files
tsv_files <- list.files(root_dir, pattern = ".*_ses-PNC1_.*space-ACPC_bundles-DSIStudio_scalarstats.tsv", 
                        full.names = TRUE, recursive = FALSE)

cat(sprintf("Found %d TSV files to process\n", length(tsv_files))) # N = 1406

# Read and combine all files
cat("Reading and combining tract scalar data...\n")

all_data <- purrr::map_dfr(tsv_files, function(file) {
  read_tract_scalar_file(file)
}, .id = "file_id")

# Check the structure of the data
cat("Data structure:\n")
print(str(all_data))
cat("Column names:\n")
print(colnames(all_data))

# Save to RData file for easier loading later on
save(all_data, file = file.path(output_dir, "all_data_pnc.RData"))
# load(file.path(output_dir, "all_data_pnc.RData"))

# ------------------------------------------------------------------------------
### Load and verify tract names with abbreviations #### 
# ------------------------------------------------------------------------------

# Load tract abbreviations
cat("\nLoading tract abbreviations...\n")
tract_abbreviations <- readxl::read_excel("/Users/joelleba/PennLINC/tractmaps/data/derivatives/tract_names/abbreviations.xlsx")
print(paste("Loaded", nrow(tract_abbreviations), "tract abbreviations"))

# Get unique tract names from all_data
all_tract_names <- unique(all_data$bundle)
cat(paste("Found", length(all_tract_names), "unique tract names in qsirecon data\n")) # Finds 67 tracts in qsirecon data

# Check which tracts from our data are in the abbreviations file
tracts_in_abbreviations <- all_tract_names %in% tract_abbreviations$new_qsirecon_tract_names
missing_tracts <- all_tract_names[!tracts_in_abbreviations]

cat(paste("Tracts found in abbreviations file:", sum(tracts_in_abbreviations), "out of", length(all_tract_names), "\n")) # finds 32 tracts out of 67

if (length(missing_tracts) > 0) {
  cat("Missing tracts (not found in abbreviations file):\n")
  print(missing_tracts)
} else {
  cat("All tracts found in abbreviations file!\n")
}

# Get list of tracts that are in both our data and abbreviations file
valid_tracts <- all_tract_names[tracts_in_abbreviations]
cat(paste("Will process", length(valid_tracts), "tracts that are in abbreviations file:\n"))
print(valid_tracts)

# Filter data to keep only valid tracts
all_data <- all_data %>%
  filter(bundle %in% valid_tracts)

cat(paste("After filtering, data has", nrow(all_data), "rows\n")) # 70080 rows


# ------------------------------------------------------------------------------
### Extract FA measures for valid tracts only ####
# ------------------------------------------------------------------------------

# Filter for FA and MD measures, keeping only tracts in abbreviations file
fa_data <- all_data %>%
  filter(variable_name == "dti_fa", bundle %in% valid_tracts) %>%
  select(subject_id, bundle, mean) %>% # getting mean FA for each bundle
  pivot_wider(
    names_from = bundle,
    values_from = mean,
    names_prefix = "fa_"
  )

cat(sprintf("FA data: %d subjects, %d tracts\n", nrow(fa_data), ncol(fa_data) - 1)) # FA data: 1406 subjects, 32 tracts

# ------------------------------------------------------------------------------
### Save group-level data ####
# ------------------------------------------------------------------------------

# Save FA data
cat("Saving FA data...\n")
write_csv(fa_data, file.path(output_dir, "pnc_tracts_fa.csv"))

cat("Done! File saved to:", output_dir, "\n")
