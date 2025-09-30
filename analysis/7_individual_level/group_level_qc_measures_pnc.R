# ------------------------------------------------------------------------------
### Generate group-level QC measures data for PNC ####
# ------------------------------------------------------------------------------

library(tidyverse)

# Define root directory and output directory
root_dir <- "/Volumes/tractmaps/data/PNC/QSIPREP-1-0-0rc1/individual_qc_measures"
output_dir <- "/Users/joelleba/PennLINC/tractmaps/data/derivatives/individual_level_pnc/cleaned"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Check that root_dir exists
if (!dir.exists(root_dir)) {
  stop("Root directory does not exist. Need to mount the cubic tractmaps dir onto the local machine.")
}

# ------------------------------------------------------------------------------
### Helper function to read and process individual QC CSV files ####
# ------------------------------------------------------------------------------

read_qc_file <- function(file_path) {
  # Read the CSV file
  data <- read_csv(file_path, show_col_types = FALSE)
  
  # Extract subject ID from filename
  subject_id <- str_extract(basename(file_path), "sub-[0-9]+")
  
  # Add subject ID column
  data <- data %>%
    mutate(subject_id = subject_id)
  
  return(data)
}

# ------------------------------------------------------------------------------
### Combine all QC measures data ####
# ------------------------------------------------------------------------------

# Get list of all CSV files
csv_files <- list.files(root_dir, pattern = ".*\\.csv$", 
                        full.names = TRUE, recursive = FALSE)

cat(sprintf("Found %d CSV files to process\n", length(csv_files))) # N = 1406

# Read and combine all files
cat("Reading and combining QC measures data...\n")

all_qc_data <- purrr::map_dfr(csv_files, function(file) {
  read_qc_file(file)
}, .id = "file_id")

# Check the structure of the data
cat("QC data structure:\n")
print(str(all_qc_data))
cat("QC column names:\n")
print(colnames(all_qc_data))

# Save to RData file for easier loading later on
save(all_qc_data, file = file.path(output_dir, "all_qc_data_pnc.RData"))
# load(file.path(output_dir, "all_qc_data_pnc.RData"))


# ------------------------------------------------------------------------------
### Save group-level QC data ####
# ------------------------------------------------------------------------------

# Create subjects x measures format
qc_subjects_measures <- all_qc_data %>%
  select(-file_id) %>%
  distinct() # Remove any duplicates

cat(sprintf("Final QC data: %d subjects, %d measures\n", # Final QC data: 1406 subjects, 60 measures
            nrow(qc_subjects_measures), 
            ncol(qc_subjects_measures) - 1))

# Save QC data
cat("Saving QC data...\n")
write_csv(qc_subjects_measures, file.path(output_dir, "pnc_qc_measures.csv"))

cat("Done! QC measures file saved to:", output_dir, "\n") 
