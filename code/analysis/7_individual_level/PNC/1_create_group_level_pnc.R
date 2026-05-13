# ------------------------------------------------------------------------------
### PNC: group-level FA and MD tract scalars + group-level QC (CSV) ####
# ------------------------------------------------------------------------------
# Uses helper_functions.R: aggregate_tract_scalars, aggregate_qc
#
# Prerequisite data (see get_data/ on CUBIC):
# - /cbica/projects/tractmaps/data/PNC/... individual scalar TSVs and QC CSVs
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
### Source helpers and set paths ####
# ------------------------------------------------------------------------------

code_dir <- "/Users/joelleba/PennLINC/tractmaps/code_pennlinc_repo/code/analysis/7_individual_level"
source(file.path(code_dir, "helper_functions.R"))

# Per-subject bundle scalarstats
scalar_input_dir <- "/Volumes/tractmaps/data/PNC/QSIRECON-1-1-0_BUNDLE-STATS/individual_tract_scalar_measures"
tsv_file_pattern <- ".*_ses-PNC1_.*space-ACPC_bundles-DSIStudio_scalarstats\\.tsv$"

# Per-subject QC
qc_input_dir <- "/Volumes/tractmaps/data/PNC/QSIPREP-1-0-0rc1/individual_qc_measures"
qc_file_pattern <- ".*\\.csv$"

# Written outputs
output_dir <- "/Volumes/tractmaps/data/PNC/derivatives/cleaned"

# Tract abbreviations
tract_abbrev_path <- "/Users/joelleba/PennLINC/tractmaps/data/derivatives/tract_names/abbreviations.xlsx"

# ------------------------------------------------------------------------------
### Tract scalars: FA, MD ####
# ------------------------------------------------------------------------------

aggregate_tract_scalars(
  input_dir = scalar_input_dir,
  file_pattern = tsv_file_pattern,
  dataset = "pnc",
  metric = "fa",
  metric_name = "dti_fa",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

aggregate_tract_scalars(
  input_dir = scalar_input_dir,
  file_pattern = tsv_file_pattern,
  dataset = "pnc",
  metric = "md",
  metric_name = "md",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

# ------------------------------------------------------------------------------
### QC measures ####
# ------------------------------------------------------------------------------

aggregate_qc(
  input_dir = qc_input_dir,
  file_pattern = qc_file_pattern,
  dataset = "pnc",
  output_dir = output_dir
)

cat("All PNC group-level steps finished (FA, MD, QC).\n")
