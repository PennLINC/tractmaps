# ------------------------------------------------------------------------------
### HCP-YA: group-level FA and MD tract scalars ####
# ------------------------------------------------------------------------------
# Uses helper_functions.R: aggregate_tract_scalars
#
# Prerequisite data (see get_data/ on CUBIC):
# - /cbica/projects/tractmaps/data/HCPYA/... individual scalar TSVs
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
### Source helpers and set paths ####
# ------------------------------------------------------------------------------

code_dir <- "/Users/joelleba/PennLINC/tractmaps/code_pennlinc_repo/code/analysis/7_individual_level"
source(file.path(code_dir, "helper_functions.R"))

# Per-subject bundle scalarstats (FA and MD)
scalar_input_dir <- "/Volumes/tractmaps/data/HCPYA/QSIRECON-1-0-0rc2_BUNDLE-STATS/scalar_stats"
tsv_file_pattern <- ".*_space-T1w_bundles-DSIStudio_scalarstats\\.tsv$"


output_dir <- "/Volumes/tractmaps/data/HCPYA/derivatives/cleaned"

tract_abbrev_path <- "/Users/joelleba/PennLINC/tractmaps/data/derivatives/tract_names/abbreviations.xlsx"

# ------------------------------------------------------------------------------
### Tract scalars: FA, MD (each run stacks TSVs + abbrev filter) ####
# ------------------------------------------------------------------------------

aggregate_tract_scalars(
  input_dir = scalar_input_dir,
  file_pattern = tsv_file_pattern,
  dataset = "hcpya",
  metric = "fa",
  metric_name = "dti_fa",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

aggregate_tract_scalars(
  input_dir = scalar_input_dir,
  file_pattern = tsv_file_pattern,
  dataset = "hcpya",
  metric = "md",
  metric_name = "md",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

cat("HCP-YA group-level steps finished (FA, MD). \n")
