# ------------------------------------------------------------------------------
### HBN: group-level FA, MD, ICVF tract scalars + group-level QC ####
# ------------------------------------------------------------------------------
# Uses helper_functions.R: aggregate_tract_scalars, aggregate_qc
#
# Prerequisite data (see code/get_data/ on CUBIC; run_unzip_* / copy scripts):
# - /cbica/projects/tractmaps/data/HBN/... QSIRECON scalar_stats + noddi TSVs, QSIPREP qc TSVs
#
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
### Source helpers and set paths ####
# ------------------------------------------------------------------------------

code_dir <- "/Users/joelleba/PennLINC/tractmaps/code_pennlinc_repo/code/analysis/7_individual_level"
source(file.path(code_dir, "helper_functions.R"))

# FA and MD: qsirecon scalar_stats
scalar_fa_md_dir <- "/Volumes/tractmaps/data/HBN/QSIRECON-1-1-1_BUNDLE-STATS/scalar_stats"
# ICVF
scalar_noddi_dir <- "/Volumes/tractmaps/data/HBN/QSIRECON-1-1-1_BUNDLE-STATS/noddi"

# One pattern for both scalar_stats and noddi
tsv_file_pattern_hbn <- ".*_ses-.*_.*space-ACPC_bundles-DSIStudio_scalarstats\\.tsv$"

# QC
qc_input_dir <- "/Volumes/tractmaps/data/HBN/QSIPREP-1-0-1/qc"
qc_file_pattern <- ".*_space-ACPC_desc-image_qc\\.tsv$"

# Written outputs
output_dir <- "/Volumes/tractmaps/data/HBN/derivatives/cleaned"

# Tract abbreviations
tract_abbrev_path <- "/Users/joelleba/PennLINC/tractmaps/data/derivatives/tract_names/abbreviations.xlsx"

# ------------------------------------------------------------------------------
### Tract metrics: FA, MD, ICVF ####
# ------------------------------------------------------------------------------

aggregate_tract_scalars(
  input_dir = scalar_fa_md_dir,
  file_pattern = tsv_file_pattern_hbn,
  dataset = "hbn",
  metric = "fa",
  metric_name = "dti_fa",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

aggregate_tract_scalars(
  input_dir = scalar_fa_md_dir,
  file_pattern = tsv_file_pattern_hbn,
  dataset = "hbn",
  metric = "md",
  metric_name = "md",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

aggregate_tract_scalars(
  input_dir = scalar_noddi_dir,
  file_pattern = tsv_file_pattern_hbn,
  dataset = "hbn",
  metric = "icvf",
  metric_name = "icvf",
  output_dir = output_dir,
  tract_abbrev_xlsx = tract_abbrev_path
)

# ------------------------------------------------------------------------------
### QC measures ####
# ------------------------------------------------------------------------------

aggregate_qc(
  input_dir = qc_input_dir,
  file_pattern = qc_file_pattern,
  dataset = "hbn",
  output_dir = output_dir,
  reader = "tsv"
)

cat("HBN group-level steps finished (FA, MD, ICVF, QC).\n")
