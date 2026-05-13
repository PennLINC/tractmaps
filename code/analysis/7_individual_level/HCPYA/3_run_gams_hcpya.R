# ------------------------------------------------------------
# Script to run cognition GAMs on HCP-YA data (FA and MD)
# ------------------------------------------------------------

# Runs func_GAM_tractmaps.R::run_gam() on cognition samples from sample_creation_hcpya.py.

# ------------------------------------------------------------
# --- Paths ---
# ------------------------------------------------------------

rm(list = ls())

local_root <- "/Users/joelleba/PennLINC/tractmaps/code_pennlinc_repo"
data_root <- "/Volumes/tractmaps"
tractmaps_results_root <- "/Users/joelleba/PennLINC/tractmaps/results"

source(file.path(local_root, "code", "analysis", "7_individual_level", "func_GAM_tractmaps.R"))

data_path <- file.path(data_root, "data", "HCPYA", "derivatives", "final_sample")
outpath <- file.path(tractmaps_results_root, "individual_level", "hcpya")

# ------------------------------------------------------------
# --- Cognition GAMs (FA, MD) ---
# ------------------------------------------------------------

gamtype <- "cognition"
metrics <- c("FA", "MD")
cognition_var <- "CogFluidComp_Unadj"

for (metric in metrics) {
  run_gam(
    dataset = "hcpya",
    gamtype = gamtype,
    metric = metric,
    cognition_var = cognition_var,
    harmonized = FALSE,
    root = NULL,
    data_path = data_path,
    outpath = outpath
  )
}
