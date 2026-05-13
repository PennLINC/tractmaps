# ------------------------------------------------------------
# Script to run age GAMs on HBN data (FA, MD, ICVF)
# ------------------------------------------------------------

# Runs func_GAM_tractmaps.R::run_gam() on COMBAT-harmonized full samples from harmonize_hbn.R:
# (Run sample_creation_hbn.py and harmonize_hbn.R first.)

# ------------------------------------------------------------
# --- Paths ---
# ------------------------------------------------------------

rm(list = ls())

local_root <- "/Users/joelleba/PennLINC/tractmaps/code_pennlinc_repo"
data_root <- "/Volumes/tractmaps"
tractmaps_results_root <- "/Users/joelleba/PennLINC/tractmaps/results"

source(file.path(local_root, "code", "analysis", "7_individual_level", "func_GAM_tractmaps.R"))

data_path <- file.path(data_root, "data", "HBN", "derivatives", "final_sample")
outpath <- file.path(tractmaps_results_root, "individual_level", "hbn")

# ------------------------------------------------------------
# --- Age GAMs (FA, MD, ICVF), harmonized samples ---
# ------------------------------------------------------------

gamtype <- "age"
metrics <- c("FA", "MD", "ICVF")

for (metric in metrics) {
  run_gam(
    dataset = "hbn",
    gamtype = gamtype,
    metric = metric,
    cognition_var = NULL,
    harmonized = TRUE,
    root = NULL,
    data_path = data_path,
    outpath = outpath
  )
}
