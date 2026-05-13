# ------------------------------------------------------------
# Script to run GAMs on PNC data
# ------------------------------------------------------------

# This script runs GAMs on PNC data to assess the relationship between tract scalar measures, age, and cognition.
# It runs the func_GAM_tractmaps.R functions to fit the GAMs.
# Inputs: PNC final_sample FA/MD for age and cognition samples.
# Outputs: Partial R2 statistics for each tract and each variable.


# ------------------------------------------------------------
# --- Paths ---
# ------------------------------------------------------------

rm(list = ls())

local_root <- "/Users/joelleba/PennLINC/tractmaps/code_pennlinc_repo"
data_root <- "/Volumes/tractmaps"
tractmaps_results_root <- "/Users/joelleba/PennLINC/tractmaps/results"

source(file.path(local_root, "code", "analysis", "7_individual_level", "func_GAM_tractmaps.R"))

data_path <- file.path(data_root, "data", "PNC", "derivatives", "final_sample")
outpath <- file.path(tractmaps_results_root, "individual_level", "pnc")

# ------------------------------------------------------------
# --- Run age GAMs ---
# ------------------------------------------------------------

gamtype <- "age"
metrics <- c("FA", "MD")

for (metric in metrics) {
  run_gam(
    dataset = "pnc",
    gamtype = gamtype,
    metric = metric,
    cognition_var = NULL,
    harmonized = FALSE,
    root = NULL,
    data_path = data_path,
    outpath = outpath
  )
}

# ------------------------------------------------------------
# --- Run cognition GAMs ---
# ------------------------------------------------------------

gamtype <- "cognition"
metrics <- c("FA", "MD")

for (metric in metrics) {
  run_gam(
    dataset = "pnc",
    gamtype = gamtype,
    metric = metric,
    cognition_var = "F3_Executive_Efficiency",
    harmonized = FALSE,
    root = NULL,
    data_path = data_path,
    outpath = outpath
  )
}
