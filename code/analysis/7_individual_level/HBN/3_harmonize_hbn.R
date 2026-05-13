# ------------------------------------------------------------------------------
# Harmonize multisite tract scalar data (COMBAT-GAM) — HBN
# ------------------------------------------------------------------------------
# Run after:
#   - create_group_level_hbn.R  → hbn_tracts_{fa,md,icvf}.csv, hbn_qc_measures.csv in cleaned/
#   - sample_creation_hbn.py     → hbn_final_sample_{fa,md,icvf}.csv in final_sample/
# Use harmonized CSVs in downstream GAM scripts (e.g. set USE_HARMONIZED <- TRUE).
#
# Uses ComBatFamily::covfam (GAM on batch with covariates) with age, sex, mean_fd.
#
# Requires: tidyverse, ComBatFamily, mgcv (via ComBatFamily)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
  library(ComBatFamily)
})

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
data_root <- "/Volumes/tractmaps/data/HBN/derivatives"
sample_dir <- file.path(data_root, "final_sample")
METRICS <- c("fa", "md", "icvf")
HARMONIZE_AGE_SAMPLE <- TRUE

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

#' Tract columns in HBN final_sample CSVs: qsirecon bundle columns {metric}_*
tract_column_names_hbn <- function(dat, path_in) {
  bn <- basename(path_in)
  m <- str_match(bn, "^hbn_final_sample_(fa|md|icvf)\\.csv$")
  if (is.na(m[1, 1])) {
    stop("Unexpected CSV name (expected hbn_final_sample_<metric>.csv): ", bn)
  }
  prefix <- paste0(m[1, 2], "_")
  nm <- names(dat)
  nm[str_starts(nm, prefix)]
}

#' Harmonize one wide sample CSV; writes *_harmonized.csv next to input
harmonize_one_file <- function(path_in, label) {
  if (!file.exists(path_in)) {
    message("Skip (missing): ", path_in)
    return(invisible(NULL))
  }

  message("\n=== ", label, " ===")
  message("Reading: ", path_in)
  dat <- read_csv(path_in, show_col_types = FALSE)

  if (!all(c("age", "sex", "mean_fd", "subject_id", "site") %in% names(dat))) {
    stop("Required columns missing: need subject_id, age, sex, mean_fd, site.")
  }

  tract_cols <- tract_column_names_hbn(dat, path_in)
  if (length(tract_cols) == 0) {
    stop("No tract columns found (expected names starting with fa_, md_, or icvf_).")
  }

  site_f <- as.factor(dat$site)
  cov_keep <- dat %>%
    transmute(
      subject_id = subject_id,
      age = as.numeric(age),
      sex = as.factor(sex),
      mean_fd = as.numeric(mean_fd),
      site = site_f
    )

  ok_row <- complete.cases(cov_keep %>% select(age, sex, mean_fd, site))
  n_drop <- sum(!ok_row)
  if (n_drop > 0) {
    message("Dropping ", n_drop, " row(s) with NA in age, sex, mean_fd, or site.")
  }
  dat <- dat[ok_row, , drop = FALSE]
  cov_keep <- cov_keep[ok_row, , drop = FALSE]

  Y <- dat %>% select(all_of(tract_cols))
  message("Tract columns: ", ncol(Y), " | Subjects: ", nrow(Y))
  n_na_tract <- sum(is.na(Y))
  message("NAs in tract block: ", n_na_tract)
  if (n_na_tract > 0) {
    stop(
      "Tract block contains NA values. ",
      "Use complete tract data or impute before running this script."
    )
  }

  cov_model <- cov_keep %>% select(age, sex, mean_fd)

  message("Running covfam (COMBAT-GAM)...")
  harm <- tryCatch(
    covfam(
      Y,
      bat = as.factor(cov_keep$site),
      covar = cov_model,
      model = "gam",
      formula = y ~ s(age, k = 3) + sex + mean_fd,
      eb = TRUE
    ),
    error = function(e) {
      stop("covfam failed: ", conditionMessage(e))
    }
  )

  Y_hat <- as.data.frame(harm$dat.covbat)
  if (ncol(Y_hat) != ncol(Y) || nrow(Y_hat) != nrow(Y)) {
    stop("Unexpected covfam output dimensions.")
  }
  names(Y_hat) <- names(Y)

  meta_cols <- setdiff(names(dat), tract_cols)
  meta_df <- dat %>% select(all_of(meta_cols))

  out <- bind_cols(meta_df, Y_hat)
  first <- c("subject_id", "age", "sex", "site", "mean_fd")
  first <- first[first %in% names(out)]
  rest <- setdiff(names(out), first)
  out <- out %>% select(all_of(c(first, rest)))

  path_out <- sub("\\.csv$", "_harmonized.csv", path_in)
  write_csv(out, path_out)
  message("Wrote: ", path_out)
  invisible(path_out)
}

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
for (metric in METRICS) {
  if (isTRUE(HARMONIZE_AGE_SAMPLE)) {
    path <- file.path(sample_dir, sprintf("hbn_final_sample_%s.csv", metric))
    harmonize_one_file(path, sprintf("HBN age sample | %s", metric))
  }
}
message("\nHarmonization finished. Point GAM scripts at *_harmonized.csv.")
