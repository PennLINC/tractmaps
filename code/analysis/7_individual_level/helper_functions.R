# ------------------------------------------------------------------------------
### Shared helpers: group-level tract scalars and QC tables ####
# ------------------------------------------------------------------------------
# Requires: tidyverse, readxl (only when tract_abbrev_xlsx is set)
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
})


# ------------------------------------------------------------------------------
### Tract filtering using abbreviations sheet ####
# ------------------------------------------------------------------------------
# When tract_abbrev_xlsx is set: load sheet, print overlap with bundles (PNC style),
# then keep only bundles listed in the sheet. If path is NULL/empty, returns data unchanged.
# ------------------------------------------------------------------------------

filter_bundles_by_abbrev_sheet <- function(
    data,
    bundle_col = "bundle",
    tract_abbrev_xlsx,
    abbrev_col = "new_qsirecon_tract_names"
) {
  if (is.null(tract_abbrev_xlsx) || is.na(tract_abbrev_xlsx) || tract_abbrev_xlsx == "") {
    return(data)
  }

  # Load tract abbreviations
  cat("\nLoading tract abbreviations...\n")
  tract_abbreviations <- readxl::read_excel(tract_abbrev_xlsx)
  if (!abbrev_col %in% names(tract_abbreviations)) {
    stop("Column '", abbrev_col, "' not found in tract abbreviations file.")
  }
  print(paste("Loaded", nrow(tract_abbreviations), "tract abbreviations"))

  # Get unique tract names from stacked scalar data
  all_tract_names <- unique(data[[bundle_col]])
  cat(paste("Found", length(all_tract_names), "unique tract names in qsirecon data\n")) # e.g. 67 in PNC

  # Check which tracts from our data are in the abbreviations file
  tracts_in_abbreviations <- all_tract_names %in% tract_abbreviations[[abbrev_col]]
  missing_tracts <- all_tract_names[!tracts_in_abbreviations]

  cat(
    paste(
      "Tracts found in abbreviations file:",
      sum(tracts_in_abbreviations),
      "out of",
      length(all_tract_names),
      "\n"
    )
  ) # e.g. 32 of 67

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
  out <- data %>% dplyr::filter(.data[[bundle_col]] %in% valid_tracts)
  cat(paste("After filtering, data has", nrow(out), "rows\n")) # e.g. 70080 in PNC
  out
}


# ------------------------------------------------------------------------------
### Extract one metric: long → wide (subject x tracts) ####
# ------------------------------------------------------------------------------
# One variable_name (e.g. dti_fa) → columns {metric}_<bundle> using mean (or value_col).
# ------------------------------------------------------------------------------

pivot_bundle_scalar_wide <- function(
    data,
    metric,
    metric_name,
    bundle_col = "bundle",
    variable_col = "variable_name",
    value_col = "mean",
    valid_bundles = NULL
) {
  out <- data
  if (!is.null(valid_bundles)) {
    out <- out %>% dplyr::filter(.data[[bundle_col]] %in% valid_bundles)
  }
  sel <- c("subject_id", bundle_col, value_col)
  out <- out %>%
    dplyr::filter(.data[[variable_col]] == metric_name) %>%
    dplyr::select(dplyr::all_of(sel)) %>%
    tidyr::pivot_wider(
      id_cols = dplyr::all_of("subject_id"),
      names_from = dplyr::all_of(bundle_col),
      values_from = dplyr::all_of(value_col),
      values_fn = function(x) mean(x, na.rm = TRUE),
      names_prefix = paste0(metric, "_")
    )
  out
}

# ------------------------------------------------------------------------------
### High-level: group tract scalar CSV (FA, MD, ICVF) ####
# ------------------------------------------------------------------------------
# Stacks qsirecon scalarstats TSVs from input_dir, optionally filters to abbrev sheet,
# pivots one scalar row-type, writes {dataset}_tracts_{metric}.csv (or output_filename).
# metric_name must match a value in the TSV's variable_name column (e.g. dti_fa, md, icvf).
# if_missing_variable: "stop" or "skip" if that variable is absent after filtering.
# ------------------------------------------------------------------------------

aggregate_tract_scalars <- function(
    input_dir,
    file_pattern,
    dataset,
    metric,
    metric_name,
    output_dir,
    output_filename = NULL,
    tract_abbrev_xlsx = NULL,
    abbrev_col = "new_qsirecon_tract_names",
    bundle_col = "bundle",
    variable_col = "variable_name",
    value_col = "mean",
    recursive = FALSE,
    if_missing_variable = c("stop", "skip")
) {
  if_missing_variable <- match.arg(if_missing_variable)

  # Check that input_dir exists
  if (!dir.exists(input_dir)) {
    stop("Root directory does not exist. Need to mount the cubic tractmaps dir onto the local machine.")
  }

  # Get list of all TSV files
  paths <- list.files(
    input_dir,
    pattern = file_pattern,
    full.names = TRUE,
    recursive = recursive
  )
  if (length(paths) == 0) {
    stop("No files matched pattern '", file_pattern, "' under ", input_dir)
  }
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # --------------------------------------------------------------------------
  ### Combine all tract scalar data ####
  # --------------------------------------------------------------------------

  cat(sprintf("Found %d TSV files to process\n", length(paths))) # e.g. N = 1406
  cat("Reading and combining tract scalar data...\n")

  # Each TSV: long format with subject_id, bundle, variable_name, mean, ...
  all_data <- purrr::map_dfr(
    paths,
    ~ readr::read_tsv(.x, show_col_types = FALSE),
    .id = "file_id"
  )

  # Filter to valid tracts if abbreviations sheet is provided
  if (!is.null(tract_abbrev_xlsx) && !is.na(tract_abbrev_xlsx) && nzchar(tract_abbrev_xlsx)) {
    all_data <- filter_bundles_by_abbrev_sheet(
      all_data,
      bundle_col = bundle_col,
      tract_abbrev_xlsx = tract_abbrev_xlsx,
      abbrev_col = abbrev_col
    )
  }

  # --------------------------------------------------------------------------
  ### Extract measures for this metric (valid tracts if abbrev was applied) ####
  # --------------------------------------------------------------------------

  available_vars <- unique(all_data[[variable_col]])
  if (!metric_name %in% available_vars) {
    msg <- sprintf(
      "metric_name '%s' not found in data; available: %s",
      metric_name,
      paste(available_vars, collapse = ", ")
    )
    if (if_missing_variable == "stop") {
      stop(msg)
    }
    cat(msg, " — skipping write.\n")
    return(invisible(NULL))
  }

  # Filter to this scalar, select subject_id / bundle / mean, then wide
  wide <- pivot_bundle_scalar_wide(
    all_data,
    metric = metric,
    metric_name = metric_name,
    bundle_col = bundle_col,
    variable_col = variable_col,
    value_col = value_col,
    valid_bundles = NULL
  )

  cat(
    sprintf(
      "%s data: %d subjects, %d tracts\n",
      toupper(metric),
      nrow(wide),
      ncol(wide) - 1L
    )
  ) 

  # --------------------------------------------------------------------------
  ### Save group-level data ####
  # --------------------------------------------------------------------------

  if (is.null(output_filename)) {
    output_filename <- sprintf("%s_tracts_%s.csv", dataset, metric)
  }
  out_path <- file.path(output_dir, output_filename)

  cat(sprintf("Saving %s data...\n", toupper(metric)))
  readr::write_csv(wide, out_path)

  cat("Done! File saved to:", output_dir, "\n")
  invisible(out_path)
}

# ------------------------------------------------------------------------------
### High-level: group QC table ####
# ------------------------------------------------------------------------------
# Stacks per-subject QC files into one subjects x measures table and writes
# {dataset}_qc_measures.csv. PNC: reader = "csv" (per-subject QC from qsiprep).
# HCP-D / HBN: often *_desc-image_qc.tsv (reader = "tsv") — see revisions/3_group_level_qc_measures.R.
# ------------------------------------------------------------------------------

aggregate_qc <- function(
    input_dir,
    file_pattern,
    dataset,
    output_dir,
    reader = c("csv", "tsv"),
    recursive = FALSE,
    output_filename = NULL
) {
  reader <- match.arg(reader)

  if (is.null(input_dir) || is.na(input_dir) || !nzchar(input_dir) || !dir.exists(input_dir)) {
    stop("Root directory does not exist. Need to mount the cubic tractmaps dir onto the local machine.")
  }

  # Get list of all QC files
  paths <- list.files(
    input_dir,
    pattern = file_pattern,
    full.names = TRUE,
    recursive = recursive
  )
  if (length(paths) == 0) {
    stop("No files matched pattern '", file_pattern, "' under ", input_dir)
  }

  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # --------------------------------------------------------------------------
  ### Combine all QC measures data ####
  # --------------------------------------------------------------------------

  kind <- if (identical(reader, "tsv")) "TSV" else "CSV"
  cat(sprintf("Found %d %s files to process\n", length(paths), kind)) # e.g. N = 1406
  cat("Reading and combining QC measures data...\n")

  all_qc <- purrr::map_dfr(
    paths,
    ~ {
      if (identical(reader, "tsv")) {
        readr::read_tsv(.x, show_col_types = FALSE)
      } else {
        readr::read_csv(.x, show_col_types = FALSE)
      }
    },
    .id = "file_id"
  )

  # Check the structure of the data
  cat("QC data structure:\n")
  print(str(all_qc))
  cat("QC column names:\n")
  print(colnames(all_qc))

  # Create subjects x measures format (drop map index, de-duplicate)
  qc_subjects_measures <- all_qc %>%
    dplyr::select(-file_id) %>%
    dplyr::distinct()

  cat(
    sprintf(
      "Final QC data: %d subjects, %d measures\n",
      nrow(qc_subjects_measures),
      ncol(qc_subjects_measures) - 1L
    )
  ) # e.g. 1406 subjects, 60 measures (excluding subject_id)

  # --------------------------------------------------------------------------
  ### Save group-level QC data ####
  # --------------------------------------------------------------------------

  if (is.null(output_filename)) {
    output_filename <- sprintf("%s_qc_measures.csv", dataset)
  }
  out_path <- file.path(output_dir, output_filename)

  cat("Saving QC data...\n")
  readr::write_csv(qc_subjects_measures, out_path)

  cat("Done! QC measures file saved to:", output_dir, "\n")
  invisible(out_path)
}
