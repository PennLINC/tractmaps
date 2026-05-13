# ------------------------------------------------------------------------------------------------
# Age GAM partial R² vs tract-level functional properties (Gini, S–A range)
# ------------------------------------------------------------------------------------------------
# For each DTI metric (FA, MD, ICVF), loads age-effect partial R² tables produced by the per-
# dataset GAM pipelines (e.g. *_*_age_partialR2_stats.csv under results/individual_level/<dataset>/).
# It then tests Spearman associations (permutation testing) between tract partial R² effects and
# each tract property (Gini coefficient; sensorimotor–association axis range), combining all
# cohorts into one correlation table.
#
# Multiple testing: Benjamini–Hochberg FDR is applied for each DWI metric table, 
# across all datasets and tract properties.
# Datasets included here:
#   - PNC: FA, MD
#   - HBN: FA, MD, ICVF
#
# Outputs (under results/individual_level/age_effects/ relative to TRACTMAPS_ROOT):
#   - One CSV per metric: {fa,md,icvf}_tract_property_correlations.csv (dataset / dti_metric /
#     effect columns plus tract-property correlations and FDR)
#   - Scatter plots per cohort × property: figures/<metric>_age/*.svg
#
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# Load packages and set paths
# ------------------------------------------------------------------------------------------------
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

_INDIVIDUAL_LEVEL = Path(__file__).resolve().parent
if str(_INDIVIDUAL_LEVEL) not in sys.path:
    sys.path.insert(0, str(_INDIVIDUAL_LEVEL))

from test_partial_r2_tract_properties import (
    load_tract_properties,
    run_metric_tract_property_corr,
)

TRACTMAPS_ROOT = Path("/Users/joelleba/PennLINC/tractmaps")
INDIVIDUAL = TRACTMAPS_ROOT / "results" / "individual_level"
OUT_AGE = INDIVIDUAL / "age_effects"

# ------------------------------------------------------------------------------------------------
# Define datasets and metric runs
# ------------------------------------------------------------------------------------------------
# (dataset, metric, individual_dataset_dir)
AGE_RUNS = [
    ("pnc", "FA", INDIVIDUAL / "pnc"),
    ("pnc", "MD", INDIVIDUAL / "pnc"),
    ("hbn", "FA", INDIVIDUAL / "hbn"),
    ("hbn", "MD", INDIVIDUAL / "hbn"),
    ("hbn", "ICVF", INDIVIDUAL / "hbn"),
]

# ------------------------------------------------------------------------------------------------
# Print partial R² results and test associations between partial R² effects and tract properties
# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    OUT_AGE.mkdir(parents=True, exist_ok=True)
    tract_properties, tract_names = load_tract_properties(TRACTMAPS_ROOT)

    by_metric: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for dataset, metric, ind_dir in AGE_RUNS:
        by_metric[metric.upper()].append((dataset, metric, Path(ind_dir)))

    for metric in sorted(by_metric.keys()):
        runs = by_metric[metric]
        csv_out = OUT_AGE / f"{metric.lower()}_tract_property_correlations.csv"
        fig_dir = OUT_AGE / "figures" / f"{metric.lower()}_age"
        print(f"\n=== Age effects | metric {metric} | {len(runs)} dataset(s) ===")
        run_metric_tract_property_corr(
            runs=runs,
            gam_kind="age",
            tract_properties=tract_properties,
            tract_names=tract_names,
            output_csv=csv_out,
            figures_dir=fig_dir,
            cognition_var_by_dataset=None,
        )

    print(f"\nAll age-effect tables under: {OUT_AGE}")
