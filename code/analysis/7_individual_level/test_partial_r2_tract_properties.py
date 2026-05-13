# ------------------------------------------------------------------------------------------------------------
#### Test and Plot Individual Level Partial R² Results vs Tract Properties (S-A range, Gini coefficient) ####
# ------------------------------------------------------------------------------------------------------------
# Script to load and plot partial R² results from individual-level GAM analyses.
# Creates correlation plots showing the relationship between tract properties 
# (Gini coefficient and S-A range) and brain-behavior associations (partial R²).
# Use across datasets (PNC, HCPYA, HBN), metrics (FA, MD, ICVF), and gam kinds (age, cognition).
# Called in: run_partial_r2_age_effects.py, run_partial_r2_cognition_effects.py.
# ------------------------------------------------------------------------------------------------------------

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
plt.switch_backend("Agg")
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils import tm_utils 

GamKind = Literal["age", "cognition"]

# Separator in GAM measure labels (plots / CSV split columns / y-axis wording)
_GAM_LABEL_SEP = " \u00b7 "

# ------------------------------------------------------------------------------------------------
# Functions to load tract properties and partial R² stats
# ------------------------------------------------------------------------------------------------

def load_tract_properties(
    tractmaps_root: str | Path,
    tract_threshold: float = 0.5,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load tract abbreviation table and tract-property tables (Gini, S–A range).

    Returns
    -------
    tract_properties : dict
        Keys 'Gini_Coefficient' and 'SA_Range'; values are DataFrames with
        Tract_Short_Name and a column matching the key name (values for plots / tests).
    tract_names : DataFrame
        Abbreviations / merge keys (includes new_qsirecon_tract_names, Tract, Tract_Long_Name).
    """
    root = Path(tractmaps_root)
    thresh_suffix = f"_thresh{int(tract_threshold * 100)}"

    tract_names = pd.read_excel(
        root / "data" / "derivatives" / "tract_names" / "abbreviations.xlsx"
    )

    gini_scores = pd.read_csv(
        root
        / "results"
        / "tract_functional_diversity"
        / "gini_coefficients"
        / "tract_gini_term_scores.csv"
    )
    if "Hemisphere" in gini_scores.columns:
        gini_scores = gini_scores.drop(columns=["Hemisphere"])
    if "Tract_Short_Name" not in gini_scores.columns and "Tract" in gini_scores.columns:
        gini_scores = gini_scores.rename(columns={"Tract": "Tract_Short_Name"})

    sa_ranges = pd.read_csv(
        root
        / "data"
        / "derivatives"
        / "tracts"
        / "tracts_sa_axis"
        / f"tract_sa_axis_ranges{thresh_suffix}.csv"
    )
    sa_ranges = sa_ranges.rename(columns={"Tract": "Tract_Short_Name"})

    tract_properties: Dict[str, pd.DataFrame] = {
        "Gini_Coefficient": gini_scores,
        "SA_Range": sa_ranges,
    }
    return tract_properties, tract_names


def build_partial_r2_stats_path(
    individual_dataset_dir: str | Path,
    dataset: str,
    metric: str,
    gam_kind: GamKind,
    cognition_var: Optional[str] = None,
) -> Path:
    """
    Path to run_gam() output CSV under results/individual_level/{dataset}/.

    Parameters
    ----------
    individual_dataset_dir : Path-like
        e.g. {tractmaps_root}/results/individual_level/pnc
    dataset : str
        Lowercase slug, e.g. 'pnc', 'hbn', 'hcpya'
    metric : str
        'FA', 'MD', or 'ICVF' (case-insensitive in filename)
    gam_kind : 'age' | 'cognition'
    cognition_var : str, optional
        Required if gam_kind == 'cognition' (column name token in filename, e.g. F3_Executive_Efficiency).
    """
    ds = dataset.strip().lower()
    m = metric.strip().lower()
    base = Path(individual_dataset_dir)
    if gam_kind == "age":
        return base / f"{ds}_final_sample_{m}_age_partialR2_stats.csv"
    if gam_kind != "cognition":
        raise ValueError(f"gam_kind must be 'age' or 'cognition', got {gam_kind!r}")
    if not cognition_var or not str(cognition_var).strip():
        raise ValueError("cognition_var is required when gam_kind == 'cognition'")
    cog = str(cognition_var).strip()
    return base / f"{ds}_final_cognition_sample_{m}_{cog}_partialR2_stats.csv"


def load_partial_r2_stats(
    path: str | Path,
    metric: str,
    tract_names: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load one partial R² stats CSV and align tracts to abbreviations (Tract_Short_Name, etc.).

    Expects columns partialR2, anovaPvaluefdr, gamPvaluefdr; first column is tract identifier.
    Strips a leading '{metric}_' prefix from tract names when present (qsirecon-style bundles).
    """
    path = Path(path)
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected CSV layout: {path}")
    first = df.columns[0]
    df = df.rename(columns={first: "tract_raw"})
    m = metric.lower().strip()
    pref = f"{m}_"

    def _strip_metric(s: object) -> str:
        s = str(s).strip()
        return s[len(pref) :] if s.startswith(pref) else s

    df["new_qsirecon_tract_names"] = df["tract_raw"].map(_strip_metric)
    out = pd.merge(
        df,
        tract_names[["new_qsirecon_tract_names", "Tract_Long_Name", "Tract"]],
        on="new_qsirecon_tract_names",
        how="left",
    )
    out = (
        out.rename(columns={"Tract": "Tract_Short_Name", "Tract_Long_Name": "Tract"})
        .set_index("Tract")
        .drop(columns=[c for c in ["tract_raw", "new_qsirecon_tract_names"] if c in out.columns])
    )
    needed = ["Tract_Short_Name", "partialR2", "anovaPvaluefdr", "gamPvaluefdr"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} after merge for {path}. Have: {list(out.columns)}")
    return out[needed].copy()

# ------------------------------------------------------------------------------------------------
# Functions to print partial R² results and compute tract-property correlations
# ------------------------------------------------------------------------------------------------

def print_partial_r2_summary(
    df: pd.DataFrame,
    title: str,
    fdr_alpha: float = 0.05,
) -> None:
    """Print partialR² range (all tracts) and count of tracts significant by Anova FDR."""
    n = len(df)
    y = pd.to_numeric(df["partialR2"], errors="coerce")
    p = pd.to_numeric(df["anovaPvaluefdr"], errors="coerce")
    sig = p < fdr_alpha
    n_sig = int(sig.sum())
    print(f"\n{title}")
    print(f"  Tracts (rows): {n}")
    if n == 0:
        return
    print(f"  partialR² range (all tracts): [{y.min():.4f}, {y.max():.4f}]")
    print(
        f"  Significant tracts (Anova FDR < {fdr_alpha}): {n_sig} / {n} ({100.0 * n_sig / n:.1f}%)"
    )
    if n_sig > 0:
        ys = y[sig]
        print(f"  partialR² range (significant only): [{ys.min():.3f}, {ys.max():.3f}]")
        n_pos = int((ys > 0).sum())
        print(f"  Positive partialR² among significant: {n_pos} / {n_sig} ({100.0 * n_pos / n_sig:.1f}%)")


def compute_tract_property_correlations(
    gam_results: Dict[str, pd.DataFrame],
    tract_properties: Dict[str, pd.DataFrame],
    n_permutations: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation Spearman correlations between partial R² and each tract property.
    Outputs a DataFrame with one row per (GAM_Measure, Tract_Property).

    """
    print("\nComputing tract-property vs partial R² correlations...")
    all_results: List[dict] = []

    for measure_name, measure_df in gam_results.items():
        for property_name, property_df in tract_properties.items():
            print(f"  {measure_name} vs {property_name}...")

            # Prepare data for this analysis
            df_plot = measure_df.copy()
            cols = ["Tract_Short_Name", property_name]
            
            # Check if the property table is missing any required columns
            missing_c = [c for c in cols if c not in property_df.columns]
            if missing_c:
                raise KeyError(f"{property_name} table missing columns {missing_c}")

            # Add metric values to df_plot using merge (note: the partial R2 results may contain fewer tracts than the tract metrics)
            df_plot = pd.merge(
                df_plot,
                property_df[cols],
                on="Tract_Short_Name",
                how="left",
            )
            # Filter out tracts without metric values
            df_plot = df_plot[df_plot[property_name].notna()].copy() # sanity check; this shouldn't actually remove any tracts

            r_value = np.nan
            p_value = np.nan
            if len(df_plot) > 0:
                # Prepare data for correlation
                x = df_plot[property_name].values
                y = df_plot["partialR2"].values.astype(float)

                # Run permutation test
                try:
                    corr_result = tm_utils.perm_corr_test(
                        x,
                        y,
                        n_permutations=n_permutations,
                        method="spearman",
                        alternative="two-sided",
                        random_state=random_state,
                    )
                    r_value = corr_result["observed_corr"]
                    p_value = corr_result["p_value"]

                # Catch any errors in permutation test
                except Exception as exc: 
                    print(f"    Error in permutation test: {exc}")

            # Store result
            all_results.append(
                {
                    "GAM_Measure": measure_name,
                    "Tract_Property": property_name,
                    "Correlation": r_value,
                    "P_Value": p_value,
                }
            )

    return pd.DataFrame(all_results)


def apply_fdr_to_correlation_table(
    results_df: pd.DataFrame,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Add BH-FDR columns over all rows in this table."""
    out = results_df.copy()
    valid_p_mask = ~out["P_Value"].isna()

    # Apply FDR correction to p-values
    if valid_p_mask.sum() > 0:
        fdr_reject, fdr_pvals = fdrcorrection(
            out.loc[valid_p_mask, "P_Value"],
            alpha=fdr_alpha,
            method="indep",
        )

        # Add FDR corrected p-values to results
        out["P_Value_FDR"] = np.nan
        out.loc[valid_p_mask, "P_Value_FDR"] = fdr_pvals

        # Add FDR significance
        out["Significant_FDR"] = False
        out.loc[valid_p_mask, "Significant_FDR"] = fdr_reject
        print(
            f"  FDR (alpha={fdr_alpha}) over {valid_p_mask.sum()} test(s): "
            f"{int(fdr_reject.sum())} significant"
        )
    else:
        # If no valid p-values, set FDR columns to NaN and print a message
        out["P_Value_FDR"] = np.nan
        out["Significant_FDR"] = False
        print("  No valid p-values for FDR correction")
    return out


def _load_gam_results_for_runs(
    runs: List[Tuple[str, str, Path]],
    gam_kind: GamKind,
    tract_names: pd.DataFrame,
    cognition_var_by_dataset: Optional[Dict[str, str]],
    summary_fdr_alpha: float,
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Validate one DTI metric across runs; load each cohort CSV.

    Returns
    -------
    metric_lower
        Normalized metric string, e.g. 'fa'.
    gam_results
        Keys ``DATA · METRIC · Age`` or ``… · cognition_col``; values from load_partial_r2_stats.
    """
    if not runs:
        raise ValueError("runs must be non-empty")
    metrics = {str(m).strip().lower() for _, m, _ in runs}
    if len(metrics) != 1:
        raise ValueError(
            "All runs must use the same DTI metric for one combined CSV; "
            f"got metrics: {sorted(metrics)}"
        )
    metric = next(iter(metrics))

    if gam_kind == "cognition":
        if not cognition_var_by_dataset:
            raise ValueError("cognition_var_by_dataset is required when gam_kind == 'cognition'")
        cog_lookup = {str(k).strip().lower(): str(v).strip() for k, v in cognition_var_by_dataset.items()}
    else:
        cog_lookup = {}

    gam_results: Dict[str, pd.DataFrame] = {}
    for dataset, m, ind_dir in runs:
        if str(m).strip().lower() != metric:
            raise ValueError(f"Inconsistent metric in runs: {m!r} vs {metric!r}")
        ind_dir = Path(ind_dir)
        cog: Optional[str] = None
        if gam_kind == "cognition":
            ds_key = str(dataset).strip().lower()
            if ds_key not in cog_lookup:
                raise KeyError(f"No cognition_var for dataset {dataset!r} in cognition_var_by_dataset")
            cog = cog_lookup[ds_key]

        path = build_partial_r2_stats_path(ind_dir, dataset, m, gam_kind, cog)
        dsu = dataset.strip().upper()
        mu = m.strip().upper()
        if gam_kind == "age":
            label = f"{dsu}{_GAM_LABEL_SEP}{mu}{_GAM_LABEL_SEP}Age"
        else:
            label = f"{dsu}{_GAM_LABEL_SEP}{mu}{_GAM_LABEL_SEP}{(cog or '').strip()}"
        if not path.is_file():
            raise FileNotFoundError(f"Partial R² file not found: {path}")
        df = load_partial_r2_stats(path, m, tract_names)
        gam_results[label] = df
        print(f"Loaded {label}: {df.shape} from {path.name}")
        print_partial_r2_summary(df, label, fdr_alpha=summary_fdr_alpha)

    return metric, gam_results


def run_metric_tract_property_corr(
    runs: List[Tuple[str, str, Path]],
    gam_kind: GamKind,
    tract_properties: Dict[str, pd.DataFrame],
    tract_names: pd.DataFrame,
    output_csv: str | Path,
    figures_dir: str | Path,
    cognition_var_by_dataset: Optional[Dict[str, str]] = None,
    n_permutations: int = 10000,
    perm_fdr_alpha: float = 0.05,
    summary_fdr_alpha: float = 0.05,
    run_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Builds a correlation table for one DWI metric across multiple datasets.
    Applies BH-FDR across all dataset × tract-property tests in that table. 

    Parameters
    ----------
    runs
        List of (dataset, metric, individual_dataset_dir); every ``metric`` must match
        (case-insensitive), e.g. all ``"FA"`` for one combined FA table.
    cognition_var_by_dataset
        Required when ``gam_kind == "cognition"``: maps dataset slug (any case) to the
        cognition column name used in ``run_gam`` output filenames (e.g. PNC vs HCP-YA).
    """
    # Load partial R² stats for each dataset
    metric, gam_results = _load_gam_results_for_runs(
        runs, gam_kind, tract_names, cognition_var_by_dataset, summary_fdr_alpha
    )

    # Compute tract-property correlations
    results_df = compute_tract_property_correlations(
        gam_results, tract_properties, n_permutations=n_permutations
    )

    # Apply BH-FDR across all dataset × tract-property tests in that table
    results_df = apply_fdr_to_correlation_table(results_df, fdr_alpha=perm_fdr_alpha)
    # Split GAM_Measure into dataset, dti_metric, effect
    parts = results_df["GAM_Measure"].str.split(_GAM_LABEL_SEP, n=2, expand=True)
    if parts.shape[1] >= 3:
        results_df.insert(0, "dataset", parts[0])
        results_df.insert(1, "dti_metric", parts[1])
        results_df.insert(2, "effect", parts[2])

    # Save CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # CSV: dataset / dti_metric / effect replace the redundant GAM_Measure string.
    results_df.drop(columns=["GAM_Measure"], errors="ignore").to_csv(output_csv, index=False)
    print(f"Saved combined correlation table ({metric.upper()}): {output_csv}")

    # Plot scatter plots
    if run_plots:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_partial_r2_vs_tract_properties(
            gam_results, tract_properties, results_df, figures_dir
        )

    return results_df, gam_results


# ------------------------------------------------------------------------------------------------
# Function to plot partial R² vs tract properties
# ------------------------------------------------------------------------------------------------

def plot_partial_r2_vs_tract_properties(
    gam_results: Dict[str, pd.DataFrame],
    tract_properties: Dict[str, pd.DataFrame],
    correlation_results: pd.DataFrame,
    figures_dir: str | Path,
    font_size: int = 18,
) -> None:
    """Plot correlations between partial R² and tract properties. Creates scatter plots: tract property (x) vs partial R² (y), colored by x; one file per measure×property."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": font_size})

    print("\nCreating correlation plots...")

    # Plot scatter plots for each measure-property combination
    for measure_name, measure_df in gam_results.items():
        for property_name, property_df in tract_properties.items():
            # Prepare data for this analysis (partial R2 and tract property values)
            print(f"  Plotting {measure_name} vs {property_name}...")
            df_plot = measure_df.copy()
            df_plot = pd.merge(
                df_plot,
                property_df[["Tract_Short_Name", property_name]],
                on="Tract_Short_Name",
                how="left",
            )
            df_plot = df_plot[df_plot[property_name].notna()].copy()

            if len(df_plot) == 0:
                print(f"    No valid data points for {measure_name} vs {property_name}")
                continue

            # Input data for plot
            x = df_plot[property_name].values
            y = df_plot["partialR2"].values.astype(float)
            tract_abbrevs = df_plot["Tract_Short_Name"].values

            # Get correlation results for this measure-property combination
            result_row = correlation_results[
                (correlation_results["GAM_Measure"] == measure_name)
                & (correlation_results["Tract_Property"] == property_name)
            ]
            if len(result_row) > 0:
                r_value = result_row["Correlation"].iloc[0]
                p_fdr = result_row["P_Value_FDR"].iloc[0]
            else:
                r_value = np.nan
                p_fdr = np.nan

            # Determine settings based on property
            reverse_cmap = property_name == "SA_Range" # reverse colormap for S-A range so that yellow is low values
            text_position = "top_left" if property_name == "SA_Range" else "top_right" # top left for S-A range, top right for Gini
            if property_name == "SA_Range":
                axis_label = "S-A range"
                colorbar_tick_interval = 25
            else:
                axis_label = "Gini coefficient"
                colorbar_tick_interval = 0.1

            # Clean measure name for filename
            clean_measure = measure_name.lower()
            for ch in ("\u00b7", " ", ".", "|", "/", "\\", ":", "(", ")"):
                clean_measure = clean_measure.replace(ch, "_")
            while "__" in clean_measure:
                clean_measure = clean_measure.replace("__", "_")
            clean_measure = clean_measure.strip("_")
            clean_property = property_name.lower().replace("-", "_").replace(" ", "_")
            
            # Create output filename and path
            output_filename = f"{clean_measure}_vs_{clean_property}.svg"
            output_path = figures_dir / output_filename

            # Create colorbar filename based on the variable used for coloring (tract property)
            colorbar_filename = f"{clean_property}_colorbar.svg"

            # Get significance data for gray coloring of non-significant tracts
            significance_pvals = df_plot["anovaPvaluefdr"].values.astype(float)

            # Set y-axis label based on effect
            if _GAM_LABEL_SEP in measure_name:
                effect = measure_name.split(_GAM_LABEL_SEP)[-1].strip()
                y_label = f"{effect} partial R²" if effect == "Age" else "Cognition partial R²"
            else:
                y_label = f"{measure_name} partial R²"

            tm_utils.plot_correlation(
                x=x,
                y=y,
                corr_value=r_value,
                p_value=p_fdr,
                x_label=axis_label,
                y_label=y_label,
                reverse_colormap=reverse_cmap,
                colorbar="separate_figure", # Creates both plot and colorbar in separate figures
                colorbar_label=axis_label,
                color_by="x", # Color points by tract properties (x-values)
                colorbar_filename=colorbar_filename,
                significance_data=significance_pvals, # P-values for significance-based coloring of tract data points (partial R²),
                point_size=30,
                point_alpha=0.8,
                significance_threshold=0.05,
                point_labels=tract_abbrevs,
                text_box_position=text_position,
                output_path=str(output_path),
                colorbar_tick_interval=colorbar_tick_interval,
                dpi=300,
                figure_size_mm=(70, 60),
            )

    print(f"Plots saved under: {figures_dir}")
