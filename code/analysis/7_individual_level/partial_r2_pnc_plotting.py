# ------------------------------------------------------------------------------------------------
#### Plot Individual Level Partial R² Results ####
# ------------------------------------------------------------------------------------------------
# Script to load and plot partial R² results from individual-level GAM analyses.
# Creates correlation plots showing the relationship between tract properties 
# (Gini coefficient and S-A range) and brain-behavior associations (partial R²).
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils import tm_utils
from statsmodels.stats.multitest import fdrcorrection
plt.switch_backend('Agg')

# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# root directory
root = "/Users/joelleba/PennLINC/tractmaps"

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Create threshold suffix for filenames
thresh_suffix = f'_thresh{int(tract_threshold * 100)}'

# create results directory if it doesn't exist
results_dir = f"{root}/results/individual_level/figures"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Custom colormaps are now handled by tm_utils.plot_correlation

# set fontsize for all plots
plt.rcParams.update({'font.size': 18})

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load tract abbreviations for display names
tract_names = pd.read_excel(f'{root}/data/raw/tract_names/abbreviations.xlsx')

# Load tract gini term scores
gini_scores = pd.read_csv(f"{root}/results/tract_functional_diversity/gini_coefficients/tract_gini_term_scores.csv")
gini_scores = gini_scores.drop(columns=['Hemisphere'])

# Load tract SA range
sa_ranges = pd.read_csv(f'{root}/data/derivatives/tracts/tracts_sa_axis/tract_sa_axis_ranges{thresh_suffix}.csv')
sa_ranges = sa_ranges.rename(columns={'Tract': 'Tract_Short_Name'})

# store gini scores and sa ranges in a dictionary
tract_properties = {
    'Gini_Coefficient': gini_scores,
    'SA_Range': sa_ranges
}

# Find all result files
partial_r2_files = list(glob.glob(f"{root}/results/individual_level/*partialR2_stats.csv"))

# ------------------------------------------------------------------------------------------------
# ---- Load results ----
# ------------------------------------------------------------------------------------------------

# Dictionary to store GAM measures
gam_results = {}

for file_path in partial_r2_files:
    # Extract filename
    filename = Path(file_path).stem
    brain_measure = "FA"
    
    # Extract measure name
    if "age" in filename:
        measure_name = "Age"
    elif "F3_Executive_Efficiency" in filename:
        measure_name = "Executive efficiency"
    else:
        measure_name = "Unknown"
    
    # Load the csv
    df = pd.read_csv(file_path, index_col=0)

    # rename columns to match tract names
    df.index = df.index.str.replace('fa_', '')

    # Convert index to column for merging
    df = df.reset_index()
    df = df.rename(columns={'gam.variable.tract.tract': 'new_qsirecon_tract_names'})
    
    # Merge with tract_types to get long and short names (needed for merging with tract properties later)
    df = pd.merge(df, 
                 tract_names[['new_qsirecon_tract_names', 'Tract_Long_Name', 'Tract']], 
                 on='new_qsirecon_tract_names',
                 how='left')
    
    # Set Tract_Long_Name as index and add Tract_Short_Name
    df = (df.rename(columns={'Tract': 'Tract_Short_Name', 'Tract_Long_Name': 'Tract'})
          .set_index('Tract')
          .drop('new_qsirecon_tract_names', axis=1))

    # Store DataFrame directly 
    gam_results[measure_name] = df
    
    print(f"Loaded {measure_name} ({brain_measure}): {df.shape}")
    
# ------------------------------------------------------------------------------------------------
# --- Significance testing with FDR correction ---
# ------------------------------------------------------------------------------------------------

def significance_testing(gam_results, tract_properties, output_dir, significance_threshold=0.05):
    """Run permutation tests for all measure-metric combinations and apply FDR correction.
    
    Parameters
    ----------
    gam_results : dict
        Dictionary containing FA partial R² results organized by measure
    tract_properties : dict
        Dictionary containing DataFrames with tract metrics (Gini coefficient and S-A range data)
    output_dir : str
        Directory to save results CSV
    significance_threshold : float, optional
        Significance threshold for partial R² results
    
    Returns
    -------
    str
        Path to saved CSV file with results
    """
    
    print("\nRunning significance testing for all measure-metric combinations...")
    
    # Collect all correlation results
    all_results = []
    
    # Loop through each measure and each tract property
    for measure_name, measure_df in gam_results.items():
        for property_name, property_df in tract_properties.items():
            
            print(f"Testing {measure_name} vs {property_name}...")
            
            # Prepare data for this analysis
            df_plot = measure_df.copy()
            
            # Add metric values to df_plot using merge (note: the partial R2 results may contain fewer tracts than the tract metrics)
            df_plot = pd.merge(df_plot, property_df[['Tract_Short_Name', property_name]], 
                              on='Tract_Short_Name', how='left')
            
            # Filter out tracts without metric values
            df_plot = df_plot[df_plot[property_name].notna()].copy() # sanity check; this shouldn't actually remove any tracts
            
            if len(df_plot) > 0:
                # Prepare data for correlation
                x = df_plot[property_name].values
                y = df_plot['partialR2'].values
                
                # Run permutation test
                try:
                    corr_result = tm_utils.perm_corr_test(x, y, n_permutations=10000, 
                                                        method='spearman', alternative='two-sided', random_state=42)
                    r_value = corr_result['observed_corr']
                    p_value = corr_result['p_value']
                    
                except Exception as e:
                    print(f"Error in permutation test for {measure_name} vs {property_name}: {e}")
                    r_value = np.nan
                    p_value = np.nan
            else:
                r_value = np.nan
                p_value = np.nan
            
            # Store result
            result = {
                'GAM_Measure': measure_name,
                'Tract_Property': property_name,
                'Correlation': r_value,
                'P_Value': p_value
            }
            all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Apply FDR correction to p-values (only on non-NaN values)
    valid_p_mask = ~results_df['P_Value'].isna()
    if valid_p_mask.sum() > 0:
        fdr_reject, fdr_pvals = fdrcorrection(results_df.loc[valid_p_mask, 'P_Value'], alpha=0.05, method='indep')
        
        # Add FDR corrected p-values to results
        results_df['P_Value_FDR'] = np.nan
        results_df.loc[valid_p_mask, 'P_Value_FDR'] = fdr_pvals
        
        # Add FDR significance
        results_df['Significant_FDR'] = False
        results_df.loc[valid_p_mask, 'Significant_FDR'] = fdr_reject
        
        print(f"FDR correction applied to {valid_p_mask.sum()} tests")
        print(f"Significant after FDR correction: {fdr_reject.sum()}")
    else:
        results_df['P_Value_FDR'] = np.nan
        results_df['Significant_FDR'] = False
        print("No valid p-values for FDR correction")
    
    # Save results to CSV
    results_csv_path = f"{output_dir}/correlation_results_with_fdr.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Correlation results saved to: {results_csv_path}")
    
    return results_csv_path

# ------------------------------------------------------------------------------------------------
# --- Print Partial R² Results ---
# ------------------------------------------------------------------------------------------------
print("Summary of Partial R² Results:\n")

# Print results for Age and Executive Efficiency
for measure_name in ['Age', 'Executive Efficiency']:
    if measure_name in gam_results:
        df = gam_results[measure_name]
        
        # Get number of significant tracts based on FDR-corrected p-values
        significant_tracts = df[df['anovaPvaluefdr'] < 0.05]
        n_significant = len(significant_tracts)
        n_total = len(df)
        pct_significant = (n_significant / n_total) * 100
        
        # Get range of partial R² in significant tracts
        if n_significant > 0:
            min_r2 = significant_tracts['partialR2'].min()
            max_r2 = significant_tracts['partialR2'].max()
            r2_range_str = f"[{min_r2:.3f}, {max_r2:.3f}]"
            
            # Get percentage of positive partial R² values among significant tracts (to know how many tracts have positive associations with age or executive efficiency)
            n_positive = (significant_tracts['partialR2'] > 0).sum()
            pct_positive = (n_positive / n_significant) * 100
        else:
            r2_range_str = "N/A (no significant tracts)"
            pct_positive = np.nan
        
        print(f"\n{measure_name}:")
        print(f"  Total tracts: {n_total}")
        print(f"  Significant tracts (FDR < 0.05): {n_significant} ({pct_significant:.1f}%)")
        print(f"  Partial R² range (significant tracts): {r2_range_str}")
        if n_significant > 0:
            print(f"  Positive partial R² (significant tracts): {n_positive}/{n_significant} ({pct_positive:.1f}%)")
    else:
        print(f"\n{measure_name}: No data available")


# ------------------------------------------------------------------------------------------------
# --- Plot correlations between partial R² and tract metric values ---
# ------------------------------------------------------------------------------------------------

### Run significance testing with FDR correction ###
results_csv_path = significance_testing(gam_results, tract_properties, results_dir, significance_threshold = 0.05)

### Create individual correlation plots ###
print("\nCreating correlation plots...")

# Load correlation results once for all plots
correlation_results = pd.read_csv(results_csv_path)

# Create individual plots for each measure-property combination
for measure_name, measure_df in gam_results.items():
    for property_name, property_df in tract_properties.items():
        
        print(f"Plotting {measure_name} vs {property_name}...")
        
        # Prepare data for this analysis
        df_plot = measure_df.copy()

        # Add metric values to df_plot using merge
        df_plot = pd.merge(df_plot, property_df[['Tract_Short_Name', property_name]], 
                          on='Tract_Short_Name', how='left')
        
        # Filter out tracts without metric values
        df_plot = df_plot[df_plot[property_name].notna()].copy()
        
        if len(df_plot) > 0:
            # Prepare data
            x = df_plot[property_name].values
            y = df_plot['partialR2'].values
            tract_abbrevs = df_plot['Tract_Short_Name'].values
            
            # Get correlation results for this measure-property combination
            result_row = correlation_results[
                (correlation_results['GAM_Measure'] == measure_name) & 
                (correlation_results['Tract_Property'] == property_name)
            ]
            
            if len(result_row) > 0:
                r_value = result_row['Correlation'].iloc[0]
                p_fdr = result_row['P_Value_FDR'].iloc[0]
            else:
                r_value = np.nan
                p_fdr = np.nan
            
            # Determine settings based on property
            reverse_cmap = property_name == 'SA_Range' # reverse colormap for S-A range so that yellow is low values
            text_position = 'top_left' if property_name == 'SA_Range' else 'top_right' # top left for S-A range, top right for Gini
            if property_name == 'SA_Range':
                axis_label = f'S-A range'
                colorbar_tick_interval = 25
            else:
                axis_label = f'Gini coefficient'
                colorbar_tick_interval = 0.1
            
            # Clean filename components
            clean_measure = measure_name.lower().replace(' ', '_').replace('.', '')
            clean_property = property_name.lower().replace('-', '_').replace(' ', '_')
            
            # Create output paths
            output_filename = f"{clean_measure}_vs_{clean_property}.svg"
            output_path = f"{results_dir}/{output_filename}"
            
            # Create colorbar filename based on the variable used for coloring (tract property)
            colorbar_filename = f"{clean_property}_colorbar.svg"
            
            # Get significance data for gray coloring of non-significant tracts
            significance_pvals = df_plot['anovaPvaluefdr'].values
            
            # Create correlation plot for this measure-property combination
            tm_utils.plot_correlation(
                x=x, 
                y=y,
                corr_value=r_value, 
                p_value=p_fdr,
                x_label=axis_label,
                y_label=f'{measure_name} partial R²',
                reverse_colormap=reverse_cmap,
                colorbar='separate_figure',  # Creates both plot and colorbar in separate figures
                colorbar_label=axis_label,
                color_by='x',  # Color points by tract properties (x-values)
                colorbar_filename=colorbar_filename, 
                significance_data=significance_pvals,  # P-values for significance-based coloring of tract data points (partial R²),
                point_size=30,
                point_alpha=0.8,
                significance_threshold=0.05, 
                point_labels=tract_abbrevs,
                text_box_position=text_position,
                output_path=output_path,
                colorbar_tick_interval=colorbar_tick_interval,
                dpi=300,
                figure_size_mm=(70, 60)
            )
        else:
            print(f"No valid data points for {measure_name} vs {property_name}")

print(f"Plots saved to: {results_dir}")