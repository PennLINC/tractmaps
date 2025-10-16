# ------------------------------------------------------------------------------------------------
# --- Tract Rewiring Null Generation ---
# ------------------------------------------------------------------------------------------------

# This script generates tract rewiring degree-preserving nulls for tract data. Specifically, 
# the number of regions connected to each tract is preserved, but the regions connected to each  
# tract are permuted within each hemisphere.

# Inputs: tract-to-region connection probabilities data
# Outputs: tract rewiring nulls pickle
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import sys
import pickle
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils import tm_utils

# ------------------------------------------------------------------------------------------------
# --- Set up paths and load data ---
# ------------------------------------------------------------------------------------------------
root = '/Users/joelleba/PennLINC/tractmaps'
data_root = f'{root}/data/derivatives/'
tracts_dir = f'{root}/data/derivatives/tracts'

# Define tract-to-region connection threshold
tract_threshold = 0.5

# Set number of nulls
n_nulls = 10000

# Load tract data
tract_connection_path = f'{tracts_dir}/tracts_probabilities/tracts_probabilities.csv'
tractdata = pd.read_csv(tract_connection_path)
print(f"Loaded tract connection data for {len(tractdata)} regions, {len(tractdata.filter(regex='left|right').columns)} tracts") # count the number of tracts

# Create output directories
nulls_dir = f'{data_root}/nulls/tract_rewiring_nulls'
if not os.path.exists(nulls_dir):
    os.makedirs(nulls_dir)
    print(f"Created output directory: {nulls_dir}")

plots_dir = f'{nulls_dir}/figures'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created figures directory: {plots_dir}")

# ------------------------------------------------------------------------------------------------
# --- Core functions ---
# ------------------------------------------------------------------------------------------------

def generate_tract_rewiring_nulls(tractdata, n_nulls=10000, tract_threshold=0.5, 
                           random_state=None, progress_bar=True, output_dir=None):
    """
    Generate and save tract rewiring degree-preserving nulls for tract data.
    
    This function creates null tract data where the *number* of regions connected to each tract
    remains the same as the original data, but the connected regions are permuted
    within each hemisphere. Left hemisphere tracts only permute among left hemisphere regions,
    and right hemisphere tracts only permute among right hemisphere regions.
    
    Parameters
    ----------
    tractdata : pandas.DataFrame
        DataFrame containing tract data with columns for each tract (ending with _left or _right)
        and rows representing brain regions (0-179 for left, 180-359 for right hemisphere)
    n_nulls : int, optional
        Number of null datasets to generate. Default: 10000
    tract_threshold : float, optional
        Threshold for considering a region connected to a tract. Default: 0.5
    random_state : int or None, optional
        Random state for reproducibility. Default: None
    progress_bar : bool, optional
        Whether to show progress bar. Default: True
    output_dir : str, optional
        Directory to save null datasets. If None, uses default path.
    
    Returns
    -------
    list of pandas.DataFrame
        List of null tract data DataFrames, each with the same structure as input tractdata
        but with permuted region connections while preserving degree (number of connections)
    """
    
    # Set output directory
    if output_dir is None:
        output_dir = f'{data_root}/nulls/tract_rewiring_nulls'
    
    # Check if null datasets already exist
    null_pickle_path = f'{output_dir}/null_tractdata_{n_nulls}nulls.pkl'
    
    if os.path.exists(null_pickle_path):
        print(f"\nLoading existing null datasets from: {null_pickle_path}")
        with open(null_pickle_path, 'rb') as f:
            null_datasets = pickle.load(f)
        print(f"Loaded {len(null_datasets)} null datasets")
        return null_datasets
    
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get tract columns (those ending with _left or _right)
    tract_columns = [col for col in tractdata.columns if col.endswith('_left') or col.endswith('_right')]
    
    # Separate left and right hemisphere regions
    left_regions = np.arange(0, 180)  # Regions 0-179
    right_regions = np.arange(180, 360)  # Regions 180-359
    
    # Initialize list to store null datasets
    null_datasets = []
    
    # Create progress bar if requested
    if progress_bar:
        iterator = tqdm(range(n_nulls), desc="Generating tract rewiring nulls")
    else:
        iterator = range(n_nulls)
    
    # Generate each null dataset
    for null_idx in iterator:
        # Create a copy of the original tractdata
        null_tractdata = tractdata.copy()
        
        # Process each tract
        for tract in tract_columns:
            # Get original connected regions for this tract
            original_connected = np.where(tractdata[tract] >= tract_threshold)[0]
            n_connected = len(original_connected)
            
            if n_connected > 0:
                # Get original probability values for connected regions
                original_probabilities = tractdata.loc[original_connected, tract].values
                
                # Determine hemisphere based on tract name and select appropriate regions
                if tract.endswith('_left'):
                    available_regions = left_regions
                elif tract.endswith('_right'):
                    available_regions = right_regions
                else:
                    # Skip tracts that don't have hemisphere designation
                    continue
                
                # Randomly sample the same number of regions from the appropriate hemisphere
                null_connected = np.random.choice(available_regions, size=n_connected, replace=False)
                
                # Create new tract column with zeros everywhere
                null_tractdata[tract] = 0.0
                
                # Assign the original probability values to the shuffled regions
                null_tractdata.loc[null_connected, tract] = original_probabilities
        
        # Add this null dataset to the list
        null_datasets.append(null_tractdata)
    
    # Save null datasets
    print(f"\nSaving {len(null_datasets)} null datasets...")
    with open(null_pickle_path, 'wb') as f:
        pickle.dump(null_datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved null datasets to: {null_pickle_path}")
    
    return null_datasets

def plot_tract_with_nulls(tract_names, tractdata, null_datasets, null_indices=[0, 1], 
                          tract_threshold=0.5, outpath=None, figsize=None):
    """
    Plot multiple tracts with their nulls on brain surfaces.
    
    Parameters
    ----------
    tract_names : str or list of str
        Name(s) of the tract(s) to plot (e.g., 'IFOF_left' or ['IFOF_left', 'OR_right'])
    tractdata : pandas.DataFrame
        Original tract data
    null_datasets : list of pandas.DataFrame
        List of null tract datasets
    null_indices : list of int, optional
        Indices of null datasets to plot. Default: [0, 1]
    tract_threshold : float, optional
        Threshold for considering a region connected to a tract. Default: 0.5
    outpath : str, optional
        Path to save the output figure. If None, figure is not saved.
    figsize : tuple, optional
        Figure size in inches. If None, automatically calculated based on number of tracts.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the brain surface plots
    """
    
    import matplotlib.image as mpimg
    import tempfile
    import os
    
    # Convert single tract name to list
    if isinstance(tract_names, str):
        tract_names = [tract_names]
    
    # Calculate grid dimensions
    n_tracts = len(tract_names)
    n_nulls = len(null_indices)
    n_plots_per_tract = 1 + n_nulls  # original + nulls
    
    # Calculate grid layout
    n_cols = n_plots_per_tract
    n_rows = n_tracts
    
    # Set default figsize if not provided
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Load glasser labels for plotting on brain surface
    lhlabels = f'{root}/data/derivatives/glasser_parcellation/HCP_MMP_L.label.gii'
    rhlabels = f'{root}/data/derivatives/glasser_parcellation/HCP_MMP_R.label.gii'
    
    # get custom colormaps
    _, tract_cmap, _, _, _, _ = tm_utils.make_colormaps()
    
    # Create temporary directory for individual plots
    temp_dir = tempfile.mkdtemp()
    temp_plot_files = []
    
    try:
        # Generate individual plots for each tract
        for tract_idx, tract_name in enumerate(tract_names):
            # Plot titles for this tract
            titles = ['Original'] + [f'Null {i+1}' for i in null_indices]
            
            # Datasets to plot for this tract
            datasets = [tractdata] + [null_datasets[i] for i in null_indices]
            
            for plot_idx, (dataset, title) in enumerate(zip(datasets, titles)):
                # Create connectivity data for this tract (all 360 regions)
                connectivity_data = np.zeros(len(dataset))
                
                # Get regions connected to this tract
                connected_regions = np.where(dataset[tract_name] >= tract_threshold)[0]
                # Populate with actual probability values instead of binary 1.0
                connectivity_data[connected_regions] = dataset[tract_name].iloc[connected_regions].values
                
                # Save individual plot to temporary file
                temp_plot_file = os.path.join(temp_dir, f'plot_{tract_idx}_{plot_idx}.png')
                temp_plot_files.append((temp_plot_file, tract_idx, plot_idx))
                
                # Plot brain surface and save
                tm_utils.conte69_plot_grid(
                    data=connectivity_data,
                    lhlabel=lhlabels,
                    rhlabel=rhlabels,
                    vmin=np.nanmin(dataset[tract_name]),
                    vmax=np.nanmax(dataset[tract_name]),
                    surf='inflated',
                    customcmap=tract_cmap,
                    shared_colorbar=False,
                    colorbartitle='Probability',
                    connection_threshold=tract_threshold,
                    outpath=temp_plot_file
                )
        
        # Create grid figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        # Load and display plots in grid
        for temp_file, tract_idx, plot_idx in temp_plot_files:
            if os.path.exists(temp_file):
                img = mpimg.imread(temp_file)
                axes[tract_idx, plot_idx].imshow(img)
                axes[tract_idx, plot_idx].axis('off')
                
                # Add column titles for first row
                if tract_idx == 0:
                    if plot_idx == 0:
                        axes[tract_idx, plot_idx].set_title('Original', fontsize=18, pad=5)
                    else:
                        axes[tract_idx, plot_idx].set_title(f'Null {null_indices[plot_idx-1]+1}', fontsize=18, pad=5)
                
                # Add row titles for first column
                if plot_idx == 0:
                    axes[tract_idx, plot_idx].text(-0.05, 0.5, tract_names[tract_idx], 
                                                  fontsize=18, ha='center', va='center', rotation=90,
                                                  transform=axes[tract_idx, plot_idx].transAxes)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save final grid figure if outpath is provided
        if outpath is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            plt.savefig(outpath, bbox_inches='tight', dpi=300)
            print(f"Saved figure to: {outpath}")
        
        plt.show()
        return fig
        
    finally:
        # Clean up temporary files
        for temp_file, _, _ in temp_plot_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        # Remove temporary directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


# ------------------------------------------------------------------------------------------------
# --- Plotting correlation results: empirical vs null distribution ---
# ------------------------------------------------------------------------------------------------
def plot_correlation_null_distribution(empirical_correlation, null_correlations, 
                                     correlation_type='spearman', outpath=None, figsize=(6, 6)):
    """Plot the distribution of null correlations with the empirical value as a vertical line.
    This can be used in the analyses scripts to visualize the results of significance testing using
    tract rewiring nulls."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out NaN values
    valid_nulls = ~np.isnan(null_correlations)
    null_correlations_clean = null_correlations[valid_nulls]
    
    if len(null_correlations_clean) == 0:
        print("No valid null correlations to plot")
        return fig
    
    # Plot null distribution as density curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(null_correlations_clean)
    x_range = np.linspace(null_correlations_clean.min(), null_correlations_clean.max(), 100)
    ax.plot(x_range, kde(x_range), color='lightblue', linewidth=2, label='Null distribution')
    ax.fill_between(x_range, kde(x_range), alpha=0.3, color='lightblue')
    
    # Add empirical value as vertical line
    ax.axvline(empirical_correlation, color='red', linewidth=2, linestyle='--', 
               label='Empirical')
    
    # Calculate p-value using the same logic as the reference implementation
    null_mean = np.nanmean(null_correlations_clean)
    empirical_deviation = abs(empirical_correlation - null_mean)
    null_deviations = abs(null_correlations_clean - null_mean)
    p_value = (1 + np.sum(null_deviations > empirical_deviation)) / (len(null_correlations_clean) + 1)
    
    # Add statistics text in top left
    stats_text += f'Empirical r = {empirical_correlation:.3f}\n'
    stats_text = f'Mean null r = {np.mean(null_correlations_clean):.3f}\n'
    stats_text += f'pval = {p_value:.4f}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel(f'{correlation_type.capitalize()} Correlation', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'S-A Range vs Euclidean Distance\n({len(null_correlations_clean)} nulls)', 
                 fontsize=14)
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Place legend below plot horizontally
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Despine the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout to accommodate legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure if outpath is provided
    if outpath is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        print(f"Saved figure to: {outpath}")
    
    plt.show()
    return fig


# ------------------------------------------------------------------------------------------------
# --- Main execution ---
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate and save nulls
    null_datasets = generate_tract_rewiring_nulls(tractdata, n_nulls=n_nulls, tract_threshold=tract_threshold, random_state=42)
    
    # Example: Plot tracts with their nulls
    print("\nGenerating example plot...")
    example_tracts = ['IFOF_left', 'OR_right', 'AF_left']
    if all(tract in tractdata.columns for tract in example_tracts):
        plot_tract_with_nulls(
            tract_names=example_tracts,
            tractdata=tractdata,
            null_datasets=null_datasets,
            null_indices=[0, 1], # plot the first two nulls (and the original is automatically plotted)
            tract_threshold=tract_threshold,
            outpath=f'{plots_dir}/example_tracts_with_nulls.png'
        )
    else:
        print(f"Some tracts not found in data. Available tracts: {list(tractdata.filter(regex='left|right').columns[:5])}")

    print("Tract rewiring null generation complete!")
 