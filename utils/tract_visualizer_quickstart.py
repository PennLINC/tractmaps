#!/usr/bin/env python3
"""
Tract Visualizer Quick Start Guide
===================================

These are some examples illustrating how to use TractVisualizer!

"""

# ------------------------------------------------------------
# Import packages
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from utils.tract_visualizer import TractVisualizer
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Initialize TractVisualizer
# ------------------------------------------------------------
root_dir = '/Users/joelleba/PennLINC/tractmaps'
viz = TractVisualizer(root_dir=root_dir, # always required to specify
                        # optional, these are the defaults
                        trk_dir=f'{root_dir}/data/raw/tracts_trk', 
                        dsi_studio_path = '/Applications/dsi_studio.app/Contents/MacOS/dsi_studio',
                        fib_file = f'{root_dir}/data/raw/tracts_trk/HCP1065.1mm.fib.gz',
                        output_dir = f'{root_dir}/results/examples', # custom output directory
                        )

# ------------------------------------------------------------
# View available tracts
# ------------------------------------------------------------
available_tracts = viz.get_available_tracts()
print(f"Available tracts: {available_tracts}")

# ------------------------------------------------------------
# Validate tract names (useful for checking if the tract names you're providing are valid)
# ------------------------------------------------------------
print(f"Validated tract names: {viz.validate_tract_names(['AF_L', 'IFOF_L', 'weird_name'])}")

# ------------------------------------------------------------
# EXAMPLE 1: Simple single color for specified tracts
# ------------------------------------------------------------
print("Creating single-color visualization...")

viz.visualize_tracts(
    tract_list=['AF_L', 'IFOF_L'],
    single_color='steelblue',
    output_name='steelblue_tracts',
    colorbar=True # this will add the tract name below the figure
)

viz.visualize_tracts(
    tract_list=['AF', 'IFOF'], # with will plot left and right tracts iteratively
    single_color='steelblue',
    output_name='steelblue_tracts',
    colorbar=True 
)

# ------------------------------------------------------------
# EXAMPLE 2: Visualize tracts with your data values
# ------------------------------------------------------------

# Your data: tract names and some metric (e.g., Gini coefficient)
my_data = pd.DataFrame({
    'Tract_Short_Name': ['AF_L', 'IFOF_L', 'SLF_I_L', 'UF_L'],
    'My_Metric': [0.8, 0.3, 0.6, 0.4]
})

# Create one image per tract, colored based on a metric
viz.visualize_tracts(
    tract_df=my_data, 
    values_column='My_Metric',
    plot_mode='iterative',
    color_scheme='viridis',
    output_name='my_tracts_viridis',
    colorbar=True
)

# create one image per tract, with both lateral and medial views
viz.visualize_tracts(
    tract_df=my_data, 
    values_column='My_Metric',
    plot_mode='iterative',
    color_scheme='viridis',
    grid_orientation='vertical', # or 'horizontal'
    output_name='my_tracts_viridis',
    colorbar=True
)

# Create one image for all tracts in the dataframe, colored based on a metric
viz.visualize_tracts(
    tract_df=my_data, 
    values_column='My_Metric',
    plot_mode='all_tracts',
    color_scheme='viridis',
    output_name='my_tracts_viridis',
    colorbar=True
)

# ------------------------------------------------------------
# EXAMPLE 3: All tracts with custom colors from a dataframe
# ------------------------------------------------------------

# Create initial dataframe
all_tracts_data = pd.DataFrame({
    'Tract_Short_Name': available_tracts, # this can be any list of tracts; names should match one of the columns in the abbreviations.xlsx file
    'My_Metric': np.random.rand(len(available_tracts)),
})

# Sort the dataframe by the metric column 
all_tracts_data = all_tracts_data.sort_values(by='My_Metric', ascending=False).reset_index(drop=True)

# Generate colors AFTER sorting, based on the sorted metric values
cmap = plt.cm.get_cmap('magma')
# Normalize the sorted metric values to 0-1 range for the colormap
min_val, max_val = all_tracts_data['My_Metric'].min(), all_tracts_data['My_Metric'].max()
if max_val > min_val:  # Avoid division by zero
    normalized_values = (all_tracts_data['My_Metric'] - min_val) / (max_val - min_val)
else:
    normalized_values = np.ones(len(all_tracts_data))  # All same value

# Generate colors based on the actual metric values
actual_colors = [cmap(val)[:3] for val in normalized_values]  # RGB tuples (0-1 range)
# Note: colors can also be matplotlib color names or hex colors

# Add the colors column (now properly aligned with sorted order)
all_tracts_data['Colors'] = actual_colors

# Use the DataFrame with custom colors
viz.visualize_tracts(
    tract_df=all_tracts_data,
    tract_name_column='Tract_Short_Name',
    values_column='My_Metric',
    color_column='Colors',  # Colors now match the sorted metric values
    plot_mode='all_tracts',
    output_name='all_tracts_custom_colors',
    colorbar=True
)

# ------------------------------------------------------------
# EXAMPLE 4: Gradient plot individual tracts in specified order
# ------------------------------------------------------------

viz.visualize_tracts(
    tract_df=all_tracts_data,
    values_column='My_Metric',
    tract_gradient_plot=True,
    gradient_n_tracts=4,  # Show 4 evenly spaced tracts based on My_Metric
    color_scheme='viridis',
    output_name='gradient',
    colorbar=True
)


