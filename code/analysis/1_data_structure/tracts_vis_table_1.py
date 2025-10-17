# ------------------------------------------------------------------------------------------------
# --- Tracts Visualizations for Table 1 ---
# ------------------------------------------------------------------------------------------------
# This script creates lateral views of all left and right tracts for Table 1 visualization.

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.tract_visualizer import TractVisualizer

# ------------------------------------------------------------------------------------------------
# --- Set up paths ---
# ------------------------------------------------------------------------------------------------

root_dir = '/Users/joelleba/PennLINC/tractmaps'

# Create output directory
output_dir = os.path.join(root_dir, 'results/table_1')
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------------------------------------------
# --- Create visualizations for all tracts ---
# ------------------------------------------------------------------------------------------------

viz = TractVisualizer(root_dir=root_dir)

# Get tract abbreviations
abbrev_df = viz._load_abbreviations()
unique_abbreviations = abbrev_df['Abbreviation'].dropna().unique().tolist()

print(f"Visualizing {len(unique_abbreviations)} bilateral tracts")

# Create one image per tract, with both lateral views (left and right) in a horizontal grid
viz.visualize_tracts(
    tract_list=unique_abbreviations, 
    single_color='#626bda',
    plot_mode='iterative',
    grid_orientation='horizontal',
    view='lateral',
    output_name='tract',
    output_dir=output_dir,
    colorbar=False
)

print(f"Images saved to {output_dir}")