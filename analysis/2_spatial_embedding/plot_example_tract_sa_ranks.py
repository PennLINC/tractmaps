# ------------------------------------------------------------------------------------------------
# #### Plot example tracts' S-A ranks on brain surface ####
# ------------------------------------------------------------------------------------------------

# This script plots the full S-A axis and the S-A ranks of example tracts on the brain surface.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils.tract_visualizer import TractVisualizer
from utils import tm_utils

# ------------------------------------------------------------------------------------------------
# --- Set inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# Paths
root = '/Users/joelleba/PennLINC/tractmaps'
results_dir = f'{root}/results/spatial_embedding/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Label files (Conte69)
lhlabels = f'{root}/data/derivatives/glasser_parcellation/HCP_MMP_L.label.gii'
rhlabels = f'{root}/data/derivatives/glasser_parcellation/HCP_MMP_R.label.gii'

# Data paths
tract_probs_path = f'{root}/data/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv'
sa_ranks_path = f'{root}/data/derivatives/glasser_parcellation/glasser_sa_axis_ranks.csv'

# Load data
tracts_df = pd.read_csv(tract_probs_path)
sa_df = pd.read_csv(sa_ranks_path)

# Extract a 360-length S-A ranks series (robust to different column names)
# Prefer a column named like 'rank' or containing 'sa' if available; otherwise first numeric column
# Load S-A axis ranks
sa_series = sa_df['sa_axis'] 

# Colormaps
_, _, _, cool_warm_cmap, _, _ = tm_utils.make_colormaps()

# ------------------------------------------------------------------------------------------------
# --- Plot whole-brain S-A axis ---
# ------------------------------------------------------------------------------------------------

# Global vmin/vmax for consistent color scaling
vmin = float(np.nanmin(sa_series.values))
vmax = float(np.nanmax(sa_series.values))

# Plot the whole-brain S-A axis in coolwarm
tm_utils.conte69_plot_grid(
    data=sa_series,
    lhlabel=lhlabels,
    rhlabel=rhlabels,
    vmin=vmin,
    vmax=vmax,
    surf='inflated',
    customcmap=cool_warm_cmap,
    shared_colorbar=True,
    colorbartitle='S-A rank',
    outpath=f'{results_dir}/brain_sa_axis.svg',
    # title='S-A axis'
)

# ------------------------------------------------------------------------------------------------
# --- Plot example tracts in brain surface ---
# ------------------------------------------------------------------------------------------------

example_tracts = ['IFOF', 'VOF']
connection_threshold = 0.5

for abbr in example_tracts:
    left_name = f'{abbr}_left'
    right_name = f'{abbr}_right'

    tm_utils.conte69_plot_grid(
        data=sa_series,
        lhlabel=lhlabels,
        rhlabel=rhlabels,
        vmin=vmin,
        vmax=vmax,
        surf='inflated',
        customcmap=cool_warm_cmap,
        shared_colorbar=True,
        colorbartitle='S-A rank',
        outpath=f'{results_dir}/brain_sa_ranks_{abbr}.svg',
        # title=f'{abbr}',
        tractdata=tracts_df,
        tracts=[left_name, right_name],
        connection_threshold=connection_threshold,
        hemisphere='left'
    )


# ------------------------------------------------------------------------------------------------
# --- Plot example tracts in glass brain ---
# ------------------------------------------------------------------------------------------------

example_tracts = ['VOF_left', 'IFOF_left']
viz = TractVisualizer(root_dir=root, 
                      output_dir=results_dir)
viz.visualize_tracts(tract_list=example_tracts, 
                    single_color='#626bda', 
                    plot_mode='iterative'
                    )


print(f"Files saved to: {results_dir}")