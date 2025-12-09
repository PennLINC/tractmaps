###################################################
#### UTILITY FUNCTIONS ####
###################################################

# This script contains general utility functions for loading, saving and plotting data.

###################################################

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib

## Save data as in https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin/parspin/utils.py#L81
def save_dir(fname, data, overwrite=True):
    """
    Saves `data` to `fname`, creating any necessary intermediate directories

    Parameters
    ----------
    fname : str or os.PathLike
        Output filename for `data`
    data : array_like
        Data to be saved to disk
    """

    fname = Path(fname).resolve()
    fname.parent.mkdir(parents=True, exist_ok=True)
    fmt = '%.10f' if data.dtype.kind == 'f' else '%d'
    if fname.exists() and not overwrite:
        warnings.warn(f'{fname} already exists; not overwriting')
        return
    np.savetxt(fname, data, delimiter=',', fmt=fmt)

### Save a dictionary
# from https://github.com/VinceBaz/bazinet_assortativity
def save_data(data, path):
    '''
    Utility function to save pickled dictionary containing the data used in
    these experiments.

    Parameters
    ----------
    data: dict
        Dictionary storing the data that we want to save.
    path: str
        path of the pickle file
    '''

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

### Load a dictionary ###
# from https://github.com/VinceBaz/bazinet_assortativity
def load_data(path):
    '''
    Utility function to load pickled dictionary containing the data used in
    these experiments.

    Parameters
    ----------
    path: str
        File path to the pickle file to be loaded.

    Returns
    -------
    data: dict
        Dictionary containing the data used in these experiments
    '''

    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data

def make_colormaps():
    """
    Create custom colormaps for brain visualization.
    
    Based on Golia Shafiei's code: https://github.com/netneurolab/shafiei_megdynamics/blob/main/code/analysis/fcn_megdynamics.py
    
    Returns
    -------
    warm_cmap : LinearSegmentedColormap
        Warm half of diverging color palette (continuous)
    tract_cmap : ListedColormap  
        Tract colormap for continuous data visualization
    categ_warm : ListedColormap
        Categorical warm colormap with 6 discrete colors
    cool_warm_cmap : ListedColormap
        Full cool-warm diverging colormap (continuous)
    categ_cool_warm : ListedColormap
        Categorical cool-warm colormap with 6 discrete colors  
    bppy_cmap : ListedColormap
        Custom FDS colormap (blue-purple-pink-yellow gradient)
    """
    
    # Helper function to create categorical colormap from continuous colormap
    def create_categorical_cmap(base_cmap, n_colors=6):
        """Create categorical colormap with discrete color blocks."""
        # Sample colors from the base colormap at evenly spaced intervals
        colors_disc = np.vstack([np.array(base_cmap(i / (n_colors - 1))[:3]) for i in range(n_colors)])
        
        # Create blocks with sizes that add up to 256 (matching original implementation)
        # Original used: 43, 43, 43, 43, 42, 42 for 6 colors
        if n_colors == 6:
            block_sizes = [43, 43, 43, 43, 42, 42]
        else:
            # General case: distribute 256 colors as evenly as possible
            block_size = 256 // n_colors
            remainder = 256 % n_colors
            block_sizes = [block_size + (1 if i < remainder else 0) for i in range(n_colors)]
        
        color_blocks = []
        for i, color in enumerate(colors_disc):
            color_blocks.append(np.ones((block_sizes[i], 3)) * color)
        
        return ListedColormap(np.vstack(color_blocks))
    
    # 1. Warm colormap (warm half of diverging palette)
    cool_warm_palette = sns.diverging_palette(h_neg=264, h_pos=360, s=75, l=50, 
                                             sep=1, n=15, center='light', as_cmap=False)
    warm_cmap = LinearSegmentedColormap.from_list('warm_cmap', 
                                                 cool_warm_palette[len(cool_warm_palette) // 2:])
    
    # 2. Tract colormap (ListedColormap version of warm_cmap for continuous data)
    warm_colors = np.vstack([np.array(warm_cmap(i)[:3]) for i in range(256)])
    tract_cmap = ListedColormap(warm_colors)
    
    # 3. Categorical warm colormap (6 discrete colors)
    categ_warm = create_categorical_cmap(warm_cmap)
    
    # 4. Full cool-warm diverging colormap
    cool_warm_diverging = sns.diverging_palette(h_neg=264, h_pos=360, s=75, l=50, 
                                               sep=1, n=15, center='light', as_cmap=True)
    cool_warm_colors = np.vstack([np.array(cool_warm_diverging(i)[:3]) for i in range(256)])
    cool_warm_cmap = ListedColormap(cool_warm_colors)
    
    # 5. Categorical cool-warm colormap (6 discrete colors)
    categ_cool_warm = create_categorical_cmap(cool_warm_diverging)
    
    # 6. blue-purple-pink-yellow colormap 
    bppy_colors = ["#648FFF", "#785EF0", "#EF63BF", "#FFB000"]  # blue, purple, pink, yellow
    bppy_base = LinearSegmentedColormap.from_list('fds_map', bppy_colors, N=256)
    bppy_colors_array = np.vstack([np.array(bppy_base(i)[:3]) for i in range(256)])
    bppy_cmap = ListedColormap(bppy_colors_array)
    
    return warm_cmap, tract_cmap, categ_warm, cool_warm_cmap, categ_cool_warm, bppy_cmap

### Function to plot brain data on Conte69 atlas (modified from netneurotools) ###
def plot_conte69(data, lhlabel, rhlabel, surf='midthickness',
                 vmin=None, vmax=None, colormap='viridis', customcmap=None,
                 colorbar=True, num_labels=4, orientation='horizontal',
                 colorbartitle=None, backgroundcolor=(1, 1, 1), foregroundcolor=(0, 0, 0), 
                 tractdata=None, tracts=None, connection_threshold=0.95, regions=None, 
                 hemisphere=None,
                 delineate_contours=False,
                 contour_color=(0, 0, 0),
                 contour_linewidth=1.0,
                 contour_opacity=0.6,
                 contour_max_edges=None,
                 **kwargs):

    """
    Plots surface `data` on Conte69 Atlas

    (This is a modified version of plotting.plot_conte69 from netneurotools 
    based on Golia Shafiei's code: https://github.com/netneurolab/shafiei_megdynamics/blob/main/code/analysis/fcn_megdynamics.py.
    The function additionally has been edited to provide the option of plotting a subset of brain regions.)

    Parameters
    ----------
    data : (N,) array_like
        Surface data for N parcels
    lhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the left hemisphere
    rhlabel : str
        Path to .gii file (generic GIFTI file) containing labels to N/2 parcels
        on the right hemisphere
    surf : {'midthickness', 'inflated', 'vinflated'}, optional
        Type of brain surface. Default: 'midthickness'
    vmin : float, optional
        Minimum value to scale the colormap. If None, the min of the data will
        be used. Default: None
    vmax : float, optional
        Maximum value to scale the colormap. If None, the max of the data will
        be used. Default: None
    colormap : str, optional
        Any colormap from matplotlib. Default: 'viridis'
    customcmap : matplotlib.colors.Colormap, optional
        Custom colormap for displaying brain regions. Default is None.
    colorbar : bool, optional
        Wheter to display a colorbar. Default: True
    num_labels : int, optional
        The number of labels to display on the colorbar.
        Available only if colorbar=True. Default: 4
    orientation : str, optional
        Defines the orientation of colorbar. Can be 'horizontal' or 'vertical'.
        Available only if colorbar=True. Default: 'horizontal'
    colorbartitle : str, optional
        The title of colorbar. Available only if colorbar=True. Default: None
    backgroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the background color. Default: (1, 1, 1)
    foregroundcolor : tuple of float values with RGB code in [0, 1], optional
        Defines the foreground color (e.g., colorbartitle color).
        Default: (0, 0, 0)
    tractdata : pd.DataFrame, optional
        DataFrame containing tracts and Glasser parcellation region IDs. Default: None.
    tracts : str or list of str, optional
        Name(s) of the tract(s) for which structurally connected brain regions should be displayed. Default: None.
    connection_threshold : float, optional
        Threshold for selecting brain regions based on structural connectivity. Default: 0.95.
    regions : list of int, optional
        List of region IDs to plot. Default: None.
    kwargs : key-value mapping
        Keyword arguments for `mayavi.mlab.triangular_mesh()`

    Returns
    -------
    scene : mayavi.Scene
        Scene object containing plot(s)
    """

    from netneurotools.datasets import fetch_conte69
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError('Cannot use plot_conte69() if mayavi is not '
                          'installed. Please install mayavi and try again.')

    opts = dict()
    opts.update(**kwargs)

    try:
        surface = fetch_conte69()[surf]
    except KeyError:
        raise ValueError('Provided surf "{}" is not valid. Must be one of '
                         '[\'midthickness\', \'inflated\', \'vinflated\']'
                         .format(surf))
    lhsurface, rhsurface = [nib.load(s) for s in surface]

    lhlabels = nib.load(lhlabel).darrays[0].data
    rhlabels = nib.load(rhlabel).darrays[0].data
    lhvert, lhface = [d.data for d in lhsurface.darrays] # vertex (32492, 3) and face (64980, 3) coordinates
    rhvert, rhface = [d.data for d in rhsurface.darrays]

    # add NaNs for subcortex
    data = np.append(np.nan, data)

    # get lh and rh vertex-level data
    lhdata = np.squeeze(data[lhlabels.astype(int)])
    rhdata = np.squeeze(data[rhlabels.astype(int)])
  
    # if tracts and tractdata are provided, subset the regions
    if tracts is not None and tractdata is not None:
        if isinstance(tracts, str):
            tracts = [tracts]
        region_ids_list = []
        for tract in tracts:
            # select brain regions structurally connected to the tract
            tract_df = tractdata.loc[tractdata[tract] >= connection_threshold, 'regionID']
            region_ids = np.array(tract_df.astype(int))
            region_ids_list.extend(region_ids)
        
        # select unique region IDs
        unique_region_ids = sorted(list(set(region_ids_list)))
      
        # if a regions list of integers is provided, subset regions from unique_region_ids that are in regions
        if regions is not None:
            unique_region_ids = [region_id for region_id in unique_region_ids if region_id in regions]

        # mask regions not connected above the threshold
        lh_mask = np.isin(lhlabels, unique_region_ids)
        rh_mask = np.isin(rhlabels, unique_region_ids)
        
        lhdata = np.where(lh_mask, lhdata, np.nan)
        rhdata = np.where(rh_mask, rhdata, np.nan)
    
    else:
        # No subset provided: set mask to None to delineate all parcel boundaries
        lh_mask = None
        rh_mask = None

    plots = []

    # plot left hemisphere
    lhplot = mlab.figure()
    lhmesh = mlab.triangular_mesh(lhvert[:, 0], lhvert[:, 1], lhvert[:, 2], lhface, figure=lhplot, 
                                    colormap=colormap, mask=np.isnan(lhdata), scalars=lhdata, 
                                    vmin=vmin, vmax=vmax, **opts)
    lhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863, 0.863, 1]
    lhmesh.update_pipeline()
    if not customcmap == None and type(customcmap) != str:
        lut = lhmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = customcmap.colors * 255
        lhmesh.module_manager.scalar_lut_manager.lut.table = lut
        mlab.draw()
    if colorbar is True:
        mlab.colorbar(object = lhmesh, title=colorbartitle, nb_labels=num_labels, orientation=orientation)
    plots.append(lhplot)
    
    # plot right hemisphere
    rhplot = mlab.figure()
    rhmesh = mlab.triangular_mesh(rhvert[:, 0], rhvert[:, 1], rhvert[:, 2], rhface, figure=rhplot, 
                                colormap=colormap, mask=np.isnan(rhdata), scalars=rhdata,
                                vmin=vmin, vmax=vmax, **opts)
    rhmesh.module_manager.scalar_lut_manager.lut.nan_color = [0.863, 0.863, 0.863, 1]
    rhmesh.update_pipeline()
    if not customcmap == None and type(customcmap) != str:
        lut = rhmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = customcmap.colors * 255
        rhmesh.module_manager.scalar_lut_manager.lut.table = lut
        mlab.draw()
    if colorbar is True:
        mlab.colorbar(object = rhmesh, title=colorbartitle, nb_labels=num_labels, orientation=orientation)
    plots.append(rhplot)
    
    for plot in plots:  
        mlab.figure(bgcolor=backgroundcolor, fgcolor=foregroundcolor, figure=plot)

    return plots

### Function to assemble plots generated by plot_conte69 in a 2x2 grid ###
def conte69_plot_grid(data, lhlabel, rhlabel, surf='midthickness',
                        vmin=None, vmax=None, colormap='viridis', customcmap=None,
                        shared_colorbar=True, subplot_colorbar=False, colorbartitle=None, 
                        backgroundcolor=(1, 1, 1), foregroundcolor=(0, 0, 0), 
                        outpath='./Glasser360.png', title=None,
                        tractdata=None, tracts=None, connection_threshold=0.95, 
                        regions=None, hemisphere=None,
                        delineate_contours=False,
                        contour_color=(0, 0, 0),
                        contour_linewidth=1.0,
                        contour_opacity=0.6,
                        contour_max_edges=None,
                        fontsize=28,
                        **kwargs):
                    
    """
    Creates a 2x2 grid of brain plots using plot_conte69 and adds a colorbar below.
    
    Parameters
    ----------
    data : array-like
        Data to plot on the brain surface.
    lhlabel, rhlabel : str
        Paths to the GIFTI label files for left and right hemispheres.
    surf : {'midthickness', 'inflated', 'vinflated'}, optional
        Type of brain surface. Default is 'midthickness'.
    vmin, vmax : float, optional
        Min and max values for color scaling.
    colormap : str, optional
        Colormap for the brain surface plots.
    customcmap : matplotlib.colors.Colormap, optional
        Custom colormap for displaying brain regions. Default is None.
    shared_colorbar : bool, optional
        Whether to include a shared colorbar. Default is True.
    subplot_colorbar : bool, optional
        Whether to include a colorbar in each subplot. Default is False.
    colorbartitle : str, optional
        Title for the colorbar.
    backgroundcolor : tuple, optional
        Background color for the plots.
    foregroundcolor : tuple, optional
        Foreground color (e.g., for the colorbar title).
    outpath : str, optional
        Path and filename to save the output figure. Default is './Glasser360.png'.
    title: str, optional
        Shared title for the plot grid.
    tractdata : pd.DataFrame, optional
        DataFrame containing tracts and Glasser parcellation region IDs. Default is None.
    tracts : str or list of str, optional
        Name(s) of the tract(s) for which structurally connected brain regions should be displayed. Default is None.
    connection_threshold : float, optional
        Threshold for selecting brain regions based on structural connectivity. Default is 0.95.
    regions : list of int, optional
        List of region IDs to plot. Default is None.
    hemisphere : {'left', 'right', None}, optional 
        If specified, plots only the left or right hemisphere. Default is None (both hemispheres are plotted).
    fontsize : int, optional
        Font size for all text elements in the figure (title, colorbar label, colorbar ticks).
        Default is 28.
    
    Returns
    -------
    Saves the final image with the grid and colorbar to the specified outpath and shows the gridplot.
    """
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError('Cannot use create_brain_plot_grid() if mayavi is not '
                          'installed. Please install mayavi and try again.')
    
    # Set font to Arial for all text elements
    plt.rcParams['font.family'] = 'Arial'
        
    # arguments for plot_conte69
    plot_args = {
        'data': data,
        'lhlabel': lhlabel,
        'rhlabel': rhlabel,
        'surf': surf,
        'vmin': vmin,
        'vmax': vmax,
        'colormap': colormap,
        'customcmap': customcmap,
        'colorbar': False,  # no colorbar in individual plots; shared colorbar is added later if desired
        'backgroundcolor': backgroundcolor,
        'foregroundcolor': foregroundcolor,
        'delineate_contours': delineate_contours,
        'contour_color': contour_color,
        'contour_linewidth': contour_linewidth,
        'contour_opacity': contour_opacity,
        'contour_max_edges': contour_max_edges,
    }
    # add tract-related arguments if they are provided
    if tractdata is not None and tracts is not None:
        plot_args.update({
            'tractdata': tractdata,
            'tracts': tracts,
            'connection_threshold': connection_threshold
        })
    if regions is not None:
        plot_args.update({'regions': regions})

    # plot brains and capture screenshots
    brains = plot_conte69(**plot_args)
    
    # lateral view
    mlab.view(azimuth=180, elevation=90, distance=450, figure=brains[0])
    lh_lateral = mlab.screenshot(figure=brains[0], mode='rgba', antialiased=True)
    mlab.view(azimuth=180, elevation=-90, distance=450, figure=brains[1])
    rh_lateral = mlab.screenshot(figure=brains[1], mode='rgba', antialiased=True)

    # medial view
    mlab.view(azimuth=0, elevation=90, distance=450, figure=brains[0]) 
    mlab.view(azimuth=0, elevation=-90, distance=450, figure=brains[1])
    lh_medial = mlab.screenshot(figure=brains[0], mode='rgba', antialiased=True)
    rh_medial = mlab.screenshot(figure=brains[1], mode='rgba', antialiased=True)

    mlab.close(all=True)

    # show each view
    if hemisphere == 'left':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(lh_lateral)
        axes[1].imshow(lh_medial)
        if shared_colorbar is True:
            cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.05])  # single colorbar axis
    elif hemisphere == 'right':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(rh_lateral)
        axes[1].imshow(rh_medial)
        if shared_colorbar is True:
            cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.05])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(lh_lateral)
        axes[0, 1].imshow(rh_lateral)
        axes[1, 0].imshow(lh_medial)
        axes[1, 1].imshow(rh_medial)
        if shared_colorbar is True:
            cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.05])

    # remove subplot axes
    for ax in axes.flat:
        ax.axis('off')

    # reduce the spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=-0.2, hspace=-0.2)
    
    # add a shared colorbar if desired
    if shared_colorbar is True:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=customcmap if customcmap is not None else colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.outline.set_visible(False)
       
        # set ticks on the colorbar
        ticks = np.linspace(norm.vmin, norm.vmax, 4)
        cbar.set_ticks(ticks)
        
        # Automatically determine appropriate decimal precision based on value range
        value_range = norm.vmax - norm.vmin
        if value_range <= 1:
            # For small ranges (like 0-1), use 1 decimal place
            tick_labels = [f'{tick:.1f}' for tick in ticks]
        else:
            # For larger ranges, use integers
            tick_labels = [f'{tick:.0f}' for tick in ticks]
        
        cbar.ax.set_xticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=fontsize)

        # add a colorbar title if desired
        cbar.set_label(colorbartitle, color=foregroundcolor, fontsize=fontsize)
    
    # save and show the figure
    plt.suptitle(title, fontsize=fontsize)
    plt.savefig(outpath)
    plt.show()


def perm_corr_test(x, y, n_permutations=10000, method='pearson', 
                               alternative='two-sided', random_state=None):
    """
    Perform permutation testing for correlation between two variables.
    
    This function computes the observed correlation between x and y, then
    performs permutation testing by randomly shuffling one of the variables
    to generate a null distribution of correlation coefficients.
    
    Parameters
    ----------
    x : array_like
        First variable for correlation testing
    y : array_like
        Second variable for correlation testing
    n_permutations : int, optional
        Number of permutations to perform. Default: 10000
    method : str, optional
        Correlation method to use. Options: 'pearson', 'spearman', 'kendall'.
        Default: 'pearson'
    alternative : str, optional
        Alternative hypothesis. Options: 'two-sided', 'greater', 'less'.
        Default: 'two-sided'
    random_state : int or None, optional
        Random state for reproducibility. Default: None
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'observed_corr': Observed correlation coefficient
        - 'p_value': P-value from permutation test
        - 'permuted_corrs': Array of permuted correlation coefficients
        - 'n_permutations': Number of permutations performed
        - 'method': Correlation method used
        - 'alternative': Alternative hypothesis tested
    """
    
    import numpy as np
    from scipy import stats
    
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        raise ValueError("At least 3 non-NaN pairs are required for correlation testing")
    
    # Compute observed correlation
    if method == 'pearson':
        observed_corr, _ = stats.pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        observed_corr, _ = stats.spearmanr(x_clean, y_clean)
    elif method == 'kendall':
        observed_corr, _ = stats.kendalltau(x_clean, y_clean)
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
    
    # Initialize array to store permuted correlations
    permuted_corrs = np.zeros(n_permutations)
    
    # Perform permutations with progress bar
    from tqdm import tqdm
    for i in tqdm(range(n_permutations), desc="Permutation testing", unit="perm"):
        # Shuffle y values while keeping x values fixed
        y_permuted = np.random.permutation(y_clean)
        
        # Compute correlation for this permutation
        if method == 'pearson':
            permuted_corrs[i], _ = stats.pearsonr(x_clean, y_permuted)
        elif method == 'spearman':
            permuted_corrs[i], _ = stats.spearmanr(x_clean, y_permuted)
        elif method == 'kendall':
            permuted_corrs[i], _ = stats.kendalltau(x_clean, y_permuted)
    
    # Compute p-value based on alternative hypothesis
    if alternative == 'two-sided':
        # Count permutations where absolute correlation is >= observed absolute correlation
        count = np.sum(np.abs(permuted_corrs) >= np.abs(observed_corr))
    elif alternative == 'greater':
        # Count permutations where correlation is >= observed correlation
        count = np.sum(permuted_corrs >= observed_corr)
    elif alternative == 'less':
        # Count permutations where correlation is <= observed correlation
        count = np.sum(permuted_corrs <= observed_corr)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    # Ensure p-value is not exactly 0
    p_value = (count + 1) / (n_permutations + 1)
    
    # Print results
    print(f"Permutation correlation test results:")
    print(f"  r = {observed_corr:.4f}")
    print(f"  p = {p_value:.4f}")
    print(f"  n = {len(x_clean)}")
    print(f"  method = {method}")
    print(f"  alternative = {alternative}")
    print(f"  n_permutations = {n_permutations}")
    
    return {
        'observed_corr': observed_corr,
        'p_value': p_value,
        'permuted_corrs': permuted_corrs,
        'n_permutations': n_permutations,
        'method': method,
        'alternative': alternative,
        'n_samples': len(x_clean)
    }


def plot_correlation(x, y, 
                    corr_value, p_value,
                    x_label="X Variable", y_label="Y Variable",
                    color_scheme=None, reverse_colormap=False,
                    colorbar='same_plot', colorbar_label=None,
                    color_by='y', colorbar_filename=None,
                    significance_data=None, significance_threshold=0.05, significance_method='below',
                    point_labels=None, 
                    text_box_position='top_right',
                    figure_size=None,
                    figure_size_mm=None,
                    point_size=30, point_alpha=0.8,
                    regression_line=True,
                    title=None,
                    output_path=None,
                    dpi=300,
                    return_figure=False,
                    colorbar_tick_interval=0.1):
    """
    Create a correlation scatter plot with customizable styling and colorbar options.
    
    Parameters
    ----------
    x : array_like
        X-axis data values
    y : array_like  
        Y-axis data values
    corr_value : float
        Pre-computed correlation coefficient
    p_value : float
        Pre-computed p-value for the correlation
    x_label : str, optional
        Label for x-axis. Default: "X Variable"
    y_label : str, optional
        Label for y-axis. Default: "Y Variable"
    color_scheme : matplotlib.colors.Colormap or str, optional
        Colormap to use for coloring points. If None, uses fds_cmap from make_colormaps().
        Can be colormap object or string name. Default: None (uses fds_cmap)
    reverse_colormap : bool, optional
        Whether to reverse the colormap. Default: False
    colorbar : {'same_plot', 'separate_figure', 'none'}, optional
        Where to place the colorbar:
        - 'same_plot': Add colorbar to the same figure (default)
        - 'separate_figure': Create colorbar in a separate figure
        - 'none': No colorbar
        Default: 'same_plot'
    colorbar_label : str, optional
        Label for the colorbar. If None and colorbar is requested, uses appropriate label
        based on color_by parameter. Default: None
    color_by : {'x', 'y'}, optional
        Which variable to use for coloring points and colorbar range.
        'x': Color by x-values (useful when x is the property of interest)
        'y': Color by y-values (default behavior)
        Default: 'y'
    colorbar_filename : str, optional
        Custom filename for the separate colorbar (when colorbar='separate_figure').
        If None, uses the output_path with '_colorbar' suffix. Should include file extension.
        Default: None
    significance_data : array_like, optional
        Array of significance values (e.g., p-values) for each point. If provided,
        non-significant points will be colored gray and significant points will use
        the main color scheme. Default: None
    significance_threshold : float, optional
        Threshold for determining significance. Default: 0.05
    significance_method : {'below', 'above'}, optional
        Method for determining significance:
        'below': Values below threshold are significant (for p-values)
        'above': Values above threshold are significant (for other metrics)
        Default: 'below'
    point_labels : array_like, optional
        Labels for individual points (e.g., tract names). If provided, will add
        non-overlapping text annotations. Default: None
    text_box_position : {'top_right', 'top_left', 'bottom_right', 'bottom_left'}, optional
        Position for correlation statistics text box. Default: 'top_right'
    figure_size : tuple, optional
        Figure size as (width, height) in inches. If None and figure_size_mm is None, 
        automatically determined based on colorbar setting: (7, 8) with colorbar, (7, 6) without. 
        Default: None
    figure_size_mm : tuple, optional
        Figure size as (width, height) in millimeters. If provided, takes precedence over 
        figure_size. This allows specifying exact physical dimensions for InDesign panels.
        Default: None
    point_size : float, optional
        Size of scatter plot points. Default: 100
    point_alpha : float, optional
        Transparency of scatter plot points (0-1). Default: 0.8
    regression_line : bool, optional
        Whether to add a regression line. Default: True
    title : str, optional
        Plot title. Default: None
    output_path : str, optional
        Path to save the plot. If None, plot is not saved. Default: None
    dpi : int, optional
        Resolution for saved figure. Default: 300
    return_figure : bool, optional
        Whether to return the figure object instead of showing it. Default: False
    colorbar_tick_interval : float, optional
        Interval for colorbar ticks (e.g., 0.1 for ticks every 0.1, 0.2 for ticks every 0.2).
        Only applies when colorbar='separate_figure'. Default: 0.1
    
    Returns
    -------
    matplotlib.figure.Figure or None
        If return_figure=True, returns the figure object. If colorbar='separate_figure',
        returns a tuple of (main_figure, colorbar_figure). Otherwise returns None.
    
    Examples
    --------
    # Basic correlation plot (colors points by y-values)
    plot_correlation(x_data, y_data, r_value, p_value, 
                    x_label="S-A Range", y_label="Gini Coefficient")
    
    # Plot with significance-based coloring and custom colorbar filename
    plot_correlation(x_data, y_data, r_value, p_value,
                    x_label="Tract Property", y_label="Behavior Score",
                    color_by='x', reverse_colormap=True, colorbar='separate_figure',
                    colorbar_filename="tract_property_colorbar.svg",
                    significance_data=p_values, significance_threshold=0.05,
                    point_labels=tract_names)
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    try:
        from .figure_formatting import setup_figure, save_figure
    except ImportError:
        # Fallback for when called as a script
        from figure_formatting import setup_figure, save_figure
    
    # Convert to numpy arrays and handle NaN values
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Handle significance data if provided
    if significance_data is not None:
        significance_data = np.asarray(significance_data)
        
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    if significance_data is not None:
        mask = mask & ~np.isnan(significance_data)
        significance_clean = significance_data[mask]
    else:
        significance_clean = None
        
    x_clean = x[mask]
    y_clean = y[mask]
    
    if point_labels is not None:
        point_labels = np.asarray(point_labels)[mask]
    
    if len(x_clean) < 2:
        raise ValueError("At least 2 non-NaN data points required for plotting")
    
    # Set up colormap
    if color_scheme is None:
        # Use fds_cmap from make_colormaps
        _, _, _, _, _, color_scheme = make_colormaps()
    elif isinstance(color_scheme, str):
        color_scheme = plt.cm.get_cmap(color_scheme)
    
    if reverse_colormap:
        color_scheme = color_scheme.reversed()
    
    # Determine figure size: use figure_size_mm if provided, otherwise figure_size, otherwise defaults
    if figure_size_mm is not None:
        # Use mm directly
        width_mm, height_mm = figure_size_mm
    elif figure_size is not None:
        # Convert from inches to mm
        width_mm = figure_size[0] * 25.4
        height_mm = figure_size[1] * 25.4
    else:
        # Use defaults based on colorbar setting
        if colorbar == 'same_plot':
            width_mm = 7 * 25.4  # 177.8 mm
            height_mm = 8 * 25.4  # 203.2 mm
        else:
            width_mm = 7 * 25.4  # 177.8 mm
            height_mm = 7 * 25.4  # 177.8 mm
    
    # Create main figure using figure formatting utilities
    # Adjust margins if colorbar is on same plot to ensure it doesn't extend beyond plot
    # margins_mm format: (left, right, bottom, top)
    # Increase left margin when significance_data is used to prevent y-axis label cutoff
    if colorbar == 'same_plot':
        if significance_data is not None:
            margins_mm = (14, 10, 8, 4)  # left, right, bottom, top 
        else:
            margins_mm = (11, 10, 8, 4)  # left, right, bottom, top
    else:
        if significance_data is not None:
            margins_mm = (14, 12, 10, 4)  # left, right, bottom, top
        else:
            margins_mm = (11, 12, 10, 4)  
    fig, ax = setup_figure(width_mm=width_mm, height_mm=height_mm, 
                                  margins_mm=margins_mm)
    
    # Add regression line if requested
    if regression_line:
        sns.regplot(x=x_clean, y=y_clean, scatter=False,
                   line_kws={'color': 'grey', 'alpha': 0.7, 'linewidth': 0.5}, ax=ax)
    
    # Determine color values based on color_by parameter
    if color_by == 'x':
        color_values = x_clean
        if colorbar_label is None:
            colorbar_label = x_label
    else:  # color_by == 'y'
        color_values = y_clean
        if colorbar_label is None:
            colorbar_label = y_label
    
    # Create scatter plot with optional significance-based coloring
    if significance_clean is not None:
        # Create significance mask
        if significance_method == 'below':
            significant_mask = significance_clean < significance_threshold
        else:  # significance_method == 'above'
            significant_mask = significance_clean > significance_threshold
        
        # Plot non-significant points first (gray)
        if (~significant_mask).sum() > 0:
            scatter_ns = ax.scatter(x_clean[~significant_mask], y_clean[~significant_mask], 
                                 c='lightgrey', alpha=0.6, s=point_size*0.8, zorder=4, 
                                 edgecolors='none', label='Non-significant')
        
        # Plot significant points with color scheme
        if significant_mask.sum() > 0:
            scatter = ax.scatter(x_clean[significant_mask], y_clean[significant_mask], 
                               c=color_values[significant_mask], cmap=color_scheme,
                               s=point_size, alpha=point_alpha, zorder=5, 
                               edgecolors='none', label='Significant')
        else:
            # If no significant points, create dummy scatter for colorbar
            scatter = ax.scatter([], [], c=[], cmap=color_scheme, s=point_size, edgecolors='none')
    else:
        # No significance data - plot all points with color scheme
        scatter = ax.scatter(x_clean, y_clean, c=color_values, cmap=color_scheme,
                            s=point_size, alpha=point_alpha, zorder=5, edgecolors='none')
    
    # Format p-value text
    if p_value < 0.001:
        p_text = '$\\it{{p}}$ < 0.001'
    else:
        p_text = f'$\\it{{p}}$ = {p_value:.3f}'
    
    # Add correlation statistics text box
    text_positions = {
        'top_right': (0.95, 0.95, 'right', 'top'),
        'top_left': (0.05, 0.95, 'left', 'top'), 
        'bottom_right': (0.95, 0.05, 'right', 'bottom'),
        'bottom_left': (0.05, 0.05, 'left', 'bottom')
    }
    
    if text_box_position in text_positions:
        x_pos, y_pos, ha, va = text_positions[text_box_position]
    else:
        x_pos, y_pos, ha, va = text_positions['top_right']
    
    # Use font size from rcParams (set by setup_figure)
    # Use label_pt size for text annotations
    fontsize = plt.rcParams.get('axes.labelsize', 7) # default font size is 7pt
    ax.text(x_pos, y_pos, f'$\\it{{r}}$ = {corr_value:.3f}\n{p_text}',
            transform=ax.transAxes, fontsize=fontsize,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='0.7', linewidth=0.5, alpha=0.9),
            va=va, ha=ha)
    
    # Add point labels if provided (non-overlapping)
    if point_labels is not None:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        kept_bboxes = []
        
        for i, label in enumerate(point_labels):
            if pd.notna(label):  # Handle potential NaN labels
                display_label = str(label).replace('_', ' ')
                # Use base_pt size from rcParams (set by setup_figure)
                fontsize = plt.rcParams.get('xtick.labelsize', 7) # default font size is 7pt
                txt = ax.text(x_clean[i], y_clean[i], display_label, 
                             fontsize=fontsize, ha='left', va='center')
                # Add offset similar to other plotting functions
                txt.set_position((x_clean[i] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03,
                                y_clean[i]))
                ax.figure.canvas.draw()
                bbox = txt.get_window_extent(renderer=renderer).expanded(1.05, 1.05)
                if any(bbox.overlaps(b) for b in kept_bboxes):
                    txt.remove()
                else:
                    kept_bboxes.append(bbox)
    
    # Handle colorbar
    colorbar_fig = None
    if colorbar == 'same_plot':
        cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', 
                           pad=0.15, fraction=0.08, aspect=20)
        cbar.set_label(colorbar_label, labelpad=4)
        cbar.ax.tick_params(width=0.5, length=2)
        cbar.outline.set_visible(False)
        
    elif colorbar == 'separate_figure':
        # Create separate colorbar figure using figure formatting utilities
        colorbar_width_mm = 40
        # Increase height to accommodate label at top and ticks at bottom
        colorbar_height_mm = 15 
        colorbar_fig, cbar_ax = setup_figure(
            width_mm=colorbar_width_mm, 
            height_mm=colorbar_height_mm,
            margins_mm=(2, 2, 6, 6), # left, right, bottom, top 
        )
        
        # Create dummy mappable for colorbar using the full range of color values
        # (not just significant ones, to maintain consistent colorbar range)
        vmin, vmax = np.min(color_values), np.max(color_values)
        print(f"Colorbar range: {vmin:.3f} to {vmax:.3f} (coloring by {color_by})")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=color_scheme, norm=norm)
        sm.set_array([])
        
        # Create colorbar on the axes
        cbar = colorbar_fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        
        # Set ticks at specified interval
        tick_start = np.ceil(vmin / colorbar_tick_interval) * colorbar_tick_interval
        tick_end = np.floor(vmax / colorbar_tick_interval) * colorbar_tick_interval + colorbar_tick_interval
        tick_values = np.arange(tick_start, tick_end, colorbar_tick_interval)
        cbar.set_ticks(tick_values)
        
        cbar.set_label(colorbar_label, labelpad=4)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(width=0.5, length=2)
        cbar.outline.set_visible(False)
        
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    
    # Styling consistent with other plots
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.tick_params(axis='both', width=0.3)
    
    # Save if path provided
    if output_path:
        save_figure(fig, output_path)
        if colorbar_fig and colorbar == 'separate_figure':
            # Use custom colorbar filename if provided, otherwise use default naming
            if colorbar_filename is not None:
                # If colorbar_filename is relative, put it in the same directory as main plot
                if not os.path.isabs(colorbar_filename):
                    output_dir = os.path.dirname(output_path)
                    colorbar_path = os.path.join(output_dir, colorbar_filename)
                else:
                    colorbar_path = colorbar_filename
            else:
                # Default: use main plot filename with '_colorbar' suffix
                colorbar_path = output_path.replace('.', '_colorbar.')
            
            print(f"Creating colorbar file: {colorbar_path}")
            save_figure(colorbar_fig, colorbar_path)
            print(f"Saved colorbar: {colorbar_path}")
        print(f"Saved plot: {output_path}")
    
    # Return figure(s) or show
    if return_figure:
        if colorbar == 'separate_figure' and colorbar_fig:
            return fig, colorbar_fig
        return fig
    else:
        if colorbar == 'separate_figure' and colorbar_fig:
            # Only show the main plot, colorbar is separate
            plt.figure(fig.number)
            plt.show()
            # Close the colorbar figure to prevent it from showing
            plt.close(colorbar_fig)
        else:
            plt.show()
        
        # Close main figure after showing (only when not returning it)
        plt.close(fig)
    


